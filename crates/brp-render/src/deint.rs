//! GPU deinterlacing: NNEDI3, bwdif, and NNEDI3 + motion-adaptive (bwdif temporal
//! check with NNEDI3 as the spatial estimate). Works per plane on planar YUV.

use crate::frames::{FrameKey, GpuFrame};
use crate::gpu::{self, GpuImage, Graveyard, SharedAllocator};
use crate::nnedi::{NnediConfig, NnediWeights};
use anyhow::{Context as _, Result};
use ash::vk;
use brp_media::FrameLayout;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

const NNEDI_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nnedi3.comp.spv"));
const NNEDI_FAST_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/nnedi3_fast.comp.spv"));
/// Shared memory the fast NNEDI3 variant needs (weights + prescreener + tile).
const NNEDI_FAST_SHARED: u32 = (2 * 16 * 128 + 2 * 16 + 280 + 9 * 80) * 4;
const BWDIF_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bwdif.comp.spv"));

const MAX_PLANES: usize = 4;
pub const OUT_FORMAT: vk::Format = vk::Format::R32_SFLOAT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeintMethod {
    /// Fast motion-adaptive (FFmpeg bwdif).
    Bwdif,
    /// Neural field interpolation, intra-field only.
    Nnedi,
    /// NNEDI3 for moving areas, neighbouring fields for static areas. Best quality.
    NnediAdaptive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeintParams {
    pub method: DeintMethod,
    pub tff: bool,
    /// Second field of the frame (field-rate output).
    pub second_field: bool,
    /// Use the lighter NNEDI3 network (16 neurons) to keep up with playback on slow
    /// GPUs. Paused frames and export use the full network.
    pub light: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NnediPush {
    size: [i32; 2],
    keep: i32,
    xdim: i32,
    ydim: i32,
    nns: i32,
    qual2: i32,
    model: i32,
    to255: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BwdifPush {
    size: [i32; 2],
    keep: i32,
    p2: i32,
    use_edi: i32,
    to_code: f32,
    max_code: f32,
}

fn bytes<T>(t: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>()) }
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

struct Slot {
    out: Vec<GpuImage>,
    edi: Vec<GpuImage>,
    nnedi_sets: Vec<vk::DescriptorSet>,
    bwdif_sets: Vec<vk::DescriptorSet>,
    key: Option<(FrameKey, FrameKey, FrameKey, DeintParams)>,
}

struct Weights {
    buffer: vk::Buffer,
    alloc: Option<Allocation>,
    meta: NnediWeights,
}

pub struct Deinterlacer {
    device: ash::Device,
    allocator: SharedAllocator,
    nnedi: Pipe,
    /// Specialised for the default network with weights in shared memory (None if the
    /// GPU has too little shared memory).
    nnedi_fast: Option<Pipe>,
    bwdif: Pipe,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    /// [full network, light network]
    weights: [Option<Weights>; 2],
    weights_error: Option<String>,
    weights_path: Option<std::path::PathBuf>,
    slots: Vec<Slot>,
}

fn make_pipe(device: &ash::Device, spv: &[u8], bindings: &[vk::DescriptorType], push_size: u32) -> Result<Pipe> {
    unsafe {
        let b: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, t)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(*t)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&b), None)?;
        let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_size)];
        let sl = [set_layout];
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&push), None)?;
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
            .map_err(|(_, e)| e)?[0];
        device.destroy_shader_module(module, None);
        Ok(Pipe { pipeline, layout, set_layout })
    }
}

impl Deinterlacer {
    pub fn new(device: &ash::Device, allocator: SharedAllocator, slots: usize, weights_path: Option<std::path::PathBuf>, max_shared: u32, integrated: bool) -> Result<Self> {
        use vk::DescriptorType as D;
        let nb = [D::COMBINED_IMAGE_SAMPLER, D::STORAGE_IMAGE, D::STORAGE_BUFFER];
        let nnedi = make_pipe(device, NNEDI_SPV, &nb, std::mem::size_of::<NnediPush>() as u32)?;
        // The shared-memory variant wins on discrete GPUs; integrated GPUs are ALU-bound
        // here and run the generic one faster (measured on an AMD iGPU).
        let nnedi_fast = if max_shared >= NNEDI_FAST_SHARED && !integrated {
            Some(make_pipe(device, NNEDI_FAST_SPV, &nb, std::mem::size_of::<NnediPush>() as u32)?)
        } else {
            log::info!("NNEDI3: generic shader (integrated GPU or {max_shared} bytes of shared memory)");
            None
        };
        let bwdif = make_pipe(
            device,
            BWDIF_SPV,
            &[D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::STORAGE_IMAGE, D::STORAGE_IMAGE],
            std::mem::size_of::<BwdifPush>() as u32,
        )?;
        let n = (slots * MAX_PLANES) as u32;
        unsafe {
            let sizes = [
                vk::DescriptorPoolSize::default().ty(D::COMBINED_IMAGE_SAMPLER).descriptor_count(n * 4),
                vk::DescriptorPoolSize::default().ty(D::STORAGE_IMAGE).descriptor_count(n * 3),
                vk::DescriptorPoolSize::default().ty(D::STORAGE_BUFFER).descriptor_count(n),
            ];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n * 2).pool_sizes(&sizes), None)?;
            let mut out_slots = Vec::new();
            for _ in 0..slots {
                let nl = vec![nnedi.set_layout; MAX_PLANES];
                let bl = vec![bwdif.set_layout; MAX_PLANES];
                let nnedi_sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&nl))?;
                let bwdif_sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&bl))?;
                out_slots.push(Slot { out: Vec::new(), edi: Vec::new(), nnedi_sets, bwdif_sets, key: None });
            }
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            Ok(Deinterlacer {
                device: device.clone(),
                allocator,
                nnedi,
                nnedi_fast,
                bwdif,
                pool,
                sampler,
                weights: [None, None],
                weights_error: None,
                weights_path,
                slots: out_slots,
            })
        }
    }

    /// Why NNEDI3 is unavailable, if it is.
    pub fn weights_error(&self) -> Option<&str> {
        self.weights_error.as_deref()
    }

    fn ensure_weights(&mut self, light: bool) -> bool {
        let i = light as usize;
        if self.weights[i].is_some() {
            return true;
        }
        if self.weights_error.is_some() {
            return false;
        }
        match self.load_weights(light) {
            Ok(w) => {
                self.weights[i] = Some(w);
                true
            }
            Err(e) => {
                log::error!("NNEDI3 unavailable: {e:#}");
                self.weights_error = Some(format!("{e:#}"));
                false
            }
        }
    }

    fn load_weights(&self, light: bool) -> Result<Weights> {
        let path = self.weights_path.clone().context("nnedi3_weights.bin not found")?;
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        // BRP_NNEDI_CFG="nsize,nns" overrides the network (for experiments).
        let cfg = std::env::var("BRP_NNEDI_CFG")
            .ok()
            .and_then(|s| {
                let v: Vec<usize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (v.len() == 2).then(|| NnediConfig { nsize: v[0], nns: v[1], ..NnediConfig::default() })
            })
            .unwrap_or_default();
        let cfg = if light { NnediConfig { nns: 0, ..cfg } } else { cfg };
        let meta = NnediWeights::load(&bytes, cfg)?;
        let size = (meta.data.len() * 4) as u64;
        unsafe {
            let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::STORAGE_BUFFER), None)?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let mut alloc = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "nnedi weights",
                requirements: req,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            self.device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
            let dst = alloc.mapped_slice_mut().context("weights buffer not mapped")?;
            for (i, v) in meta.data.iter().enumerate() {
                dst[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            log::info!("NNEDI3 weights loaded: {}x{} window, {} neurons", meta.xdim, meta.ydim, meta.nns);
            Ok(Weights { buffer, alloc: Some(alloc), meta })
        }
    }

    /// Record deinterlacing of `cur` into slot `si`. Returns the output plane views
    /// (layout GENERAL) or None when the frame can't be processed (e.g. packed RGB).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        cb: vk::CommandBuffer,
        si: usize,
        params: DeintParams,
        prev: &GpuFrame,
        cur: &GpuFrame,
        next: &GpuFrame,
        graveyard: &mut Graveyard,
        now: u64,
    ) -> Result<Option<Vec<vk::ImageView>>> {
        let FrameLayout::Yuv { bits, .. } = cur.layout else { return Ok(None) };
        let mut params = params;
        if params.method != DeintMethod::Bwdif && !self.ensure_weights(params.light) {
            params.method = DeintMethod::Bwdif;
        }
        let key = (prev.key, cur.key, next.key, params);
        let n = cur.planes.len().min(MAX_PLANES);

        // (Re)create output images when the frame shape changes.
        let needs = {
            let s = &self.slots[si];
            s.out.len() != n || s.out.iter().zip(&cur.planes).any(|(o, p)| o.w != p.w || o.h != p.h)
        };
        if needs {
            let slot = &mut self.slots[si];
            for img in slot.out.drain(..).chain(slot.edi.drain(..)) {
                graveyard.bury(now, img);
            }
            for p in &cur.planes[..n] {
                for target in 0..2 {
                    let img = gpu::create_image(
                        &self.device,
                        &self.allocator,
                        p.w,
                        p.h,
                        OUT_FORMAT,
                        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                        "deint plane",
                    )?;
                    gpu::image_barrier(
                        &self.device,
                        cb,
                        img.image,
                        (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
                        (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, vk::ImageLayout::GENERAL),
                    );
                    if target == 0 { slot.out.push(img) } else { slot.edi.push(img) }
                }
            }
            slot.key = None;
        }
        if self.slots[si].key == Some(key) {
            return Ok(Some(self.slots[si].out.iter().map(|o| o.view).collect()));
        }

        let tff = params.tff as i32;
        let parity = tff ^ (!params.second_field) as i32; // FFmpeg: tff ^ !is_second
        let keep = parity;
        let p2 = parity ^ tff;
        let wide = bits > 8;
        let max_code = ((1u32 << bits) - 1) as f32;
        let to_code = if wide { 65535.0 } else { 255.0 };
        let to255 = to_code * 255.0 / max_code;

        let d = self.device.clone();
        for p in 0..n {
            let (w, h) = (cur.planes[p].w as i32, cur.planes[p].h as i32);
            let slot = &self.slots[si];
            let img_info = |v: vk::ImageView, l: vk::ImageLayout, s: vk::Sampler| [vk::DescriptorImageInfo::default().image_view(v).image_layout(l).sampler(s)];
            let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let g = vk::ImageLayout::GENERAL;
            // Light mode: NNEDI3 on luma only, chroma with bwdif (a third of the work saved).
            let use_nnedi = params.method != DeintMethod::Bwdif && !(params.light && p > 0);
            let nnedi_only = use_nnedi && params.method == DeintMethod::Nnedi;
            let nnedi_dst = if nnedi_only { slot.out[p].view } else { slot.edi[p].view };

            unsafe {
                if use_nnedi {
                    let wt = self.weights[params.light as usize].as_ref().unwrap();
                    let i0 = img_info(cur.planes[p].view, ro, self.sampler);
                    let i1 = img_info(nnedi_dst, g, vk::Sampler::null());
                    let bi = [vk::DescriptorBufferInfo::default().buffer(wt.buffer).range(vk::WHOLE_SIZE)];
                    let set = slot.nnedi_sets[p];
                    d.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&i0),
                            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&i1),
                            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&bi),
                        ],
                        &[],
                    );
                    let m = &wt.meta;
                    let push = NnediPush {
                        size: [w, h],
                        keep,
                        xdim: m.xdim as i32,
                        ydim: m.ydim as i32,
                        nns: m.nns as i32,
                        qual2: 0,
                        model: m.model_offset as i32,
                        to255,
                    };
                    // Fast variant: compiled for the default network (32x4, 32 neurons).
                    let pipe = match &self.nnedi_fast {
                        Some(f) if m.xdim == 32 && m.ydim == 4 && m.nns == 32 => f,
                        _ => &self.nnedi,
                    };
                    d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &[set], &[]);
                    d.cmd_push_constants(cb, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(&push));
                    d.cmd_dispatch(cb, (w as u32).div_ceil(32), (h as u32).div_ceil(2).div_ceil(4), 1);
                }
                if !nnedi_only {
                    if use_nnedi {
                        // NNEDI3 writes edi, bwdif reads it.
                        gpu::image_barrier(
                            &d,
                            cb,
                            slot.edi[p].image,
                            (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, g),
                            (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_READ, g),
                        );
                    }
                    let set = slot.bwdif_sets[p];
                    let i0 = img_info(prev.planes[p].view, ro, self.sampler);
                    let i1 = img_info(cur.planes[p].view, ro, self.sampler);
                    let i2 = img_info(next.planes[p].view, ro, self.sampler);
                    let i3 = img_info(slot.edi[p].view, g, vk::Sampler::null());
                    let i4 = img_info(slot.out[p].view, g, vk::Sampler::null());
                    fn wr<'a>(set: vk::DescriptorSet, b: u32, t: vk::DescriptorType, i: &'a [vk::DescriptorImageInfo]) -> vk::WriteDescriptorSet<'a> {
                        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b).descriptor_type(t).image_info(i)
                    }
                    use vk::DescriptorType as D;
                    let writes = [
                        wr(set, 0, D::COMBINED_IMAGE_SAMPLER, &i0),
                        wr(set, 1, D::COMBINED_IMAGE_SAMPLER, &i1),
                        wr(set, 2, D::COMBINED_IMAGE_SAMPLER, &i2),
                        wr(set, 3, D::STORAGE_IMAGE, &i3),
                        wr(set, 4, D::STORAGE_IMAGE, &i4),
                    ];
                    d.update_descriptor_sets(&writes, &[]);
                    let push = BwdifPush { size: [w, h], keep, p2, use_edi: use_nnedi as i32, to_code, max_code };
                    d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.bwdif.pipeline);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.bwdif.layout, 0, &[set], &[]);
                    d.cmd_push_constants(cb, self.bwdif.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(&push));
                    d.cmd_dispatch(cb, (w as u32).div_ceil(16), (h as u32).div_ceil(8), 1);
                }
            }
            // Output is read by the display shader (or by readback in tests).
            gpu::image_barrier(
                &d,
                cb,
                self.slots[si].out[p].image,
                (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE, g),
                (
                    vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    g,
                ),
            );
        }
        self.slots[si].key = Some(key);
        Ok(Some(self.slots[si].out.iter().map(|o| o.view).collect()))
    }

    /// Output images of a slot (for readback in tests).
    pub fn outputs(&self, si: usize) -> &[GpuImage] {
        &self.slots[si].out
    }

    pub fn invalidate(&mut self) {
        for s in &mut self.slots {
            s.key = None;
        }
    }
}

impl Drop for Deinterlacer {
    fn drop(&mut self) {
        unsafe {
            for s in &mut self.slots {
                for img in s.out.drain(..).chain(s.edi.drain(..)) {
                    gpu::destroy_image(&self.device, &self.allocator, img);
                }
            }
            for mut w in self.weights.iter_mut().filter_map(Option::take) {
                if let Some(a) = w.alloc.take() {
                    let _ = self.allocator.lock().unwrap().free(a);
                }
                self.device.destroy_buffer(w.buffer, None);
            }
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            for p in [Some(&self.nnedi), self.nnedi_fast.as_ref(), Some(&self.bwdif)].into_iter().flatten() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
                self.device.destroy_descriptor_set_layout(p.set_layout, None);
            }
        }
    }
}
