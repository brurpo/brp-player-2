//! Effect chain: runs mpv-format user shaders on the GPU.
//!
//! Per run: planes -> [LUMA/CHROMA textures -> LUMA/CHROMA hooks] -> MAIN (RGB in the
//! source transfer) -> MAIN/RGB hooks -> output. All passes are compute shaders.
//! Shaders compile in the background; an effect joins the chain once it is ready.

use crate::color::VideoPush;
use crate::gpu::{self, GpuImage, Graveyard, SharedAllocator};
use crate::usershader::{self, BindKind, Generated, ParamKind, ShaderFile, Transfer};
use crate::workspace::{SourceColor, Space};
use anyhow::{anyhow, Result};
use ash::vk;
use brp_media::{ColorInfo, FrameLayout};
use crossbeam_channel::{Receiver, Sender};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::collections::HashMap;
use std::sync::Arc;

const PLANES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/planes.comp.spv"));
const MERGE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/merge.comp.spv"));
const SPACE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/space.comp.spv"));
const POOL_SETS: u32 = 2048;
const PARAM_BUF: u64 = 256 * 1024;
const WORK_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// One effect in the chain, as the UI sees it.
#[derive(Clone)]
pub struct EffectSpec {
    /// Unique per instance (the same file can be added twice).
    pub id: u64,
    pub file: Arc<ShaderFile>,
    pub params: Vec<f64>,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EffectStatus {
    Compiling,
    Ready,
    Error(String),
}

/// Input of one chain run.
pub struct ChainFrame<'a> {
    pub planes: &'a [vk::ImageView],
    /// Layout of `planes` (SHADER_READ_ONLY_OPTIMAL for cached frames, GENERAL for
    /// deinterlaced output).
    pub planes_layout: vk::ImageLayout,
    pub plane_sizes: &'a [(u32, u32)],
    pub layout: FrameLayout,
    pub color: ColorInfo,
    /// Deterministic frame counter passed to shaders as `frame`.
    pub seq: i64,
    /// Output (display or export) size, for OUTPUT in expressions.
    pub target: (u32, u32),
    /// Source video size (`input_size`).
    pub input: (u32, u32),
    /// Colour space of `planes` (decoded frames: `Source`; DESIGN.md §7a).
    pub space: Space,
    /// Space the output must be in (None: whatever the last effect left).
    pub want: Option<Space>,
}

pub struct ChainOutput {
    pub view: vk::ImageView,
    pub w: u32,
    pub h: u32,
    /// Colour space of the output.
    pub space: Space,
}

/// The post-RGB passes of `file`, in the order they run (mpv's stage order MAIN/RGB,
/// LINEAR, SIGMOID, file order within a stage), with their hook and colour space.
pub fn main_passes(file: &ShaderFile) -> Vec<(usize, String, Space)> {
    let mut v: Vec<(u8, usize, String, Space)> = Vec::new();
    for (pi, pass) in file.passes.iter().enumerate() {
        // A pass hooking several of these stages runs at the first.
        if let Some(h) = pass.hooks.iter().filter(|h| Space::hook_rank(h).is_some()).min_by_key(|h| Space::hook_rank(h)) {
            v.push((Space::hook_rank(h).unwrap(), pi, h.clone(), Space::of_hook(h, file.space).unwrap()));
        }
    }
    v.sort_by_key(|x| (x.0, x.1));
    v.into_iter().map(|(_, pi, h, s)| (pi, h, s)).collect()
}

/// Space the chain's output is in for these effects (the enabled ones), from `input`,
/// converting lazily as the chain does, then to `want` if given.
pub fn out_space(specs: &[EffectSpec], input: Space, want: Option<Space>, color: &ColorInfo) -> Space {
    let sc = SourceColor::of(color);
    let mut cur = input;
    for s in specs.iter().filter(|s| s.enabled) {
        for (_, _, sp) in main_passes(&s.file) {
            if !sc.same(cur, sp) {
                cur = sp;
            }
        }
    }
    match want {
        Some(w) if !sc.same(cur, w) => w,
        _ => cur,
    }
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

struct Compiled {
    pipe: Pipe,
    generated: Generated,
}

struct Effect {
    spec: EffectSpec,
    status: EffectStatus,
    /// (pass, hook) -> pipeline.
    compiled: HashMap<(usize, String), Compiled>,
    storage: HashMap<String, GpuImage>,
    /// Param values baked into the compiled code (DEFINE params / const mode).
    baked: Vec<f64>,
    const_mode: bool,
    job: u64,
}

struct JobResult {
    id: u64,
    job: u64,
    passes: Result<Vec<((usize, String), Generated, Vec<u32>, bool)>>,
}

struct PoolImage {
    img: GpuImage,
    used: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UserPush {
    out_size: [f32; 2],
    input_size: [f32; 2],
    target_size: [f32; 2],
    tex_offset: [f32; 2],
    frame: i32,
    random: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PlanesPush {
    chroma: i32,
    norm: f32,
}

fn bytes<T>(t: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(t as *const T as *const u8, std::mem::size_of::<T>()) }
}

pub fn vk_format(glsl: &str) -> vk::Format {
    match glsl {
        "r8" => vk::Format::R8_UNORM,
        "rg8" => vk::Format::R8G8_UNORM,
        "rgba8" => vk::Format::R8G8B8A8_UNORM,
        "r16" => vk::Format::R16_UNORM,
        "rgba16" => vk::Format::R16G16B16A16_UNORM,
        "r16f" => vk::Format::R16_SFLOAT,
        "rg16f" => vk::Format::R16G16_SFLOAT,
        "rgba16f" => vk::Format::R16G16B16A16_SFLOAT,
        "r32f" => vk::Format::R32_SFLOAT,
        "rg32f" => vk::Format::R32G32_SFLOAT,
        "rgba32f" => vk::Format::R32G32B32A32_SFLOAT,
        _ => vk::Format::R16G16B16A16_SFLOAT,
    }
}

fn make_pipe(device: &ash::Device, cache: vk::PipelineCache, spv: &[u32], bindings: &[vk::DescriptorType], push: u32) -> Result<Pipe> {
    unsafe {
        let b: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(i, t)| vk::DescriptorSetLayoutBinding::default().binding(i as u32).descriptor_type(*t).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&b), None)?;
        let pr = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push)];
        let sl = [set_layout];
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pr), None)?;
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let res = device.create_compute_pipelines(cache, &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None);
        device.destroy_shader_module(module, None);
        let pipeline = res.map_err(|(_, e)| e)?[0];
        Ok(Pipe { pipeline, layout, set_layout })
    }
}

fn retire(graveyard: &mut Graveyard, now: u64, p: &Pipe) {
    graveyard.bury_pipeline(now, p.pipeline, p.layout, p.set_layout);
}

fn destroy_pipe(device: &ash::Device, p: &Pipe) {
    unsafe {
        device.destroy_pipeline(p.pipeline, None);
        device.destroy_pipeline_layout(p.layout, None);
        device.destroy_descriptor_set_layout(p.set_layout, None);
    }
}

fn barrier(device: &ash::Device, cb: vk::CommandBuffer) {
    let m = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::CLEAR)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&m))) };
}

/// Does `params` differ from `baked` in a way that needs a recompile?
fn needs_recompile(file: &ShaderFile, const_mode: bool, baked: &[f64], params: &[f64]) -> bool {
    file.params.iter().enumerate().any(|(i, p)| (const_mode || p.kind == ParamKind::Define) && baked.get(i) != params.get(i))
}

/// Compile an effect file (default parameters) for `transfer` into the disk shader
/// cache, ahead of use (start-up), without building pipelines.
pub fn precompile(file: &ShaderFile, transfer: Transfer) -> Result<()> {
    let params: Vec<f64> = file.params.iter().map(|p| p.default).collect();
    compile_effect(file, &params, transfer).map(|_| ())
}

fn compile_effect(file: &ShaderFile, params: &[f64], transfer: Transfer) -> Result<Vec<((usize, String), Generated, Vec<u32>, bool)>> {
    let mut out = Vec::new();
    for (pi, pass) in file.passes.iter().enumerate() {
        let binds = usershader::bind_kinds(file, pass);
        for hook in &pass.hooks {
            let (g, spv, mode) = usershader::build_pass(file, pi, hook, &binds, params, transfer)
                .map_err(|e| anyhow!("pass {pi} ({}): {}", pass.desc, e.to_string().lines().take(6).collect::<Vec<_>>().join("\n")))?;
            out.push(((pi, hook.clone()), g, spv, mode == usershader::ParamMode::Const));
        }
    }
    Ok(out)
}

pub struct EffectChain {
    device: ash::Device,
    /// The driver's pipeline cache (kept on disk: see `shader_cache`).
    pipeline_cache: vk::PipelineCache,
    allocator: SharedAllocator,
    effects: Vec<Effect>,
    transfer: Transfer,
    tx: Sender<JobResult>,
    rx: Receiver<JobResult>,
    next_job: u64,
    planes_pipe: Pipe,
    merge_pipe: Pipe,
    space_pipe: Pipe,
    pools: Vec<vk::DescriptorPool>,
    param_bufs: Vec<(vk::Buffer, Option<Allocation>)>,
    param_used: u64,
    ubo_align: u64,
    images: Vec<PoolImage>,
    sampler: vk::Sampler,
    reset_storage: bool,
    /// Picture number at the last history reset: stateful shaders get
    /// `frame = seq - frame_base` (libplacebo: executions since start, so 0 = no history).
    frame_base: i64,
}

impl EffectChain {
    pub fn new(device: &ash::Device, allocator: SharedAllocator, slots: usize, ubo_align: u64, pipeline_cache: vk::PipelineCache) -> Result<Self> {
        use vk::DescriptorType as D;
        let words = |b: &[u8]| ash::util::read_spv(&mut std::io::Cursor::new(b)).expect("spirv");
        let planes_pipe = make_pipe(device, pipeline_cache, &words(PLANES_SPV), &[D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::STORAGE_IMAGE], 8)?;
        let merge_pipe = make_pipe(
            device,
            pipeline_cache,
            &words(MERGE_SPV),
            &[D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::COMBINED_IMAGE_SAMPLER, D::STORAGE_IMAGE],
            std::mem::size_of::<VideoPush>() as u32,
        )?;
        let space_pipe = make_pipe(device, pipeline_cache, &words(SPACE_SPV), &[D::COMBINED_IMAGE_SAMPLER, D::STORAGE_IMAGE], 32)?;
        let mut pools = Vec::new();
        let mut param_bufs = Vec::new();
        unsafe {
            for _ in 0..slots {
                let sizes = [
                    vk::DescriptorPoolSize::default().ty(D::COMBINED_IMAGE_SAMPLER).descriptor_count(POOL_SETS * 8),
                    vk::DescriptorPoolSize::default().ty(D::STORAGE_IMAGE).descriptor_count(POOL_SETS * 4),
                    vk::DescriptorPoolSize::default().ty(D::UNIFORM_BUFFER).descriptor_count(POOL_SETS),
                ];
                pools.push(device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(POOL_SETS).pool_sizes(&sizes), None)?);
                let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(PARAM_BUF).usage(vk::BufferUsageFlags::UNIFORM_BUFFER), None)?;
                let req = device.get_buffer_memory_requirements(buffer);
                let alloc = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                    name: "effect params",
                    requirements: req,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;
                device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
                param_bufs.push((buffer, Some(alloc)));
            }
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let (tx, rx) = crossbeam_channel::unbounded();
            Ok(EffectChain {
                device: device.clone(),
                pipeline_cache,
                allocator,
                effects: Vec::new(),
                transfer: Transfer::Sdr,
                tx,
                rx,
                next_job: 1,
                planes_pipe,
                merge_pipe,
                space_pipe,
                pools,
                param_bufs,
                param_used: 0,
                ubo_align: ubo_align.max(16),
                images: Vec::new(),
                sampler,
                reset_storage: false,
                frame_base: 0,
            })
        }
    }

    /// Replace the chain. Effects keep their compiled code when only uniform
    /// parameters or the enabled flag changed.
    pub fn set_effects(&mut self, specs: &[EffectSpec], transfer: Transfer, graveyard: &mut Graveyard, now: u64) {
        let transfer_changed = transfer != self.transfer;
        self.transfer = transfer;
        let mut old: HashMap<u64, Effect> = self.effects.drain(..).map(|e| (e.spec.id, e)).collect();
        for spec in specs {
            let mut e = match old.remove(&spec.id) {
                Some(mut e) if Arc::ptr_eq(&e.spec.file, &spec.file) => {
                    let recompile = transfer_changed || needs_recompile(&spec.file, e.const_mode, &e.baked, &spec.params);
                    e.spec = spec.clone();
                    if recompile {
                        Self::drop_compiled(&mut e, graveyard, now);
                        e.status = EffectStatus::Compiling;
                        e.job = 0;
                    }
                    e
                }
                other => {
                    if let Some(mut o) = other {
                        self.drop_effect(&mut o, graveyard, now);
                    }
                    Effect {
                        spec: spec.clone(),
                        status: EffectStatus::Compiling,
                        compiled: HashMap::new(),
                        storage: HashMap::new(),
                        baked: Vec::new(),
                        const_mode: false,
                        job: 0,
                    }
                }
            };
            if e.status == EffectStatus::Compiling && e.job == 0 {
                e.job = self.next_job;
                self.next_job += 1;
                e.baked = spec.params.clone();
                let (tx, file, params, job, id) = (self.tx.clone(), spec.file.clone(), spec.params.clone(), e.job, spec.id);
                std::thread::Builder::new()
                    .name("brp-shader-compile".into())
                    .spawn(move || {
                        let t = std::time::Instant::now();
                        // A panic while compiling becomes the effect's error (else it would
                        // show as compiling forever).
                        let passes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile_effect(&file, &params, transfer)))
                            .unwrap_or_else(|p| Err(anyhow::anyhow!("internal error while compiling: {}", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
                        log::info!("compiled {} in {:?}{}", file.name, t.elapsed(), if passes.is_err() { " (failed)" } else { "" });
                        let _ = tx.send(JobResult { id, job, passes });
                    })
                    .ok();
            }
            self.effects.push(e);
        }
        for (_, mut e) in old {
            self.drop_effect(&mut e, graveyard, now);
        }
    }

    /// Integrate finished background compiles. Returns true if something changed.
    /// Replaced pipelines go to `graveyard` (frame `now`).
    pub fn poll(&mut self, graveyard: &mut Graveyard, now: u64) -> bool {
        let mut changed = false;
        while let Ok(r) = self.rx.try_recv() {
            let Some(i) = self.effects.iter().position(|e| e.spec.id == r.id && e.job == r.job) else { continue };
            changed = true;
            match r.passes {
                Ok(passes) => {
                    let mut compiled = HashMap::new();
                    let mut error = None;
                    let mut const_mode = false;
                    for (key, generated, spv, cmode) in passes {
                        const_mode |= cmode;
                        use vk::DescriptorType as D;
                        let mut kinds = Vec::new();
                        if generated.has_params {
                            kinds.push(D::UNIFORM_BUFFER);
                        } else {
                            kinds.push(D::STORAGE_IMAGE); // placeholder at binding 0 (unused)
                        }
                        for (_, k) in &generated.binds {
                            kinds.push(if matches!(k, BindKind::Storage(_)) { D::STORAGE_IMAGE } else { D::COMBINED_IMAGE_SAMPLER });
                        }
                        kinds.push(D::STORAGE_IMAGE);
                        match make_pipe(&self.device, self.pipeline_cache, &spv, &kinds, std::mem::size_of::<UserPush>() as u32) {
                            Ok(pipe) => {
                                compiled.insert(key, Compiled { pipe, generated });
                            }
                            Err(e) => {
                                error = Some(format!("pipeline: {e}"));
                                break;
                            }
                        }
                    }
                    let e = &mut self.effects[i];
                    let old = std::mem::replace(&mut e.compiled, compiled);
                    for (_, c) in old {
                        retire(graveyard, now, &c.pipe);
                    }
                    e.const_mode = const_mode;
                    e.status = match error {
                        Some(m) => EffectStatus::Error(m),
                        None => EffectStatus::Ready,
                    };
                }
                Err(err) => self.effects[i].status = EffectStatus::Error(format!("{err:#}")),
            }
        }
        changed
    }

    pub fn status(&self) -> Vec<(u64, EffectStatus)> {
        self.effects.iter().map(|e| (e.spec.id, e.status.clone())).collect()
    }

    fn active(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter().filter(|e| e.spec.enabled && e.status == EffectStatus::Ready)
    }

    pub fn has_active(&self) -> bool {
        self.active().next().is_some()
    }

    /// Some enabled effect keeps state between frames (STORAGE textures).
    pub fn stateful(&self) -> bool {
        self.active().any(|e| e.spec.file.is_stateful())
    }

    /// Frames to replay after a jump for stateful effects.
    pub fn warmup(&self) -> u32 {
        self.active().filter(|e| e.spec.file.is_stateful()).map(|e| e.spec.file.warmup.unwrap_or(2)).max().unwrap_or(0)
    }

    /// Hash of everything that changes the output for a given frame.
    pub fn key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.transfer.hash(&mut h);
        for e in self.active() {
            e.spec.id.hash(&mut h);
            for p in &e.spec.params {
                p.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    /// The image behind a chain texture view (for readback).
    pub fn image_of(&self, view: vk::ImageView) -> Option<vk::Image> {
        self.images.iter().find(|p| p.img.view == view).map(|p| p.img.image)
    }

    /// Clear the history of stateful effects at the start of the next run.
    pub fn reset_history(&mut self) {
        self.reset_storage = true;
    }

    /// Start a command buffer's worth of runs in slot `si`.
    pub fn begin(&mut self, si: usize) -> Result<()> {
        unsafe { self.device.reset_descriptor_pool(self.pools[si], vk::DescriptorPoolResetFlags::empty())? };
        self.param_used = 0;
        Ok(())
    }

    fn alloc_image(&mut self, cb: vk::CommandBuffer, w: u32, h: u32) -> Result<(vk::ImageView, vk::Image)> {
        if let Some(p) = self.images.iter_mut().find(|p| !p.used && p.img.w == w && p.img.h == h) {
            p.used = true;
            return Ok((p.img.view, p.img.image));
        }
        let img = gpu::create_image(&self.device, &self.allocator, w, h, WORK_FORMAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC, "effect texture")?;
        gpu::image_barrier(
            &self.device,
            cb,
            img.image,
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
            (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE, vk::ImageLayout::GENERAL),
        );
        let r = (img.view, img.image);
        self.images.push(PoolImage { img, used: true });
        Ok(r)
    }

    fn alloc_set(&self, si: usize, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        let l = [layout];
        Ok(unsafe { self.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pools[si]).set_layouts(&l))?[0] })
    }

    fn write_image(&self, set: vk::DescriptorSet, binding: u32, view: vk::ImageView, layout: vk::ImageLayout, storage: bool) {
        let info = [vk::DescriptorImageInfo::default().image_view(view).image_layout(layout).sampler(if storage { vk::Sampler::null() } else { self.sampler })];
        let t = if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::COMBINED_IMAGE_SAMPLER };
        let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(t).image_info(&info);
        unsafe { self.device.update_descriptor_sets(&[w], &[]) };
    }

    /// Process one frame. Several runs may be recorded into the same command buffer
    /// (history warm-up); call [`begin`] once per command buffer first.
    pub fn run(&mut self, cb: vk::CommandBuffer, si: usize, f: &ChainFrame<'_>) -> Result<Option<ChainOutput>> {
        if !self.has_active() {
            return Ok(None);
        }
        self.run_merge(cb, si, f)
    }

    /// Like `run`, but with no active effect still converts the input to RGB (the
    /// merge pass): used to bring frames to the GPU for background processing.
    pub fn run_merge(&mut self, cb: vk::CommandBuffer, si: usize, f: &ChainFrame<'_>) -> Result<Option<ChainOutput>> {
        if f.planes.is_empty() {
            return Ok(None);
        }
        let d = self.device.clone();
        for p in &mut self.images {
            p.used = false;
        }
        barrier(&d, cb);
        if self.reset_storage {
            self.reset_storage = false;
            self.frame_base = f.seq;
            let range = gpu::color_range();
            for e in &self.effects {
                for img in e.storage.values() {
                    unsafe { d.cmd_clear_color_image(cb, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[range]) };
                }
            }
            barrier(&d, cb);
        }

        let yuv = matches!(f.layout, FrameLayout::Yuv { .. });
        let wants_planes = yuv
            && self.active().any(|e| e.spec.file.passes.iter().any(|p| p.hooks.iter().any(|h| h == "LUMA" || h == "CHROMA")));
        let push = crate::color::video_push(f.layout, &f.color, 0.0);
        let norm = push.misc[0];
        let mut reg: HashMap<String, (vk::ImageView, u32, u32)> = HashMap::new();
        let p0 = f.plane_sizes[0];

        let plane = |i: usize| f.planes.get(i).copied().unwrap_or(f.planes[0]);
        let main = if wants_planes {
            // LUMA and CHROMA textures.
            let (lv, _) = self.alloc_image(cb, p0.0, p0.1)?;
            let p1 = f.plane_sizes.get(1).copied().unwrap_or(p0);
            let (cv, _) = self.alloc_image(cb, p1.0, p1.1)?;
            for (chroma, dst, (w, h), a, b) in [(0, lv, p0, plane(0), plane(0)), (1, cv, p1, plane(1), plane(2))] {
                let set = self.alloc_set(si, self.planes_pipe.set_layout)?;
                self.write_image(set, 0, a, f.planes_layout, false);
                self.write_image(set, 1, b, f.planes_layout, false);
                self.write_image(set, 2, dst, vk::ImageLayout::GENERAL, true);
                unsafe {
                    d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.planes_pipe.pipeline);
                    d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.planes_pipe.layout, 0, &[set], &[]);
                    d.cmd_push_constants(cb, self.planes_pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(&PlanesPush { chroma, norm }));
                    d.cmd_dispatch(cb, w.div_ceil(16), h.div_ceil(16), 1);
                }
            }
            barrier(&d, cb);
            reg.insert("LUMA".into(), (lv, p0.0, p0.1));
            reg.insert("CHROMA".into(), (cv, p1.0, p1.1));
            self.run_stage(cb, si, f, "LUMA", &mut reg)?;
            self.run_stage(cb, si, f, "CHROMA", &mut reg)?;
            let (lv, lw, lh) = reg["LUMA"];
            let (cv, _, _) = reg["CHROMA"];
            let alpha = if f.layout.has_alpha() { (plane(3), f.planes_layout) } else { (lv, vk::ImageLayout::GENERAL) };
            self.merge(cb, si, f, &push, [(lv, vk::ImageLayout::GENERAL), (cv, vk::ImageLayout::GENERAL), (cv, vk::ImageLayout::GENERAL), alpha], true, (lw, lh))?
        } else {
            let l = f.planes_layout;
            self.merge(cb, si, f, &push, [(plane(0), l), (plane(1), l), (plane(2), l), (plane(3), l)], false, p0)?
        };
        reg.insert("MAIN".into(), main);
        let mut space = f.space;
        self.run_main(cb, si, f, &mut reg, &mut space)?;
        let sc = SourceColor::of(&f.color);
        if let Some(want) = f.want.filter(|w| !sc.same(space, *w)) {
            let old = reg["MAIN"].0;
            let m = self.convert(cb, si, reg["MAIN"], space, want, &sc)?;
            reg.insert("MAIN".into(), m);
            self.release_dead(old, &reg);
            space = want;
        }
        let (view, w, h) = reg["MAIN"];
        Ok(Some(ChainOutput { view, w, h, space }))
    }

    /// A MAIN picture that was replaced (by a pass or a conversion): if nothing in
    /// `reg` still names it, later work in this run may reuse its image. (Passes are
    /// separated by barriers, so reading it before and writing it after is safe.)
    /// A conversion then costs one picture, not two, and a chain ping-pongs between two.
    fn release_dead(&mut self, view: vk::ImageView, reg: &HashMap<String, (vk::ImageView, u32, u32)>) {
        if reg.values().any(|t| t.0 == view) {
            return;
        }
        if let Some(p) = self.images.iter_mut().find(|p| p.img.view == view) {
            p.used = false;
        }
    }

    /// `src` from one working colour space to another, into a new pool image.
    fn convert(&mut self, cb: vk::CommandBuffer, si: usize, src: (vk::ImageView, u32, u32), from: Space, to: Space, sc: &SourceColor) -> Result<(vk::ImageView, u32, u32)> {
        let (w, h) = (src.1, src.2);
        let (dst, _) = self.alloc_image(cb, w, h)?;
        let set = self.alloc_set(si, self.space_pipe.set_layout)?;
        self.write_image(set, 0, src.0, vk::ImageLayout::GENERAL, false);
        self.write_image(set, 1, dst, vk::ImageLayout::GENERAL, true);
        let mut push = [0u8; 32];
        for (i, v) in sc.words(from, to).iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        push[16..20].copy_from_slice(&sc.peak.to_le_bytes());
        let d = &self.device;
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.space_pipe.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.space_pipe.layout, 0, &[set], &[]);
            d.cmd_push_constants(cb, self.space_pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            d.cmd_dispatch(cb, w.div_ceil(16), h.div_ceil(16), 1);
        }
        barrier(d, cb);
        Ok((dst, w, h))
    }

    /// No effect active (removed, all disabled, effects switch off): free the working
    /// pictures and the history textures, so the direct path holds no float picture.
    /// With effects active, only the history of the inactive ones goes. Images may
    /// still be read by frames in flight: they go to `graveyard`.
    pub fn release_idle(&mut self, graveyard: &mut Graveyard, now: u64) {
        let any = self.has_active();
        for e in &mut self.effects {
            if !any || !(e.spec.enabled && e.status == EffectStatus::Ready) {
                for (_, img) in e.storage.drain() {
                    graveyard.bury(now, img);
                }
            }
        }
        if !any {
            for p in self.images.drain(..) {
                graveyard.bury(now, p.img);
            }
        }
    }

    /// The MAIN stage: each active effect's post-RGB passes in stack order, the
    /// picture converted to a pass's colour space first when it is in another
    /// (DESIGN.md §7a). `space` is the picture's space, before and after.
    fn run_main(&mut self, cb: vk::CommandBuffer, si: usize, f: &ChainFrame<'_>, reg: &mut HashMap<String, (vk::ImageView, u32, u32)>, space: &mut Space) -> Result<()> {
        let sc = SourceColor::of(&f.color);
        let ids: Vec<usize> = (0..self.effects.len()).filter(|&i| self.effects[i].spec.enabled && self.effects[i].status == EffectStatus::Ready).collect();
        for ei in ids {
            let file = self.effects[ei].spec.file.clone();
            for (pi, hook, sp) in main_passes(&file) {
                if !sc.same(*space, sp) {
                    let old = reg["MAIN"].0;
                    let m = self.convert(cb, si, reg["MAIN"], *space, sp, &sc)?;
                    reg.insert("MAIN".into(), m);
                    self.release_dead(old, reg);
                    *space = sp;
                }
                if hook != "MAIN" {
                    // BIND LINEAR (etc.) is the picture in that stage.
                    let m = reg["MAIN"];
                    reg.insert(hook.clone(), m);
                }
                let old = reg["MAIN"].0;
                let ran = self.run_pass(cb, si, f, ei, pi, &hook, "MAIN", reg)?;
                if hook != "MAIN" {
                    // The stage name follows the picture.
                    let m = reg["MAIN"];
                    reg.insert(hook.clone(), m);
                }
                self.release_dead(old, reg);
                if !ran {
                    break;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn merge(
        &mut self,
        cb: vk::CommandBuffer,
        si: usize,
        _f: &ChainFrame<'_>,
        push: &VideoPush,
        inputs: [(vk::ImageView, vk::ImageLayout); 4],
        packed: bool,
        size: (u32, u32),
    ) -> Result<(vk::ImageView, u32, u32)> {
        let (dst, _) = self.alloc_image(cb, size.0, size.1)?;
        let set = self.alloc_set(si, self.merge_pipe.set_layout)?;
        for (i, (v, l)) in inputs.iter().enumerate() {
            self.write_image(set, i as u32, *v, *l, false);
        }
        self.write_image(set, 4, dst, vk::ImageLayout::GENERAL, true);
        let mut p = *push;
        p.misc[3] = packed as i32 as f32;
        let d = &self.device;
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.merge_pipe.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.merge_pipe.layout, 0, &[set], &[]);
            d.cmd_push_constants(cb, self.merge_pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, p.bytes());
            d.cmd_dispatch(cb, size.0.div_ceil(16), size.1.div_ceil(16), 1);
        }
        barrier(d, cb);
        Ok((dst, size.0, size.1))
    }

    /// Hooks of `stage` (LUMA, CHROMA) of every active effect, in stack order.
    fn run_stage(&mut self, cb: vk::CommandBuffer, si: usize, f: &ChainFrame<'_>, stage: &str, reg: &mut HashMap<String, (vk::ImageView, u32, u32)>) -> Result<()> {
        let ids: Vec<usize> = (0..self.effects.len()).filter(|&i| self.effects[i].spec.enabled && self.effects[i].status == EffectStatus::Ready).collect();
        for ei in ids {
            let file = self.effects[ei].spec.file.clone();
            for (pi, pass) in file.passes.iter().enumerate() {
                let Some(hook) = pass.hooks.iter().find(|h| h.as_str() == stage) else { continue };
                if !self.run_pass(cb, si, f, ei, pi, hook, stage, reg)? {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Pass `pi` of effect `ei`, hooking `hook`, on the picture `reg[stage]`; its
    /// output replaces that picture (or is saved under its SAVE name). False if the
    /// effect cannot run (a texture it binds is missing): its other passes are skipped.
    #[allow(clippy::too_many_arguments)]
    fn run_pass(&mut self, cb: vk::CommandBuffer, si: usize, f: &ChainFrame<'_>, ei: usize, pi: usize, hook: &str, stage: &str, reg: &mut HashMap<String, (vk::ImageView, u32, u32)>) -> Result<bool> {
        let d = self.device.clone();
        let file = self.effects[ei].spec.file.clone();
        let pass = &file.passes[pi];
        let hooked = reg[stage];
        let size_of = |name: &str| -> Option<(f64, f64)> {
            match name {
                "HOOKED" => Some((hooked.1 as f64, hooked.2 as f64)),
                "OUTPUT" => Some((f.target.0 as f64, f.target.1 as f64)),
                "NATIVE" | "NATIVE_CROPPED" => Some((f.input.0 as f64, f.input.1 as f64)),
                n => reg.get(n).map(|t| (t.1 as f64, t.2 as f64)),
            }
        };
        let spec_params = &self.effects[ei].spec.params;
        let param_of = |name: &str| -> Option<f64> {
            let i = file.params.iter().position(|p| p.name == name)?;
            Some(spec_params.get(i).copied().unwrap_or(file.params[i].default))
        };
        if let Some(when) = &pass.when {
            match when.eval(&size_of, &param_of) {
                Ok(v) if v == 0.0 => return Ok(true),
                Ok(_) => {}
                Err(e) => {
                    log::warn!("{} pass {pi}: WHEN: {e}", file.name);
                    return Ok(true);
                }
            }
        }
        let eval = |e: &Option<usershader::Rpn>, default: u32| -> u32 {
            e.as_ref().and_then(|x| x.eval(&size_of, &param_of).ok()).map(|v| v.round().max(1.0) as u32).unwrap_or(default)
        };
        let (w, h) = (eval(&pass.width, hooked.1), eval(&pass.height, hooked.2));
        let (out, _) = self.alloc_image(cb, w, h)?;

        // Storage textures (persistent per effect).
        for t in file.textures.iter().filter(|t| t.storage) {
            if !self.effects[ei].storage.contains_key(&t.name) {
                let img = gpu::create_image(
                    &d,
                    &self.allocator,
                    t.w,
                    t.h,
                    vk_format(&t.format),
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                    "effect storage",
                )?;
                gpu::image_barrier(
                    &d,
                    cb,
                    img.image,
                    (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
                    (vk::PipelineStageFlags2::CLEAR, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::GENERAL),
                );
                unsafe { d.cmd_clear_color_image(cb, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[gpu::color_range()]) };
                barrier(&d, cb);
                self.effects[ei].storage.insert(t.name.clone(), img);
            }
        }

        let e = &self.effects[ei];
        let Some(c) = e.compiled.get(&(pi, hook.to_string())) else { return Ok(true) };
        let set = self.alloc_set(si, c.pipe.set_layout)?;
        // Parameters.
        if c.generated.has_params {
            let off = self.param_used;
            let mut buf = Vec::new();
            for (i, p) in file.params.iter().enumerate() {
                let v = e.spec.params.get(i).copied().unwrap_or(p.default);
                match p.kind {
                    ParamKind::Float => buf.extend_from_slice(&(v as f32).to_le_bytes()),
                    ParamKind::Int => buf.extend_from_slice(&(v.round() as i32).to_le_bytes()),
                    ParamKind::Uint => buf.extend_from_slice(&(v.round().max(0.0) as u32).to_le_bytes()),
                    ParamKind::Define => {}
                }
            }
            let len = (buf.len() as u64).max(16);
            if off + len > PARAM_BUF {
                anyhow::bail!("effect parameter buffer full");
            }
            let (pbuf, palloc) = &mut self.param_bufs[si];
            let mapped = palloc.as_mut().unwrap().mapped_slice_mut().unwrap();
            mapped[off as usize..off as usize + buf.len()].copy_from_slice(&buf);
            self.param_used = (off + len).next_multiple_of(self.ubo_align);
            let bi = [vk::DescriptorBufferInfo::default().buffer(*pbuf).offset(off).range(len)];
            let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&bi);
            unsafe { d.update_descriptor_sets(&[w], &[]) };
        } else {
            // Binding 0 is a placeholder storage image; point it at the output.
            self.write_image(set, 0, out, vk::ImageLayout::GENERAL, true);
        }
        for (bi, (name, kind)) in c.generated.binds.iter().enumerate() {
            let b = bi as u32 + 1;
            match kind {
                BindKind::Storage(_) => match self.effects[ei].storage.get(name) {
                    Some(img) => self.write_image(set, b, img.view, vk::ImageLayout::GENERAL, true),
                    None => anyhow::bail!("{}: storage texture {name} missing", file.name),
                },
                BindKind::Sampled => {
                    let tex = if name == "HOOKED" { Some(hooked) } else { reg.get(name).copied() };
                    match tex {
                        Some(t) => self.write_image(set, b, t.0, vk::ImageLayout::GENERAL, false),
                        None => {
                            log::warn!("{} pass {pi}: texture '{name}' is not available; effect skipped", file.name);
                            return Ok(false);
                        }
                    }
                }
            }
        }
        let c = &self.effects[ei].compiled[&(pi, hook.to_string())];
        self.write_image(set, c.generated.out_binding, out, vk::ImageLayout::GENERAL, true);
        let random = {
            let mut x = (f.seq as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03;
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        let push = UserPush {
            out_size: [w as f32, h as f32],
            input_size: [f.input.0 as f32, f.input.1 as f32],
            target_size: [f.target.0 as f32, f.target.1 as f32],
            tex_offset: [0.0, 0.0],
            // Stateful effects count from the history reset (deterministic, and
            // `frame == 0` still means "no history"); others use the picture number.
            frame: if file.is_stateful() { (f.seq - self.frame_base).max(0) as i32 } else { f.seq as i32 },
            random,
        };
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, c.pipe.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, c.pipe.layout, 0, &[set], &[]);
            d.cmd_push_constants(cb, c.pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(&push));
            d.cmd_dispatch(cb, w.div_ceil(c.generated.block[0]), h.div_ceil(c.generated.block[1]), 1);
        }
        barrier(&d, cb);
        let save = pass.save.clone().unwrap_or_else(|| stage.to_string());
        let save = if save == "HOOKED" || save == hook { stage.to_string() } else { save };
        reg.insert(save, (out, w, h));
        Ok(true)
    }

    /// Pipelines may still be used by frames in flight: retired, not destroyed.
    fn drop_compiled(e: &mut Effect, graveyard: &mut Graveyard, now: u64) {
        for (_, c) in e.compiled.drain() {
            retire(graveyard, now, &c.pipe);
        }
    }

    fn drop_effect(&self, e: &mut Effect, graveyard: &mut Graveyard, now: u64) {
        Self::drop_compiled(e, graveyard, now);
        for (_, img) in e.storage.drain() {
            graveyard.bury(now, img);
        }
    }
}

impl Drop for EffectChain {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            let mut effects = std::mem::take(&mut self.effects);
            for e in &mut effects {
                for (_, c) in e.compiled.drain() {
                    destroy_pipe(&self.device, &c.pipe);
                }
                for (_, img) in e.storage.drain() {
                    gpu::destroy_image(&self.device, &self.allocator, img);
                }
            }
            for p in self.images.drain(..) {
                gpu::destroy_image(&self.device, &self.allocator, p.img);
            }
            for (b, a) in &mut self.param_bufs {
                if let Some(a) = a.take() {
                    let _ = self.allocator.lock().unwrap().free(a);
                }
                self.device.destroy_buffer(*b, None);
            }
            for p in &self.pools {
                self.device.destroy_descriptor_pool(*p, None);
            }
            self.device.destroy_sampler(self.sampler, None);
            destroy_pipe(&self.device, &self.planes_pipe);
            destroy_pipe(&self.device, &self.merge_pipe);
            destroy_pipe(&self.device, &self.space_pipe);
        }
    }
}
