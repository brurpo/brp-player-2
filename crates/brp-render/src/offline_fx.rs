//! Windowless effect chain: feeds frames through user shaders and reads the result
//! back. Used to check effects numerically, and later by export.

use crate::context::GpuContext;
use crate::effects::{ChainFrame, ChainOutput, EffectChain};
use crate::frames::FrameCache;
use crate::deint::Deinterlacer;
use crate::gpu::{Graveyard, Staging};
use crate::{EffectSpec, EffectStatus, Space, Transfer};
use anyhow::{Context as _, Result};
use ash::vk;
use brp_media::{ColorInfo, VideoFrame};
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// RGBA result of a chain run (values as the last pass wrote them).
pub struct Rgba {
    pub w: u32,
    pub h: u32,
    pub data: Vec<[f32; 4]>,
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => {
            if m == 0.0 {
                s * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

pub struct OfflineEffects {
    ctx: std::sync::Arc<GpuContext>,
    pool: vk::CommandPool,
    cache: FrameCache,
    staging: Staging,
    graveyard: Graveyard,
    chain: Option<EffectChain>,
    frame_no: u64,
    /// Deinterlacing of source frames (before anything else), created on first use.
    deint: Option<Deinterlacer>,
    /// Deinterlacer runs recorded since `begin` (each uses its own slot).
    deint_runs: usize,
}

/// Deinterlaced pictures one recording may hold: each run needs its own descriptor
/// sets and output images until the work has run.
const DEINT_SLOTS: usize = 16;

impl OfflineEffects {
    /// Several runners can share one device (each has its own chain and frame cache).
    pub fn new(ctx: std::sync::Arc<GpuContext>) -> Result<Self> {
        unsafe {
            let pool = ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let props = ctx.instance.get_physical_device_properties(ctx.physical);
            let chain = EffectChain::new(&ctx.device, (*ctx.allocator).clone(), 1, props.limits.min_uniform_buffer_offset_alignment, ctx.pipeline_cache)?;
            Ok(OfflineEffects { ctx, pool, cache: FrameCache::new(16), staging: Staging::new(), graveyard: Graveyard::default(), chain: Some(chain), frame_no: 0, deint: None, deint_runs: 0 })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.ctx.device_name
    }

    /// Set the chain and wait until every effect compiled. Returns the compile errors.
    pub fn set_effects(&mut self, specs: &[EffectSpec], transfer: Transfer) -> Vec<String> {
        let chain = self.chain.as_mut().unwrap();
        chain.set_effects(specs, transfer, &mut self.graveyard, self.frame_no);
        // Compiles run on their own threads; one that never reports (a crashed
        // compiler) must not hold the caller forever.
        let t = std::time::Instant::now();
        loop {
            chain.poll(&mut self.graveyard, self.frame_no);
            let st = chain.status();
            if st.iter().all(|(_, s)| *s != EffectStatus::Compiling) {
                return st.into_iter().filter_map(|(_, s)| if let EffectStatus::Error(e) = s { Some(e) } else { None }).collect();
            }
            if t.elapsed() > std::time::Duration::from_secs(120) {
                return vec!["shader compile did not finish (2 minutes)".into()];
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    pub fn reset_history(&mut self) {
        self.chain.as_mut().unwrap().reset_history();
    }

    /// Start recording work into someone else's command buffer: reserves upload
    /// space for `frames` (those not on the GPU yet). Then `record` (any number of
    /// times), submit, wait, and `end`.
    pub fn begin(&mut self, frames: &[&VideoFrame]) -> Result<()> {
        self.frame_no += 1;
        self.deint_runs = 0;
        let bytes = self.cache.upload_size(1, frames);
        self.staging.reset(&self.ctx.device, &(*self.ctx.allocator).clone(), bytes)?;
        self.chain.as_mut().unwrap().begin(0)
    }

    /// Record `input` (in colour space `space`; decoded frames are `Source`) through
    /// the chain (just converted to RGB when no effect is active) into `cb`, the
    /// output converted to `want` if given (DESIGN.md §7a). The output image belongs
    /// to the chain: use it before the next `record`.
    pub fn record(&mut self, cb: vk::CommandBuffer, input: FxInput<'_>, color: ColorInfo, seq: i64, target: (u32, u32), space: Space, want: Option<Space>) -> Result<ChainOutput> {
        let device = self.ctx.device.clone();
        let alloc = (*self.ctx.allocator).clone();
        let chain = self.chain.as_mut().unwrap();
        match input {
            FxInput::Frame { frame, prev, next, deint } => {
                for f in [Some(frame), prev, next].into_iter().flatten() {
                    self.cache.ensure(&device, &alloc, cb, &mut self.staging, &mut self.graveyard, self.frame_no, 1, f)?;
                }
                let gi = self.cache.find((1, frame.index())).context("frame not uploaded")?;
                let sizes: Vec<(u32, u32)> = self.cache.get(gi).planes.iter().map(|p| (p.w, p.h)).collect();
                let mut views = self.cache.get(gi).views();
                let mut planes_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                // Deinterlaced first, with its neighbours (missing ones: the frame itself).
                if let Some(params) = deint {
                    if self.deint.is_none() {
                        let props = unsafe { self.ctx.instance.get_physical_device_properties(self.ctx.physical) };
                        let integrated = props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU;
                        self.deint = Some(Deinterlacer::new(&device, alloc.clone(), DEINT_SLOTS, crate::find_nnedi_weights(), props.limits.max_compute_shared_memory_size, integrated)?);
                    }
                    if self.deint_runs >= DEINT_SLOTS {
                        anyhow::bail!("more than {DEINT_SLOTS} deinterlaced pictures in one recording");
                    }
                    let si = self.deint_runs;
                    self.deint_runs += 1;
                    let find = |f: Option<&VideoFrame>| f.and_then(|f| self.cache.find((1, f.index()))).unwrap_or(gi);
                    let (pi, ni) = (find(prev), find(next));
                    let d = self.deint.as_mut().unwrap();
                    // No reuse of an earlier result: a recording that never ran may
                    // have left the slot's key behind.
                    d.invalidate();
                    if let Some(out) = d.run(cb, si, params, self.cache.get(pi), self.cache.get(gi), self.cache.get(ni), &mut self.graveyard, self.frame_no)? {
                        views = out;
                        planes_layout = vk::ImageLayout::GENERAL;
                    }
                }
                let cf = ChainFrame {
                    planes: &views,
                    planes_layout,
                    plane_sizes: &sizes,
                    layout: frame.layout,
                    color,
                    seq,
                    target,
                    input: (frame.width(), frame.height()),
                    space,
                    want,
                };
                chain.run_merge(cb, 0, &cf)?.context("chain produced nothing")
            }
            FxInput::Picture(pic) => {
                let sizes = [(pic.w, pic.h)];
                let cf = ChainFrame {
                    planes: &[pic.view],
                    planes_layout: vk::ImageLayout::GENERAL,
                    plane_sizes: &sizes,
                    layout: brp_media::FrameLayout::Rgba16,
                    color,
                    seq,
                    target,
                    input: (pic.w, pic.h),
                    space,
                    want,
                };
                chain.run_merge(cb, 0, &cf)?.context("chain produced nothing")
            }
        }
    }

    /// The recorded work was dropped without running: frames it was to upload are
    /// not on the GPU after all. Then `end`.
    pub fn abort(&mut self) {
        self.cache.forget(&mut self.graveyard, self.frame_no);
    }

    /// The recorded work has finished: release what it no longer needs.
    pub fn end(&mut self) {
        let alloc = (*self.ctx.allocator).clone();
        self.graveyard.collect(&self.ctx.device, &alloc, self.frame_no + 10, 1);
    }

    pub fn has_effects(&self) -> bool {
        self.chain.as_ref().unwrap().has_active()
    }

    /// Run the chain on `frame` (picture number `seq`) and read the output back.
    /// Also returns the GPU time of the submission.
    pub fn run(&mut self, frame: &VideoFrame, color: ColorInfo, seq: i64, target: (u32, u32)) -> Result<(Rgba, std::time::Duration)> {
        let device = self.ctx.device.clone();
        let alloc = (*self.ctx.allocator).clone();
        self.frame_no += 1;
        unsafe {
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let bytes = self.cache.upload_size(1, &[frame]);
            self.staging.reset(&device, &alloc, bytes)?;
            self.cache.ensure(&device, &alloc, cb, &mut self.staging, &mut self.graveyard, self.frame_no, 1, frame)?;
            let gi = self.cache.find((1, frame.index())).unwrap();
            let views = self.cache.get(gi).views();
            let sizes: Vec<(u32, u32)> = self.cache.get(gi).planes.iter().map(|p| (p.w, p.h)).collect();
            let chain = self.chain.as_mut().unwrap();
            chain.begin(0)?;
            let cf = ChainFrame {
                planes: &views,
                planes_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                plane_sizes: &sizes,
                layout: frame.layout,
                color,
                seq,
                target,
                input: (frame.width(), frame.height()),
                space: Space::Source,
                want: Some(Space::Source),
            };
            let out = chain.run(cb, 0, &cf)?.context("no active effect")?;
            let (w, h) = (out.w, out.h);
            let image = chain.image_of(out.view).context("output image")?;
            let total = w as u64 * h as u64 * 8;
            let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(total).usage(vk::BufferUsageFlags::TRANSFER_DST), None)?;
            let req = device.get_buffer_memory_requirements(buffer);
            let ralloc = alloc.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "readback",
                requirements: req,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            device.bind_buffer_memory(buffer, ralloc.memory(), ralloc.offset())?;
            let mb = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb)));
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::GENERAL, buffer, &[region]);
            device.end_command_buffer(cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let t = std::time::Instant::now();
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            {
                let _q = self.ctx.lock_queue();
                device.queue_submit2(self.ctx.queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)?;
            }
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            let elapsed = t.elapsed();
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.pool, &[cb]);
            let raw = ralloc.mapped_slice().context("readback not mapped")?;
            let data = raw[..total as usize]
                .chunks_exact(8)
                .map(|c| {
                    let g = |i: usize| f16_to_f32(u16::from_le_bytes([c[i], c[i + 1]]));
                    [g(0), g(2), g(4), g(6)]
                })
                .collect();
            alloc.lock().unwrap().free(ralloc)?;
            device.destroy_buffer(buffer, None);
            self.graveyard.collect(&device, &alloc, self.frame_no + 10, 1);
            Ok((Rgba { w, h, data }, elapsed))
        }
    }
}

impl Drop for OfflineEffects {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.wait_idle();
            self.chain = None;
            self.deint = None;
            let alloc = (*self.ctx.allocator).clone();
            self.cache.destroy(&self.ctx.device, &alloc);
            self.staging.destroy(&self.ctx.device, &alloc);
            self.graveyard.destroy_all(&self.ctx.device, &alloc);
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}

/// What `OfflineEffects::record` processes.
pub enum FxInput<'a> {
    /// A decoded frame (uploaded if not on the GPU yet), deinterlaced first when
    /// `deint` is set (with its neighbouring frames).
    Frame { frame: &'a VideoFrame, prev: Option<&'a VideoFrame>, next: Option<&'a VideoFrame>, deint: Option<crate::DeintParams> },
    /// A picture already on the GPU.
    Picture(&'a crate::compute::Picture),
}
