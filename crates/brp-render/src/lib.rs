//! Vulkan renderer: video pass + egui overlay, presented to a window.

mod color;
mod composite;
mod context;
mod deint;
pub mod effects;
pub mod gpudec;
mod frames;
pub mod hdr;
mod gpu;
mod nnedi;
pub mod offline;
pub mod compute;
pub mod offline_fx;
pub mod popup;
pub mod shader_cache;
pub mod usershader;
mod video;
pub mod workspace;

pub use color::{video_push, VideoPush};
pub use context::{GpuContext, GpuDevice, GpuInstance};
pub use popup::{Backdrop, GlassParams, PopupRenderer};
pub use deint::{DeintMethod, DeintParams};
pub use nnedi::{NnediConfig, NnediWeights};

use anyhow::Result;
use ash::vk;
use std::sync::Arc;
use brp_media::{ColorInfo, VideoFrame};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use deint::Deinterlacer;
use effects::EffectChain;
pub use effects::{EffectSpec, EffectStatus};
pub use usershader::Transfer;
pub use workspace::Space;
use frames::FrameCache;
use gpu::{Graveyard, Staging};
use video::VideoPass;

const FRAMES_IN_FLIGHT: usize = 2;

/// What to draw for the video this frame.
pub struct VideoDraw<'a> {
    pub frame: &'a VideoFrame,
    pub color: ColorInfo,
    /// Destination rectangle in physical pixels: x0, y0, x1, y1.
    pub rect: [f32; 4],
    /// Identifies the media source, so a new file never reuses an old upload.
    pub source_id: u64,
    /// Pixel-exact sampling (used at 1:1 and above).
    pub nearest: bool,
    /// Clockwise display rotation in degrees (0, 90, 180, 270).
    pub rotation: u32,
    /// Deinterlace with these neighbours (None = show the frame as is). Missing
    /// neighbours (start/end of file) fall back to the current frame.
    pub deint: Option<DeintParams>,
    pub prev: Option<&'a VideoFrame>,
    pub next: Option<&'a VideoFrame>,
    /// Deterministic position (frame, or 2*frame+field when showing fields); `frame`
    /// in shaders, and consecutive values mean "next picture" for stateful effects.
    pub seq: i64,
    /// Pictures just before this one, oldest first, used to rebuild the history of
    /// stateful effects after a jump.
    pub history: Vec<HistoryFrame<'a>>,
    /// Effect output size for OUTPUT in shader expressions (display or export size).
    pub target: (u32, u32),
    /// Show this GPU picture instead of uploading `frame` (background effects made
    /// it on the GPU; `frame` still gives timing and metadata).
    pub picture: Option<Arc<compute::Picture>>,
    /// Show the frame without the live effects (while scrubbing: only the frame the
    /// user stops on is processed).
    pub raw: bool,
}

pub struct HistoryFrame<'a> {
    pub frame: &'a VideoFrame,
    pub prev: Option<&'a VideoFrame>,
    pub next: Option<&'a VideoFrame>,
    pub seq: i64,
    pub second_field: bool,
}

#[derive(Clone, Copy)]
struct Frame {
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    fence: vk::Fence,
}

pub struct Renderer {
    // Drop order matters: everything using the allocator before the context.
    egui: Option<egui_ash_renderer::Renderer<egui_ash_renderer::allocator::GpuAllocator>>,
    video: Option<VideoPass>,
    deint: Option<Deinterlacer>,
    effects: Option<EffectChain>,
    /// (chain key, seq) of the last processed picture, and its output.
    chain_last: Option<(u64, i64)>,
    chain_out: Option<effects::ChainOutput>,
    frame_cache: FrameCache,
    staging: Vec<Staging>,
    graveyard: Graveyard,
    /// Frames submitted so far (for deferred destruction).
    frame_no: u64,
    frames: Vec<Frame>,
    /// GPU pictures read by each frame in flight (kept until its fence).
    kept: Vec<Option<Arc<compute::Picture>>>,
    frame_index: usize,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_done: Vec<vk::Semaphore>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    upload_pool: vk::CommandPool,
    needs_rebuild: bool,
    pub clear_color: [f32; 4],
    /// Blurred copy of each frame for popups (see `popup`), made while `backdrop_on`.
    backdrop: Arc<Backdrop>,
    backdrop_on: bool,
    backdrop_imgs: Option<popup::BackdropImages>,
    /// The swapchain images can be blitted from (format and usage allow it).
    can_capture: bool,
    /// HDR screens (DESIGN.md §7b): the screen the window is on when it shows HDR (kept
    /// up to date by a thread, `hdr_watch`), the surface's HDR10 format if it has one,
    /// and while HDR is used, the layers the window is drawn in and the pass that puts
    /// them into the HDR10 swapchain.
    hdr: Option<hdr::HdrScreen>,
    hdr_watch: Arc<std::sync::Mutex<Option<hdr::HdrScreen>>>,
    hdr10_format: Option<vk::Format>,
    composite: Option<composite::CompositePass>,
    layers: Option<composite::Layers>,
    /// The swapchain's format (`format` is the SDR one the UI is drawn in).
    swap_format: vk::Format,
    ctx: Arc<GpuContext>,
}

impl Renderer {
    pub fn new(inst: GpuInstance, window: &(impl HasDisplayHandle + HasWindowHandle), width: u32, height: u32, gpu: Option<&str>) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let ctx = Arc::new(GpuContext::new(inst, window, gpu)?);
        log::debug!("renderer: context {:?}", t0.elapsed());
        unsafe {
            let device = &ctx.device;
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let upload_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;
            let cmds = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES_IN_FLIGHT as u32),
            )?;
            let mut frames = Vec::new();
            for cmd in cmds {
                frames.push(Frame {
                    cmd,
                    image_available: device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                    fence: device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?,
                });
            }

            let formats = ctx.surface_fn.get_physical_device_surface_formats(ctx.physical, ctx.surface)?;
            let format = formats
                .iter()
                .find(|f| {
                    matches!(f.format, vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM)
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .map(|f| f.format)
                .unwrap_or(formats[0].format);
            // HDR10 (PQ, Rec.2020), used while the window is on an HDR screen.
            let hdr10_format = formats
                .iter()
                .find(|f| f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT && matches!(f.format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32))
                .map(|f| f.format);
            let hwnd = match window.window_handle().map(|h| h.as_raw()) {
                Ok(raw_window_handle::RawWindowHandle::Win32(h)) => h.hwnd.get(),
                _ => 0,
            };
            let th = std::time::Instant::now();
            let hdr = if hdr10_format.is_some() { hdr::screen_of(hwnd) } else { None };
            log::info!("HDR: surface HDR10 {:?}, screen {:?} ({:?})", hdr10_format, hdr, th.elapsed());
            let hdr_watch = Arc::new(std::sync::Mutex::new(hdr));
            if hdr10_format.is_some() {
                // The window may move to another screen, or HDR be switched on or off:
                // checked every 2 s off the drawing thread (a few ms each).
                let watch = Arc::downgrade(&hdr_watch);
                let _ = std::thread::Builder::new().name("brp-hdr-watch".into()).spawn(move || loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    let Some(w) = watch.upgrade() else { break };
                    let now = hdr::screen_of(hwnd);
                    *w.lock().unwrap() = now;
                });
            }
            let composite = match hdr10_format {
                Some(f) => Some(composite::CompositePass::new(device, &[f, format], FRAMES_IN_FLIGHT)?),
                None => None,
            };

            log::debug!("renderer: pools+formats {:?}", t0.elapsed());
            let video = VideoPass::new(device, format, FRAMES_IN_FLIGHT)?;
            let props = ctx.instance.get_physical_device_properties(ctx.physical);
            let integrated = props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU;
            let deint = Deinterlacer::new(device, (*ctx.allocator).clone(), FRAMES_IN_FLIGHT, find_nnedi_weights(), props.limits.max_compute_shared_memory_size, integrated)?;
            let chain = EffectChain::new(device, (*ctx.allocator).clone(), FRAMES_IN_FLIGHT, props.limits.min_uniform_buffer_offset_alignment, ctx.pipeline_cache)?;
            let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
                (*ctx.allocator).clone(),
                device.clone(),
                egui_ash_renderer::RenderMode::DynamicRendering(egui_ash_renderer::DynamicRendering {
                    color_attachment_format: format,
                    depth_attachment_format: None,
                    stencil_attachment_format: None,
                }),
                egui_ash_renderer::Options {
                    in_flight_frames: FRAMES_IN_FLIGHT,
                    srgb_framebuffer: false,
                    ..Default::default()
                },
            )?;

            log::debug!("renderer: pipelines {:?}", t0.elapsed());
            let mut r = Renderer {
                egui: Some(egui),
                video: Some(video),
                deint: Some(deint),
                effects: Some(chain),
                chain_last: None,
                chain_out: None,
                frame_cache: FrameCache::new(12),
                staging: (0..FRAMES_IN_FLIGHT).map(|_| Staging::new()).collect(),
                graveyard: Graveyard::default(),
                frame_no: 0,
                kept: vec![None; FRAMES_IN_FLIGHT],
                frames,
                frame_index: 0,
                swapchain: vk::SwapchainKHR::null(),
                images: Vec::new(),
                views: Vec::new(),
                render_done: Vec::new(),
                format,
                extent: vk::Extent2D { width, height },
                pool,
                upload_pool,
                needs_rebuild: false,
                clear_color: [0.07, 0.07, 0.08, 1.0],
                backdrop: Arc::new(Backdrop::default()),
                backdrop_on: false,
                backdrop_imgs: None,
                can_capture: false,
                hdr,
                hdr_watch,
                hdr10_format,
                composite,
                layers: None,
                swap_format: format,
                ctx,
            };
            let fp = ctx_props_features(&r.ctx, format);
            let caps = r.ctx.surface_fn.get_physical_device_surface_capabilities(r.ctx.physical, r.ctx.surface)?;
            r.can_capture = fp.contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
                && caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_SRC);
            if !r.can_capture {
                log::info!("popups: no backdrop on this GPU (format {format:?} features {fp:?}, swapchain usage {:?}): plain panels", caps.supported_usage_flags);
            }
            let ts = std::time::Instant::now();
            r.build_swapchain(width, height)?;
            log::debug!("renderer: swapchain {:?}", ts.elapsed());
            Ok(r)
        }
    }

    /// Integrated GPU (shared memory with the CPU, usually much slower).
    pub fn is_integrated(&self) -> bool {
        unsafe { self.ctx.instance.get_physical_device_properties(self.ctx.physical).device_type != vk::PhysicalDeviceType::DISCRETE_GPU }
    }

    /// The GPU context, shared with background compute (AI) on the same device.
    pub fn gpu(&self) -> Arc<GpuContext> {
        self.ctx.clone()
    }

    pub fn gpu_name(&self) -> &str {
        &self.ctx.device_name
    }

    pub fn list_gpus(&self) -> Vec<GpuDevice> {
        self.ctx.list_devices()
    }

    /// The blurred picture popups show through their glass.
    pub fn backdrop(&self) -> Arc<Backdrop> {
        self.backdrop.clone()
    }

    /// Keep the backdrop up to date (while popups are open). No effect if the
    /// swapchain cannot be copied from.
    pub fn set_backdrop(&mut self, on: bool) {
        self.backdrop_on = on && self.can_capture;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width != self.extent.width || height != self.extent.height {
            self.extent = vk::Extent2D { width, height };
            self.needs_rebuild = true;
        }
    }

    /// Upload/free egui textures. Call before `render`.
    pub fn update_textures(&mut self, mut delta: egui::TexturesDelta) -> Result<()> {
        let result = self.apply_textures(&delta);
        // egui panics if a delta is dropped without being marked handled.
        delta.clear();
        result
    }

    fn apply_textures(&mut self, delta: &egui::TexturesDelta) -> Result<()> {
        let egui = self.egui.as_mut().unwrap();
        for (id, deltas) in &delta.set {
            for d in deltas.iter() {
                let _q = self.ctx.lock_queue();
                egui.set_texture(self.ctx.queue, self.upload_pool, *id, d)?;
            }
        }
        if !delta.free.is_empty() {
            // Only this window's frames can use egui's textures: wait for those, not for
            // the whole GPU (the AI worker's long submissions stalled every redraw).
            unsafe { self.ctx.device.wait_for_fences(&self.frames.iter().map(|f| f.fence).collect::<Vec<_>>(), true, u64::MAX)? };
            for id in &delta.free {
                egui.free_texture(*id)?;
            }
        }
        Ok(())
    }

    /// Draw one frame. Returns false if nothing was presented (minimised window).
    pub fn render(&mut self, video: Option<VideoDraw<'_>>, primitives: &[egui::ClippedPrimitive], pixels_per_point: f32) -> Result<bool> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(false);
        }
        // Moved to another screen, or HDR switched on or off: a new swapchain.
        if self.hdr10_format.is_some() {
            let now = *self.hdr_watch.lock().unwrap();
            if now != self.hdr {
                log::info!("HDR screen: {:?} -> {:?}", self.hdr, now);
                self.hdr = now;
                self.needs_rebuild = true;
            }
        }
        if self.needs_rebuild {
            self.build_swapchain(self.extent.width, self.extent.height)?;
        }
        let device = self.ctx.device.clone();
        let fi = self.frame_index;
        let frame = self.frames[fi];
        unsafe {
            device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
            self.kept[fi] = None;
            let (image_index, _suboptimal) =
                match self.ctx.swapchain_fn.acquire_next_image(self.swapchain, u64::MAX, frame.image_available, vk::Fence::null()) {
                    Ok(r) => r,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.needs_rebuild = true;
                        return Ok(false);
                    }
                    Err(e) => return Err(e.into()),
                };
            device.reset_fences(&[frame.fence])?;
            let cb = frame.cmd;
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;

            self.frame_no += 1;
            self.graveyard.collect(&device, &self.ctx.allocator, self.frame_no, FRAMES_IN_FLIGHT as u64 + 1);
            if self.frame_no % 300 == 0 && std::env::var_os("BRP_MEM_LOG").is_some() {
                log_gpu_memory(&self.ctx.allocator);
            }
            let video_push = match &video {
                Some(v) => self.prepare_video(cb, fi, v)?,
                None => None,
            };

            let image = self.images[image_index as usize];
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            // Popups open: keep a blurred copy of each frame for their glass.
            if self.backdrop_on && self.backdrop_imgs.is_none() {
                match popup::BackdropImages::new(&self.ctx, self.format, self.extent.width, self.extent.height) {
                    Ok(imgs) => {
                        let mut st = self.backdrop.state.lock().unwrap();
                        st.view = imgs.eighth.view;
                        st.size = (imgs.eighth.w, imgs.eighth.h);
                        st.source = (self.extent.width, self.extent.height);
                        st.generation += 1;
                        st.valid = false;
                        drop(st);
                        self.backdrop_imgs = Some(imgs);
                    }
                    Err(e) => {
                        log::warn!("popup backdrop: {e:#}");
                        self.can_capture = false;
                        self.backdrop_on = false;
                    }
                }
            }
            let (last_layout, last_stage) = if self.layers.is_some() {
                self.record_hdr(cb, fi, image_index as usize, video_push, primitives, pixels_per_point)?
            } else {
            let to_color = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_color)));

            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: self.clear_color } })];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachment));
            device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport { width: self.extent.width as f32, height: self.extent.height as f32, max_depth: 1.0, ..Default::default() }],
            );
            device.cmd_set_scissor(cb, 0, &[area]);
            if let Some(push) = &video_push {
                self.video.as_ref().unwrap().draw(cb, fi, push, false);
            }
            self.egui.as_mut().unwrap().cmd_draw(cb, self.extent, pixels_per_point, primitives)?;
            device.cmd_end_rendering(cb);

            let mut last_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            let mut last_stage = (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
            if let (true, Some(imgs)) = (self.backdrop_on, &self.backdrop_imgs) {
                use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
                gpu::image_barrier(&device, cb, image, (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL), (S::BLIT, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL));
                imgs.capture(&device, cb, image, self.extent);
                last_layout = L::TRANSFER_SRC_OPTIMAL;
                last_stage = (S::BLIT, A::NONE);
                self.backdrop.state.lock().unwrap().valid = true;
            }
            (last_layout, last_stage)
            };

            let to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(last_stage.0)
                .src_access_mask(last_stage.1)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(last_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_present)));
            device.end_command_buffer(cb)?;

            let render_done = self.render_done[image_index as usize];
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.image_available)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [vk::SemaphoreSubmitInfo::default().semaphore(render_done).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let _q = self.ctx.lock_queue();
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            device.queue_submit2(
                self.ctx.queue,
                &[vk::SubmitInfo2::default().wait_semaphore_infos(&wait).command_buffer_infos(&cbs).signal_semaphore_infos(&signal)],
                frame.fence,
            )?;

            let swapchains = [self.swapchain];
            let indices = [image_index];
            let waits = [render_done];
            match self.ctx.swapchain_fn.queue_present(
                self.ctx.queue,
                &vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices),
            ) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_rebuild = true,
                Err(e) => return Err(e.into()),
            }
        }
        self.frame_index = (fi + 1) % FRAMES_IN_FLIGHT;
        Ok(true)
    }

    /// HDR screen (DESIGN.md §7b): the video as light into the scene layer, the UI as
    /// for an SDR window into its own transparent layer, both put together into the
    /// HDR10 swapchain image; while popups are open, also an SDR picture of the window
    /// for their glass. Returns the swapchain image's last layout and access.
    #[allow(clippy::type_complexity)]
    unsafe fn record_hdr(
        &mut self,
        cb: vk::CommandBuffer,
        fi: usize,
        image_index: usize,
        video_push: Option<VideoPush>,
        primitives: &[egui::ClippedPrimitive],
        pixels_per_point: f32,
    ) -> Result<(vk::ImageLayout, (vk::PipelineStageFlags2, vk::AccessFlags2))> {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let device = self.ctx.device.clone();
        let screen = self.hdr.unwrap_or(hdr::HdrScreen { sdr_nits: 203.0, peak_nits: 1000.0 });
        let (scene, ui) = {
            let l = self.layers.as_ref().unwrap();
            ((l.scene.image, l.scene.view), (l.ui.image, l.ui.view))
        };
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent };
        let viewport = vk::Viewport { width: self.extent.width as f32, height: self.extent.height as f32, max_depth: 1.0, ..Default::default() };
        let attach = |view: vk::ImageView, load: vk::AttachmentLoadOp, clear: [f32; 4]| {
            [vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(load)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: clear } })]
        };
        unsafe {
            // The last frame's composite read the layers.
            for img in [scene.0, ui.0] {
                gpu::image_barrier(&device, cb, img, (S::FRAGMENT_SHADER, A::NONE, L::UNDEFINED), (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL));
            }
            // Scene: the background (the clear colour as light) and the video.
            let srgb = |e: f32| if e <= 0.04045 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) };
            let c = self.clear_color;
            let a = attach(scene.1, vk::AttachmentLoadOp::CLEAR, [srgb(c[0]), srgb(c[1]), srgb(c[2]), 1.0]);
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&a));
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            if let Some(mut push) = video_push {
                push.mode[3] |= 2;
                push.m[0][3] = screen.sdr_nits;
                push.m[1][3] = screen.peak_nits;
                self.video.as_ref().unwrap().draw(cb, fi, &push, true);
            }
            device.cmd_end_rendering(cb);
            // UI: exactly as on an SDR window, over nothing.
            let a = attach(ui.1, vk::AttachmentLoadOp::CLEAR, [0.0; 4]);
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&a));
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            self.egui.as_mut().unwrap().cmd_draw(cb, self.extent, pixels_per_point, primitives)?;
            device.cmd_end_rendering(cb);
            for img in [scene.0, ui.0] {
                gpu::image_barrier(&device, cb, img, (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL), (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::SHADER_READ_ONLY_OPTIMAL));
            }
            let comp = self.composite.as_ref().unwrap();
            comp.bind(fi, self.layers.as_ref().unwrap());
            // Into the swapchain image, as HDR10.
            let image = self.images[image_index];
            gpu::image_barrier(&device, cb, image, (S::COLOR_ATTACHMENT_OUTPUT, A::NONE, L::UNDEFINED), (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL));
            let a = attach(self.views[image_index], vk::AttachmentLoadOp::DONT_CARE, [0.0; 4]);
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&a));
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
            comp.draw(cb, fi, self.swap_format, false, screen.sdr_nits);
            device.cmd_end_rendering(cb);

            // Popups' glass: an SDR picture of the window (as it would be on an SDR screen).
            if let (true, Some(_)) = (self.backdrop_on, &self.backdrop_imgs) {
                if self.layers.as_ref().unwrap().sdr.is_none() {
                    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
                    let img = gpu::create_image(&device, &self.ctx.allocator, self.extent.width, self.extent.height, self.format, usage, "hdr backdrop")?;
                    self.layers.as_mut().unwrap().sdr = Some(img);
                }
                let (simg, sview) = {
                    let s = self.layers.as_ref().unwrap().sdr.as_ref().unwrap();
                    (s.image, s.view)
                };
                gpu::image_barrier(&device, cb, simg, (S::BLIT, A::NONE, L::UNDEFINED), (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL));
                let a = attach(sview, vk::AttachmentLoadOp::DONT_CARE, [0.0; 4]);
                device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&a));
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(cb, 0, &[area]);
                self.composite.as_ref().unwrap().draw(cb, fi, self.format, true, screen.sdr_nits);
                device.cmd_end_rendering(cb);
                gpu::image_barrier(&device, cb, simg, (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL), (S::BLIT, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL));
                self.backdrop_imgs.as_ref().unwrap().capture(&device, cb, simg, self.extent);
                self.backdrop.state.lock().unwrap().valid = true;
            }
        }
        Ok((L::COLOR_ATTACHMENT_OPTIMAL, (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE)))
    }

    /// Upload the frames this draw needs, deinterlace if asked, bind the result for
    /// the video pass. Recorded before rendering starts.
    fn prepare_video(&mut self, cb: vk::CommandBuffer, fi: usize, v: &VideoDraw<'_>) -> Result<Option<VideoPush>> {
        let device = self.ctx.device.clone();
        let alloc = (*self.ctx.allocator).clone();
        let yuv = matches!(v.frame.layout, brp_media::FrameLayout::Yuv { .. });
        let deint = v.deint.filter(|_| yuv);
        let chain_on = !v.raw && self.effects.as_ref().is_some_and(|c| c.has_active());
        if let Some(pic) = v.picture.clone() {
            return self.prepare_picture(cb, fi, v, pic, chain_on);
        }

        // Everything this draw may need on the GPU.
        let mut list: Vec<&VideoFrame> = vec![v.frame];
        if deint.is_some() {
            list.extend(v.prev);
            list.extend(v.next);
        }
        if chain_on {
            for h in &v.history {
                list.push(h.frame);
                if deint.is_some() {
                    list.extend(h.prev);
                    list.extend(h.next);
                }
            }
        }
        let bytes = self.frame_cache.upload_size(v.source_id, &list);
        self.staging[fi].reset(&device, &alloc, bytes)?;
        for f in &list {
            self.frame_cache.ensure(&device, &alloc, cb, &mut self.staging[fi], &mut self.graveyard, self.frame_no, v.source_id, f)?;
        }

        let (w, h) = (self.extent.width as f32, self.extent.height as f32);
        let rect = [v.rect[0] / w * 2.0 - 1.0, v.rect[1] / h * 2.0 - 1.0, v.rect[2] / w * 2.0 - 1.0, v.rect[3] / h * 2.0 - 1.0];

        if chain_on {
            // Key: chain + source + how frames are prepared (+ output size for OUTPUT).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut hs = std::collections::hash_map::DefaultHasher::new();
                self.effects.as_ref().unwrap().key().hash(&mut hs);
                v.source_id.hash(&mut hs);
                deint.map(|d| (d.method as u8, d.tff, d.light)).hash(&mut hs);
                v.target.hash(&mut hs);
                // More history arriving (after a jump) must trigger a re-run.
                v.history.len().hash(&mut hs);
                hs.finish()
            };
            if self.chain_last != Some((key, v.seq)) || self.chain_out.is_none() {
                self.effects.as_mut().unwrap().begin(fi)?;
                let chain = self.effects.as_ref().unwrap();
                let warm = chain.stateful() && self.chain_last != Some((key, v.seq - 1));
                if warm {
                    self.effects.as_mut().unwrap().reset_history();
                    for hf in v.history.iter().filter(|hf| hf.seq < v.seq) {
                        let d = deint.map(|d| DeintParams { second_field: hf.second_field, ..d });
                        self.run_chain(cb, fi, v, hf.frame, hf.prev, hf.next, d, hf.seq)?;
                    }
                }
                self.chain_out = self.run_chain(cb, fi, v, v.frame, v.prev, v.next, deint, v.seq)?;
                self.chain_last = Some((key, v.seq));
            }
            if let Some(out) = &self.chain_out {
                let view = out.view;
                self.video.as_mut().unwrap().bind(fi, &[view], vk::ImageLayout::GENERAL, v.nearest);
                let mut push = color::video_push(brp_media::FrameLayout::Rgba16, &v.color, 12.0);
                if !v.frame.layout.has_alpha() {
                    push.mode[2] = 0;
                    push.swz[3] = -1;
                }
                push.misc[3] = (v.rotation / 90 % 4) as f32;
                push.rect = rect;
                return Ok(Some(push));
            }
        } else {
            self.chain_last = None;
            self.chain_out = None;
        }

        let (views, layout, _) = self.input_planes(cb, fi, v.source_id, v.frame, v.prev, v.next, deint)?;
        self.video.as_mut().unwrap().bind(fi, &views, layout, v.nearest);
        let mut push = color::video_push(v.frame.layout, &v.color, 12.0);
        push.misc[3] = (v.rotation / 90 % 4) as f32;
        push.rect = rect;
        Ok(Some(push))
    }

    /// A GPU picture made by background effects: sampled directly (no upload, no
    /// deinterlacing), through the live effects if any.
    fn prepare_picture(&mut self, cb: vk::CommandBuffer, fi: usize, v: &VideoDraw<'_>, pic: Arc<compute::Picture>, chain_on: bool) -> Result<Option<VideoPush>> {
        let (w, h) = (self.extent.width as f32, self.extent.height as f32);
        let rect = [v.rect[0] / w * 2.0 - 1.0, v.rect[1] / h * 2.0 - 1.0, v.rect[2] / w * 2.0 - 1.0, v.rect[3] / h * 2.0 - 1.0];
        let mut view = pic.view;
        if chain_on {
            let key = {
                use std::hash::{Hash, Hasher};
                let mut hs = std::collections::hash_map::DefaultHasher::new();
                self.effects.as_ref().unwrap().key().hash(&mut hs);
                v.source_id.hash(&mut hs);
                (Arc::as_ptr(&pic) as usize).hash(&mut hs);
                v.target.hash(&mut hs);
                hs.finish()
            };
            if self.chain_last != Some((key, v.seq)) || self.chain_out.is_none() {
                self.effects.as_mut().unwrap().begin(fi)?;
                let sizes = [(pic.w, pic.h)];
                let cf = effects::ChainFrame {
                    planes: &[pic.view],
                    planes_layout: vk::ImageLayout::GENERAL,
                    plane_sizes: &sizes,
                    layout: brp_media::FrameLayout::Rgba16,
                    color: v.color,
                    seq: v.seq,
                    target: v.target,
                    input: (pic.w, pic.h),
                    space: Space::Source,
                    want: Some(Space::Source),
                };
                self.chain_out = self.effects.as_mut().unwrap().run(cb, fi, &cf)?;
                self.chain_last = Some((key, v.seq));
            }
            if let Some(out) = &self.chain_out {
                view = out.view;
            }
        } else {
            self.chain_last = None;
            self.chain_out = None;
        }
        self.kept[fi] = Some(pic.clone());
        self.video.as_mut().unwrap().bind(fi, &[view], vk::ImageLayout::GENERAL, v.nearest);
        let mut push = color::video_push(brp_media::FrameLayout::Rgba16, &v.color, 12.0);
        if !pic.alpha {
            push.mode[2] = 0;
            push.swz[3] = -1;
        }
        push.misc[3] = (v.rotation / 90 % 4) as f32;
        push.rect = rect;
        Ok(Some(push))
    }

    /// Planes of `frame` as the display or the effect chain sees them: from the frame
    /// cache, or deinterlaced. Returns views, their layout and plane sizes.
    #[allow(clippy::too_many_arguments)]
    fn input_planes(
        &mut self,
        cb: vk::CommandBuffer,
        fi: usize,
        source: u64,
        frame: &VideoFrame,
        prev: Option<&VideoFrame>,
        next: Option<&VideoFrame>,
        deint: Option<DeintParams>,
    ) -> Result<(Vec<vk::ImageView>, vk::ImageLayout, Vec<(u32, u32)>)> {
        let cur_i = self.frame_cache.find((source, frame.index())).expect("uploaded");
        let sizes: Vec<(u32, u32)> = self.frame_cache.get(cur_i).planes.iter().map(|p| (p.w, p.h)).collect();
        let find_or_cur = |f: Option<&VideoFrame>| f.and_then(|f| self.frame_cache.find((source, f.index()))).unwrap_or(cur_i);
        if let Some(params) = deint {
            let (pi, ni) = (find_or_cur(prev), find_or_cur(next));
            let fc = &self.frame_cache;
            if let Some(out) = self.deint.as_mut().unwrap().run(cb, fi, params, fc.get(pi), fc.get(cur_i), fc.get(ni), &mut self.graveyard, self.frame_no)? {
                return Ok((out, vk::ImageLayout::GENERAL, sizes));
            }
        }
        Ok((self.frame_cache.get(cur_i).views(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, sizes))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chain(
        &mut self,
        cb: vk::CommandBuffer,
        fi: usize,
        v: &VideoDraw<'_>,
        frame: &VideoFrame,
        prev: Option<&VideoFrame>,
        next: Option<&VideoFrame>,
        deint: Option<DeintParams>,
        seq: i64,
    ) -> Result<Option<effects::ChainOutput>> {
        let (views, layout, sizes) = self.input_planes(cb, fi, v.source_id, frame, prev, next, deint)?;
        let cf = effects::ChainFrame {
            planes: &views,
            planes_layout: layout,
            plane_sizes: &sizes,
            layout: frame.layout,
            color: v.color,
            seq,
            target: v.target,
            input: (frame.width(), frame.height()),
            space: Space::Source,
            want: Some(Space::Source),
        };
        self.effects.as_mut().unwrap().run(cb, fi, &cf)
    }

    /// Replace the effect chain (compiles new shaders in the background).
    pub fn set_effects(&mut self, specs: &[EffectSpec], transfer: Transfer) {
        let now = self.frame_no;
        if let Some(c) = self.effects.as_mut() {
            c.set_effects(specs, transfer, &mut self.graveyard, now);
            c.release_idle(&mut self.graveyard, now);
        }
        self.chain_last = None;
    }

    /// Pick up finished shader compiles. True if the picture may change.
    pub fn poll_effects(&mut self) -> bool {
        let now = self.frame_no;
        let changed = self.effects.as_mut().is_some_and(|c| c.poll(&mut self.graveyard, now));
        if changed {
            if let Some(c) = self.effects.as_mut() {
                c.release_idle(&mut self.graveyard, now);
            }
            self.chain_last = None;
        }
        changed
    }

    pub fn effect_status(&self) -> Vec<(u64, EffectStatus)> {
        self.effects.as_ref().map(|c| c.status()).unwrap_or_default()
    }

    /// Size of the picture the effect chain produced last (upscalers change it).
    pub fn effect_output_size(&self) -> Option<(u32, u32)> {
        self.chain_out.as_ref().map(|o| (o.w, o.h))
    }

    /// Pictures of history stateful effects need after a jump (0 if none).
    pub fn effect_warmup(&self) -> u32 {
        self.effects.as_ref().filter(|c| c.stateful()).map(|c| c.warmup()).unwrap_or(0)
    }

    /// Why NNEDI3 can't be used (weights missing/invalid), if so.
    pub fn nnedi_error(&self) -> Option<&str> {
        self.deint.as_ref().and_then(|d| d.weights_error())
    }

    fn build_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            let device = &self.ctx.device.clone();
            self.ctx.wait_idle()?;
            let caps = self.ctx.surface_fn.get_physical_device_surface_capabilities(self.ctx.physical, self.ctx.surface)?;
            // The backdrop follows the window size (made again on the next captured frame).
            if let Some(imgs) = self.backdrop_imgs.take() {
                imgs.destroy(&self.ctx);
                let mut st = self.backdrop.state.lock().unwrap();
                st.valid = false;
                st.generation += 1;
            }
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            self.extent = extent;
            self.needs_rebuild = false;
            // HDR screen: an HDR10 swapchain and the layers the window is drawn in
            // (DESIGN.md §7b); otherwise as always.
            if let Some(l) = self.layers.take() {
                l.destroy(device, &self.ctx.allocator);
            }
            let hdr10 = self.hdr10_format.filter(|_| self.hdr.is_some() && self.composite.is_some());
            let space = if hdr10.is_some() { vk::ColorSpaceKHR::HDR10_ST2084_EXT } else { vk::ColorSpaceKHR::SRGB_NONLINEAR };
            self.swap_format = hdr10.unwrap_or(self.format);
            if extent.width == 0 || extent.height == 0 {
                return Ok(());
            }
            if hdr10.is_some() {
                self.layers = Some(composite::Layers::new(device, &self.ctx.allocator, self.format, extent.width, extent.height)?);
            }
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let old = self.swapchain;
            let swapchain = self.ctx.swapchain_fn.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.ctx.surface)
                    .min_image_count(count)
                    .image_format(self.swap_format)
                    .image_color_space(space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(if self.can_capture { vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC } else { vk::ImageUsageFlags::COLOR_ATTACHMENT })
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )?;
            self.destroy_swapchain_resources();
            if old != vk::SwapchainKHR::null() {
                self.ctx.swapchain_fn.destroy_swapchain(old, None);
            }
            self.swapchain = swapchain;
            self.images = self.ctx.swapchain_fn.get_swapchain_images(swapchain)?;
            for &img in &self.images {
                self.views.push(device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(img)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.swap_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1),
                        ),
                    None,
                )?);
                self.render_done.push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
        }
        Ok(())
    }

    fn destroy_swapchain_resources(&mut self) {
        unsafe {
            for v in self.views.drain(..) {
                self.ctx.device.destroy_image_view(v, None);
            }
            for s in self.render_done.drain(..) {
                self.ctx.device.destroy_semaphore(s, None);
            }
        }
        self.images.clear();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.wait_idle();
            self.egui = None;
            self.video = None;
            self.deint = None;
            self.effects = None;
            self.composite = None;
            if let Some(l) = self.layers.take() {
                l.destroy(&self.ctx.device, &self.ctx.allocator);
            }
            let alloc = (*self.ctx.allocator).clone();
            self.frame_cache.destroy(&self.ctx.device, &alloc);
            for s in &mut self.staging {
                s.destroy(&self.ctx.device, &alloc);
            }
            self.graveyard.destroy_all(&self.ctx.device, &alloc);
            if let Some(imgs) = self.backdrop_imgs.take() {
                imgs.destroy(&self.ctx);
                self.backdrop.state.lock().unwrap().valid = false;
            }
            self.destroy_swapchain_resources();
            self.ctx.swapchain_fn.destroy_swapchain(self.swapchain, None);
            for f in &self.frames {
                self.ctx.device.destroy_semaphore(f.image_available, None);
                self.ctx.device.destroy_fence(f.fence, None);
            }
            self.ctx.device.destroy_command_pool(self.pool, None);
            self.ctx.device.destroy_command_pool(self.upload_pool, None);
        }
    }
}

/// `nnedi3_weights.bin`: `BRP_NNEDI_WEIGHTS`, or next to the executable.
pub(crate) fn find_nnedi_weights() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("BRP_NNEDI_WEIGHTS") {
        return Some(p.into());
    }
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.join("nnedi3_weights.bin");
    p.exists().then_some(p)
}

/// Optimal-tiling features of `format` on the context's GPU.
fn ctx_props_features(ctx: &GpuContext, format: vk::Format) -> vk::FormatFeatureFlags {
    unsafe { ctx.instance.get_physical_device_format_properties(ctx.physical, format).optimal_tiling_features }
}

/// Debug (BRP_MEM_LOG): GPU memory held through the allocator, by allocation name.
fn log_gpu_memory(allocator: &gpu::SharedAllocator) {
    let r = allocator.lock().unwrap().generate_report();
    let mut by: std::collections::HashMap<&str, (u64, usize)> = std::collections::HashMap::new();
    for a in &r.allocations {
        let e = by.entry(a.name.as_str()).or_default();
        e.0 += a.size;
        e.1 += 1;
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    let top: Vec<String> = v.iter().take(10).map(|(n, (b, c))| format!("{n} {}MB/{c}", b >> 20)).collect();
    log::info!(
        "GPU memory: {} MB used of {} MB in {} blocks, {} allocations; {}",
        r.total_allocated_bytes >> 20,
        r.total_capacity_bytes >> 20,
        r.blocks.len(),
        r.allocations.len(),
        top.join(", ")
    );
}
