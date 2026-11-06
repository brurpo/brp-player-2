//! Vulkan renderer: video pass + egui overlay, presented to a window.

mod color;
mod context;
mod deint;
mod frames;
mod gpu;
mod nnedi;
pub mod offline;
mod video;

pub use color::{video_push, VideoPush};
pub use context::{GpuContext, GpuDevice, GpuInstance};
pub use deint::{DeintMethod, DeintParams};
pub use nnedi::{NnediConfig, NnediWeights};

use anyhow::Result;
use ash::vk;
use brp_media::{ColorInfo, VideoFrame};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use deint::Deinterlacer;
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
    frame_cache: FrameCache,
    staging: Vec<Staging>,
    graveyard: Graveyard,
    /// Frames submitted so far (for deferred destruction).
    frame_no: u64,
    frames: Vec<Frame>,
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
    ctx: GpuContext,
}

impl Renderer {
    pub fn new(inst: GpuInstance, window: &(impl HasDisplayHandle + HasWindowHandle), width: u32, height: u32, gpu: Option<&str>) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let ctx = GpuContext::new(inst, window, gpu)?;
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

            log::debug!("renderer: pools+formats {:?}", t0.elapsed());
            let video = VideoPass::new(device, format, FRAMES_IN_FLIGHT)?;
            let props = ctx.instance.get_physical_device_properties(ctx.physical);
            let integrated = props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU;
            let deint = Deinterlacer::new(device, (*ctx.allocator).clone(), FRAMES_IN_FLIGHT, find_nnedi_weights(), props.limits.max_compute_shared_memory_size, integrated)?;
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
                frame_cache: FrameCache::new(12),
                staging: (0..FRAMES_IN_FLIGHT).map(|_| Staging::new()).collect(),
                graveyard: Graveyard::default(),
                frame_no: 0,
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
                ctx,
            };
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

    pub fn gpu_name(&self) -> &str {
        &self.ctx.device_name
    }

    pub fn list_gpus(&self) -> Vec<GpuDevice> {
        self.ctx.list_devices()
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
                egui.set_texture(self.ctx.queue, self.upload_pool, *id, d)?;
            }
        }
        if !delta.free.is_empty() {
            unsafe { self.ctx.device.device_wait_idle()? };
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
        if self.needs_rebuild {
            self.build_swapchain(self.extent.width, self.extent.height)?;
        }
        let device = self.ctx.device.clone();
        let fi = self.frame_index;
        let frame = self.frames[fi];
        unsafe {
            device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
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
            let video_push = match &video {
                Some(v) => self.prepare_video(cb, fi, v)?,
                None => None,
            };

            let image = self.images[image_index as usize];
            let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
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
                self.video.as_ref().unwrap().draw(cb, fi, push);
            }
            self.egui.as_mut().unwrap().cmd_draw(cb, self.extent, pixels_per_point, primitives)?;
            device.cmd_end_rendering(cb);

            let to_present = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
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

    /// Upload the frames this draw needs, deinterlace if asked, bind the result for
    /// the video pass. Recorded before rendering starts.
    fn prepare_video(&mut self, cb: vk::CommandBuffer, fi: usize, v: &VideoDraw<'_>) -> Result<Option<VideoPush>> {
        let device = self.ctx.device.clone();
        let alloc = (*self.ctx.allocator).clone();
        let deint = v.deint.filter(|_| matches!(v.frame.layout, brp_media::FrameLayout::Yuv { .. }));
        let mut list: Vec<&VideoFrame> = vec![v.frame];
        if deint.is_some() {
            list.extend(v.prev);
            list.extend(v.next);
        }
        let bytes = self.frame_cache.upload_size(v.source_id, &list);
        self.staging[fi].reset(&device, &alloc, bytes)?;
        for f in &list {
            self.frame_cache.ensure(&device, &alloc, cb, &mut self.staging[fi], &mut self.graveyard, self.frame_no, v.source_id, f)?;
        }
        let cur_i = self.frame_cache.find((v.source_id, v.frame.index())).expect("just uploaded");
        let find_or_cur = |f: Option<&VideoFrame>| f.and_then(|f| self.frame_cache.find((v.source_id, f.index()))).unwrap_or(cur_i);
        let (views, layout) = match deint {
            Some(params) => {
                let (pi, ni) = (find_or_cur(v.prev), find_or_cur(v.next));
                let fc = &self.frame_cache;
                match self.deint.as_mut().unwrap().run(cb, fi, params, fc.get(pi), fc.get(cur_i), fc.get(ni), &mut self.graveyard, self.frame_no)? {
                    Some(out) => (out, vk::ImageLayout::GENERAL),
                    None => (fc.get(cur_i).views(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                }
            }
            None => (self.frame_cache.get(cur_i).views(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        };
        self.video.as_mut().unwrap().bind(fi, &views, layout, v.nearest);

        let (w, h) = (self.extent.width as f32, self.extent.height as f32);
        let mut push = color::video_push(v.frame.layout, &v.color, 12.0);
        push.misc[3] = (v.rotation / 90 % 4) as f32;
        push.rect = [v.rect[0] / w * 2.0 - 1.0, v.rect[1] / h * 2.0 - 1.0, v.rect[2] / w * 2.0 - 1.0, v.rect[3] / h * 2.0 - 1.0];
        Ok(Some(push))
    }

    /// Why NNEDI3 can't be used (weights missing/invalid), if so.
    pub fn nnedi_error(&self) -> Option<&str> {
        self.deint.as_ref().and_then(|d| d.weights_error())
    }

    fn build_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            let device = &self.ctx.device.clone();
            device.device_wait_idle()?;
            let caps = self.ctx.surface_fn.get_physical_device_surface_capabilities(self.ctx.physical, self.ctx.surface)?;
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
            if extent.width == 0 || extent.height == 0 {
                return Ok(());
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
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
                        .format(self.format)
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
            let _ = self.ctx.device.device_wait_idle();
            self.egui = None;
            self.video = None;
            self.deint = None;
            let alloc = (*self.ctx.allocator).clone();
            self.frame_cache.destroy(&self.ctx.device, &alloc);
            for s in &mut self.staging {
                s.destroy(&self.ctx.device, &alloc);
            }
            self.graveyard.destroy_all(&self.ctx.device, &alloc);
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
fn find_nnedi_weights() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("BRP_NNEDI_WEIGHTS") {
        return Some(p.into());
    }
    let exe = std::env::current_exe().ok()?;
    let p = exe.parent()?.join("nnedi3_weights.bin");
    p.exists().then_some(p)
}
