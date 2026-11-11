//! Popup windows (Info, Settings, Effects) drawn as glass, each with its own surface
//! and swapchain on the player window's device.
//!
//! The player window keeps a small blurred copy of what it last presented (the
//! "backdrop": half-, quarter- and eighth-size images made by three blits right after
//! its frame). A popup samples the eighth image under itself with smooth (cubic
//! B-spline) reads, bends it near its rounded border and draws egui on top. Outside
//! the player window the popup stays translucent and the compositor blurs the desktop.
//! Cost: three blits per player frame while a popup is open,
//! and one full-window pass of ~20 (up to 60 at the rim) texture reads per pixel per
//! popup frame.

use crate::gpu::{self, GpuImage};
use crate::GpuContext;
use anyhow::{Context as _, Result};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::{Arc, Mutex};

const VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glass.vert.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glass.frag.spv"));
const FRAMES: usize = 2;

/// The player window's blurred picture, shared with its popups.
#[derive(Default)]
pub struct Backdrop {
    pub(crate) state: Mutex<BackdropState>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct BackdropState {
    pub view: vk::ImageView,
    /// Size of the sampled (eighth) image.
    pub size: (u32, u32),
    /// Size of the player window it was made from.
    pub source: (u32, u32),
    /// Changes whenever `view` does.
    pub generation: u64,
    /// Holds a picture (written by at least one frame).
    pub valid: bool,
}

/// Images the player renderer blits its frame into (owned by it).
pub(crate) struct BackdropImages {
    pub half: GpuImage,
    pub quarter: GpuImage,
    /// The one popups sample (smoothly, see `glass.frag`).
    pub eighth: GpuImage,
}

impl BackdropImages {
    pub fn new(ctx: &GpuContext, format: vk::Format, w: u32, h: u32) -> Result<Self> {
        let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let half = gpu::create_image(&ctx.device, &ctx.allocator, (w / 2).max(1), (h / 2).max(1), format, usage, "backdrop/2")?;
        let quarter = gpu::create_image(&ctx.device, &ctx.allocator, (w / 4).max(1), (h / 4).max(1), format, usage, "backdrop/4")?;
        let eighth = gpu::create_image(&ctx.device, &ctx.allocator, (w / 8).max(1), (h / 8).max(1), format, usage, "backdrop/8")?;
        Ok(BackdropImages { half, quarter, eighth })
    }

    pub fn destroy(self, ctx: &GpuContext) {
        gpu::destroy_image(&ctx.device, &ctx.allocator, self.half);
        gpu::destroy_image(&ctx.device, &ctx.allocator, self.quarter);
        gpu::destroy_image(&ctx.device, &ctx.allocator, self.eighth);
    }

    /// Record: `src` (in TRANSFER_SRC_OPTIMAL, `extent`) → half → quarter → eighth, the eighth
    /// left readable by fragment shaders (popups are later submissions on this queue).
    pub fn capture(&self, device: &ash::Device, cb: vk::CommandBuffer, src: vk::Image, extent: vk::Extent2D) {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let blit = |from: vk::Image, fw: u32, fh: u32, to: vk::Image, tw: u32, th: u32| unsafe {
            let sub = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
            let region = vk::ImageBlit::default()
                .src_subresource(sub)
                .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: fw as i32, y: fh as i32, z: 1 }])
                .dst_subresource(sub)
                .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: tw as i32, y: th as i32, z: 1 }]);
            device.cmd_blit_image(cb, from, L::TRANSFER_SRC_OPTIMAL, to, L::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR);
        };
        let (h, q, e) = (&self.half, &self.quarter, &self.eighth);
        // Contents are fully replaced: old layouts can be discarded. Earlier reads (blits,
        // popups' sampling) must finish first.
        gpu::image_barrier(device, cb, h.image, (S::BLIT, A::NONE, L::UNDEFINED), (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL));
        gpu::image_barrier(device, cb, q.image, (S::BLIT, A::NONE, L::UNDEFINED), (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL));
        gpu::image_barrier(device, cb, e.image, (S::FRAGMENT_SHADER, A::NONE, L::UNDEFINED), (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL));
        blit(src, extent.width, extent.height, h.image, h.w, h.h);
        gpu::image_barrier(device, cb, h.image, (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL), (S::BLIT, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL));
        blit(h.image, h.w, h.h, q.image, q.w, q.h);
        gpu::image_barrier(device, cb, q.image, (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL), (S::BLIT, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL));
        blit(q.image, q.w, q.h, e.image, e.w, e.h);
        gpu::image_barrier(
            device,
            cb,
            e.image,
            (S::BLIT, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL),
            (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::SHADER_READ_ONLY_OPTIMAL),
        );
    }
}

/// Where the popup is and how its glass looks this frame.
#[derive(Clone, Copy)]
pub struct GlassParams {
    /// Popup client origin minus player client origin, physical pixels.
    pub offset: [f32; 2],
    /// Corner radius in points.
    pub radius: f32,
    /// Straight RGB and how much of it to mix over the picture.
    pub tint: [f32; 4],
    /// Bend at the border, in points.
    pub refraction: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GlassPush {
    place: [f32; 4],
    shape: [f32; 4],
    tint: [f32; 4],
    misc: [f32; 4],
}

#[derive(Clone, Copy)]
struct Frame {
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    fence: vk::Fence,
    set: vk::DescriptorSet,
    /// Backdrop generation the set points at (0: the placeholder).
    bound: u64,
}

pub struct PopupRenderer {
    egui: Option<egui_ash_renderer::Renderer<egui_ash_renderer::allocator::GpuAllocator>>,
    /// Waits for the GPU instead of the window's thread (see `wake`).
    waiter: crate::wake::GpuWaiter,
    free_later: Vec<egui::TextureId>,
    frames: Vec<Frame>,
    frame_index: usize,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    render_done: Vec<vk::Semaphore>,
    format: vk::Format,
    extent: vk::Extent2D,
    composite: vk::CompositeAlphaFlagsKHR,
    /// MAILBOX when available: the newest frame is the one shown (a moving popup must
    /// not show a frame drawn for where it was); FIFO otherwise.
    present_mode: vk::PresentModeKHR,
    needs_rebuild: bool,
    pool: vk::CommandPool,
    upload_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    /// 1×1 stand-in bound while there is no backdrop.
    placeholder: Option<GpuImage>,
    backdrop: Option<Arc<Backdrop>>,
    ctx: Arc<GpuContext>,
}

impl PopupRenderer {
    /// `backdrop`: the player window's picture to show through the glass.
    pub fn new(ctx: Arc<GpuContext>, window: &(impl HasDisplayHandle + HasWindowHandle), width: u32, height: u32, backdrop: Option<Arc<Backdrop>>) -> Result<Self> {
        unsafe {
            let display = window.display_handle().map_err(|e| anyhow::anyhow!("display handle: {e:?}"))?.as_raw();
            let handle = window.window_handle().map_err(|e| anyhow::anyhow!("window handle: {e:?}"))?.as_raw();
            let surface = ash_window::create_surface(&ctx.entry, &ctx.instance, display, handle, None)?;
            let ok = ctx.surface_fn.get_physical_device_surface_support(ctx.physical, ctx.queue_family, surface).unwrap_or(false);
            if !ok {
                ctx.surface_fn.destroy_surface(surface, None);
                anyhow::bail!("the GPU cannot present to the popup window");
            }
            let formats = ctx.surface_fn.get_physical_device_surface_formats(ctx.physical, surface)?;
            let format = formats
                .iter()
                .find(|f| matches!(f.format, vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .map(|f| f.format)
                .context("no 8-bit surface format")?;
            let caps = ctx.surface_fn.get_physical_device_surface_capabilities(ctx.physical, surface)?;
            let modes = ctx.surface_fn.get_physical_device_surface_present_modes(ctx.physical, surface).unwrap_or_default();
            let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) { vk::PresentModeKHR::MAILBOX } else { vk::PresentModeKHR::FIFO };
            let composite = if caps.supported_composite_alpha.contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED) {
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
            } else {
                vk::CompositeAlphaFlagsKHR::OPAQUE
            };
            log::info!(
                "popup: {} ({}), {:?}, composite alpha supported {:?}",
                ctx.device_name,
                if composite == vk::CompositeAlphaFlagsKHR::OPAQUE { "opaque window: plain panel outside the player" } else { "translucent window" },
                present_mode,
                caps.supported_composite_alpha
            );
            let device = &ctx.device;
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let upload_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;

            // Glass pipeline: one sampled image, push constants, writes RGBA without blending.
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(std::mem::size_of::<GlassPush>() as u32)];
            let set_layouts = [set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push), None)?;
            let module = |code: &[u8]| {
                let words = ash::util::read_spv(&mut std::io::Cursor::new(code)).expect("spirv");
                device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            };
            let vs = module(VERT)?;
            let fs = module(FRAG)?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_att = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
            let fmts = [format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&fmts);
            let info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&ia)
                .viewport_state(&vp)
                .rasterization_state(&rs)
                .multisample_state(&ms)
                .color_blend_state(&blend)
                .dynamic_state(&dynamic)
                .layout(layout)
                .push_next(&mut rendering);
            let pipeline = device.create_graphics_pipelines(ctx.pipeline_cache, &[info], None).map_err(|(_, e)| e)?[0];
            device.destroy_shader_module(vs, None);
            device.destroy_shader_module(fs, None);
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(FRAMES as u32)];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES as u32).pool_sizes(&sizes), None)?;
            let layouts = [set_layout; FRAMES];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(desc_pool).set_layouts(&layouts))?;

            // Placeholder: 1×1, cleared, readable.
            let ph = gpu::create_image(
                &ctx.device,
                &ctx.allocator,
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                "glass placeholder",
            )?;
            let cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(upload_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
            gpu::image_barrier(device, cb, ph.image, (S::NONE, A::NONE, L::UNDEFINED), (S::CLEAR, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL));
            device.cmd_clear_color_image(cb, ph.image, L::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0; 4] }, &[gpu::color_range()]);
            gpu::image_barrier(device, cb, ph.image, (S::CLEAR, A::TRANSFER_WRITE, L::TRANSFER_DST_OPTIMAL), (S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::SHADER_READ_ONLY_OPTIMAL));
            device.end_command_buffer(cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            {
                let _q = ctx.lock_queue();
                let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
                device.queue_submit2(ctx.queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)?;
            }
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(upload_pool, &[cb]);

            let cmds = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(FRAMES as u32))?;
            let mut frames = Vec::new();
            for (cmd, set) in cmds.into_iter().zip(sets) {
                frames.push(Frame {
                    cmd,
                    image_available: device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                    fence: device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?,
                    set,
                    bound: u64::MAX,
                });
            }
            let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
                (*ctx.allocator).clone(),
                device.clone(),
                egui_ash_renderer::RenderMode::DynamicRendering(egui_ash_renderer::DynamicRendering {
                    color_attachment_format: format,
                    depth_attachment_format: None,
                    stencil_attachment_format: None,
                }),
                egui_ash_renderer::Options { in_flight_frames: FRAMES, srgb_framebuffer: false, ..Default::default() },
            )?;
            let mut r = PopupRenderer {
                egui: Some(egui),
                waiter: crate::wake::GpuWaiter::new(device.clone()),
                free_later: Vec::new(),
                frames,
                frame_index: 0,
                surface,
                swapchain: vk::SwapchainKHR::null(),
                images: Vec::new(),
                views: Vec::new(),
                render_done: Vec::new(),
                format,
                extent: vk::Extent2D { width, height },
                composite,
                present_mode,
                needs_rebuild: false,
                pool,
                upload_pool,
                pipeline,
                layout,
                set_layout,
                desc_pool,
                sampler,
                placeholder: Some(ph),
                backdrop,
                ctx,
            };
            r.build_swapchain(width, height)?;
            Ok(r)
        }
    }

    /// The window can show what is behind it (per-pixel alpha reaches the compositor).
    pub fn translucent(&self) -> bool {
        self.composite == vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width != self.extent.width || height != self.extent.height {
            self.extent = vk::Extent2D { width, height };
            self.needs_rebuild = true;
        }
    }

    pub fn update_textures(&mut self, mut delta: egui::TexturesDelta) -> Result<()> {
        let result = (|| {
            let egui = self.egui.as_mut().unwrap();
            for (id, deltas) in &delta.set {
                for d in deltas.iter() {
                    let _q = self.ctx.lock_queue();
                    egui.set_texture(self.ctx.queue, self.upload_pool, *id, d)?;
                }
            }
            // Freed once no frame in flight uses them (not waited for here).
            self.free_later.extend(delta.free.iter().copied());
            self.free_textures()
        })();
        delta.clear();
        result
    }

    fn free_textures(&mut self) -> Result<()> {
        if self.free_later.is_empty() {
            return Ok(());
        }
        let device = &self.ctx.device;
        if self.frames.iter().all(|f| unsafe { device.get_fence_status(f.fence) }.unwrap_or(true)) {
            let egui = self.egui.as_mut().unwrap();
            for id in self.free_later.drain(..) {
                egui.free_texture(id)?;
            }
        }
        Ok(())
    }

    /// Called from the waiting thread when the window should draw again.
    pub fn set_waker(&self, w: crate::wake::Waker) {
        self.waiter.set_waker(w);
    }

    /// Draw the glass and egui. Returns false when nothing was presented.
    pub fn render(&mut self, glass: &GlassParams, primitives: &[egui::ClippedPrimitive], pixels_per_point: f32) -> Result<bool> {
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(false);
        }
        if self.needs_rebuild {
            self.build_swapchain(self.extent.width, self.extent.height)?;
            if self.extent.width == 0 || self.extent.height == 0 {
                return Ok(false);
            }
        }
        let device = self.ctx.device.clone();
        let fi = self.frame_index;
        unsafe {
            // Not waiting for the GPU on the window's thread (see `wake`).
            if !device.get_fence_status(self.frames[fi].fence)? {
                self.waiter.wake_when(Some(self.frames[fi].fence));
                return Ok(false);
            }
            self.free_textures()?;
            let frame = self.frames[fi];
            let (image_index, _) = match self.ctx.swapchain_fn.acquire_next_image(self.swapchain, 0, frame.image_available, vk::Fence::null()) {
                Ok(r) => r,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_rebuild = true;
                    return Ok(false);
                }
                Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => {
                    self.waiter.wake_when(None);
                    return Ok(false);
                }
                Err(e) => return Err(e.into()),
            };
            device.reset_fences(&[frame.fence])?;

            // What the glass shows: the player's picture (if any yet) or nothing.
            let bd = self.backdrop.as_ref().map(|b| *b.state.lock().unwrap()).filter(|s| s.valid);
            let want = bd.map_or(0, |s| s.generation);
            if frame.bound != want {
                let (view, layout) = match bd {
                    Some(s) => (s.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    None => (self.placeholder.as_ref().unwrap().view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                };
                let info = [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(layout)];
                let write = vk::WriteDescriptorSet::default().dst_set(frame.set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info);
                device.update_descriptor_sets(&[write], &[]);
                self.frames[fi].bound = want;
            }
            let push = GlassPush {
                place: match bd {
                    Some(s) => [glass.offset[0], glass.offset[1], s.source.0 as f32, s.source.1 as f32],
                    None => [0.0, 0.0, 1.0, 1.0],
                },
                shape: [self.extent.width as f32, self.extent.height as f32, glass.radius * pixels_per_point, pixels_per_point],
                tint: glass.tint,
                misc: match bd {
                    Some(s) => [1.0, 1.0 / s.size.0 as f32, 1.0 / s.size.1 as f32, glass.refraction * pixels_per_point],
                    None => [0.0, 0.0, 0.0, 0.0],
                },
            };
            // An opaque window (no per-pixel alpha) ignores the alpha written: outside the
            // player it shows the premultiplied tint as a plain dark panel, while the glass
            // over the player is unaffected.

            let cb = frame.cmd;
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let image = self.images[image_index as usize];
            use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
            gpu::image_barrier(&device, cb, image, (S::COLOR_ATTACHMENT_OUTPUT, A::NONE, L::UNDEFINED), (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL));
            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[image_index as usize])
                .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent };
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&attachment));
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { width: self.extent.width as f32, height: self.extent.height as f32, max_depth: 1.0, ..Default::default() }]);
            device.cmd_set_scissor(cb, 0, &[area]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[frame.set], &[]);
            let bytes = std::slice::from_raw_parts(&push as *const GlassPush as *const u8, std::mem::size_of::<GlassPush>());
            device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytes);
            device.cmd_draw(cb, 3, 1, 0, 0);
            self.egui.as_mut().unwrap().cmd_draw(cb, self.extent, pixels_per_point, primitives)?;

            device.cmd_end_rendering(cb);
            gpu::image_barrier(&device, cb, image, (S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::COLOR_ATTACHMENT_OPTIMAL), (S::BOTTOM_OF_PIPE, A::NONE, L::PRESENT_SRC_KHR));
            device.end_command_buffer(cb)?;

            let render_done = self.render_done[image_index as usize];
            let wait = [vk::SemaphoreSubmitInfo::default().semaphore(frame.image_available).stage_mask(S::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [vk::SemaphoreSubmitInfo::default().semaphore(render_done).stage_mask(S::ALL_COMMANDS)];
            let _q = self.ctx.lock_queue();
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            device.queue_submit2(self.ctx.queue, &[vk::SubmitInfo2::default().wait_semaphore_infos(&wait).command_buffer_infos(&cbs).signal_semaphore_infos(&signal)], frame.fence)?;

            let swapchains = [self.swapchain];
            let indices = [image_index];
            let waits = [render_done];
            match self.ctx.swapchain_fn.queue_present(self.ctx.queue, &vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&swapchains).image_indices(&indices)) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_rebuild = true,
                Err(e) => return Err(e.into()),
            }
        }
        self.frame_index = (fi + 1) % FRAMES;
        Ok(true)
    }

    fn wait_frames(&self) {
        unsafe {
            let fences: Vec<_> = self.frames.iter().map(|f| f.fence).collect();
            let _ = self.ctx.device.wait_for_fences(&fences, true, u64::MAX);
        }
    }

    fn build_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.wait_frames();
            // Presentation of the old images may still wait on their semaphores.
            {
                let _q = self.ctx.lock_queue();
                let _ = self.ctx.device.queue_wait_idle(self.ctx.queue);
            }
            let caps = self.ctx.surface_fn.get_physical_device_surface_capabilities(self.ctx.physical, self.surface)?;
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
                    .surface(self.surface)
                    .min_image_count(count)
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(self.composite)
                    .present_mode(self.present_mode)
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
                self.views.push(self.ctx.device.create_image_view(
                    &vk::ImageViewCreateInfo::default().image(img).view_type(vk::ImageViewType::TYPE_2D).format(self.format).subresource_range(gpu::color_range()),
                    None,
                )?);
                self.render_done.push(self.ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
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

impl Drop for PopupRenderer {
    fn drop(&mut self) {
        unsafe {
            self.wait_frames();
            self.waiter.stop();
            {
                let _q = self.ctx.lock_queue();
                let _ = self.ctx.device.queue_wait_idle(self.ctx.queue);
            }
            self.egui = None;
            self.destroy_swapchain_resources();
            let ctx = self.ctx.clone();
            let d = &ctx.device;
            if self.swapchain != vk::SwapchainKHR::null() {
                ctx.swapchain_fn.destroy_swapchain(self.swapchain, None);
            }
            ctx.surface_fn.destroy_surface(self.surface, None);
            for f in &self.frames {
                d.destroy_semaphore(f.image_available, None);
                d.destroy_fence(f.fence, None);
            }
            if let Some(ph) = self.placeholder.take() {
                gpu::destroy_image(d, &ctx.allocator, ph);
            }
            d.destroy_sampler(self.sampler, None);
            d.destroy_descriptor_pool(self.desc_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_command_pool(self.upload_pool, None);
        }
    }
}
