//! HDR screens (DESIGN.md §7b): the window is drawn in two layers, the video as
//! linear light (`scene`, float) and the UI as for an SDR window (`ui`, transparent),
//! and this pass puts them together into the HDR10 swapchain (or into an SDR picture
//! for the popups' glass).

use crate::gpu::{self, GpuImage, SharedAllocator};
use crate::video::SCENE_FORMAT;
use anyhow::Result;
use ash::vk;

pub(crate) const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/composite.vert.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/composite.frag.spv"));

/// The layers, sized to the window (made again when it changes size).
pub struct Layers {
    pub scene: GpuImage,
    pub ui: GpuImage,
    /// SDR copy of the whole window for the popups' backdrop (made while they are open).
    pub sdr: Option<GpuImage>,
}

impl Layers {
    pub fn new(device: &ash::Device, alloc: &SharedAllocator, ui_format: vk::Format, w: u32, h: u32) -> Result<Layers> {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let scene = gpu::create_image(device, alloc, w, h, SCENE_FORMAT, usage, "hdr scene")?;
        let ui = gpu::create_image(device, alloc, w, h, ui_format, usage, "hdr ui")?;
        Ok(Layers { scene, ui, sdr: None })
    }

    pub fn destroy(self, device: &ash::Device, alloc: &SharedAllocator) {
        gpu::destroy_image(device, alloc, self.scene);
        gpu::destroy_image(device, alloc, self.ui);
        if let Some(s) = self.sdr {
            gpu::destroy_image(device, alloc, s);
        }
    }
}

pub struct CompositePass {
    device: ash::Device,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    sets: Vec<vk::DescriptorSet>,
    /// (target format, pipeline).
    pipelines: Vec<(vk::Format, vk::Pipeline)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    mode: [i32; 4],
    f: [f32; 4],
}

impl CompositePass {
    /// `formats`: the targets it draws into (the HDR10 swapchain, the SDR backdrop).
    pub fn new(device: &ash::Device, formats: &[vk::Format], slots: usize) -> Result<CompositePass> {
        unsafe {
            let bindings: Vec<_> = (0..2)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(std::mem::size_of::<Push>() as u32)];
            let sl = [set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&push), None)?;
            let module = |code: &[u8]| {
                let words = ash::util::read_spv(&mut std::io::Cursor::new(code)).expect("spirv");
                device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            };
            let (vs, fs) = (module(VERT_SPV)?, module(FRAG)?);
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
            let mut pipelines = Vec::new();
            for &format in formats {
                let f = [format];
                let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&f);
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
                let p = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0];
                pipelines.push((format, p));
            }
            device.destroy_shader_module(vs, None);
            device.destroy_shader_module(fs, None);
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(2 * slots as u32)];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(slots as u32).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; slots];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            Ok(CompositePass { device: device.clone(), layout, set_layout, pool, sampler, sets, pipelines })
        }
    }

    /// Point slot `i` at the layers (SHADER_READ_ONLY_OPTIMAL).
    pub fn bind(&self, i: usize, layers: &Layers) {
        let infos = [layers.scene.view, layers.ui.view]
            .map(|v| [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(self.sets[i]).dst_binding(b as u32).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info))
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Record the draw (inside rendering into a `format` target). `sdr_out`: SDR
    /// picture instead of HDR10; `sdr_nits`: brightness of SDR white.
    pub fn draw(&self, cb: vk::CommandBuffer, i: usize, format: vk::Format, sdr_out: bool, sdr_nits: f32) {
        let Some(&(_, pipeline)) = self.pipelines.iter().find(|(f, _)| *f == format) else { return };
        let push = Push { mode: [sdr_out as i32, 0, 0, 0], f: [sdr_nits, 0.0, 0.0, 0.0] };
        let bytes = unsafe { std::slice::from_raw_parts(&push as *const Push as *const u8, std::mem::size_of::<Push>()) };
        unsafe {
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.sets[i]], &[]);
            self.device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytes);
            self.device.cmd_draw(cb, 3, 1, 0, 0);
        }
    }
}

impl Drop for CompositePass {
    fn drop(&mut self) {
        unsafe {
            for (_, p) in &self.pipelines {
                self.device.destroy_pipeline(*p, None);
            }
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
