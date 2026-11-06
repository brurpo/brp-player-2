//! Video pass: draws planes (a cached frame or processed output) converted to display RGB.

use crate::color::VideoPush;
use anyhow::Result;
use ash::vk;

const VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/video.vert.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/video.frag.spv"));

/// One descriptor set per frame in flight.
pub struct Slot {
    set: vk::DescriptorSet,
    /// What the set currently points at (views, layout, nearest).
    bound: Option<(Vec<vk::ImageView>, vk::ImageLayout, bool)>,
}

pub struct VideoPass {
    device: ash::Device,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sampler_linear: vk::Sampler,
    sampler_nearest: vk::Sampler,
    pub slots: Vec<Slot>,
}

impl VideoPass {
    pub fn new(device: &ash::Device, color_format: vk::Format, slots: usize) -> Result<Self> {
        unsafe {
            let bindings: Vec<_> = (0..4)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                })
                .collect();
            let set_layout =
                device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .size(std::mem::size_of::<VideoPush>() as u32)];
            let set_layouts = [set_layout];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push),
                None,
            )?;

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
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
            let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let rs = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_att = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
            let formats = [color_format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
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
            let pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0];
            device.destroy_shader_module(vs, None);
            device.destroy_shader_module(fs, None);

            let sampler = |filter: vk::Filter| {
                device.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
            };
            let sampler_linear = sampler(vk::Filter::LINEAR)?;
            let sampler_nearest = sampler(vk::Filter::NEAREST)?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(4 * slots as u32)];
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(slots as u32).pool_sizes(&pool_sizes),
                None,
            )?;
            let layouts = vec![set_layout; slots];
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
            )?;
            let slots = sets
                .into_iter()
                .map(|set| Slot {
                    set,
                    bound: None,
                })
                .collect();

            Ok(VideoPass {
                device: device.clone(),
                pipeline,
                layout,
                set_layout,
                pool,
                sampler_linear,
                sampler_nearest,
                slots,
            })
        }
    }

    /// Point slot `i` at these planes (skipped when unchanged).
    pub fn bind(&mut self, i: usize, views: &[vk::ImageView], layout: vk::ImageLayout, nearest: bool) {
        // Always rewrite: a destroyed view's handle value can be reused by a new one,
        // so comparing handles is not a safe way to skip this (and it is cheap).
        let want = (views.to_vec(), layout, nearest);
        if views.is_empty() {
            return;
        }
        let sampler = if nearest { self.sampler_nearest } else { self.sampler_linear };
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = (0..4)
            .map(|b| {
                // Unused bindings point at plane 0 so every descriptor is valid.
                let v = *views.get(b).unwrap_or(&views[0]);
                [vk::DescriptorImageInfo::default().sampler(sampler).image_view(v).image_layout(layout)]
            })
            .collect();
        let set = self.slots[i].set;
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        self.slots[i].bound = Some(want);
    }

    /// Record the draw (inside rendering).
    pub fn draw(&self, cb: vk::CommandBuffer, i: usize, push: &VideoPush) {
        unsafe {
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.slots[i].set], &[]);
            self.device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, push.bytes());
            self.device.cmd_draw(cb, 4, 1, 0, 0);
        }
    }
}

impl Drop for VideoPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_sampler(self.sampler_linear, None);
            self.device.destroy_sampler(self.sampler_nearest, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
