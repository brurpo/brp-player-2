//! OpenColorIO on the GPU (DESIGN.md §7d). The player asks OpenColorIO (`brp-ocio`) for
//! a conversion as GLSL defining `vec4 OCIOMain(vec4)` plus its LUT textures; this
//! module uploads the textures once and builds the passes that run it:
//!
//! - `OcioPass`, the window's last pass: the video (drawn raw, in the file's own
//!   encoding, into the scene layer) through OCIO, the background and the checkerboard,
//!   the UI over it, into the swapchain (SDR, or HDR10 from display-referred XYZ).
//! - `OcioCompute`, for export: a picture through OCIO, on the compute queue.

use crate::composite::Layers;
use crate::context::GpuContext;
use crate::gpu::SharedAllocator;
use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

/// A conversion as OpenColorIO gives it (see `brp_ocio::shader`).
#[derive(Clone, Debug)]
pub struct OcioShader {
    /// Vulkan GLSL defining `vec4 OCIOMain(vec4)`; textures in descriptor set 1.
    pub glsl: String,
    pub textures: Vec<OcioTexture>,
}

#[derive(Clone, Debug)]
pub struct OcioTexture {
    pub binding: u32,
    /// 1, 2 or 3.
    pub dims: u8,
    pub w: u32,
    pub h: u32,
    pub d: u32,
    /// 1 (red) or 3 (RGB) values per texel.
    pub channels: u8,
    pub linear: bool,
    pub data: Vec<f32>,
}

/// Descriptor set the OCIO textures are declared in (ours is 0).
pub const OCIO_SET: u32 = 1;

struct Lut {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
    sampler: vk::Sampler,
    binding: u32,
}

/// The LUT textures of one shader, uploaded, and the descriptor set binding them.
struct Luts {
    device: ash::Device,
    allocator: SharedAllocator,
    luts: Vec<Lut>,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl Luts {
    fn new(ctx: &GpuContext, s: &OcioShader, stage: vk::ShaderStageFlags) -> Result<Luts> {
        let device = ctx.device.clone();
        let allocator: SharedAllocator = (*ctx.allocator).clone();
        unsafe {
            // RGBA float32 where it can be filtered, else half float (values of 3D LUTs
            // are 0..1; 1D LUTs on half-float domains need the range, rarely the bits).
            let props = ctx.instance.get_physical_device_format_properties(ctx.physical, vk::Format::R32G32B32A32_SFLOAT);
            let rgba = if props.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) { vk::Format::R32G32B32A32_SFLOAT } else { vk::Format::R16G16B16A16_SFLOAT };
            let red = vk::Format::R32_SFLOAT;
            let mut luts = Vec::new();
            let mut uploads: Vec<(vk::Image, vk::Extent3D, Vec<u8>)> = Vec::new();
            for t in &s.textures {
                let format = if t.channels == 1 { red } else { rgba };
                let (ty, vty) = match t.dims {
                    1 => (vk::ImageType::TYPE_1D, vk::ImageViewType::TYPE_1D),
                    3 => (vk::ImageType::TYPE_3D, vk::ImageViewType::TYPE_3D),
                    _ => (vk::ImageType::TYPE_2D, vk::ImageViewType::TYPE_2D),
                };
                let extent = vk::Extent3D { width: t.w.max(1), height: if t.dims >= 2 { t.h.max(1) } else { 1 }, depth: if t.dims == 3 { t.d.max(1) } else { 1 } };
                let image = device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(ty)
                        .format(format)
                        .extent(extent)
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )?;
                let req = device.get_image_memory_requirements(image);
                let alloc = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                    name: "ocio lut",
                    requirements: req,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;
                device.bind_image_memory(image, alloc.memory(), alloc.offset())?;
                let view = device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vty)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)),
                    None,
                )?;
                let filter = if t.linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
                let sampler = device.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )?;
                // Texels: red as is, RGB widened to RGBA.
                let n = (extent.width * extent.height * extent.depth) as usize;
                let bytes: Vec<u8> = if t.channels == 1 {
                    t.data.iter().take(n).flat_map(|v| v.to_le_bytes()).collect()
                } else if format == vk::Format::R32G32B32A32_SFLOAT {
                    t.data.chunks_exact(3).take(n).flat_map(|c| [c[0], c[1], c[2], 1.0]).flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    t.data.chunks_exact(3).take(n).flat_map(|c| [c[0], c[1], c[2], 1.0]).flat_map(|v| f32_to_f16(v).to_le_bytes()).collect()
                };
                uploads.push((image, extent, bytes));
                luts.push(Lut { image, view, alloc: Some(alloc), sampler, binding: t.binding });
            }
            upload(ctx, &uploads)?;

            let bindings: Vec<_> = luts
                .iter()
                .map(|l| vk::DescriptorSetLayoutBinding::default().binding(l.binding).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(stage))
                .collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(luts.len().max(1) as u32)];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None)?;
            let sl = [set_layout];
            let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl))?[0];
            let infos: Vec<[vk::DescriptorImageInfo; 1]> =
                luts.iter().map(|l| [vk::DescriptorImageInfo::default().sampler(l.sampler).image_view(l.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]).collect();
            let writes: Vec<_> = luts
                .iter()
                .zip(&infos)
                .map(|(l, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(l.binding).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info))
                .collect();
            device.update_descriptor_sets(&writes, &[]);
            Ok(Luts { device, allocator, luts, set_layout, pool, set })
        }
    }
}

impl Drop for Luts {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            for l in &mut self.luts {
                self.device.destroy_sampler(l.sampler, None);
                self.device.destroy_image_view(l.view, None);
                self.device.destroy_image(l.image, None);
                if let Some(a) = l.alloc.take() {
                    let _ = self.allocator.lock().unwrap().free(a);
                }
            }
        }
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        s
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((e as u16) << 10) | ((m + 0x1000) >> 13) as u16
    }
}

/// Copy texel data into fresh images (left SHADER_READ_ONLY_OPTIMAL); waits.
unsafe fn upload(ctx: &GpuContext, items: &[(vk::Image, vk::Extent3D, Vec<u8>)]) -> Result<()> {
    if items.is_empty() {
        return Ok(());
    }
    unsafe {
        let device = &ctx.device;
        let total: usize = items.iter().map(|i| i.2.len().next_multiple_of(16)).sum();
        let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(total as u64).usage(vk::BufferUsageFlags::TRANSFER_SRC), None)?;
        let req = device.get_buffer_memory_requirements(buffer);
        let mut alloc = ctx.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: "ocio upload",
            requirements: req,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
        let map = alloc.mapped_slice_mut().context("upload buffer not mapped")?;
        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let mut off = 0usize;
        for (image, extent, bytes) in items {
            map[off..off + bytes.len()].copy_from_slice(bytes);
            let to_dst = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(*image)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_dst)));
            let region = vk::BufferImageCopy::default()
                .buffer_offset(off as u64)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(*extent);
            device.cmd_copy_buffer_to_image(cb, buffer, *image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            let to_read = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(*image)
                .subresource_range(range);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_read)));
            off += bytes.len().next_multiple_of(16);
        }
        device.end_command_buffer(cb)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let r = {
            let _q = ctx.lock_queue();
            device.queue_submit2(ctx.queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)
        };
        if r.is_ok() {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        ctx.allocator.lock().unwrap().free(alloc)?;
        device.destroy_buffer(buffer, None);
        r?;
        Ok(())
    }
}

/// How `OcioPass` writes the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcioOut {
    /// The display's code values, as OCIO makes them (SDR swapchain).
    Direct,
    /// From display-referred XYZ: HDR10 (PQ) for the HDR swapchain.
    Hdr10,
    /// From display-referred XYZ: an SDR picture (popups' backdrop on HDR screens).
    SdrFromXyz,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PassPush {
    mode: [i32; 4],
    /// x: SDR white (nits), y: light per unit of the XYZ output, z: checker size (px).
    f: [f32; 4],
    /// Background colour (sRGB code values).
    bg: [f32; 4],
}

/// OCIO's output with NaN and infinities as 0 (its GPU code can make them from
/// out-of-gamut values where its CPU code does not).
const FINITE: &str = "vec3 brp_finite(vec3 c) {
    bvec3 bad = bvec3(isnan(c.r) || isinf(c.r), isnan(c.g) || isinf(c.g), isnan(c.b) || isinf(c.b));
    return mix(c, vec3(0.0), bad);
}
";

const PASS_MAIN: &str = r#"
const mat3 BRP_XYZ_TO_709 = mat3(
    3.2409699419, -0.9692436363, 0.0556300797,
   -1.5373831776, 1.8759675015, -0.2039769589,
   -0.4986107603, 0.0415550574, 1.0569715142);
const mat3 BRP_709_TO_2020 = mat3(
    0.6274038959, 0.0690972894, 0.0163914389,
    0.3292830384, 0.9195403951, 0.0880133079,
    0.0433130657, 0.0113623156, 0.8955952532);
vec3 brp_srgb_encode(vec3 l) {
    l = clamp(l, 0.0, 1.0);
    return mix(l * 12.92, 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, l));
}
vec3 brp_srgb_decode(vec3 e) {
    return mix(e / 12.92, pow((e + 0.055) / 1.055, vec3(2.4)), step(0.04045, e));
}
vec3 brp_pq_encode(vec3 nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 s = texelFetch(brp_scene, p, 0);
    vec4 u = texelFetch(brp_ui, p, 0);
    bool xyz = pc.mode.x != 0;
    vec3 code = vec3(0.0);   // display code values (direct)
    vec3 lin = vec3(0.0);    // display light, Rec.709, 1.0 = SDR white (from XYZ)
    if (s.a < -0.5) {
        // Outside the picture: the window's background.
        code = pc.bg.rgb;
        lin = brp_srgb_decode(pc.bg.rgb);
    } else {
        vec3 c = brp_finite(OCIOMain(vec4(s.rgb, 1.0)).rgb);
        float a = clamp(s.a, 0.0, 1.0);
        ivec2 cell = ivec2(gl_FragCoord.xy / max(pc.f.z, 1.0));
        float ck = ((cell.x + cell.y) & 1) == 0 ? 0.20 : 0.30;
        if (xyz) {
            lin = mix(brp_srgb_decode(vec3(ck)), BRP_XYZ_TO_709 * c * pc.f.y, a);
        } else {
            code = mix(vec3(ck), c, a);
        }
    }
    if (!xyz) {
        o_color = vec4(u.rgb + (1.0 - u.a) * clamp(code, 0.0, 1.0), 1.0);
        return;
    }
    // The UI over the picture as on an SDR window; light above SDR white added back
    // (as composite.frag).
    vec3 sdr = clamp(lin, 0.0, 1.0);
    vec3 g = u.rgb + (1.0 - u.a) * brp_srgb_encode(sdr);
    if (pc.mode.x == 2) {
        o_color = vec4(clamp(g, 0.0, 1.0), 1.0);
        return;
    }
    vec3 l = brp_srgb_decode(clamp(g, 0.0, 1.0)) + (1.0 - u.a) * (lin - sdr);
    o_color = vec4(brp_pq_encode(max(BRP_709_TO_2020 * l, 0.0) * pc.f.x), 1.0);
}
"#;

/// The window's last pass with OCIO (see the module doc).
pub(crate) struct OcioPass {
    device: ash::Device,
    luts: Luts,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    sets: Vec<vk::DescriptorSet>,
    pipelines: Vec<(vk::Format, vk::Pipeline)>,
    /// The shader writes display-referred XYZ (HDR screens), not code values.
    pub xyz: bool,
}

impl OcioPass {
    pub fn new(ctx: &GpuContext, s: &OcioShader, xyz: bool, formats: &[vk::Format], slots: usize) -> Result<OcioPass> {
        let device = ctx.device.clone();
        let luts = Luts::new(ctx, s, vk::ShaderStageFlags::FRAGMENT)?;
        let src = format!(
            "#version 460\nlayout(set = 0, binding = 0) uniform sampler2D brp_scene;\nlayout(set = 0, binding = 1) uniform sampler2D brp_ui;\n\
             layout(push_constant) uniform BrpPC {{ ivec4 mode; vec4 f; vec4 bg; }} pc;\nlayout(location = 0) out vec4 o_color;\n\n{}\n{FINITE}
{PASS_MAIN}",
            s.glsl
        );
        let frag = crate::usershader::compile_fragment(&src, "ocio display")?;
        unsafe {
            let bindings: Vec<_> = (0..2)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT))
                .collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(std::mem::size_of::<PassPush>() as u32)];
            let sls = [set_layout, luts.set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&push), None)?;
            let vert = ash::util::read_spv(&mut std::io::Cursor::new(crate::composite::VERT_SPV))?;
            let vs = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert), None)?;
            let fs = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag), None)?;
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
            let mut err = None;
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
                match device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) {
                    Ok(p) => pipelines.push((format, p[0])),
                    Err((_, e)) => err = Some(e),
                }
            }
            device.destroy_shader_module(vs, None);
            device.destroy_shader_module(fs, None);
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST).address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(2 * slots as u32)];
            let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(slots as u32).pool_sizes(&sizes), None)?;
            let layouts = vec![set_layout; slots];
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            let pass = OcioPass { device, luts, set_layout, layout, pool, sampler, sets, pipelines, xyz };
            if let Some(e) = err {
                return Err(anyhow::anyhow!("OCIO pipeline: {e}"));
            }
            Ok(pass)
        }
    }

    /// Point slot `i` at the layers (SHADER_READ_ONLY_OPTIMAL).
    pub fn bind(&self, i: usize, layers: &Layers) {
        let infos = [layers.scene.view, layers.ui.view].map(|v| [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]);
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(b, info)| vk::WriteDescriptorSet::default().dst_set(self.sets[i]).dst_binding(b as u32).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info))
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Record the draw (inside rendering into a `format` target). `scale`: light per
    /// unit of the XYZ output, relative to SDR white.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(&self, cb: vk::CommandBuffer, i: usize, format: vk::Format, out: OcioOut, sdr_nits: f32, scale: f32, checker_px: f32, bg: [f32; 4]) {
        let Some(&(_, pipeline)) = self.pipelines.iter().find(|(f, _)| *f == format) else { return };
        let mode = match out {
            OcioOut::Direct => 0,
            OcioOut::Hdr10 => 1,
            OcioOut::SdrFromXyz => 2,
        };
        let push = PassPush { mode: [mode, 0, 0, 0], f: [sdr_nits, scale, checker_px, 0.0], bg };
        let bytes = unsafe { std::slice::from_raw_parts(&push as *const PassPush as *const u8, std::mem::size_of::<PassPush>()) };
        unsafe {
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.sets[i], self.luts.set], &[]);
            self.device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::FRAGMENT, 0, bytes);
            self.device.cmd_draw(cb, 3, 1, 0, 0);
        }
    }
}

impl Drop for OcioPass {
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

/// A picture (rgba16f) through OCIO, for export (alpha kept).
pub struct OcioCompute {
    device: ash::Device,
    luts: Luts,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

unsafe impl Send for OcioCompute {}
unsafe impl Sync for OcioCompute {}

impl OcioCompute {
    pub fn new(ctx: &Arc<GpuContext>, s: &OcioShader) -> Result<Arc<OcioCompute>> {
        let device = ctx.device.clone();
        let luts = Luts::new(ctx, s, vk::ShaderStageFlags::COMPUTE)?;
        let src = format!(
            "#version 460\nlayout(local_size_x = 16, local_size_y = 16) in;\nlayout(set = 0, binding = 0, rgba16f) uniform readonly image2D brp_src;\n\
             layout(set = 0, binding = 1, rgba16f) uniform writeonly image2D brp_dst;\n\n{}\n{FINITE}\n\
             void main() {{\n    ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n    if (any(greaterThanEqual(p, imageSize(brp_dst)))) return;\n\
                 vec4 c = imageLoad(brp_src, p);\n    imageStore(brp_dst, p, vec4(brp_finite(OCIOMain(vec4(c.rgb, 1.0)).rgb), c.a));\n}}\n",
            s.glsl
        );
        let spv = crate::usershader::compile_glsl(&src, "ocio export")?;
        unsafe {
            let bindings: Vec<_> = (0..2)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let sls = [set_layout, luts.set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls), None)?;
            let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let r = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None);
            device.destroy_shader_module(module, None);
            let pipeline = r.map_err(|(_, e)| e)?[0];
            Ok(Arc::new(OcioCompute { device, luts, set_layout, layout, pipeline }))
        }
    }

    /// Record `src` through OCIO into `dst` (same size, GENERAL layout).
    pub fn record(self: &Arc<Self>, rec: &mut crate::compute::Recorder, src: vk::ImageView, dst: vk::ImageView, w: u32, h: u32) -> Result<()> {
        rec.keep(self.clone());
        rec.dispatch_raw(self.pipeline, self.layout, self.set_layout, &[src, dst], &[self.luts.set], [w.div_ceil(16), h.div_ceil(16), 1])
    }
}

impl Drop for OcioCompute {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
