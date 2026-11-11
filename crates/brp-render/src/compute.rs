//! General GPU compute on the renderer's device: buffers addressed by pointer,
//! GLSL compute kernels, and a recorder that chains many dispatches into one
//! submission on the compute queue. Used by the AI path (neural networks, motion,
//! temporal processing) so pictures stay on the GPU from decode to screen.
//!
//! Kernels receive their buffers as device addresses in push constants
//! (`GL_EXT_buffer_reference`): no descriptor sets for buffers. Storage images (for
//! exchanging pictures with the renderer) are bound through a small descriptor set.

use crate::context::GpuContext;
use anyhow::{bail, Context as _, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A GPU buffer with its device address. Freed on drop (only drop buffers the GPU
/// is done with: the recorder waits for its submissions).
pub struct Buffer {
    ctx: Arc<GpuContext>,
    pub buffer: vk::Buffer,
    alloc: Option<Allocation>,
    pub size: u64,
    pub addr: u64,
}

impl Buffer {
    pub fn new(ctx: &Arc<GpuContext>, size: u64, location: MemoryLocation, name: &str) -> Result<Buffer> {
        let size = size.max(4);
        unsafe {
            let device = &ctx.device;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)?;
            let req = device.get_buffer_memory_requirements(buffer);
            let alloc = ctx.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name,
                requirements: req,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
            let alloc = match alloc {
                Ok(a) => a,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    bail!("GPU memory for {name} ({} MB): {e}", size >> 20);
                }
            };
            device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
            let addr = device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok(Buffer { ctx: ctx.clone(), buffer, alloc: Some(alloc), size, addr })
        }
    }

    /// Host-visible bytes (buffers made with a CPU-side location).
    pub fn mapped(&mut self) -> Option<&mut [u8]> {
        let size = self.size as usize;
        self.alloc.as_mut()?.mapped_slice_mut().map(|s| &mut s[..size])
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(a) = self.alloc.take() {
            let _ = self.ctx.allocator.lock().unwrap().free(a);
        }
        unsafe { self.ctx.device.destroy_buffer(self.buffer, None) };
    }
}

/// A compiled compute kernel.
pub struct Kernel {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    images: u32,
    /// Workgroup size (x, y, z), as declared in the source.
    pub local: [u32; 3],
}

/// Most timestamps per recording when profiling.
const PROFILE_MAX: u32 = 8192;

/// Largest push-constant block kernels may use (the guaranteed Vulkan minimum).
pub const PUSH_MAX: usize = 128;

/// GLSL lines every kernel starts with: buffer pointers to float / vec4 / uint
/// data and the 16-bit float type when available.
pub const PRELUDE: &str = r#"#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32 { uint v[]; };
"#;

pub struct Compute {
    pub ctx: Arc<GpuContext>,
    kernels: Mutex<HashMap<String, Arc<Kernel>>>,
}

impl Compute {
    pub fn new(ctx: Arc<GpuContext>) -> Arc<Compute> {
        Arc::new(Compute { ctx, kernels: Mutex::new(HashMap::new()) })
    }

    /// The kernel `name` (compiled from `source` the first time). `images` storage
    /// images are bound at set 0, bindings 0.. (format given in the source).
    pub fn kernel(&self, name: &str, source: &str, images: u32) -> Result<Arc<Kernel>> {
        if let Some(k) = self.kernels.lock().unwrap().get(name) {
            return Ok(k.clone());
        }
        let local = workgroup_size(source).with_context(|| format!("kernel {name}: no local_size"))?;
        let t0 = std::time::Instant::now();
        let spv = crate::usershader::compile_glsl(source, name).with_context(|| format!("kernel {name}"))?;
        let t_glsl = t0.elapsed();
        let device = &self.ctx.device;
        unsafe {
            let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)?;
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..images)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(PUSH_MAX as u32)];
            let set_layouts = [set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let stats = std::env::var_os("BRP_PIPELINE_STATS").is_some();
            let flags = if stats { vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR } else { vk::PipelineCreateFlags::empty() };
            let res = device.create_compute_pipelines(self.ctx.pipeline_cache, &[vk::ComputePipelineCreateInfo::default().flags(flags).stage(stage).layout(layout)], None);
            device.destroy_shader_module(module, None);
            let pipeline = res.map_err(|(_, e)| e).with_context(|| format!("pipeline {name}"))?[0];
            log::debug!("kernel {name}: GLSL to SPIR-V {:.0} ms, pipeline {:.0} ms", t_glsl.as_secs_f64() * 1e3, (t0.elapsed() - t_glsl).as_secs_f64() * 1e3);
            if stats {
                let pe = ash::khr::pipeline_executable_properties::Device::new(&self.ctx.instance, device);
                let execs = pe.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline)).unwrap_or_default();
                for i in 0..execs.len() as u32 {
                    let st = pe.get_pipeline_executable_statistics(&vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i)).unwrap_or_default();
                    let text: Vec<String> = st
                        .iter()
                        .map(|s| {
                            let v = match s.format {
                                vk::PipelineExecutableStatisticFormatKHR::BOOL32 => s.value.b32.to_string(),
                                vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                                vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                                _ => s.value.f64.to_string(),
                            };
                            format!("{}={v}", s.name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default())
                        })
                        .collect();
                    log::info!("kernel {name}: {}", text.join(", "));
                }
            }
            let k = Arc::new(Kernel { pipeline, layout, set_layout, images, local });
            self.kernels.lock().unwrap().insert(name.to_string(), k.clone());
            Ok(k)
        }
    }

    /// A buffer in GPU memory.
    pub fn buffer(&self, bytes: u64, name: &str) -> Result<Buffer> {
        Buffer::new(&self.ctx, bytes, MemoryLocation::GpuOnly, name)
    }

    /// A buffer the CPU reads (results copied into it by a recording; read with
    /// `Buffer::mapped` once that has run).
    pub fn readback(&self, bytes: u64, name: &str) -> Result<Buffer> {
        Buffer::new(&self.ctx, bytes, MemoryLocation::GpuToCpu, name)
    }

    /// A buffer the CPU writes and the GPU reads (uploads; fill with `Buffer::mapped`).
    pub fn staging(&self, bytes: u64, name: &str) -> Result<Buffer> {
        Buffer::new(&self.ctx, bytes, MemoryLocation::CpuToGpu, name)
    }

    /// A GPU buffer holding `data` (uploaded through a staging buffer; waits).
    pub fn upload(&self, data: &[u8], name: &str) -> Result<Buffer> {
        let dst = self.buffer(data.len() as u64, name)?;
        let mut staging = Buffer::new(&self.ctx, data.len() as u64, MemoryLocation::CpuToGpu, "upload")?;
        staging.mapped().context("staging not mapped")?[..data.len()].copy_from_slice(data);
        let mut rec = Recorder::new(self)?;
        rec.copy(&staging, 0, &dst, 0, data.len() as u64);
        rec.submit_wait()?;
        Ok(dst)
    }

    /// The first `bytes` of `src`, read back to the CPU (waits).
    pub fn download(&self, src: &Buffer, offset: u64, bytes: u64) -> Result<Vec<u8>> {
        let mut staging = Buffer::new(&self.ctx, bytes, MemoryLocation::GpuToCpu, "download")?;
        let mut rec = Recorder::new(self)?;
        rec.copy(src, offset, &staging, 0, bytes);
        rec.submit_wait()?;
        Ok(staging.mapped().context("readback not mapped")?[..bytes as usize].to_vec())
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        let _ = self.ctx.wait_idle();
        let device = &self.ctx.device;
        for (_, k) in self.kernels.lock().unwrap().drain() {
            unsafe {
                device.destroy_pipeline(k.pipeline, None);
                device.destroy_pipeline_layout(k.layout, None);
                device.destroy_descriptor_set_layout(k.set_layout, None);
            }
        }
    }
}

/// `layout(local_size_x = X, local_size_y = Y, local_size_z = Z) in;` from the source.
fn workgroup_size(src: &str) -> Option<[u32; 3]> {
    let line = src.lines().find(|l| l.contains("local_size_x"))?;
    let get = |k: &str| -> u32 {
        line.split(k).nth(1).and_then(|r| r.trim_start().strip_prefix('=')).and_then(|r| r.trim_start().split(|c: char| !c.is_ascii_digit()).next()).and_then(|n| n.parse().ok()).unwrap_or(1)
    };
    Some([get("local_size_x"), get("local_size_y"), get("local_size_z")])
}

/// Push-constant block built field by field (std430-like scalar packing: 8-byte
/// values must sit at 8-byte offsets — put pointers first).
#[derive(Default, Clone)]
pub struct Push(pub Vec<u8>);

impl Push {
    pub fn new() -> Push {
        Push(Vec::with_capacity(PUSH_MAX))
    }
    pub fn ptr(self, b: &Buffer) -> Push {
        self.ptr_at(b, 0)
    }
    /// Pointer `bytes` into the buffer.
    pub fn ptr_at(mut self, b: &Buffer, bytes: u64) -> Push {
        while self.0.len() % 8 != 0 {
            self.0.push(0);
        }
        self.0.extend((b.addr + bytes).to_le_bytes());
        self
    }
    pub fn i(mut self, v: i32) -> Push {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn u(mut self, v: u32) -> Push {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn f(mut self, v: f32) -> Push {
        self.0.extend(v.to_le_bytes());
        self
    }
}

/// Records dispatches (with a barrier after each) into one command buffer.
pub struct Recorder<'a> {
    compute: &'a Compute,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
    dispatches: usize,
    /// Kept alive until the submission is done (buffers the recorded commands use).
    keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    /// Fences of the submissions `flush` made.
    submitted: Vec<vk::Fence>,
    first_submit: Option<std::time::Instant>,
    /// Profiling (BRP_GPU_PROFILE): timestamp after each dispatch, with its label.
    queries: Option<(vk::QueryPool, Vec<String>)>,
    pub label: String,
}

impl<'a> Recorder<'a> {
    pub fn new(compute: &'a Compute) -> Result<Recorder<'a>> {
        let ctx = &compute.ctx;
        let device = &ctx.device;
        unsafe {
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
            let cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(256)];
            let desc_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(128).pool_sizes(&sizes), None)?;
            let queries = if std::env::var_os("BRP_GPU_PROFILE").is_some() {
                let qp = device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PROFILE_MAX), None)?;
                device.cmd_reset_query_pool(cb, qp, 0, PROFILE_MAX);
                device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, qp, 0);
                Some((qp, vec![String::new()]))
            } else {
                None
            };
            Ok(Recorder { compute, pool, cb, desc_pool, dispatches: 0, keep: Vec::new(), submitted: Vec::new(), first_submit: None, queries, label: String::new() })
        }
    }

    /// Keep `v` alive until this recording has run (or is dropped unsubmitted).
    pub fn keep<T: Send + Sync + 'static>(&mut self, v: Arc<T>) {
        self.keep.push(v);
    }

    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.cb
    }

    /// Run `kernel` over `groups` workgroups, then a barrier (later dispatches see
    /// its writes).
    pub fn dispatch(&mut self, kernel: &Kernel, groups: [u32; 3], push: &Push) -> Result<()> {
        self.dispatch_images(kernel, groups, push, &[])
    }

    /// Like `dispatch`, binding storage images (views in GENERAL layout).
    pub fn dispatch_images(&mut self, kernel: &Kernel, groups: [u32; 3], push: &Push, images: &[vk::ImageView]) -> Result<()> {
        if push.0.len() > PUSH_MAX {
            bail!("push constants: {} bytes", push.0.len());
        }
        if images.len() as u32 != kernel.images {
            bail!("kernel takes {} images, {} given", kernel.images, images.len());
        }
        let device = &self.compute.ctx.device;
        unsafe {
            device.cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            if kernel.images > 0 {
                let layouts = [kernel.set_layout];
                let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts))?[0];
                let infos: Vec<[vk::DescriptorImageInfo; 1]> = images.iter().map(|v| [vk::DescriptorImageInfo::default().image_view(*v).image_layout(vk::ImageLayout::GENERAL)]).collect();
                let writes: Vec<vk::WriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(info))
                    .collect();
                device.update_descriptor_sets(&writes, &[]);
                device.cmd_bind_descriptor_sets(self.cb, vk::PipelineBindPoint::COMPUTE, kernel.layout, 0, &[set], &[]);
            }
            let mut bytes = push.0.clone();
            bytes.resize(PUSH_MAX, 0);
            device.cmd_push_constants(self.cb, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            device.cmd_dispatch(self.cb, groups[0].max(1), groups[1].max(1), groups[2].max(1));
            if let Some((qp, labels)) = &mut self.queries {
                if (labels.len() as u32) < PROFILE_MAX {
                    device.cmd_write_timestamp2(self.cb, vk::PipelineStageFlags2::COMPUTE_SHADER, *qp, labels.len() as u32);
                    labels.push(self.label.clone());
                }
            }
            self.barrier();
        }
        self.dispatches += 1;
        Ok(())
    }

    /// Workgroups covering `n` (x, y, z) items with `kernel`'s workgroup size.
    pub fn groups(kernel: &Kernel, n: [u32; 3]) -> [u32; 3] {
        [n[0].div_ceil(kernel.local[0]), n[1].div_ceil(kernel.local[1]), n[2].div_ceil(kernel.local[2])]
    }

    pub fn copy(&mut self, src: &Buffer, src_offset: u64, dst: &Buffer, dst_offset: u64, bytes: u64) {
        unsafe {
            self.compute.ctx.device.cmd_copy_buffer(self.cb, src.buffer, dst.buffer, &[vk::BufferCopy { src_offset, dst_offset, size: bytes }]);
        }
        self.barrier();
    }

    /// Copy a picture's pixels (RGBA16F, 8 bytes each, rows packed) into `dst`.
    pub fn copy_picture_to_buffer(&mut self, pic: &Picture, dst: &Buffer) {
        self.barrier();
        unsafe {
            self.compute.ctx.device.cmd_copy_image_to_buffer(self.cb, pic.image(), vk::ImageLayout::GENERAL, dst.buffer, &[picture_region(pic)]);
        }
        self.barrier();
    }

    /// Fill a picture from `src` (RGBA16F, 8 bytes a pixel, rows packed).
    pub fn copy_buffer_to_picture(&mut self, src: &Buffer, pic: &Picture) {
        self.barrier();
        unsafe {
            self.compute.ctx.device.cmd_copy_buffer_to_image(self.cb, src.buffer, pic.image(), vk::ImageLayout::GENERAL, &[picture_region(pic)]);
        }
        self.barrier();
    }

    /// Compute and transfer writes are visible to later compute, transfer and
    /// host reads.
    pub fn barrier(&mut self) {
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::HOST_READ);
        unsafe {
            self.compute.ctx.device.cmd_pipeline_barrier2(self.cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb)));
        }
    }

    /// Submit what is recorded so far (without waiting) and go on in a new command
    /// buffer. Long work is split this way so no single submission runs long:
    /// Windows resets a GPU that does not respond for about 2 seconds.
    pub fn flush(&mut self) -> Result<()> {
        let ctx = &self.compute.ctx;
        let device = &ctx.device;
        unsafe {
            device.end_command_buffer(self.cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cb)];
            let r = {
                let _q = ctx.lock_compute();
                device.queue_submit2(ctx.compute_queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)
            };
            if let Err(e) = r {
                device.destroy_fence(fence, None);
                return Err(e.into());
            }
            self.submitted.push(fence);
            self.first_submit.get_or_insert_with(std::time::Instant::now);
            self.cb = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            device.begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            // Later work sees what the earlier submissions wrote (same queue: a
            // barrier's first scope includes them).
            let mb = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
            device.cmd_pipeline_barrier2(self.cb, &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb)));
        }
        Ok(())
    }

    /// Submit what is recorded and wait for everything submitted (debugging: an
    /// error names the work that failed).
    pub fn sync(&mut self) -> Result<()> {
        self.flush()?;
        let device = &self.compute.ctx.device;
        let mut r = Ok(());
        for f in self.submitted.drain(..) {
            unsafe {
                if let Err(e) = device.wait_for_fences(&[f], true, u64::MAX) {
                    r = Err(e.into());
                }
                device.destroy_fence(f, None);
            }
        }
        r
    }

    /// Wait for the submissions `flush` made (before anything they use is freed).
    fn wait_submitted(&mut self) {
        let device = &self.compute.ctx.device;
        for f in self.submitted.drain(..) {
            unsafe {
                let _ = device.wait_for_fences(&[f], true, u64::MAX);
                device.destroy_fence(f, None);
            }
        }
    }

    /// Submit on the compute queue and wait for the GPU to finish.
    pub fn submit_wait(mut self) -> Result<std::time::Duration> {
        let ctx = &self.compute.ctx;
        let device = &ctx.device;
        unsafe {
            device.end_command_buffer(self.cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cb)];
            // Timed from the first submission (with `flush`, earlier parts are running).
            let t = self.first_submit.unwrap_or_else(std::time::Instant::now);
            let sub = {
                let _q = ctx.lock_compute();
                device.queue_submit2(ctx.compute_queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)
            };
            if let Err(e) = sub {
                device.destroy_fence(fence, None);
                self.wait_submitted();
                return Err(e.into());
            }
            let r = device.wait_for_fences(&[fence], true, u64::MAX);
            let elapsed = t.elapsed();
            device.destroy_fence(fence, None);
            self.wait_submitted();
            r?;
            if let Some((qp, labels)) = &self.queries {
                let mut data = vec![0u64; labels.len()];
                device.get_query_pool_results(*qp, 0, &mut data, vk::QueryResultFlags::TYPE_64)?;
                let period = ctx.instance.get_physical_device_properties(ctx.physical).limits.timestamp_period as f64;
                let mut by: HashMap<&str, (f64, usize)> = HashMap::new();
                for i in 1..labels.len() {
                    let e = by.entry(labels[i].as_str()).or_default();
                    e.0 += data[i].saturating_sub(data[i - 1]) as f64 * period / 1e6;
                    e.1 += 1;
                }
                let mut v: Vec<_> = by.into_iter().collect();
                v.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
                let total: f64 = v.iter().map(|x| x.1 .0).sum();
                log::info!("GPU profile: {total:.1} ms in {} dispatches", labels.len() - 1);
                for (label, (ms, n)) in v.iter().take(12) {
                    log::info!("  {:>8.2} ms {:>5.1}%  {n:>5}x  {}", ms, ms / total * 100.0, if label.is_empty() { "-" } else { label });
                }
            }
            Ok(elapsed)
        }
    }
}

impl Drop for Recorder<'_> {
    fn drop(&mut self) {
        // Submitted recorders have been waited for; unsubmitted ones never ran, except
        // parts already flushed: wait for those before freeing what they use.
        self.wait_submitted();
        unsafe {
            let device = &self.compute.ctx.device;
            if let Some((qp, _)) = self.queries.take() {
                device.destroy_query_pool(qp, None);
            }
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_command_pool(self.pool, None);
        }
    }
}

/// A picture on the GPU (RGBA, half float, GENERAL layout): written by compute
/// kernels as a storage image, sampled by the renderer and the effect chain.
/// Destroyed when the last holder drops it (holders keep it while GPU work using it
/// may be in flight).
pub struct Picture {
    ctx: Arc<GpuContext>,
    img: Option<crate::gpu::GpuImage>,
    pub view: vk::ImageView,
    pub w: u32,
    pub h: u32,
    /// The alpha channel carries transparency (else it is 1).
    pub alpha: bool,
}

impl Picture {
    /// A new picture; `rec` gets its layout transition.
    pub fn new(compute: &Compute, rec: &mut Recorder, w: u32, h: u32, alpha: bool) -> Result<Arc<Picture>> {
        let ctx = compute.ctx.clone();
        let img = crate::gpu::create_image(
            &ctx.device,
            &ctx.allocator,
            w.max(1),
            h.max(1),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            "picture",
        )?;
        crate::gpu::image_barrier(
            &ctx.device,
            rec.command_buffer(),
            img.image,
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
            (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ, vk::ImageLayout::GENERAL),
        );
        let view = img.view;
        Ok(Arc::new(Picture { ctx, img: Some(img), view, w, h, alpha }))
    }

    pub fn image(&self) -> vk::Image {
        self.img.as_ref().map_or(vk::Image::null(), |i| i.image)
    }

    /// Bytes of its pixels in a buffer (RGBA16F, rows packed).
    pub fn byte_size(&self) -> u64 {
        self.w.max(1) as u64 * self.h.max(1) as u64 * 8
    }
}

/// The whole of a picture, for buffer copies.
fn picture_region(pic: &Picture) -> vk::BufferImageCopy {
    vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
        .image_extent(vk::Extent3D { width: pic.w.max(1), height: pic.h.max(1), depth: 1 })
}

impl Drop for Picture {
    fn drop(&mut self) {
        if let Some(img) = self.img.take() {
            crate::gpu::destroy_image(&self.ctx.device, &self.ctx.allocator, img);
        }
    }
}
