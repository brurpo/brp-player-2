//! Offline (windowless) processing: run the GPU deinterlacer on frames and read the
//! result back. Used by golden tests against FFmpeg, and later by export.

use crate::context::GpuContext;
use crate::deint::{DeintParams, Deinterlacer};
use crate::frames::FrameCache;
use crate::gpu::{Graveyard, Staging};
use anyhow::{Context as _, Result};
use ash::vk;
use brp_media::VideoFrame;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

pub struct Offline {
    ctx: GpuContext,
    pool: vk::CommandPool,
    cache: FrameCache,
    staging: Staging,
    graveyard: Graveyard,
    deint: Option<Deinterlacer>,
    frame_no: u64,
}

/// One processed plane: width, height, samples in the source's normalised domain
/// (what an R8/R16 UNORM texture would return).
pub struct Plane {
    pub w: u32,
    pub h: u32,
    pub data: Vec<f32>,
}

impl Offline {
    pub fn new(ctx: GpuContext, weights: Option<std::path::PathBuf>) -> Result<Self> {
        unsafe {
            let pool = ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let props = ctx.instance.get_physical_device_properties(ctx.physical);
            let integrated = props.device_type != vk::PhysicalDeviceType::DISCRETE_GPU;
            let deint = Deinterlacer::new(&ctx.device, (*ctx.allocator).clone(), 1, weights, props.limits.max_compute_shared_memory_size, integrated)?;
            Ok(Offline { ctx, pool, cache: FrameCache::new(8), staging: Staging::new(), graveyard: Graveyard::default(), deint: Some(deint), frame_no: 0 })
        }
    }

    pub fn device_name(&self) -> &str {
        &self.ctx.device_name
    }

    /// Deinterlace `cur` (with neighbours) and return its planes. Also returns the GPU time.
    pub fn deinterlace(&mut self, prev: &VideoFrame, cur: &VideoFrame, next: &VideoFrame, params: DeintParams) -> Result<(Vec<Plane>, std::time::Duration)> {
        let device = self.ctx.device.clone();
        let alloc = (*self.ctx.allocator).clone();
        self.frame_no += 1;
        unsafe {
            let cb = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let list = [prev, cur, next];
            let bytes = self.cache.upload_size(1, &list);
            self.staging.reset(&device, &alloc, bytes)?;
            for f in list {
                self.cache.ensure(&device, &alloc, cb, &mut self.staging, &mut self.graveyard, self.frame_no, 1, f)?;
            }
            let (pi, ci, ni) = (
                self.cache.find((1, prev.index())).unwrap(),
                self.cache.find((1, cur.index())).unwrap(),
                self.cache.find((1, next.index())).unwrap(),
            );
            let d = self.deint.as_mut().unwrap();
            d.invalidate();
            d.run(cb, 0, params, self.cache.get(pi), self.cache.get(ci), self.cache.get(ni), &mut self.graveyard, self.frame_no)?
                .context("frame layout cannot be deinterlaced")?;

            // Read back every output plane.
            let outs = d.outputs(0);
            let sizes: Vec<(u32, u32)> = outs.iter().map(|o| (o.w, o.h)).collect();
            let total: u64 = sizes.iter().map(|(w, h)| (*w as u64) * (*h as u64) * 4).sum();
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
            let mut off = 0u64;
            for o in outs {
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(off)
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_extent(vk::Extent3D { width: o.w, height: o.h, depth: 1 });
                device.cmd_copy_image_to_buffer(cb, o.image, vk::ImageLayout::GENERAL, buffer, &[region]);
                off += o.w as u64 * o.h as u64 * 4;
            }
            device.end_command_buffer(cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let t = std::time::Instant::now();
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            device.queue_submit2(self.ctx.queue, &[vk::SubmitInfo2::default().command_buffer_infos(&cbs)], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            let elapsed = t.elapsed();
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.pool, &[cb]);

            let raw = ralloc.mapped_slice().context("readback not mapped")?;
            let mut planes = Vec::new();
            let mut off = 0usize;
            for (w, h) in sizes {
                let n = (w * h) as usize;
                let data = raw[off..off + n * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                planes.push(Plane { w, h, data });
                off += n * 4;
            }
            alloc.lock().unwrap().free(ralloc)?;
            device.destroy_buffer(buffer, None);
            self.graveyard.collect(&device, &alloc, self.frame_no + 10, 1);
            Ok((planes, elapsed))
        }
    }
}

impl Drop for Offline {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.deint = None;
            let alloc = (*self.ctx.allocator).clone();
            self.cache.destroy(&self.ctx.device, &alloc);
            self.staging.destroy(&self.ctx.device, &alloc);
            self.graveyard.destroy_all(&self.ctx.device, &alloc);
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}
