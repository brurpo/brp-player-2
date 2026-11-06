//! GPU frame cache: decoded frames uploaded once, reused by display and processing
//! (deinterlacing needs frames N-1, N, N+1; effects will need more).

use crate::gpu::{self, GpuImage, Graveyard, SharedAllocator, Staging};
use anyhow::Result;
use ash::vk;
use brp_media::{FrameLayout, VideoFrame};

pub type FrameKey = (u64, i64);

pub struct GpuFrame {
    pub key: FrameKey,
    pub planes: Vec<GpuImage>,
    pub layout: FrameLayout,
    last_used: u64,
}

impl GpuFrame {
    pub fn views(&self) -> Vec<vk::ImageView> {
        self.planes.iter().map(|p| p.view).collect()
    }
}

pub struct FrameCache {
    entries: Vec<GpuFrame>,
    capacity: usize,
}

fn plane_format(layout: FrameLayout, bytes_per_sample: usize) -> vk::Format {
    match layout {
        FrameLayout::Yuv { .. } if bytes_per_sample == 2 => vk::Format::R16_UNORM,
        FrameLayout::Yuv { .. } => vk::Format::R8_UNORM,
        FrameLayout::Rgba8 { .. } => vk::Format::R8G8B8A8_UNORM,
        FrameLayout::Rgba16 => vk::Format::R16G16B16A16_UNORM,
    }
}

fn texel_size(f: vk::Format) -> usize {
    match f {
        vk::Format::R8_UNORM => 1,
        vk::Format::R16_UNORM => 2,
        vk::Format::R8G8B8A8_UNORM => 4,
        _ => 8,
    }
}

impl FrameCache {
    pub fn new(capacity: usize) -> Self {
        FrameCache { entries: Vec::new(), capacity }
    }

    pub fn find(&self, key: FrameKey) -> Option<usize> {
        self.entries.iter().position(|e| e.key == key)
    }

    pub fn get(&self, i: usize) -> &GpuFrame {
        &self.entries[i]
    }

    /// Bytes needed in staging to upload those of `frames` that are not cached yet.
    pub fn upload_size(&self, source: u64, frames: &[&VideoFrame]) -> u64 {
        let mut seen = Vec::new();
        frames
            .iter()
            .filter(|f| {
                let k = (source, f.index());
                let new = self.find(k).is_none() && !seen.contains(&k);
                seen.push(k);
                new
            })
            .map(|f| (0..f.plane_count()).map(|p| f.plane(p).0.len().next_multiple_of(16) as u64).sum::<u64>())
            .sum()
    }

    /// Make sure `frame` is on the GPU, uploading it if needed (recorded into `cb`).
    /// Look entries up with [`find`] afterwards: indices change on eviction.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure(
        &mut self,
        device: &ash::Device,
        allocator: &SharedAllocator,
        cb: vk::CommandBuffer,
        staging: &mut Staging,
        graveyard: &mut Graveyard,
        now: u64,
        source: u64,
        frame: &VideoFrame,
    ) -> Result<()> {
        let key = (source, frame.index());
        if let Some(i) = self.find(key) {
            self.entries[i].last_used = now;
            return Ok(());
        }
        // Evict the least recently used entry if full (deferred destroy). Never evict
        // something already used this frame; grow instead.
        if self.entries.len() >= self.capacity {
            let lru = self.entries.iter().enumerate().filter(|(_, e)| e.last_used < now).min_by_key(|(_, e)| e.last_used);
            if let Some((i, _)) = lru {
                let old = self.entries.swap_remove(i);
                for p in old.planes {
                    graveyard.bury(old.last_used, p);
                }
            }
        }

        let n = frame.plane_count();
        let fmt = plane_format(frame.layout, frame.bytes_per_sample());
        let mut planes = Vec::with_capacity(n);
        for p in 0..n {
            let (w, h) = frame.plane_size(p);
            planes.push(gpu::create_image(
                device,
                allocator,
                w,
                h,
                fmt,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                "frame plane",
            )?);
        }
        for (p, img) in planes.iter().enumerate() {
            let (data, stride) = frame.plane(p);
            let off = staging.push(data);
            gpu::image_barrier(
                device,
                cb,
                img.image,
                (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
                (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            );
            let region = vk::BufferImageCopy::default()
                .buffer_offset(off)
                .buffer_row_length((stride / texel_size(fmt)) as u32)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: img.w, height: img.h, depth: 1 });
            unsafe {
                device.cmd_copy_buffer_to_image(cb, staging.buffer, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }
            gpu::image_barrier(
                device,
                cb,
                img.image,
                (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                (
                    vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ),
            );
        }
        self.entries.push(GpuFrame { key, planes, layout: frame.layout, last_used: now });
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &SharedAllocator) {
        for e in self.entries.drain(..) {
            for p in e.planes {
                gpu::destroy_image(device, allocator, p);
            }
        }
    }
}
