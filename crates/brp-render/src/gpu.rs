//! Small GPU resource helpers shared by the passes.

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

pub type SharedAllocator = Arc<Mutex<Allocator>>;

pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    alloc: Option<Allocation>,
    pub w: u32,
    pub h: u32,
    pub format: vk::Format,
}

pub fn create_image(
    device: &ash::Device,
    allocator: &SharedAllocator,
    w: u32,
    h: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    name: &str,
) -> Result<GpuImage> {
    unsafe {
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage),
            None,
        )?;
        let req = device.get_image_memory_requirements(image);
        let alloc = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        device.bind_image_memory(image, alloc.memory(), alloc.offset())?;
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(color_range()),
            None,
        )?;
        Ok(GpuImage { image, view, alloc: Some(alloc), w, h, format })
    }
}

pub fn destroy_image(device: &ash::Device, allocator: &SharedAllocator, mut img: GpuImage) {
    unsafe {
        device.destroy_image_view(img.view, None);
        device.destroy_image(img.image, None);
    }
    if let Some(a) = img.alloc.take() {
        let _ = allocator.lock().unwrap().free(a);
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

pub fn image_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    (src_stage, src_access, old): (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
    (dst_stage, dst_access, new): (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
) {
    let b = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old)
        .new_layout(new)
        .image(image)
        .subresource_range(color_range());
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&b))) };
}

/// Host-visible upload buffer, one per frame in flight. Reset at the start of each frame.
pub struct Staging {
    pub buffer: vk::Buffer,
    alloc: Option<Allocation>,
    size: u64,
    used: u64,
}

impl Staging {
    pub fn new() -> Self {
        Staging { buffer: vk::Buffer::null(), alloc: None, size: 0, used: 0 }
    }

    /// Make room for `bytes` this frame. Only call before any copy is recorded.
    pub fn reset(&mut self, device: &ash::Device, allocator: &SharedAllocator, bytes: u64) -> Result<()> {
        self.used = 0;
        if self.size >= bytes {
            return Ok(());
        }
        self.destroy(device, allocator);
        let size = (bytes + (8 << 20)).next_power_of_two();
        unsafe {
            let buffer = device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC), None)?;
            let req = device.get_buffer_memory_requirements(buffer);
            let alloc = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "staging",
                requirements: req,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())?;
            self.buffer = buffer;
            self.alloc = Some(alloc);
            self.size = size;
        }
        Ok(())
    }

    /// Copy bytes in; returns their offset in the buffer.
    pub fn push(&mut self, data: &[u8]) -> u64 {
        let off = self.used;
        let mapped = self.alloc.as_mut().unwrap().mapped_slice_mut().unwrap();
        mapped[off as usize..off as usize + data.len()].copy_from_slice(data);
        self.used = (off + data.len() as u64).next_multiple_of(16);
        off
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &SharedAllocator) {
        if let Some(a) = self.alloc.take() {
            let _ = allocator.lock().unwrap().free(a);
            unsafe { device.destroy_buffer(self.buffer, None) };
        }
        self.size = 0;
    }
}

/// Images whose destruction waits until no frame in flight can still use them.
#[derive(Default)]
pub struct Graveyard {
    items: Vec<(u64, GpuImage)>,
}

impl Graveyard {
    pub fn bury(&mut self, frame: u64, img: GpuImage) {
        self.items.push((frame, img));
    }

    /// Destroy everything last used at least `frames_in_flight` frames ago.
    pub fn collect(&mut self, device: &ash::Device, allocator: &SharedAllocator, now: u64, frames_in_flight: u64) {
        let (dead, alive): (Vec<_>, Vec<_>) = std::mem::take(&mut self.items).into_iter().partition(|(f, _)| f + frames_in_flight <= now);
        self.items = alive;
        for (_, img) in dead {
            destroy_image(device, allocator, img);
        }
    }

    pub fn destroy_all(&mut self, device: &ash::Device, allocator: &SharedAllocator) {
        for (_, img) in self.items.drain(..) {
            destroy_image(device, allocator, img);
        }
    }
}
