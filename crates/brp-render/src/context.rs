//! Vulkan instance, device, queue and allocator.

use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

pub struct GpuDevice {
    pub name: String,
    pub discrete: bool,
}

pub struct GpuContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_fn: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub allocator: std::mem::ManuallyDrop<Arc<Mutex<Allocator>>>,
    pub swapchain_fn: ash::khr::swapchain::Device,
    pub device_name: String,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { (*data).message_as_c_str() }.map(|s| s.to_string_lossy()).unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("vulkan: {msg}");
    } else {
        log::warn!("vulkan: {msg}");
    }
    vk::FALSE
}

/// Vulkan instance, created without a window so it can start on a background
/// thread while the window opens (driver start-up is the slowest part of launch).
pub struct GpuInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl GpuInstance {
    pub fn new() -> Result<Self> {
        let t0 = std::time::Instant::now();
        let lap = |what: &str| log::debug!("vk init: {what} at {:?}", t0.elapsed());
        unsafe {
            let entry = ash::Entry::load().context("Vulkan loader not found")?;
            lap("loader");
            let available = entry.enumerate_instance_extension_properties(None)?;
            let has_ext = |name: &CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            // Every surface extension this system offers (the window type is not known yet).
            let mut exts: Vec<*const i8> = [
                ash::khr::surface::NAME,
                ash::khr::win32_surface::NAME,
                ash::khr::xlib_surface::NAME,
                ash::khr::xcb_surface::NAME,
                ash::khr::wayland_surface::NAME,
                ash::ext::metal_surface::NAME,
            ]
            .into_iter()
            .filter(|n| has_ext(n))
            .map(|n| n.as_ptr())
            .collect();
            let validation = cfg!(debug_assertions) && std::env::var_os("BRP_NO_VALIDATION").is_none();
            let layer = c"VK_LAYER_KHRONOS_validation";
            let has_layer = entry
                .enumerate_instance_layer_properties()?
                .iter()
                .any(|l| l.layer_name_as_c_str() == Ok(layer));
            let use_debug = validation && has_layer && has_ext(ash::ext::debug_utils::NAME);
            if use_debug {
                exts.push(ash::ext::debug_utils::NAME.as_ptr());
            }
            let mut flags = vk::InstanceCreateFlags::empty();
            if has_ext(ash::khr::portability_enumeration::NAME) {
                exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let layers = if use_debug { vec![layer.as_ptr()] } else { vec![] };
            let app = vk::ApplicationInfo::default()
                .application_name(c"BRP Player")
                .engine_name(c"brp")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&exts)
                    .enabled_layer_names(&layers)
                    .flags(flags),
                None,
            )?;

            lap("instance");
            let debug = if use_debug {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let m = du.create_debug_utils_messenger(
                    &vk::DebugUtilsMessengerCreateInfoEXT::default()
                        .message_severity(
                            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                        )
                        .message_type(
                            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                        )
                        .pfn_user_callback(Some(debug_callback)),
                    None,
                )?;
                Some((du, m))
            } else {
                None
            };

            Ok(GpuInstance { entry, instance, debug })
        }
    }
}

impl GpuContext {
    /// `preferred` picks a GPU by (part of) its name; otherwise a discrete GPU wins.
    pub fn new(inst: GpuInstance, window: &(impl HasDisplayHandle + HasWindowHandle), preferred: Option<&str>) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let lap = |what: &str| log::debug!("vk init: {what} at {:?}", t0.elapsed());
        let GpuInstance { entry, instance, debug } = inst;
        unsafe {
            let display = window.display_handle().map_err(|e| anyhow::anyhow!("display handle: {e:?}"))?.as_raw();
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let surface = ash_window::create_surface(&entry, &instance, display, window.window_handle().map_err(|e| anyhow::anyhow!("window handle: {e:?}"))?.as_raw(), None)?;

            lap("surface");
            // Pick a device that can present to this surface and supports Vulkan 1.3.
            let mut best: Option<(i32, vk::PhysicalDevice, u32, String)> = None;
            for pd in instance.enumerate_physical_devices()? {
                let props = instance.get_physical_device_properties(pd);
                if props.api_version < vk::API_VERSION_1_3 {
                    continue;
                }
                let name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let families = instance.get_physical_device_queue_family_properties(pd);
                let Some(qf) = families.iter().enumerate().find_map(|(i, f)| {
                    let ok = f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface_fn.get_physical_device_surface_support(pd, i as u32, surface).unwrap_or(false);
                    ok.then_some(i as u32)
                }) else {
                    continue;
                };
                let mut score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 100,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 50,
                    _ => 10,
                };
                if let Some(p) = preferred {
                    if name.to_lowercase().contains(&p.to_lowercase()) {
                        score += 1000;
                    }
                }
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, pd, qf, name));
                }
            }
            let (_, physical, queue_family, device_name) = best.context("no Vulkan 1.3 GPU can present to this window")?;
            log::info!("GPU: {device_name}");

            lap("pick device");
            let mut dev_exts = vec![ash::khr::swapchain::NAME.as_ptr()];
            let dev_available = instance.enumerate_device_extension_properties(physical)?;
            if dev_available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_subset::NAME)) {
                dev_exts.push(ash::khr::portability_subset::NAME.as_ptr());
            }
            let prio = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
            // User shaders use storage images like r16f (extended formats).
            let supported = instance.get_physical_device_features(physical);
            // User shaders also use textureGatherOffset(s) (mpv nlmeans), which needs
            // shaderImageGatherExtended; without it results are undefined.
            let base = vk::PhysicalDeviceFeatures::default()
                .shader_storage_image_extended_formats(supported.shader_storage_image_extended_formats != 0)
                .shader_image_gather_extended(supported.shader_image_gather_extended != 0);
            let device = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&dev_exts)
                    .enabled_features(&base)
                    .push_next(&mut f13),
                None,
            )?;
            lap("device");
            let queue = device.get_device_queue(queue_family, 0);
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: physical,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })?;
            lap("allocator");
            let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);

            Ok(GpuContext {
                entry,
                instance,
                debug,
                surface_fn,
                surface,
                physical,
                device,
                queue,
                queue_family,
                allocator: std::mem::ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
                swapchain_fn,
                device_name,
            })
        }
    }

    /// A context without a window (tests, offline processing, export).
    pub fn headless(inst: GpuInstance, preferred: Option<&str>) -> Result<Self> {
        let GpuInstance { entry, instance, debug } = inst;
        unsafe {
            let mut best: Option<(i32, vk::PhysicalDevice, u32, String)> = None;
            for pd in instance.enumerate_physical_devices()? {
                let props = instance.get_physical_device_properties(pd);
                if props.api_version < vk::API_VERSION_1_3 {
                    continue;
                }
                let name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let Some(qf) = instance.get_physical_device_queue_family_properties(pd).iter().position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                }) else {
                    continue;
                };
                let mut score = if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU { 100 } else { 50 };
                if preferred.is_some_and(|p| name.to_lowercase().contains(&p.to_lowercase())) {
                    score += 1000;
                }
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, pd, qf as u32, name));
                }
            }
            let (_, physical, queue_family, device_name) = best.context("no Vulkan 1.3 GPU")?;
            let prio = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
            let device = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues).push_next(&mut f13),
                None,
            )?;
            let queue = device.get_device_queue(queue_family, 0);
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: physical,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })?;
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
            Ok(GpuContext {
                entry,
                instance,
                debug,
                surface_fn,
                surface: vk::SurfaceKHR::null(),
                physical,
                device,
                queue,
                queue_family,
                allocator: std::mem::ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
                swapchain_fn,
                device_name,
            })
        }
    }

    /// All Vulkan GPUs on the system (for the settings menu).
    pub fn list_devices(&self) -> Vec<GpuDevice> {
        unsafe {
            self.instance
                .enumerate_physical_devices()
                .unwrap_or_default()
                .into_iter()
                .map(|pd| {
                    let p = self.instance.get_physical_device_properties(pd);
                    GpuDevice {
                        name: p.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                        discrete: p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                    }
                })
                .collect()
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // The allocator must go before the device (users of clones must be dropped first).
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_fn.destroy_surface(self.surface, None);
            }
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
