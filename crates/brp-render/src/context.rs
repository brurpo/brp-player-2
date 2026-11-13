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
    /// Second queue of the same family for background compute (AI), or `queue`
    /// itself when the family has only one.
    pub compute_queue: vk::Queue,
    /// Submissions to a queue must not overlap: hold the lock of the queue used.
    pub queue_lock: Mutex<()>,
    compute_lock: Mutex<()>,
    /// 16-bit storage buffers and float16 arithmetic are available (fp16 tensors).
    pub fp16: bool,
    /// Cooperative matrices (tensor cores): 16x16x16, fp16 inputs, fp32 accumulation,
    /// per subgroup.
    pub coopmat: bool,
    pub subgroup_size: u32,
    /// Cooperative matrices also sum in fp16 (twice the rate on GeForce cards).
    pub coopmat_acc16: bool,
    /// VK_EXT_memory_budget: how much memory this process may use right now.
    pub memory_budget: bool,
    /// The driver's compiled pipelines, kept on disk (see `shader_cache`).
    pub pipeline_cache: vk::PipelineCache,
}

/// What `create_device` made.
struct Device {
    device: ash::Device,
    queue: vk::Queue,
    compute_queue: vk::Queue,
    allocator: Allocator,
    fp16: bool,
    coopmat: bool,
    subgroup_size: u32,
    coopmat_acc16: bool,
    memory_budget: bool,
}

/// Logical device with the features the renderer, user shaders and compute need,
/// one or two queues of `queue_family`, and an allocator (buffer device addresses on).
unsafe fn create_device(entry: &ash::Entry, instance: &ash::Instance, physical: vk::PhysicalDevice, queue_family: u32, exts: &[*const std::ffi::c_char]) -> Result<Device> {
    unsafe {
        let families = instance.get_physical_device_queue_family_properties(physical);
        let two = families[queue_family as usize].queue_count >= 2;
        let prio = [1.0f32, 0.5];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(if two { &prio[..] } else { &prio[..1] })];
        // What the GPU has.
        let (mut s11, mut s12) = (vk::PhysicalDeviceVulkan11Features::default(), vk::PhysicalDeviceVulkan12Features::default());
        let mut s2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut s11).push_next(&mut s12);
        instance.get_physical_device_features2(physical, &mut s2);
        let supported = s2.features;
        let fp16 = s11.storage_buffer16_bit_access != 0 && s12.shader_float16 != 0;
        // Cooperative matrices: the 16x16x16 fp16 x fp16 -> fp32 subgroup shape.
        let available = instance.enumerate_device_extension_properties(physical).unwrap_or_default();
        let has_cm_ext = available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::cooperative_matrix::NAME));
        let mut coopmat = false;
        let mut coopmat_acc16 = false;
        if has_cm_ext && fp16 {
            let cm = ash::khr::cooperative_matrix::Instance::new(entry, instance);
            let props = cm.get_physical_device_cooperative_matrix_properties(physical).unwrap_or_default();
            let has = |acc: vk::ComponentTypeKHR| {
                props.iter().any(|p| {
                    (p.m_size, p.n_size, p.k_size) == (16, 16, 16)
                        && p.a_type == vk::ComponentTypeKHR::FLOAT16
                        && p.b_type == vk::ComponentTypeKHR::FLOAT16
                        && p.c_type == acc
                        && p.result_type == acc
                        && p.scope == vk::ScopeKHR::SUBGROUP
                })
            };
            coopmat = has(vk::ComponentTypeKHR::FLOAT32);
            coopmat_acc16 = coopmat && has(vk::ComponentTypeKHR::FLOAT16);
        }
        let memory_budget = available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
        // (Whether they are faster than the plain kernels is measured once the
        // device is up: `coopcheck`.)
        if std::env::var_os("BRP_NO_COOPMAT").is_some() {
            coopmat = false;
            coopmat_acc16 = false;
        }
        let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
        instance.get_physical_device_properties2(physical, &mut p2);
        let subgroup_size = sg.subgroup_size;
        let mut exts = exts.to_vec();
        if memory_budget {
            exts.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        let mut fcm = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        if coopmat {
            exts.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(fp16);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true).shader_float16(fp16).scalar_block_layout(s12.scalar_block_layout != 0)
            .vulkan_memory_model(s12.vulkan_memory_model != 0);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
        // BRP_PIPELINE_STATS: the driver's statistics per compute kernel (registers...).
        let mut fstats = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
        let stats = std::env::var_os("BRP_PIPELINE_STATS").is_some();
        if stats {
            exts.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        }
        // User shaders use storage images like r16f (extended formats), and
        // textureGatherOffset(s) (mpv nlmeans), which needs shaderImageGatherExtended.
        let base = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(supported.shader_storage_image_extended_formats != 0)
            .shader_image_gather_extended(supported.shader_image_gather_extended != 0)
            .shader_int64(supported.shader_int64 != 0);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&exts)
            .enabled_features(&base)
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if stats {
            info = info.push_next(&mut fstats);
        }
        if coopmat {
            info = info.push_next(&mut fcm);
        }
        let device = instance.create_device(physical, &info, None)?;
        let queue = device.get_device_queue(queue_family, 0);
        let compute_queue = if two { device.get_device_queue(queue_family, 1) } else { queue };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;
        log::info!("GPU compute: {} queue(s), fp16 {fp16}, cooperative matrices {coopmat}, subgroup {subgroup_size}", if two { 2 } else { 1 });
        Ok(Device { device, queue, compute_queue, allocator, fp16, coopmat, subgroup_size, coopmat_acc16, memory_budget })
    }
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
            // HDR10 swapchains on HDR screens (Renderer, DESIGN.md §7b).
            if has_ext(ash::ext::swapchain_colorspace::NAME) {
                exts.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
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
    /// Size of the GPU's own memory (its largest device-local heap; on integrated
    /// GPUs, the share of system memory it may use).
    pub fn vram(&self) -> u64 {
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.physical) };
        props.memory_heaps[..props.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(2 << 30)
    }

    /// GPU memory this process may still allocate now (the system's budget for it,
    /// other programs' use counted, minus what it uses), if the driver says.
    pub fn vram_free(&self) -> Option<u64> {
        if !self.memory_budget {
            return None;
        }
        let mut mb = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut mb);
        unsafe { self.instance.get_physical_device_memory_properties2(self.physical, &mut p2) };
        let props = p2.memory_properties;
        (0..props.memory_heap_count as usize)
            .filter(|&i| props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .max_by_key(|&i| props.memory_heaps[i].size)
            .map(|i| mb.heap_budget[i].saturating_sub(mb.heap_usage[i]))
    }

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
            let Device { device, queue, compute_queue, allocator, fp16, coopmat, subgroup_size, coopmat_acc16, memory_budget } = create_device(&entry, &instance, physical, queue_family, &dev_exts)?;
            let pipeline_cache = crate::shader_cache::load_pipeline_cache(&device, &instance.get_physical_device_properties(physical));
            lap("device");
            let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);

            Ok(crate::coopcheck::check(GpuContext {
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
                compute_queue,
                queue_lock: Mutex::new(()),
                compute_lock: Mutex::new(()),
                fp16,
                coopmat,
                subgroup_size,
                coopmat_acc16,
                memory_budget,
                pipeline_cache,
            }))
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
            let Device { device, queue, compute_queue, allocator, fp16, coopmat, subgroup_size, coopmat_acc16, memory_budget } = create_device(&entry, &instance, physical, queue_family, &[])?;
            let pipeline_cache = crate::shader_cache::load_pipeline_cache(&device, &instance.get_physical_device_properties(physical));
            let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
            let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
            Ok(crate::coopcheck::check(GpuContext {
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
                compute_queue,
                queue_lock: Mutex::new(()),
                compute_lock: Mutex::new(()),
                fp16,
                coopmat,
                subgroup_size,
                coopmat_acc16,
                memory_budget,
                pipeline_cache,
            }))
        }
    }

    /// Write the driver's compiled pipelines to disk (also done when the context goes).
    pub fn save_pipeline_cache(&self) {
        crate::shader_cache::save_pipeline_cache(&self.device, self.pipeline_cache);
    }

    pub fn properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe { self.instance.get_physical_device_properties(self.physical) }
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
            self.save_pipeline_cache();
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
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

impl GpuContext {
    /// Lock for submitting to the compute queue (the main queue's lock when they are
    /// the same queue).
    pub fn lock_compute(&self) -> std::sync::MutexGuard<'_, ()> {
        if self.compute_queue == self.queue { self.queue_lock.lock().unwrap() } else { self.compute_lock.lock().unwrap() }
    }

    pub fn lock_queue(&self) -> std::sync::MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap()
    }
}

impl GpuContext {
    /// Wait until the GPU is idle. Holds both queue locks: no other thread may submit
    /// meanwhile (vkDeviceWaitIdle needs every queue).
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        let _q = self.queue_lock.lock().unwrap();
        let _c = (self.compute_queue != self.queue).then(|| self.compute_lock.lock().unwrap());
        unsafe { self.device.device_wait_idle() }
    }
}
