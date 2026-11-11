//! Compiled shaders kept on disk between runs, so a shader is compiled once, not at
//! every launch:
//! - SPIR-V, by a hash of the GLSL source and compile options: a shader is compiled
//!   again only when its source changes (a new build of the app, an edited effect).
//! - The driver's pipeline cache (`pipelines.bin`), which turns SPIR-V into the GPU's
//!   own code; its header names the GPU and driver, and a cache from another driver
//!   is ignored (rebuilt after a driver update).
//! - A stamp of what the start-up warm-up last built for (`warm.stamp`: this build of
//!   the app, the GPU and its driver): see `needs_warm_up`.
//!
//! The folder is `%LOCALAPPDATA%\BRP Player\shader_cache` (else `$XDG_CACHE_HOME` or
//! `~/.cache` + `brp-player/shader_cache`); `BRP_SHADER_CACHE` sets another, `off`
//! turns caching off.

use ash::vk;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Bumped when the cache's format or the compile settings change.
const FORMAT: &str = "brp-shader-cache-1";

/// GLSL compiles that missed the cache (this process).
pub static COMPILED: AtomicUsize = AtomicUsize::new(0);
/// Shaders found in the cache (this process).
pub static HITS: AtomicUsize = AtomicUsize::new(0);

pub fn dir() -> Option<PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = match std::env::var_os("BRP_SHADER_CACHE") {
            Some(v) if v == "off" => return None,
            Some(v) => PathBuf::from(v),
            None => {
                if let Some(local) = std::env::var_os("LOCALAPPDATA") {
                    PathBuf::from(local).join("BRP Player").join("shader_cache")
                } else {
                    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
                    base.join("brp-player").join("shader_cache")
                }
            }
        };
        match std::fs::create_dir_all(&d) {
            Ok(()) => Some(d),
            Err(e) => {
                log::warn!("shader cache off: {}: {e}", d.display());
                None
            }
        }
    })
    .clone()
}

/// 128-bit FNV-1a (two 64-bit lanes with different offsets), as 32 hex digits.
fn hash(parts: &[&[u8]]) -> String {
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64);
    for p in parts {
        for &x in p.iter().chain(&[0xff]) {
            a = (a ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
            b = (b ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
        }
    }
    format!("{a:016x}{b:016x}")
}

/// Write via a temporary file and a rename: a reader never sees half a file, and two
/// processes writing the same entry leave a whole one.
fn write_atomic(path: &std::path::Path, bytes: &[u8]) {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// The SPIR-V of `glsl` compiled with `options` (a description of the settings),
/// from the cache or made by `compile` (then stored).
pub fn spirv(glsl: &str, options: &str, compile: impl FnOnce() -> anyhow::Result<Vec<u32>>) -> anyhow::Result<Vec<u32>> {
    let path = dir().map(|d| d.join(format!("{}.spv", hash(&[FORMAT.as_bytes(), options.as_bytes(), glsl.as_bytes()]))));
    if let Some(p) = &path {
        if let Ok(b) = std::fs::read(p) {
            // A whole SPIR-V module: its magic number first.
            if b.len() >= 20 && b.len() % 4 == 0 && b[..4] == 0x0723_0203u32.to_le_bytes() {
                HITS.fetch_add(1, Ordering::Relaxed);
                // Marked as used (old entries are removed: see `prune`).
                if let Ok(f) = std::fs::File::options().write(true).open(p) {
                    let _ = f.set_modified(std::time::SystemTime::now());
                }
                return Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect());
            }
        }
    }
    let spv = compile()?;
    COMPILED.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = &path {
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        write_atomic(p, &bytes);
    }
    Ok(spv)
}

/// What a pipeline cache is valid for: the GPU and its driver.
fn device_id(props: &vk::PhysicalDeviceProperties) -> String {
    let uuid: String = props.pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!("{:04x}:{:04x} driver {:08x} cache {uuid}", props.vendor_id, props.device_id, props.driver_version)
}

/// The driver's pipeline cache, with what was saved for this GPU and driver.
pub fn load_pipeline_cache(device: &ash::Device, props: &vk::PhysicalDeviceProperties) -> vk::PipelineCache {
    let t = std::time::Instant::now();
    let data = dir().and_then(|d| std::fs::read(d.join("pipelines.bin")).ok()).filter(|b| {
        // Header (Vulkan spec): length, version, vendor, device, cache UUID.
        b.len() >= 32
            && u32::from_le_bytes(b[8..12].try_into().unwrap()) == props.vendor_id
            && u32::from_le_bytes(b[12..16].try_into().unwrap()) == props.device_id
            && b[16..32] == props.pipeline_cache_uuid
    });
    let info = vk::PipelineCacheCreateInfo::default();
    let info = match &data {
        Some(d) => info.initial_data(d),
        None => info,
    };
    let cache = unsafe {
        device.create_pipeline_cache(&info, None).or_else(|_| device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)).unwrap_or(vk::PipelineCache::null())
    };
    log::debug!("pipeline cache: {} KB loaded in {:.1} ms", data.as_ref().map_or(0, |d| d.len() >> 10), t.elapsed().as_secs_f64() * 1e3);
    cache
}

pub fn save_pipeline_cache(device: &ash::Device, cache: vk::PipelineCache) {
    if cache == vk::PipelineCache::null() {
        return;
    }
    let Some(d) = dir() else { return };
    if let Ok(data) = unsafe { device.get_pipeline_cache_data(cache) } {
        if !data.is_empty() {
            write_atomic(&d.join("pipelines.bin"), &data);
        }
    }
}

/// What the warm-up builds for: this executable (path, size, time) and the GPU with
/// its driver. A different build may have changed shaders; a different driver
/// compiles them to new GPU code.
fn warm_stamp(props: &vk::PhysicalDeviceProperties) -> String {
    let exe = std::env::current_exe().ok();
    let meta = exe.as_ref().and_then(|p| std::fs::metadata(p).ok());
    let modified = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
    format!("{FORMAT}\n{}\n{} {modified}\n{}\n", exe.map(|p| p.display().to_string()).unwrap_or_default(), meta.map_or(0, |m| m.len()), device_id(props))
}

/// The shaders need building at start-up: never built for this build of the app on
/// this GPU and driver (or `BRP_WARM_UP=1`). Without a cache, never.
pub fn needs_warm_up(props: &vk::PhysicalDeviceProperties) -> bool {
    let Some(d) = dir() else { return false };
    std::env::var_os("BRP_WARM_UP").is_some() || std::fs::read_to_string(d.join("warm.stamp")).ok().as_deref() != Some(warm_stamp(props).as_str())
}

/// The warm-up is done for this build, GPU and driver.
pub fn warm_up_done(props: &vk::PhysicalDeviceProperties) {
    if let Some(d) = dir() {
        write_atomic(&d.join("warm.stamp"), warm_stamp(props).as_bytes());
    }
}

/// Remove shaders not used for 60 days (earlier builds' versions): run after the
/// warm-up, which has just marked everything the app uses now.
pub fn prune() {
    let Some(d) = dir() else { return };
    let Ok(entries) = std::fs::read_dir(&d) else { return };
    let limit = std::time::Duration::from_secs(60 * 24 * 3600);
    for e in entries.flatten() {
        let p = e.path();
        let old = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).is_some_and(|age| age > limit);
        if old && p.extension().is_some_and(|x| x == "spv") {
            let _ = std::fs::remove_file(&p);
        }
    }
}
