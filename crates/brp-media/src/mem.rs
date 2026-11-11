//! System memory and the decoded-frame cache budget.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Share of total RAM the frame cache of a file may use by default.
pub const DEFAULT_CACHE_SHARE: f64 = 0.70;

/// Bytes; 0 = not set yet (the default is computed on first use).
static CACHE_BUDGET: AtomicUsize = AtomicUsize::new(0);

/// Total physical memory in bytes (not the free part), if the system says.
pub fn total_ram() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        }
        let mut s: MemoryStatusEx = unsafe { std::mem::zeroed() };
        s.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
            return Some(s.total_phys);
        }
        None
    }
    #[cfg(not(windows))]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

/// Default budget: 70% of total RAM (2 GB if unknown).
pub fn default_cache_budget() -> usize {
    match total_ram() {
        Some(t) => (t as f64 * DEFAULT_CACHE_SHARE) as usize,
        None => 2048 << 20,
    }
}

/// Most bytes of decoded frames each open file keeps.
pub fn cache_budget() -> usize {
    match CACHE_BUDGET.load(Ordering::Relaxed) {
        0 => {
            let b = default_cache_budget();
            CACHE_BUDGET.store(b, Ordering::Relaxed);
            b
        }
        b => b,
    }
}

/// Change the budget (takes effect on the next eviction).
pub fn set_cache_budget(bytes: usize) {
    CACHE_BUDGET.store(bytes.max(256 << 20), Ordering::Relaxed);
}

/// Share of total RAM processed effect pictures may use (per window).
pub const DEFAULT_FX_SHARE: f64 = 0.25;

/// Bytes; 0 = not set yet.
static FX_BUDGET: AtomicUsize = AtomicUsize::new(0);

/// Most bytes of processed effect pictures (AI stack results) kept in RAM, so they
/// are not computed again: 25% of total RAM (1 GB if unknown).
pub fn fx_cache_budget() -> usize {
    match FX_BUDGET.load(Ordering::Relaxed) {
        0 => {
            let b = match total_ram() {
                Some(t) => (t as f64 * DEFAULT_FX_SHARE) as usize,
                None => 1024 << 20,
            };
            FX_BUDGET.store(b, Ordering::Relaxed);
            b
        }
        b => b,
    }
}

/// Change the processed-picture budget (`BRP_FX_RAM_MB` sets it at start).
pub fn set_fx_cache_budget(bytes: usize) {
    FX_BUDGET.store(bytes.max(64 << 20), Ordering::Relaxed);
}

/// Physical memory free now, if the system says (caches stop growing when low).
pub fn avail_ram() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        }
        let mut s: MemoryStatusEx = unsafe { std::mem::zeroed() };
        s.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut s) } != 0 {
            return Some(s.avail_phys);
        }
        None
    }
    #[cfg(not(windows))]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}
