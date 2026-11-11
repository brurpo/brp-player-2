//! HDR screens: whether the window's monitor shows HDR, its peak brightness, and the
//! brightness the system gives SDR white there (DESIGN.md §7b).
//!
//! `BRP_HDR=0` never uses HDR output; `BRP_HDR=1` uses it whenever the surface offers
//! HDR10, even if the system says the screen is not in HDR (for tests).

/// An HDR screen as the system reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrScreen {
    /// Brightness of SDR white (the UI, SDR video), nits: Windows' "SDR content
    /// brightness".
    pub sdr_nits: f32,
    /// Peak brightness the screen reports, nits.
    pub peak_nits: f32,
}

/// Used when the system does not say.
const DEFAULT_SDR_NITS: f32 = 203.0;
const DEFAULT_PEAK_NITS: f32 = 1000.0;

/// The HDR screen `hwnd` (a Win32 window handle) is on, if HDR is on there.
pub fn screen_of(hwnd: isize) -> Option<HdrScreen> {
    match std::env::var("BRP_HDR").ok().as_deref() {
        Some("0") => return None,
        Some("1") => return Some(query(hwnd).unwrap_or(HdrScreen { sdr_nits: DEFAULT_SDR_NITS, peak_nits: DEFAULT_PEAK_NITS })),
        _ => {}
    }
    query(hwnd)
}

#[cfg(windows)]
fn query(hwnd: isize) -> Option<HdrScreen> {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, IDXGIOutput6};
    use windows::Win32::Graphics::Gdi::{MonitorFromWindow, MONITOR_DEFAULTTONEAREST};
    use windows::core::Interface;
    unsafe {
        let monitor = MonitorFromWindow(HWND(hwnd as *mut _), MONITOR_DEFAULTTONEAREST);
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        for a in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(a) else { break };
            for o in 0.. {
                let Ok(output) = adapter.EnumOutputs(o) else { break };
                let Ok(o6) = output.cast::<IDXGIOutput6>() else { continue };
                let Ok(d) = o6.GetDesc1() else { continue };
                if d.Monitor != monitor {
                    continue;
                }
                if d.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 {
                    return None;
                }
                let peak = if d.MaxLuminance > 80.0 { d.MaxLuminance } else { DEFAULT_PEAK_NITS };
                let sdr = sdr_white(&d.DeviceName).unwrap_or(DEFAULT_SDR_NITS);
                return Some(HdrScreen { sdr_nits: sdr, peak_nits: peak.max(sdr) });
            }
        }
        None
    }
}

/// Windows' SDR white level for the display `gdi_name` (e.g. `\\.\DISPLAY1`), nits.
#[cfg(windows)]
fn sdr_white(gdi_name: &[u16; 32]) -> Option<f32> {
    use windows::Win32::Devices::Display::*;
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        if GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm).0 != 0 {
            return None;
        }
        let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); np as usize];
        let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); nm as usize];
        if QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut np, paths.as_mut_ptr(), &mut nm, modes.as_mut_ptr(), None).0 != 0 {
            return None;
        }
        for p in &paths[..np as usize] {
            let mut src = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
            src.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            src.header.size = std::mem::size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
            src.header.adapterId = p.sourceInfo.adapterId;
            src.header.id = p.sourceInfo.id;
            if DisplayConfigGetDeviceInfo(&mut src.header) != 0 || src.viewGdiDeviceName != *gdi_name {
                continue;
            }
            let mut white = DISPLAYCONFIG_SDR_WHITE_LEVEL::default();
            white.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
            white.header.size = std::mem::size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32;
            white.header.adapterId = p.targetInfo.adapterId;
            white.header.id = p.targetInfo.id;
            if DisplayConfigGetDeviceInfo(&mut white.header) != 0 {
                return None;
            }
            // In thousandths of 80 nits.
            return Some(white.SDRWhiteLevel as f32 * 80.0 / 1000.0);
        }
        None
    }
}

#[cfg(not(windows))]
fn query(_hwnd: isize) -> Option<HdrScreen> {
    // Other systems: not detected yet (BRP_HDR=1 forces it where the surface allows).
    None
}

/// Ask Windows to give this process's GPU work priority over other programs' (a
/// training run or a render in the background), so the window stays smooth while
/// the GPU is shared. BRP_GPU_PRIORITY=0 leaves it normal.
pub fn raise_gpu_priority() {
    #[cfg(windows)]
    {
        if std::env::var("BRP_GPU_PRIORITY").is_ok_and(|v| v == "0") {
            return;
        }
        #[link(name = "gdi32")]
        unsafe extern "system" {
            fn D3DKMTSetProcessSchedulingPriorityClass(process: isize, class: i32) -> i32;
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        // D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH (REALTIME needs administrator rights).
        let status = unsafe { D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), 4) };
        if status == 0 {
            log::info!("GPU scheduling priority: high");
        } else {
            log::info!("GPU scheduling priority: not raised ({status:#x})");
        }
    }
}
