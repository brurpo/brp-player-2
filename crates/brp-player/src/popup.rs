//! Info, Settings and Effects as small borderless windows of their own, drawn as
//! glass (see `brp_render::popup`). They can be moved anywhere, also outside the
//! player window; on Windows they are owned by it (stay above it, minimise with it)
//! and get the system's rounded corners, shadow and acrylic backdrop.

use crate::player::Player;
use crate::ui::{self, Action, PopupKind, UiState};
use brp_render::{Backdrop, GlassParams, GpuContext, PopupRenderer};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

/// Glass look: dark blue-grey frost, bend at the border, corner radius (points).
const TINT: [f32; 4] = [0.045, 0.05, 0.07, 0.55];
const REFRACTION: f32 = 14.0;
const RADIUS: f32 = 8.0;
/// Opening animation.
const APPEAR: Duration = Duration::from_millis(180);

pub struct PopupWin {
    pub kind: PopupKind,
    // Renderer before window: its surface must go first.
    renderer: PopupRenderer,
    pub window: Arc<Window>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    opened: Instant,
    /// Shown on screen (after the first frame at the right size).
    visible: bool,
    frames: u32,
    pub repaint_at: Option<Instant>,
    /// Size asked of the window last (physical), to avoid asking again.
    asked: Option<PhysicalSize<u32>>,
    /// Moving by the title bar: (cursor, window position) when it started, in screen pixels.
    drag: Option<((i32, i32), PhysicalPosition<i32>)>,
}

/// What a popup frame asks of the player window.
#[derive(Default)]
pub struct Outcome {
    pub close: bool,
    /// The user did something here: the player window may need a redraw.
    pub input: bool,
    pub actions: Vec<Action>,
}

impl PopupWin {
    pub fn new(el: &ActiveEventLoop, kind: PopupKind, owner: &Window, gpu: Arc<GpuContext>, backdrop: Arc<Backdrop>) -> anyhow::Result<Self> {
        let scale = owner.scale_factor();
        let w = (kind.width() as f64 * scale).round() as u32;
        let h = (240.0 * scale).round() as u32;
        // Effects on the left, the others on the right, under the top bar.
        let origin = owner.inner_position().unwrap_or_default();
        let ow = owner.inner_size().width as i32;
        let top = origin.y + (52.0 * scale) as i32;
        // Settings a little lower and to the left of Info, so both stay visible.
        let (x, top) = match kind {
            PopupKind::Effects => (origin.x + (14.0 * scale) as i32, top),
            PopupKind::Info => (origin.x + ow - w as i32 - (14.0 * scale) as i32, top),
            PopupKind::Settings => (origin.x + ow - w as i32 - (40.0 * scale) as i32, top + (26.0 * scale) as i32),
            PopupKind::Export => (origin.x + ow - w as i32 - (80.0 * scale) as i32, top),
            PopupKind::Batch => (origin.x + (14.0 * scale) as i32, top + (40.0 * scale) as i32),
            PopupKind::Color => (origin.x + ow - w as i32 - (60.0 * scale) as i32, top + (12.0 * scale) as i32),
        };
        let mut attrs = Window::default_attributes()
            .with_title(format!("BRP Player — {:?}", kind))
            .with_decorations(false)
            .with_transparent(true)
            .with_theme(Some(winit::window::Theme::Dark))
            .with_resizable(false)
            .with_visible(false)
            .with_inner_size(PhysicalSize::new(w, h))
            .with_position(PhysicalPosition::new(x, top));
        #[cfg(windows)]
        {
            use winit::platform::windows::{BackdropType, Color, CornerPreference, WindowAttributesExtWindows};
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(h) = owner.window_handle() {
                if let RawWindowHandle::Win32(h) = h.as_raw() {
                    attrs = attrs.with_owner_window(h.hwnd.get());
                }
            }
            let backdrop = if std::env::var_os("BRP_ACRYLIC").is_some() { BackdropType::TransientWindow } else { BackdropType::None };
            attrs = attrs
                .with_skip_taskbar(true)
                // The system shadow extends the frame into the window, which turns its
                // translucent part flat grey: off unless asked for.
                .with_undecorated_shadow(std::env::var_os("BRP_POPUP_SHADOW").is_some())
                .with_system_backdrop(backdrop)
                .with_corner_preference(CornerPreference::Round)
                .with_border_color(Some(Color::from_rgb(58, 62, 78)));
        }
        let window = Arc::new(el.create_window(attrs)?);
        #[cfg(windows)]
        blur_behind(&window);
        let size = window.inner_size();
        let renderer = PopupRenderer::new(gpu, &*window, size.width, size.height, Some(backdrop))?;
        let egui_ctx = egui::Context::default();
        ui::setup_style(&egui_ctx, true);
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, Some(8192));
        window.request_redraw();
        Ok(PopupWin { kind, renderer, window, egui_ctx, egui_state, opened: Instant::now(), visible: false, frames: 0, repaint_at: None, asked: None, drag: None })
    }

    pub fn id(&self) -> WindowId {
        self.window.id()
    }

    /// On screen yet (hidden windows get no redraw requests: draw them directly).
    pub fn visible(&self) -> bool {
        self.visible
    }

    /// A window event for this popup. True if it should be redrawn.
    pub fn event(&mut self, event: &WindowEvent) -> bool {
        let resp = self.egui_state.on_window_event(&self.window, event);
        match event {
            WindowEvent::Resized(s) => {
                self.renderer.resize(s.width, s.height);
                true
            }
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => true,
            _ => resp.repaint,
        }
    }

    /// Lies over the player window (then it shows the player's picture and must follow it).
    pub fn overlaps(&self, owner: &Window) -> bool {
        let (Ok(a), Ok(b)) = (self.window.inner_position(), owner.inner_position()) else { return true };
        let (sa, sb) = (self.window.inner_size(), owner.inner_size());
        a.x < b.x + sb.width as i32 && b.x < a.x + sa.width as i32 && a.y < b.y + sb.height as i32 && b.y < a.y + sa.height as i32
    }

    pub fn redraw(&mut self, owner: &Window, p: &mut Player, s: &mut UiState) -> Outcome {
        let mut out = Outcome::default();
        let raw = self.egui_state.take_egui_input(&self.window);
        out.input = raw.events.iter().any(|e| !matches!(e, egui::Event::PointerMoved(_) | egui::Event::MouseMoved(_)));
        let appear = (self.opened.elapsed().as_secs_f32() / APPEAR.as_secs_f32()).min(1.0);
        let appear = 1.0 - (1.0 - appear).powi(3);
        let kind = self.kind;
        let mut resp = ui::PopupResponse::default();
        let full = self.egui_ctx.run_ui(raw, |root| {
            resp = ui::popup_ui(root, kind, p, s, &mut out.actions, appear);
        });
        self.egui_state.handle_platform_output(&self.window, full.platform_output);
        out.close = resp.close;
        // Moving: we move the window ourselves, one frame at a time, each frame drawn for
        // the position it will be shown at, so the glass never lags behind (the system
        // move loop shifts the window before our picture for the new place is ready).
        if resp.drag {
            match (mouse::cursor(), self.window.outer_position()) {
                (Some(c), Ok(w)) => self.drag = Some((c, w)),
                _ => {
                    let _ = self.window.drag_window();
                }
            }
        }
        let mut move_to: Option<PhysicalPosition<i32>> = None;
        if let Some((c0, w0)) = self.drag {
            match mouse::cursor().filter(|_| mouse::left_down()) {
                Some(c) => move_to = Some(PhysicalPosition::new(w0.x + c.0 - c0.0, w0.y + c.1 - c0.1)),
                None => self.drag = None,
            }
        }

        // Follow the content's size.
        let scale = self.window.scale_factor() as f32;
        let want = PhysicalSize::new((resp.size.x * scale).round().max(64.0) as u32, (resp.size.y * scale).round().max(48.0) as u32);
        let have = self.window.inner_size();
        let size_ok = have.width.abs_diff(want.width) <= 1 && have.height.abs_diff(want.height) <= 1;
        if !size_ok && self.asked != Some(want) {
            self.asked = Some(want);
            if let Some(got) = self.window.request_inner_size(want) {
                self.renderer.resize(got.width, got.height);
            }
        }

        let prims = self.egui_ctx.tessellate(full.shapes, full.pixels_per_point);
        if let Err(e) = self.renderer.update_textures(full.textures_delta) {
            log::error!("popup textures: {e}");
        }
        // Where the window will be when this frame shows.
        let shift = match (move_to, self.window.outer_position()) {
            (Some(to), Ok(now)) => (to.x - now.x, to.y - now.y),
            _ => (0, 0),
        };
        let offset = match (self.window.inner_position(), owner.inner_position()) {
            (Ok(a), Ok(b)) => [(a.x + shift.0 - b.x) as f32, (a.y + shift.1 - b.y) as f32],
            _ => [-1e6, -1e6],
        };
        // Move first (synchronously), then present the frame drawn for the new place right
        // away, so both reach the compositor for the same screen refresh.
        if shift != (0, 0) {
            if let Some(to) = move_to {
                if !mouse::move_window(&self.window, to) {
                    self.window.set_outer_position(to);
                }
            }
        }
        let test = std::env::var_os("BRP_GLASS_TEST").is_some();
        // BRP_GLASS_TEST: plain background, no tint, bend or content (alignment checks).
        let glass = if test {
            GlassParams { offset, radius: RADIUS, tint: [0.0; 4], refraction: 0.0 }
        } else {
            GlassParams { offset, radius: RADIUS, tint: TINT, refraction: REFRACTION }
        };
        let prims = if test { Vec::new() } else { prims };
        match self.renderer.render(&glass, &prims, full.pixels_per_point) {
            Ok(true) => {
                self.frames += 1;
                // Show once laid out at its size (or after a few tries).
                if !self.visible && (size_ok || self.frames >= 4) {
                    self.visible = true;
                    self.opened = Instant::now();
                    self.window.set_visible(true);
                }
            }
            Ok(false) => {}
            Err(e) => log::error!("popup render: {e:#}"),
        }
        if self.drag.is_some() {
            // Wait for that refresh: paces the drag to the display and keeps the next move
            // from landing before this frame is on screen.
            mouse::wait_compositor();
        }

        let delay = full.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(Duration::MAX);
        let animating = self.opened.elapsed() < APPEAR || !self.visible || !size_ok;
        self.repaint_at = if self.drag.is_some() {
            // Follow the pointer every refresh (paced by `wait_compositor`).
            Some(Instant::now())
        } else if animating || delay.is_zero() {
            // Next display refresh, not at once: the swapchain is MAILBOX (presents do
            // not wait), so "at once" redrew ~900 times a second while the AI spinner
            // turned, starving the player window (playback took seconds to start).
            let hz = self.window.current_monitor().and_then(|m| m.refresh_rate_millihertz()).map_or(60.0, |m| m as f64 / 1000.0).clamp(30.0, 240.0);
            Some(Instant::now() + Duration::from_secs_f64(1.0 / hz))
        } else {
            (delay < Duration::from_secs(3600)).then(|| Instant::now() + delay)
        };
        out
    }
}

/// The compositor blurs what is behind the window's translucent pixels, focused or
/// not (the system "acrylic" backdrop turns into a flat colour when the window is
/// inactive, which a popup usually is). `BRP_ACRYLIC=1` uses acrylic instead.
#[cfg(windows)]
fn blur_behind(window: &Window) {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    if std::env::var_os("BRP_ACRYLIC").is_some() {
        return;
    }
    #[repr(C)]
    struct AccentPolicy {
        state: u32,
        flags: u32,
        gradient: u32,
        animation: u32,
    }
    #[repr(C)]
    struct CompositionData {
        attribute: u32,
        data: *mut std::ffi::c_void,
        size: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleA(name: *const u8) -> isize;
        fn LoadLibraryA(name: *const u8) -> isize;
        fn GetProcAddress(module: isize, name: *const u8) -> *const std::ffi::c_void;
    }
    type SetAttr = unsafe extern "system" fn(isize, *mut CompositionData) -> i32;
    // Not in the import library (undocumented): look it up.
    let f = unsafe {
        let mut m = GetModuleHandleA(c"user32.dll".as_ptr() as *const u8);
        if m == 0 {
            m = LoadLibraryA(c"user32.dll".as_ptr() as *const u8);
        }
        let p = if m == 0 { std::ptr::null() } else { GetProcAddress(m, c"SetWindowCompositionAttribute".as_ptr() as *const u8) };
        if p.is_null() {
            return;
        }
        std::mem::transmute::<*const std::ffi::c_void, SetAttr>(p)
    };
    const WCA_ACCENT_POLICY: u32 = 19;
    const ACCENT_ENABLE_BLURBEHIND: u32 = 3;
    let Ok(h) = window.window_handle() else { return };
    let RawWindowHandle::Win32(h) = h.as_raw() else { return };
    let mut policy = AccentPolicy { state: ACCENT_ENABLE_BLURBEHIND, flags: 0, gradient: 0, animation: 0 };
    let mut data = CompositionData { attribute: WCA_ACCENT_POLICY, data: &mut policy as *mut _ as *mut _, size: std::mem::size_of::<AccentPolicy>() };
    if unsafe { f(h.hwnd.get(), &mut data) } == 0 {
        log::debug!("popup: blur behind not available");
    }
}

/// Pointer position on screen and the primary button, read straight from the system
/// (the window moves under the pointer while dragging, so window-relative positions
/// would chase themselves).
mod mouse {
    #[cfg(windows)]
    mod sys {
        #[repr(C)]
        pub struct Point {
            pub x: i32,
            pub y: i32,
        }
        #[link(name = "user32")]
        unsafe extern "system" {
            pub fn GetCursorPos(p: *mut Point) -> i32;
            pub fn GetAsyncKeyState(key: i32) -> i16;
            pub fn GetSystemMetrics(index: i32) -> i32;
            pub fn SetWindowPos(hwnd: isize, after: isize, x: i32, y: i32, cx: i32, cy: i32, flags: u32) -> i32;
        }
        #[link(name = "dwmapi")]
        unsafe extern "system" {
            pub fn DwmFlush() -> i32;
        }
    }

    #[cfg(windows)]
    pub fn cursor() -> Option<(i32, i32)> {
        let mut p = sys::Point { x: 0, y: 0 };
        (unsafe { sys::GetCursorPos(&mut p) } != 0).then_some((p.x, p.y))
    }

    #[cfg(windows)]
    pub fn left_down() -> bool {
        // The primary button is the right one when the user swapped them.
        const SM_SWAPBUTTON: i32 = 23;
        let key = if unsafe { sys::GetSystemMetrics(SM_SWAPBUTTON) } != 0 { 0x02 } else { 0x01 };
        unsafe { sys::GetAsyncKeyState(key) as u16 & 0x8000 != 0 }
    }

    /// Move a window now (winit posts the move to happen later). False if not possible.
    #[cfg(windows)]
    pub fn move_window(window: &winit::window::Window, to: winit::dpi::PhysicalPosition<i32>) -> bool {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        const SWP_NOSIZE: u32 = 0x1;
        const SWP_NOZORDER: u32 = 0x4;
        const SWP_NOACTIVATE: u32 = 0x10;
        let Ok(h) = window.window_handle() else { return false };
        let RawWindowHandle::Win32(h) = h.as_raw() else { return false };
        unsafe { sys::SetWindowPos(h.hwnd.get(), 0, to.x, to.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE) != 0 }
    }

    /// Block until the compositor has shown its next frame.
    #[cfg(windows)]
    pub fn wait_compositor() {
        unsafe { sys::DwmFlush() };
    }

    #[cfg(not(windows))]
    pub fn move_window(_: &winit::window::Window, _: winit::dpi::PhysicalPosition<i32>) -> bool {
        false
    }

    #[cfg(not(windows))]
    pub fn wait_compositor() {}

    #[cfg(not(windows))]
    pub fn cursor() -> Option<(i32, i32)> {
        None
    }

    #[cfg(not(windows))]
    pub fn left_down() -> bool {
        false
    }
}
