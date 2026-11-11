#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ai;
mod color;
mod color_ui;
mod effects;
mod export;
mod export_ui;
mod i18n;
mod icons;
mod instance;
mod player;
mod popup;
mod settings;
mod ui;
mod update;
mod warm;

use brp_render::{Renderer, VideoDraw};
use player::Player;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use ui::{Action, UiState};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Fullscreen, Window, WindowId, WindowLevel};

const MEDIA_EXTS: &[&str] = &["mp4", "mkv", "mov", "avi", "mxf", "webm", "gif", "m4v", "ts", "mts", "m2ts", "mpg", "mpeg", "wmv", "flv"];

/// Video and still images (any numbered image opens its whole sequence).
fn media_exts() -> Vec<&'static str> {
    MEDIA_EXTS.iter().chain(brp_media::sequence::IMAGE_EXTS).copied().collect()
}

/// Events sent to the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    /// Another launch asked to open a file (or just to come to the front).
    Instance(instance::Msg),
}

struct Win {
    /// Closed from the custom title bar: the app drops it after this event.
    close: bool,
    /// Was maximised before fullscreen (restored after).
    was_maximized: bool,
    window: Arc<Window>,
    // Popups before the renderer: they use its device and backdrop.
    popups: Vec<popup::PopupWin>,
    /// None only while switching GPUs.
    renderer: Option<Renderer>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    player: Player,
    ui: UiState,
    needs_tick: bool,
    repaint_at: Option<Instant>,
    presented_once: bool,
    /// Debug stats: (window start, presents, distinct pictures shown, last picture).
    stats: (Instant, u32, u32, (i64, u8)),
}

struct App {
    start: Instant,
    /// Vulkan instance being created in the background since process start.
    gpu_init: Option<std::thread::JoinHandle<anyhow::Result<brp_render::GpuInstance>>>,
    initial_file: Option<String>,
    /// `--frame=N`: open paused on this frame.
    start_frame: Option<i64>,
    wins: HashMap<WindowId, Win>,
    /// Most recently focused first.
    focus_order: Vec<WindowId>,
    /// Popup window -> its player window.
    popup_owner: HashMap<WindowId, WindowId>,
    /// Saved choices; new windows start from them.
    settings: settings::Settings,
    /// Settings changed and not written yet (written a second later, or on exit).
    settings_pending: Option<Instant>,
}

impl App {
    fn create_window(&mut self, el: &ActiveEventLoop, file: Option<String>) -> anyhow::Result<WindowId> {
        let mut attrs = Window::default_attributes()
            .with_title("BRP Player")
            .with_inner_size(LogicalSize::new(1280.0, 760.0))
            .with_min_inner_size(LogicalSize::new(480.0, 300.0))
            .with_window_icon(icons::window_icon())
            // Our own title bar (see `ui`): no system frame.
            .with_decorations(false);
        #[cfg(windows)]
        {
            use winit::platform::windows::{CornerPreference, WindowAttributesExtWindows};
            attrs = attrs.with_undecorated_shadow(true).with_corner_preference(CornerPreference::Round).with_taskbar_icon(icons::taskbar_icon());
        }
        // The first window comes back where it was (if that place is still on a screen).
        let mut restore_max = false;
        if self.wins.is_empty() {
            if let Some(pl) = self.settings.window {
                let on_screen = el.available_monitors().any(|m| {
                    let (p, s) = (m.position(), m.size());
                    let (cx, cy) = (pl.x + 60, pl.y + 20);
                    cx >= p.x && cy >= p.y && cx < p.x + s.width as i32 && cy < p.y + s.height as i32
                });
                if on_screen {
                    attrs = attrs
                        .with_position(winit::dpi::PhysicalPosition::new(pl.x, pl.y))
                        .with_inner_size(winit::dpi::PhysicalSize::new(pl.w, pl.h));
                    restore_max = pl.maximized;
                }
            }
        }
        // Cascade extra windows so they don't sit exactly on top of each other.
        if let Some(prev) = self.focus_order.first().and_then(|id| self.wins.get(id)) {
            if let Ok(pos) = prev.window.outer_position() {
                attrs = attrs.with_position(winit::dpi::PhysicalPosition::new(pos.x + 40, pos.y + 40));
            }
        }
        let window = Arc::new(el.create_window(attrs)?);
        // BRP_TEST_MAXIMIZED: open maximized (as most people play).
        if restore_max || std::env::var_os("BRP_TEST_MAXIMIZED").is_some() {
            window.set_maximized(true);
        }
        let t_window = self.start.elapsed();
        // Open the media first: its first frame decodes while the GPU driver sets up.
        let mut player = Player::new();
        player.audio_device = self.settings.audio_device.clone();
        player.volume = self.settings.volume;
        player.muted = self.settings.muted;
        player.loop_on = self.settings.loop_on;
        player.deint_mode = self.settings.deint_mode;
        player.field_rate = self.settings.field_rate;
        // --fx=A.glsl,ai:upscale:Model=1,B.glsl:Strength=0.5 : start with this stack (shader
        // files from the library, AI effects as ai:<name>; parameters after colons).
        if let Some(list) = std::env::args().find_map(|a| a.strip_prefix("--fx=").map(str::to_string)) {
            player.fx.add_from_arg(&list);
        }
        if let Some(f) = file {
            player.open(&f);
            if let Some(m) = std::env::args().find_map(|a| a.strip_prefix("--deint=").map(str::to_string)) {
                player.deint_mode = match m.as_str() {
                    "off" => player::DeintMode::Off,
                    "best" => player::DeintMode::Best,
                    "nnedi" => player::DeintMode::Nnedi,
                    "bwdif" => player::DeintMode::Bwdif,
                    _ => player::DeintMode::Auto,
                };
            }
            if let Some(n) = self.start_frame.take() {
                player.pause();
                player.seek(n);
            }
            player.tick();
        }
        let size = window.inner_size();
        let gpu = std::env::var("BRP_GPU").ok().or_else(|| self.settings.gpu.clone());
        let inst = match self.gpu_init.take() {
            Some(h) => h.join().map_err(|_| anyhow::anyhow!("GPU init thread panicked"))??,
            None => brp_render::GpuInstance::new()?,
        };
        let renderer = Renderer::new(inst, &*window, size.width, size.height, gpu.as_deref())?;
        log::info!("startup: window {:?}, renderer {:?} ({})", t_window, self.start.elapsed(), renderer.gpu_name());

        let egui_ctx = egui::Context::default();
        ui::setup_style(&egui_ctx, false);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(8192),
        );
        let mut ui = UiState::new(renderer.gpu_name().to_string(), &egui_ctx, self.settings.clone());
        // Decoders on the GPU (DXV, ProRes): registered before any file opens.
        brp_media::gpudec::set_factory(Some(brp_render::gpudec::factory(&renderer.gpu())));
        player.ai.ctx = Some(renderer.gpu());
        warm::start(&renderer.gpu());
        if self.wins.is_empty() && std::env::args().any(|a| a == "--lock") {
            ui.locked = true;
        }
        // --fullscreen : start in fullscreen.
        let start_fullscreen = self.wins.is_empty() && std::env::args().any(|a| a == "--fullscreen");
        // BRP_TEST_FX_MENU: the "Add effect" list open (checking the layout).
        ui.fx_add_open = std::env::var_os("BRP_TEST_FX_MENU").is_some();
        // BRP_TEST_OCIO=1: OpenColorIO on for this run (the setting is not saved).
        if std::env::var_os("BRP_TEST_OCIO").is_some() {
            ui.ocio.enabled = true;
        }
        // BRP_TEST_OCIO_CONFIG=<file>: that config of one's own for this run (not saved).
        if let Some(c) = std::env::var_os("BRP_TEST_OCIO_CONFIG") {
            ui.settings.ocio_custom = Some(c.to_string_lossy().into_owned());
            ui.ocio.set_custom(Some(c.into()));
        }
        // BRP_TEST_BATCH=a;b : these files in the batch queue (checking the layout);
        // BRP_TEST_BATCH_START=1 starts it.
        if let Some(list) = std::env::var_os("BRP_TEST_BATCH") {
            for f in list.to_string_lossy().split(';').filter(|f| !f.is_empty()) {
                ui.export.batch.add(f.to_string());
            }
            if let Some(d) = std::env::var_os("BRP_TEST_BATCH_DIR") {
                ui.export.batch.out_dir = Some(d.into());
            }
            if let Some(n) = std::env::var("BRP_TEST_BATCH_SELECT").ok().and_then(|n| n.parse().ok()) {
                ui.export.select_first(n);
            }
            if std::env::var_os("BRP_TEST_BATCH_START").is_some() {
                let ocio = ui.ocio_job();
                ui.export.batch.start(player.fx.all_stages(), player.deint_mode, player.field_rate, ocio);
            }
        }
        // --show=info,settings,fx : open these popups at start.
        if let Some(list) = std::env::args().find_map(|a| a.strip_prefix("--show=").map(str::to_string)) {
            for k in list.split(',') {
                match k.trim() {
                    "info" => ui.set_shown(ui::PopupKind::Info, true),
                    "settings" => ui.set_shown(ui::PopupKind::Settings, true),
                    "fx" | "effects" => ui.set_shown(ui::PopupKind::Effects, true),
                    "export" => ui.set_shown(ui::PopupKind::Export, true),
                    "batch" => ui.set_shown(ui::PopupKind::Batch, true),
                    "color" | "ocio" => ui.set_shown(ui::PopupKind::Color, true),
                    _ => {}
                }
            }
        }
        window.request_redraw();
        let id = window.id();
        self.wins.insert(
            id,
            Win { close: false, was_maximized: false, window, popups: Vec::new(), renderer: Some(renderer), egui_ctx, egui_state, player, ui, needs_tick: true, repaint_at: None, presented_once: false, stats: (Instant::now(), 0, 0, (-1, 0)) },
        );
        if start_fullscreen {
            if let Some(w) = self.wins.get_mut(&id) {
                w.set_fullscreen(true);
            }
        }
        self.focus_order.insert(0, id);
        Ok(id)
    }

    /// Route a file from another launch: the most recently used unlocked window
    /// gets it; if every window is locked, open a new one.
    fn route(&mut self, el: &ActiveEventLoop, msg: instance::Msg) {
        match msg {
            instance::Msg::Open(path) => {
                let target = self.focus_order.iter().copied().find(|id| self.wins.get(id).is_some_and(|w| !w.ui.locked));
                match target {
                    Some(id) => {
                        let w = self.wins.get_mut(&id).unwrap();
                        w.player.open(&path);
                        bring_to_front(&w.window);
                        w.window.request_redraw();
                    }
                    None => {
                        if let Err(e) = self.create_window(el, Some(path)) {
                            log::error!("new window: {e:#}");
                        }
                    }
                }
            }
            instance::Msg::Focus => {
                if let Some(w) = self.focus_order.first().and_then(|id| self.wins.get(id)) {
                    bring_to_front(&w.window);
                }
            }
        }
    }
}

/// Show a file (or the folder of an image sequence) in the file browser.
fn reveal(path: &str) {
    let p = std::path::Path::new(path);
    let target = if path.contains('%') { p.parent().map(|d| d.to_path_buf()).unwrap_or_default() } else { p.to_path_buf() };
    #[cfg(windows)]
    let r = if target.is_file() { std::process::Command::new("explorer").arg(format!("/select,{}", target.display())).spawn() } else { std::process::Command::new("explorer").arg(&target).spawn() };
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg("-R").arg(&target).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(target.parent().unwrap_or(&target)).spawn();
    if let Err(e) = r {
        log::warn!("show {path}: {e}");
    }
}

/// Focus a window for a file from another launch. Its place, size, maximised and
/// fullscreen state stay as they are; only a minimised window is restored.
fn bring_to_front(w: &Window) {
    if w.is_minimized() == Some(true) {
        w.set_minimized(false);
    }
    w.focus_window();
}

impl Win {
    /// Fullscreen covering the whole monitor, taskbar included: a borderless window
    /// the size of the monitor, kept above other windows (a maximised window going
    /// fullscreen, or one Windows does not treat as the fullscreen app, leaves the
    /// taskbar on top).
    fn set_fullscreen(&mut self, on: bool) {
        if on {
            self.was_maximized = self.window.is_maximized();
            if self.was_maximized {
                self.window.set_maximized(false);
            }
            let monitor = self.window.current_monitor();
            self.window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
            self.window.set_window_level(WindowLevel::AlwaysOnTop);
            self.window.focus_window();
        } else {
            self.window.set_fullscreen(None);
            self.window.set_window_level(WindowLevel::Normal);
            if self.was_maximized {
                self.window.set_maximized(true);
            }
        }
        self.ui.fullscreen = on;
    }

    /// A file dropped on the player window or one of its popups.
    fn on_drop(&mut self, path: &std::path::Path) {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("glsl")) {
            // A shader: add it to the chain (copied into the user shader folder).
            match self.player.fx.import(path) {
                Ok(_) => self.ui.set_shown(ui::PopupKind::Effects, true),
                Err(e) => log::error!("import shader: {e:#}"),
            }
        } else {
            self.player.open(&path.to_string_lossy());
            self.window.focus_window();
        }
        self.window.request_redraw();
    }

    fn open_dialog(&mut self) {
        let was_playing = self.player.playing;
        self.player.pause();
        // Owned by the player window: above it even in fullscreen (always on top),
        // and the window itself is left as it is.
        let pick = rfd::FileDialog::new()
            .set_parent(&*self.window)
            .set_title(self.ui.lang.t("open_title"))
            .add_filter(self.ui.lang.t("files_media"), &media_exts())
            .add_filter("Video", MEDIA_EXTS)
            .add_filter(self.ui.lang.t("files_images"), brp_media::sequence::IMAGE_EXTS)
            .add_filter("*", &["*"])
            .pick_file();
        match pick {
            Some(path) => self.player.open(&path.to_string_lossy()),
            None if was_playing => self.player.play(),
            None => {}
        }
    }

    /// Open popup windows for the popups shown, close the others. If a window cannot
    /// be made, popups are drawn inside the player window instead.
    fn sync_popups(&mut self, el: &ActiveEventLoop) {
        self.popups.retain(|p| self.ui.shown(p.kind));
        if !self.ui.detached {
            return;
        }
        let Some(r) = &self.renderer else { return };
        for k in ui::PopupKind::ALL {
            if self.ui.shown(k) && !self.popups.iter().any(|p| p.kind == k) {
                match popup::PopupWin::new(el, k, &self.window, r.gpu(), r.backdrop()) {
                    Ok(p) => self.popups.push(p),
                    Err(e) => {
                        log::warn!("popup window: {e:#} (drawing popups inside the player window)");
                        self.popups.clear();
                        self.ui.detached = false;
                        self.window.request_redraw();
                        return;
                    }
                }
                // The backdrop starts with the player's next frame.
                self.window.request_redraw();
            }
        }
    }

    /// Draw one popup. True if the player window should redraw too.
    fn redraw_popup(&mut self, id: WindowId, el: &ActiveEventLoop) -> bool {
        let Some(i) = self.popups.iter().position(|p| p.id() == id) else { return false };
        let t = Instant::now();
        let out = self.popups[i].redraw(&self.window, &mut self.player, &mut self.ui);
        if t.elapsed() > Duration::from_millis(50) {
            log::debug!("slow redraw: popup {:?} {:?}", self.popups[i].kind, t.elapsed());
        }
        let main = out.input || !out.actions.is_empty() || out.close || self.player.fx.dirty;
        if out.close {
            let k = self.popups[i].kind;
            self.ui.set_shown(k, false);
            self.popups.remove(i);
            // Give the keyboard back to the player.
            self.window.focus_window();
        }
        self.apply_actions(out.actions, el);
        self.sync_popups(el);
        main
    }

    /// Run video, effects and AI on another GPU: new renderer (and popups), new egui
    /// textures, AI on the new device. Falls back to the previous GPU on failure.
    fn switch_gpu(&mut self, name: Option<String>) {
        let previous = self.renderer.as_ref().map(|r| r.gpu_name().to_string());
        let t = Instant::now();
        // Everything on the old device goes first (the window's swapchain with it).
        self.popups.clear();
        self.player.ai.clear_gpu();
        self.player.ai.ctx = None;
        self.renderer = None;
        let size = self.window.inner_size();
        let make = |pick: Option<&str>| brp_render::GpuInstance::new().and_then(|i| Renderer::new(i, &*self.window, size.width, size.height, pick));
        let renderer = match make(name.as_deref()) {
            Ok(r) => {
                self.ui.settings.gpu = name;
                self.ui.settings_dirty = true;
                r
            }
            Err(e) => {
                log::error!("switching GPU: {e:#}");
                self.ui.toast(self.ui.lang.t("gpu_failed"));
                match make(previous.as_deref()) {
                    Ok(r) => r,
                    Err(e) => {
                        log::error!("GPU: {e:#}");
                        return;
                    }
                }
            }
        };
        log::info!("GPU: now {} ({:?})", renderer.gpu_name(), t.elapsed());
        // egui's textures lived on the old device: start a fresh context.
        self.egui_ctx = egui::Context::default();
        ui::setup_style(&self.egui_ctx, false);
        self.egui_state = egui_winit::State::new(self.egui_ctx.clone(), egui::ViewportId::ROOT, &self.window, Some(self.window.scale_factor() as f32), None, Some(8192));
        self.ui.icons = icons::Icons::load(&self.egui_ctx);
        self.ui.gpu_name = renderer.gpu_name().to_string();
        self.ui.gpus.clear();
        self.ui.toast(format!("GPU · {}", renderer.gpu_name()));
        self.player.ai.ctx = Some(renderer.gpu());
        brp_media::gpudec::set_factory(Some(brp_render::gpudec::factory(&renderer.gpu())));
        self.player.fx.dirty = true;
        self.renderer = Some(renderer);
        self.window.request_redraw();
    }

    fn apply_actions(&mut self, actions: Vec<Action>, el: &ActiveEventLoop) {
        for a in actions {
            match a {
                Action::Open => self.open_dialog(),
                Action::OcioLoadConfig => {
                    let pick = rfd::FileDialog::new().set_parent(&*self.window).set_title("OpenColorIO config").add_filter("OCIO config", &["ocio"]).pick_file();
                    if let Some(p) = pick {
                        self.ui.settings.ocio_custom = Some(p.to_string_lossy().into_owned());
                        self.ui.settings_dirty = true;
                        self.ui.ocio.set_custom(Some(p));
                    }
                }
                Action::SetLayer(name) => {
                    self.player.set_layer(&name);
                    self.window.request_redraw();
                }
                Action::ToggleFullscreen => {
                    let fs = self.window.fullscreen().is_none();
                    self.set_fullscreen(fs);
                }
                Action::ExitFullscreen => self.set_fullscreen(false),
                Action::FxBrowse => {
                    if let Some(p) = rfd::FileDialog::new().add_filter("GLSL shader", &["glsl"]).pick_file() {
                        if let Err(e) = self.player.fx.import(&p) {
                            log::error!("import shader: {e:#}");
                        }
                    }
                }
                Action::FxSavePreset => {
                    let dir = effects::user_shader_dir().join("presets");
                    let _ = std::fs::create_dir_all(&dir);
                    if let Some(p) = rfd::FileDialog::new().set_directory(&dir).add_filter("BRP effects preset", &["json"]).set_file_name("preset.json").save_file() {
                        if let Err(e) = self.player.fx.save_preset(&p) {
                            log::error!("save preset: {e:#}");
                        }
                    }
                }
                Action::FxLoadPreset => {
                    let dir = effects::user_shader_dir().join("presets");
                    if let Some(p) = rfd::FileDialog::new().set_directory(&dir).add_filter("BRP effects preset", &["json"]).pick_file() {
                        match self.player.fx.load_preset(&p) {
                            Ok(missing) if !missing.is_empty() => log::warn!("{}: {}", self.ui.lang.t("fx_preset_missing"), missing.join(", ")),
                            Ok(_) => {}
                            Err(e) => log::error!("load preset: {e:#}"),
                        }
                    }
                }
                Action::SetGpu(name) => self.switch_gpu(name),
                Action::DragWindow => {
                    let _ = self.window.drag_window();
                }
                Action::Resize(dir) => {
                    let _ = self.window.drag_resize_window(dir);
                }
                Action::Minimize => self.window.set_minimized(true),
                Action::ToggleMaximize => self.window.set_maximized(!self.window.is_maximized()),
                Action::Close => self.close = true,
                Action::RestartForUpdate => {
                    update::restart();
                    self.close = true;
                }
                Action::ExportBrowse => {
                    let e = &mut self.ui.export;
                    let c = export::container(&e.settings.container);
                    let cur = std::path::PathBuf::from(&e.output);
                    let mut d = rfd::FileDialog::new().set_title(self.ui.lang.t("export")).add_filter(c.label, &[c.ext]);
                    if let Some(dir) = cur.parent().filter(|d| d.is_dir()) {
                        d = d.set_directory(dir);
                    }
                    if let Some(name) = cur.file_name() {
                        d = d.set_file_name(name.to_string_lossy());
                    }
                    if let Some(p) = d.save_file() {
                        e.output = p.to_string_lossy().into_owned();
                    }
                }
                Action::BatchAddFiles => {
                    if let Some(files) = rfd::FileDialog::new().set_title(self.ui.lang.t("batch_add")).add_filter(self.ui.lang.t("files_media"), &media_exts()).add_filter("Video", MEDIA_EXTS).add_filter(self.ui.lang.t("files_images"), brp_media::sequence::IMAGE_EXTS).add_filter("*", &["*"]).pick_files() {
                        for f in files {
                            self.ui.export.batch.add(f.to_string_lossy().into_owned());
                        }
                    }
                }
                Action::BatchFolder => {
                    if let Some(dir) = rfd::FileDialog::new().set_title(self.ui.lang.t("batch_folder")).pick_folder() {
                        self.ui.settings.batch_dir = Some(dir.to_string_lossy().into_owned());
                        self.ui.settings_dirty = true;
                        self.ui.export.batch.out_dir = Some(dir);
                    }
                }
                Action::Reveal(path) => reveal(&path),
            }
        }
        let _ = el;
    }

    fn redraw(&mut self, start: Instant, el: &ActiveEventLoop) {
        // BRP_TEST_DROP_AFTER=secs:path : that long after start, open `path` as if it
        // was dropped on the window (logs the window's place before and after).
        if let Some((at, path)) = std::env::var("BRP_TEST_DROP_AFTER").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.to_string())))) {
            if start.elapsed().as_secs_f64() >= at && !self.ui.test_dropped {
                self.ui.test_dropped = true;
                let place = |w: &Window| (w.outer_position().ok(), w.inner_size(), w.is_maximized(), w.fullscreen().is_some());
                log::info!("test drop: window before {:?}", place(&self.window));
                self.on_drop(std::path::Path::new(&path));
                log::info!("test drop: window after {:?}", place(&self.window));
            }
        }
        // BRP_TEST_FX_BYPASS=off_secs:on_secs : all effects off, then back on (as the
        // Effects panel switch), that long after start.
        if let Some((off, on)) = std::env::var("BRP_TEST_FX_BYPASS").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))) {
            let t = start.elapsed().as_secs_f64();
            let want = t >= off && t < on;
            if want != self.player.fx.bypass && (t < on || self.player.fx.bypass) {
                self.player.fx.set_bypass(want);
                log::info!("test: all effects {}", if want { "off" } else { "on" });
            }
        }
        // BRP_TEST_OCIO_TOGGLE=off_secs:on_secs : OpenColorIO off, then on again (as the
        // Color panel's switch, not saved), that long after start.
        if let Some((off, on)) = std::env::var("BRP_TEST_OCIO_TOGGLE").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))) {
            let t = start.elapsed().as_secs_f64();
            let want_on = !(t >= off && t < on);
            if want_on != self.ui.ocio.enabled && (t < on || !self.ui.ocio.enabled) && t >= off {
                self.ui.ocio.enabled = want_on;
                log::info!("test: OCIO {}", if want_on { "on" } else { "off" });
            }
        }
        // BRP_TEST_EXR_LAYER=secs:name : show that EXR layer that long after start (as
        // picking it from the layer chip).
        if let Some((at, name)) = std::env::var("BRP_TEST_EXR_LAYER").ok().and_then(|v| v.split_once(':').and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.to_string())))) {
            let shown = self.player.source.as_ref().and_then(|s| s.info.layers.get(s.info.layer).map(|(n, _)| n.clone()));
            if start.elapsed().as_secs_f64() >= at && !self.ui.test_layer_done && shown.is_some() {
                self.ui.test_layer_done = true;
                log::info!("test: EXR layer {name:?}");
                self.player.set_layer(&name);
            }
        }
        self.ui.maximized = self.window.is_maximized();
        let size = self.window.inner_size();
        let px = [size.width as f32, size.height as f32];
        self.player.tick();

        let raw = self.egui_state.take_egui_input(&self.window);
        let mut actions = Vec::new();
        let out = self.egui_ctx.run_ui(raw, |root| {
            actions = ui::draw(root, &mut self.player, &mut self.ui);
        });
        self.egui_state.handle_platform_output(&self.window, out.platform_output);

        // Scrubbing shows raw frames; the frame let go on is processed.
        self.player.scrubbing = self.ui.scrubbing();
        let ctx_before = self.egui_ctx.clone();
        self.apply_actions(actions, el);
        self.sync_popups(el);
        if !ctx_before.eq(&self.egui_ctx) {
            // GPU switched: this frame belongs to the old egui context. Draw anew.
            let mut d = out.textures_delta;
            d.clear();
            self.window.request_redraw();
            return;
        }
        if self.ui.gpus.is_empty() {
            if let Some(r) = &self.renderer {
                self.ui.gpus = r.list_gpus().into_iter().map(|g| (g.name, g.discrete)).collect();
            }
        }
        if self.renderer.is_none() {
            // egui insists every texture delta is handled.
            let mut d = out.textures_delta;
            d.clear();
            return;
        }
        let renderer = self.renderer.as_mut().unwrap();
        renderer.set_backdrop(!self.popups.is_empty());

        // OpenColorIO as the display transform (DESIGN.md §7d): off, or the conversion for
        // this file and screen once built.
        self.ui.screen_hdr = renderer.hdr_screen().is_some();
        if let Some(e) = renderer.ocio_error() {
            self.ui.ocio.error = Some(e.to_string());
        }
        if !self.ui.ocio.enabled {
            renderer.set_ocio(None);
            // Off: what the window had is gone (on again, it is sent again).
            self.ui.ocio.window_off();
        } else {
            let screen = if self.ui.screen_hdr { brp_ocio::auto::Screen::Hdr } else { brp_ocio::auto::Screen::Sdr };
            if let Some(setup) = self.ui.ocio.update(&self.ui.settings, self.player.source.as_ref().map(|s| &s.info), screen) {
                renderer.set_ocio(Some(setup));
            }
            if self.ui.ocio.busy() {
                self.window.request_redraw();
            }
        }

        // Effect chain changes and background shader compiles.
        if self.player.fx.dirty {
            // Up to the last AI effect: background stages; after it: live shaders.
            let (stages, live) = self.player.fx.split();
            renderer.set_effects(&live, self.player.transfer());
            let transfer = self.player.transfer();
            if self.player.ai.set_stages(stages, transfer) {
                self.player.sub = 0;
            }
            self.player.fx.dirty = false;
        }
        // Deinterlacing comes before the AI stack (the setting, or the detection,
        // may change at any time).
        if self.player.ai.set_source_deint(self.player.ai_source_deint()) {
            self.player.sub = 0;
        }
        let fx_changed = renderer.poll_effects();
        self.player.fx_warmup = renderer.effect_warmup();
        self.ui.fx_status = renderer.effect_status();

        // Pick up seeks made by the UI this frame.
        let waiting = self.player.tick() || fx_changed || self.ui.fx_status.iter().any(|(_, s)| *s == brp_render::EffectStatus::Compiling);

        let title = match &self.player.source {
            Some(s) => format!("{} — BRP Player", s.info.file_name),
            None => "BRP Player".to_string(),
        };
        if self.window.title() != title {
            self.window.set_title(&title);
        }

        let prims = self.egui_ctx.tessellate(out.shapes, out.pixels_per_point);
        if let Err(e) = renderer.update_textures(out.textures_delta) {
            log::error!("egui textures: {e}");
        }
        // AI picture when ready (it replaces the decoded frame; until then the decoded
        // frame is shown). The AI path deinterlaces its source itself, so none on top.
        let ai_picture = self.player.ai_frame();
        let ai_size = ai_picture.as_ref().map(|p| (p.w, p.h));
        let source_id = self.player.source_id;
        let frame = self.player.shown.clone();
        let warm = if ai_size.is_some() || self.player.scrubbing { 0 } else { self.player.fx_warmup };
        let hist = if warm > 0 { self.player.history(warm) } else { Vec::new() };
        let deint_method = if ai_size.is_some() { None } else { self.player.deint_method() };
        let (prev, next) = if deint_method.is_some() { self.player.neighbours() } else { (None, None) };
        let video = match (&frame, &self.player.source) {
            (Some(f), Some(src)) => self.player.video_rect(self.ui.video_area(px)).map(|rect| VideoDraw {
                deint: deint_method.map(|method| brp_render::DeintParams {
                    method,
                    tff: self.player.field_tff(f),
                    second_field: self.player.field_rate_active() && self.player.field == 1,
                    // Weak GPU: lighter network while playing; exact network when paused.
                    light: self.player.playing && renderer.is_integrated(),
                }),
                prev: prev.as_deref(),
                next: next.as_deref(),
                seq: self.player.seq(),
                history: hist
                    .iter()
                    .map(|(f, p, n, s, second)| brp_render::HistoryFrame { frame: f, prev: p.as_deref(), next: n.as_deref(), seq: *s, second_field: *second })
                    .collect(),
                target: (((rect[2] - rect[0]).abs().round() as u32).max(1), ((rect[3] - rect[1]).abs().round() as u32).max(1)),
                frame: f,
                color: src.info.color,
                rect,
                source_id,
                nearest: self.player.pixel_exact(self.ui.video_area(px)),
                rotation: src.info.rotation,
                picture: ai_picture.clone(),
                raw: self.player.scrubbing,
            }),
            _ => None,
        };
        match renderer.render(video, &prims, out.pixels_per_point) {
            Ok(true) if !self.presented_once => {
                self.presented_once = true;
                log::info!("startup: first present {:?}", start.elapsed());
            }
            Ok(_) => {}
            Err(e) => log::error!("render: {e}"),
        }
        // Popups over the player window show its picture: follow it.
        for p in &self.popups {
            if p.overlaps(&self.window) {
                p.window.request_redraw();
            }
        }
        let fx_output = renderer.effect_output_size().or(ai_size).filter(|&(w, h)| {
            self.player.source.as_ref().is_some_and(|s| (w, h) != (s.info.width, s.info.height) && (h, w) != (s.info.width, s.info.height))
        });
        if fx_output != self.player.fx_output {
            self.player.fx_output = fx_output;
            self.window.request_redraw();
        }
        if self.player.playing {
            // Position of the picture: frame, field or in-between.
            let pic = (self.player.seq(), 0u8);
            self.stats.1 += 1;
            if pic != self.stats.3 {
                self.stats.2 += 1;
                self.stats.3 = pic;
            }
            let secs = self.stats.0.elapsed().as_secs_f64();
            if secs >= 2.0 {
                log::debug!("playback: {:.1} presents/s, {:.1} pictures/s", self.stats.1 as f64 / secs, self.stats.2 as f64 / secs);
                self.stats = (Instant::now(), 0, 0, pic);
            }
        }

        let delay = out.viewport_output.get(&egui::ViewportId::ROOT).map(|v| v.repaint_delay).unwrap_or(Duration::MAX);
        self.needs_tick = waiting || self.player.playing || self.ui.has_pending_click() || delay.is_zero();
        self.repaint_at = (delay < Duration::from_secs(3600)).then(|| Instant::now() + delay);
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.wins.is_empty() {
            let file = self.initial_file.take();
            if let Err(e) = self.create_window(el, file) {
                log::error!("could not start: {e:#}");
                el.exit();
            }
        }
    }

    fn user_event(&mut self, el: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Instance(msg) => self.route(el, msg),
        }
    }

    fn new_events(&mut self, _el: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::ResumeTimeReached { .. } = cause {
            let now = Instant::now();
            for w in self.wins.values() {
                if w.repaint_at.is_some_and(|t| t <= now) {
                    w.window.request_redraw();
                }
                for p in &w.popups {
                    if p.repaint_at.is_some_and(|t| t <= now) {
                        p.window.request_redraw();
                    }
                }
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let start = self.start;
        // A popup's event: handled by its player window.
        if let Some(&owner) = self.popup_owner.get(&id) {
            let Some(w) = self.wins.get_mut(&owner) else {
                self.popup_owner.remove(&id);
                return;
            };
            match event {
                WindowEvent::CloseRequested => {
                    if let Some(k) = w.popups.iter().find(|p| p.id() == id).map(|p| p.kind) {
                        w.ui.set_shown(k, false);
                    }
                    w.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    if w.redraw_popup(id, el) {
                        w.window.request_redraw();
                    }
                }
                WindowEvent::DroppedFile(path) => {
                    // On the Batch window: into the queue. Elsewhere: as on the player window.
                    if w.popups.iter().any(|p| p.id() == id && p.kind == ui::PopupKind::Batch) {
                        w.ui.export.batch.add(path.to_string_lossy().into_owned());
                        w.window.request_redraw();
                    } else {
                        w.on_drop(&path);
                    }
                }
                ev => {
                    if let Some(p) = w.popups.iter_mut().find(|p| p.id() == id) {
                        if p.event(&ev) {
                            p.window.request_redraw();
                        }
                    }
                }
            }
            self.sync_popup_owners(owner);
            self.save_settings(owner);
            return;
        }
        if let WindowEvent::CloseRequested = event {
            // Its place and choices are kept for next time.
            self.save_settings(id);
            if self.settings_pending.take().is_some() {
                self.settings.save();
            }
            // Dropping the window frees its GPU resources and stops its decoders.
            if let Some(w) = self.wins.remove(&id) {
                self.popup_owner.retain(|_, o| *o != id);
                drop(w);
            }
            self.focus_order.retain(|w| *w != id);
            if self.wins.is_empty() {
                el.exit();
            }
            return;
        }
        if let WindowEvent::Focused(true) = event {
            self.focus_order.retain(|w| *w != id);
            self.focus_order.insert(0, id);
        }
        let Some(w) = self.wins.get_mut(&id) else { return };
        let resp = w.egui_state.on_window_event(&w.window, &event);
        match event {
            WindowEvent::Resized(s) => {
                if let Some(r) = w.renderer.as_mut() {
                    r.resize(s.width, s.height);
                }
                w.window.request_redraw();
            }
            WindowEvent::Moved(_) => {
                // Popups over the window show its picture at the new offset.
                for p in &w.popups {
                    p.window.request_redraw();
                }
            }
            WindowEvent::DroppedFile(path) => w.on_drop(&path),
            WindowEvent::RedrawRequested => {
                let t = Instant::now();
                w.redraw(start, el);
                if t.elapsed() > Duration::from_millis(50) {
                    log::debug!("slow redraw: main window {:?}", t.elapsed());
                }
            }
            _ => {}
        }
        if resp.repaint {
            w.window.request_redraw();
        }
        self.sync_popup_owners(id);
        self.save_settings(id);
        if self.wins.get(&id).is_some_and(|w| w.close) {
            self.window_event(el, id, WindowEvent::CloseRequested);
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        self.flush_settings();
        // New popups are drawn before they are shown (Windows sends no paint to hidden windows).
        let ids: Vec<WindowId> = self.wins.keys().copied().collect();
        for id in ids {
            let Some(w) = self.wins.get_mut(&id) else { continue };
            let hidden: Vec<WindowId> = w.popups.iter().filter(|p| !p.visible()).map(|p| p.id()).collect();
            for pid in hidden {
                if w.redraw_popup(pid, el) {
                    w.window.request_redraw();
                }
            }
            self.sync_popup_owners(id);
        }
        let mut wake: Option<Instant> = None;
        let now = Instant::now();
        let mut at = |t: Instant| wake = Some(wake.map_or(t, |x: Instant| x.min(t)));
        for w in self.wins.values() {
            if w.needs_tick {
                // FIFO presentation paces this to the display refresh.
                w.window.request_redraw();
            } else if let Some(t) = w.repaint_at {
                at(t);
            }
            for p in &w.popups {
                match p.repaint_at {
                    Some(t) if t <= now => p.window.request_redraw(),
                    Some(t) => at(t),
                    None => {}
                }
            }
        }
        if let Some(t) = self.settings_pending {
            let t = t + Duration::from_secs(1);
            wake = Some(wake.map_or(t, |x: Instant| x.min(t)));
        }
        // BRP_TEST_PLAY_AFTER (see Player::tick): keep ticking to press play on time.
        if std::env::var_os("BRP_TEST_PLAY_AFTER").is_some() || std::env::var_os("BRP_TEST_SEEK_AFTER").is_some() {
            for w in self.wins.values() {
                w.window.request_redraw();
            }
        }
        el.set_control_flow(match wake {
            Some(t) => ControlFlow::WaitUntil(t),
            None => ControlFlow::Wait,
        });
    }
}

impl App {
    /// Keep the popup -> player window map in step with the window's popups.
    fn sync_popup_owners(&mut self, owner: WindowId) {
        let Some(w) = self.wins.get(&owner) else { return };
        self.popup_owner.retain(|_, o| *o != owner);
        for p in &w.popups {
            self.popup_owner.insert(p.id(), owner);
        }
    }

    /// Gather a window's choices and place; when they changed, write them soon
    /// (`flush_settings`). Later windows start from them.
    fn save_settings(&mut self, id: WindowId) {
        let Some(w) = self.wins.get_mut(&id) else { return };
        w.ui.settings_dirty = false;
        let mut s = w.ui.settings.clone();
        s.lang = w.ui.lang;
        s.volume = w.player.volume;
        s.muted = w.player.muted;
        s.loop_on = w.player.loop_on;
        s.deint_mode = w.player.deint_mode;
        s.field_rate = w.player.field_rate;
        // Position and size as a normal window (not maximised, fullscreen or minimised).
        let win = &w.window;
        let normal = win.fullscreen().is_none() && !win.is_maximized() && win.is_minimized() != Some(true);
        if normal {
            if let Ok(p) = win.outer_position() {
                let size = win.inner_size();
                if size.width > 0 && size.height > 0 {
                    s.window = Some(settings::WindowPlace { x: p.x, y: p.y, w: size.width, h: size.height, maximized: false });
                }
            }
        } else if win.fullscreen().is_none() && win.is_minimized() != Some(true) {
            if let Some(pl) = s.window.as_mut() {
                pl.maximized = win.is_maximized();
            }
        }
        if s != self.settings {
            w.ui.settings = s.clone();
            self.settings = s;
            self.settings_pending.get_or_insert_with(Instant::now);
        }
    }

    /// Write pending settings once they have been still for a second.
    fn flush_settings(&mut self) {
        if self.settings_pending.is_some_and(|t| t.elapsed() >= Duration::from_secs(1)) {
            self.settings_pending = None;
            self.settings.save();
        }
    }
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let initial_file = std::env::args().skip(1).find(|a| !a.starts_with("--"));
    let new_window = std::env::args().any(|a| a == "--new-window");

    // Export without a window: --export=<file> or --batch-dir=<folder>.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a.starts_with("--export=") || a.starts_with("--batch-dir=") || a.starts_with("--verify=")) {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
        brp_media::init();
        let settings = settings::Settings::load();
        let saved: Vec<export::ExportSettings> = settings.export_presets.iter().map(export::ExportSettings::from_json).collect();
        let gpu = std::env::var("BRP_GPU").ok().or(settings.gpu.clone());
        let verify = args.iter().any(|a| a.starts_with("--verify="));
        std::process::exit(if verify { export::verify(&args, &saved, gpu) } else { export::cli(&args, &saved, gpu) });
    }

    // A player is already running: give it the file and leave (takes a few ms).
    if !new_window && instance::hand_off(initial_file.as_deref()) {
        return Ok(());
    }

    // Driver start-up is the slowest part of launch: begin it before anything else.
    let gpu_init = std::thread::Builder::new().name("brp-gpu-init".into()).spawn(brp_render::GpuInstance::new).ok();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu=warn")).init();
    brp_media::init();

    let el = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy: EventLoopProxy<UserEvent> = el.create_proxy();
    if !new_window {
        instance::listen(move |msg| {
            let _ = proxy.send_event(UserEvent::Instance(msg));
        });
    }
    let start_frame = std::env::args().find_map(|a| a.strip_prefix("--frame=").and_then(|v| v.parse().ok()));
    let settings = settings::Settings::load();
    if let Some(b) = settings.cache_bytes {
        brp_media::mem::set_cache_budget(b as usize);
    }
    log::info!("frame cache: {} MB per file", brp_media::mem::cache_budget() >> 20);
    // BRP_FX_RAM_MB: RAM kept for processed effect pictures (default 25% of RAM).
    if let Some(mb) = std::env::var("BRP_FX_RAM_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
        brp_media::mem::set_fx_cache_budget(mb << 20);
    }
    let mut app = App { start, gpu_init, initial_file, start_frame, wins: HashMap::new(), focus_order: Vec::new(), popup_owner: HashMap::new(), settings, settings_pending: None };
    el.run_app(&mut app)?;
    // Drop the windows (GPU resources, AI threads) before the process exits, while
    // the GPU driver is still loaded.
    app.wins.clear();
    if app.settings_pending.is_some() {
        app.settings.save();
    }
    Ok(())
}
