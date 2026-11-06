#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod effects;
mod i18n;
mod icons;
mod instance;
mod player;
mod ui;

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
use winit::window::{Fullscreen, Window, WindowId};

const MEDIA_EXTS: &[&str] = &["mp4", "mkv", "mov", "avi", "mxf", "webm", "gif", "m4v", "ts", "mts", "m2ts", "mpg", "mpeg", "wmv", "flv"];

/// Events sent to the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    /// Another launch asked to open a file (or just to come to the front).
    Instance(instance::Msg),
}

struct Win {
    window: Arc<Window>,
    renderer: Renderer,
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
}

impl App {
    fn create_window(&mut self, el: &ActiveEventLoop, file: Option<String>) -> anyhow::Result<WindowId> {
        let mut attrs = Window::default_attributes()
            .with_title("BRP Player")
            .with_inner_size(LogicalSize::new(1280.0, 760.0))
            .with_min_inner_size(LogicalSize::new(480.0, 300.0))
            .with_window_icon(icons::window_icon());
        // Cascade extra windows so they don't sit exactly on top of each other.
        if let Some(prev) = self.focus_order.first().and_then(|id| self.wins.get(id)) {
            if let Ok(pos) = prev.window.outer_position() {
                attrs = attrs.with_position(winit::dpi::PhysicalPosition::new(pos.x + 40, pos.y + 40));
            }
        }
        let window = Arc::new(el.create_window(attrs)?);
        let t_window = self.start.elapsed();
        // Open the media first: its first frame decodes while the GPU driver sets up.
        let mut player = Player::new();
        // --fx=A.glsl,B.glsl : start with these effects from the shader library.
        if let Some(list) = std::env::args().find_map(|a| a.strip_prefix("--fx=").map(str::to_string)) {
            let lib = effects::library();
            for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
                match lib.iter().find(|p| p.file_name().is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(name))) {
                    Some(p) => {
                        if let Err(e) = player.fx.add(p) {
                            log::error!("--fx {name}: {e:#}");
                        }
                    }
                    None => log::error!("--fx: {name} not found in {:?}", effects::shader_dirs()),
                }
            }
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
        let gpu = std::env::var("BRP_GPU").ok();
        let inst = match self.gpu_init.take() {
            Some(h) => h.join().map_err(|_| anyhow::anyhow!("GPU init thread panicked"))??,
            None => brp_render::GpuInstance::new()?,
        };
        let renderer = Renderer::new(inst, &*window, size.width, size.height, gpu.as_deref())?;
        log::info!("startup: window {:?}, renderer {:?} ({})", t_window, self.start.elapsed(), renderer.gpu_name());

        let egui_ctx = egui::Context::default();
        ui::setup_style(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(8192),
        );
        let mut ui = UiState::new(renderer.gpu_name().to_string(), &egui_ctx);
        // New windows inherit the language of the last used one.
        if let Some(prev) = self.focus_order.first().and_then(|id| self.wins.get(id)) {
            ui.lang = prev.ui.lang;
        }
        if self.wins.is_empty() && std::env::args().any(|a| a == "--lock") {
            ui.locked = true;
        }
        window.request_redraw();
        let id = window.id();
        self.wins.insert(
            id,
            Win { window, renderer, egui_ctx, egui_state, player, ui, needs_tick: true, repaint_at: None, presented_once: false, stats: (Instant::now(), 0, 0, (-1, 0)) },
        );
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

fn bring_to_front(w: &Window) {
    w.set_minimized(false);
    w.focus_window();
}

impl Win {
    fn open_dialog(&mut self) {
        let was_playing = self.player.playing;
        self.player.pause();
        let pick = rfd::FileDialog::new()
            .set_title(self.ui.lang.t("open_title"))
            .add_filter("Video", MEDIA_EXTS)
            .add_filter("*", &["*"])
            .pick_file();
        match pick {
            Some(path) => self.player.open(&path.to_string_lossy()),
            None if was_playing => self.player.play(),
            None => {}
        }
    }

    fn redraw(&mut self, start: Instant) {
        let size = self.window.inner_size();
        let px = [size.width as f32, size.height as f32];
        self.player.tick();

        let raw = self.egui_state.take_egui_input(&self.window);
        let mut actions = Vec::new();
        let out = self.egui_ctx.run_ui(raw, |root| {
            actions = ui::draw(root, &mut self.player, &mut self.ui, px);
        });
        self.egui_state.handle_platform_output(&self.window, out.platform_output);

        for a in actions {
            match a {
                Action::Open => self.open_dialog(),
                Action::ToggleFullscreen => {
                    let fs = self.window.fullscreen().is_none();
                    self.window.set_fullscreen(fs.then_some(Fullscreen::Borderless(None)));
                    self.ui.fullscreen = fs;
                }
                Action::ExitFullscreen => {
                    self.window.set_fullscreen(None);
                    self.ui.fullscreen = false;
                }
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
            }
        }

        // Effect chain changes and background shader compiles.
        if self.player.fx.dirty {
            self.renderer.set_effects(&self.player.fx.specs(), self.player.transfer());
            self.player.fx.dirty = false;
        }
        let fx_changed = self.renderer.poll_effects();
        self.player.fx_warmup = self.renderer.effect_warmup();
        self.ui.fx_status = self.renderer.effect_status();

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
        if let Err(e) = self.renderer.update_textures(out.textures_delta) {
            log::error!("egui textures: {e}");
        }
        let frame = self.player.shown.clone();
        let warm = self.player.fx_warmup;
        let hist = if warm > 0 { self.player.history(warm) } else { Vec::new() };
        let deint_method = self.player.deint_method();
        let (prev, next) = if deint_method.is_some() { self.player.neighbours() } else { (None, None) };
        let video = match (&frame, &self.player.source) {
            (Some(f), Some(src)) => self.player.video_rect(px[0], px[1]).map(|rect| VideoDraw {
                deint: deint_method.map(|method| brp_render::DeintParams {
                    method,
                    tff: self.player.field_tff(f),
                    second_field: self.player.field_rate_active() && self.player.field == 1,
                    // Weak GPU: lighter network while playing; exact network when paused.
                    light: self.player.playing && self.renderer.is_integrated(),
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
                source_id: self.player.source_id,
                nearest: self.player.pixel_exact(px[0], px[1]),
                rotation: src.info.rotation,
            }),
            _ => None,
        };
        match self.renderer.render(video, &prims, out.pixels_per_point) {
            Ok(true) if !self.presented_once => {
                self.presented_once = true;
                log::info!("startup: first present {:?}", start.elapsed());
            }
            Ok(_) => {}
            Err(e) => log::error!("render: {e}"),
        }
        if self.player.playing {
            let pic = (self.player.shown.as_ref().map_or(-1, |f| f.index()), self.player.field);
            self.stats.1 += 1;
            if pic != self.stats.3 {
                self.stats.2 += 1;
                self.stats.3 = pic;
            }
            let el = self.stats.0.elapsed().as_secs_f64();
            if el >= 2.0 {
                log::debug!("playback: {:.1} presents/s, {:.1} pictures/s", self.stats.1 as f64 / el, self.stats.2 as f64 / el);
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
            for w in self.wins.values() {
                w.window.request_redraw();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let start = self.start;
        if let WindowEvent::CloseRequested = event {
            // Dropping the window frees its GPU resources and stops its decoders.
            self.wins.remove(&id);
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
                w.renderer.resize(s.width, s.height);
                w.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("glsl")) => {
                // A shader: add it to the chain (copied into the user shader folder).
                match w.player.fx.import(&path) {
                    Ok(_) => w.ui.show_fx = true,
                    Err(e) => log::error!("import shader: {e:#}"),
                }
                w.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                w.player.open(&path.to_string_lossy());
                w.window.focus_window();
                w.window.request_redraw();
            }
            WindowEvent::RedrawRequested => w.redraw(start),
            _ => {}
        }
        if resp.repaint {
            w.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let mut wake: Option<Instant> = None;
        for w in self.wins.values() {
            if w.needs_tick {
                // FIFO presentation paces this to the display refresh.
                w.window.request_redraw();
            } else if let Some(t) = w.repaint_at {
                wake = Some(wake.map_or(t, |x| x.min(t)));
            }
        }
        el.set_control_flow(match wake {
            Some(t) => ControlFlow::WaitUntil(t),
            None => ControlFlow::Wait,
        });
    }
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let initial_file = std::env::args().skip(1).find(|a| !a.starts_with("--"));
    let new_window = std::env::args().any(|a| a == "--new-window");

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
    let mut app = App { start, gpu_init, initial_file, start_frame, wins: HashMap::new(), focus_order: Vec::new() };
    el.run_app(&mut app)?;
    // Drop the windows (GPU resources) before the process exits.
    app.wins.clear();
    Ok(())
}
