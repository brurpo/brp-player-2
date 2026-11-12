//! The interface: a full-window viewer with overlay controls that fade during
//! playback, a zoomable timeline with In/Out handles, and the Info, Settings and
//! Effects popups (drawn in their own glass windows, see `popup`).

use crate::i18n::Lang;
use crate::icons::{self, Icons};
use crate::player::{DeintMode, Player};
use crate::settings::Settings;
use egui::{pos2, vec2, Align, Align2, Color32, CornerRadius, FontId, Frame, Id, Key, LayerId, Layout, Margin, Order, PointerButton, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, UiBuilder, Vec2};
use std::time::{Duration, Instant};

pub const ACCENT: Color32 = Color32::from_rgb(0x4f, 0x8c, 0xff);
/// Second colour of the accent gradient.
const ACCENT_2: Color32 = Color32::from_rgb(0x8a, 0x5c, 0xff);
/// In/Out handles.
const HANDLE: Color32 = Color32::from_rgb(0xff, 0xc1, 0x3b);
/// Outside the In/Out range.
const OUTSIDE: Color32 = Color32::from_rgb(0xe5, 0x3e, 0x46);
/// Title bar and bottom bar in a window (they do not cover the picture).
const BAR_BG: Color32 = Color32::from_rgb(17, 18, 23);
const TITLE_H: f32 = 38.0;
const HIDE_AFTER: Duration = Duration::from_millis(2200);
const CLICK_DELAY: Duration = Duration::from_millis(220);
const TOAST_FOR: Duration = Duration::from_millis(1300);

pub enum Action {
    Open,
    ToggleFullscreen,
    ExitFullscreen,
    FxBrowse,
    FxSavePreset,
    FxLoadPreset,
    /// Run video, effects and AI on this GPU (None: the best one).
    SetGpu(Option<String>),
    /// Custom title bar: move, resize, minimise, maximise, close the window.
    DragWindow,
    Resize(winit::window::ResizeDirection),
    Minimize,
    ToggleMaximize,
    Close,
    /// Export: pick the output file.
    ExportBrowse,
    /// Batch: add files, pick the output folder.
    BatchAddFiles,
    BatchFolder,
    /// OCIO: pick a config file of one's own.
    OcioLoadConfig,
    /// EXR: show this layer (by name).
    SetLayer(String),
    /// Show a file in the system's file browser.
    Reveal(String),
    /// An update was installed: start the new version and close this one.
    RestartForUpdate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum PopupKind {
    Info,
    Settings,
    Effects,
    Export,
    Batch,
    /// OpenColorIO (DESIGN.md §7d).
    Color,
}

impl PopupKind {
    pub const ALL: [PopupKind; 6] = [PopupKind::Info, PopupKind::Settings, PopupKind::Effects, PopupKind::Export, PopupKind::Batch, PopupKind::Color];

    pub fn title_key(self) -> &'static str {
        match self {
            PopupKind::Info => "info",
            PopupKind::Settings => "settings",
            PopupKind::Effects => "effects",
            PopupKind::Export => "export",
            PopupKind::Batch => "batch",
            PopupKind::Color => "color",
        }
    }

    /// Content width in points (the height follows the content).
    pub fn width(self) -> f32 {
        match self {
            PopupKind::Info => 360.0,
            PopupKind::Settings => 440.0,
            PopupKind::Effects => 420.0,
            PopupKind::Export => 440.0,
            PopupKind::Batch => 900.0,
            PopupKind::Color => 460.0,
        }
    }
}

/// What the timeline shows and what a drag on it does.
#[derive(Default)]
struct Timeline {
    /// First output frame in view.
    start: f64,
    /// Output frames in view (0: the whole clip).
    len: f64,
    drag: Option<TlDrag>,
}

#[derive(Clone, Copy)]
enum TlDrag {
    Scrub,
    In,
    Out,
    /// View start and pointer x when the drag began.
    Pan(f64, f32),
}

pub struct UiState {
    pub lang: Lang,
    pub settings: Settings,
    /// OpenColorIO: on or off, the Color panel's choices (DESIGN.md §7d).
    pub ocio: crate::color::Ocio,
    /// Settings changed: save them.
    pub settings_dirty: bool,
    pub show_info: bool,
    pub show_settings: bool,
    pub show_fx: bool,
    pub show_color: bool,
    /// The Color panel's (or Settings') OCIO picker whose tree is open.
    pub ocio_open: Option<String>,
    /// The window is on an HDR screen (set each frame; OCIO's Auto display).
    pub screen_hdr: bool,
    pub show_export: bool,
    pub show_batch: bool,
    /// Export and batch export (see `export_ui`).
    pub export: crate::export_ui::ExportUi,
    /// New versions and downloaded components (models, GPL ffmpeg.exe).
    pub update: crate::update::Updater,
    /// Popups are separate glass windows (false: drawn inside the player window).
    pub detached: bool,
    pub fx_status: Vec<(u64, brp_render::EffectStatus)>,
    fx_library: Vec<std::path::PathBuf>,
    /// Output devices, listed when Settings opens.
    audio_devices: Vec<String>,
    /// GPUs (name, dedicated), listed when Settings opens.
    pub gpus: Vec<(String, bool)>,
    pub locked: bool,
    pub last_activity: Instant,
    pub pos_text: String,
    pub fps_text: String,
    pub gpu_name: String,
    pub fullscreen: bool,
    pub maximized: bool,
    /// Where the picture goes (physical pixels, x0 y0 x1 y1), set each frame.
    area: [f32; 4],
    scrub_resume: Option<bool>,
    pending_click: Option<Instant>,
    last_pointer: Option<egui::Pos2>,
    tl: Timeline,
    toast: Option<(String, Instant)>,
    /// Effect being dragged to a new place in the stack.
    fx_drag: Option<u64>,
    /// The "Add effect" list is open in the Effects window.
    pub fx_add_open: bool,
    /// BRP_TEST_DROP_AFTER done.
    pub test_dropped: bool,
    pub test_layer_done: bool,
    pub test_dumped: bool,
    pub icons: Icons,
}

impl UiState {
    pub fn new(gpu_name: String, ctx: &egui::Context, settings: Settings) -> Self {
        let mut s = UiState {
            icons: Icons::load(ctx),
            lang: settings.lang,
            ocio: crate::color::Ocio::new(&settings),
            settings,
            settings_dirty: false,
            show_info: false,
            show_settings: false,
            show_fx: false,
            show_color: false,
            ocio_open: None,
            screen_hdr: false,
            show_export: false,
            show_batch: false,
            export: Default::default(),
            update: crate::update::Updater::start(),
            detached: true,
            fx_status: Vec::new(),
            fx_library: Vec::new(),
            audio_devices: Vec::new(),
            gpus: Vec::new(),
            locked: false,
            last_activity: Instant::now(),
            pos_text: String::new(),
            fps_text: String::new(),
            gpu_name,
            fullscreen: false,
            maximized: false,
            area: [0.0; 4],
            scrub_resume: None,
            pending_click: None,
            last_pointer: None,
            tl: Timeline::default(),
            toast: None,
            fx_drag: None,
            fx_add_open: false,
            test_dropped: false,
            test_layer_done: false,
            test_dumped: false,
        };
        s.export.load(&s.settings);
        s
    }

    /// Where the picture goes (physical pixels): between the bars, or the whole
    /// window (fullscreen, or before the first frame).
    pub fn video_area(&self, px: [f32; 2]) -> [f32; 4] {
        let a = self.area;
        if a[2] - a[0] < 2.0 || a[3] - a[1] < 2.0 { [0.0, 0.0, px[0], px[1]] } else { a }
    }

    /// Dragging the playhead or an In/Out handle.
    pub fn scrubbing(&self) -> bool {
        matches!(self.tl.drag, Some(TlDrag::Scrub | TlDrag::In | TlDrag::Out))
    }

    /// True when a delayed single click is pending (needs a repaint to resolve).
    pub fn has_pending_click(&self) -> bool {
        self.pending_click.is_some()
    }

    pub fn shown(&self, k: PopupKind) -> bool {
        match k {
            PopupKind::Info => self.show_info,
            PopupKind::Settings => self.show_settings,
            PopupKind::Effects => self.show_fx,
            PopupKind::Export => self.show_export,
            PopupKind::Batch => self.show_batch,
            PopupKind::Color => self.show_color,
        }
    }

    pub fn set_shown(&mut self, k: PopupKind, on: bool) {
        if on && !self.shown(k) {
            match k {
                PopupKind::Settings => {
                    self.audio_devices = brp_media::output_devices();
                    self.gpus.clear();
                }
                PopupKind::Effects => self.fx_library = crate::effects::library(),
                PopupKind::Export | PopupKind::Batch => crate::export::probe_hardware(),
                PopupKind::Info | PopupKind::Color => {}
            }
        }
        match k {
            PopupKind::Info => self.show_info = on,
            PopupKind::Settings => self.show_settings = on,
            PopupKind::Effects => self.show_fx = on,
            PopupKind::Export => self.show_export = on,
            PopupKind::Batch => self.show_batch = on,
            PopupKind::Color => self.show_color = on,
        }
    }

    pub fn toggle(&mut self, k: PopupKind) {
        let on = !self.shown(k);
        self.set_shown(k, on);
    }

    /// A short message over the picture.
    pub fn toast(&mut self, text: impl Into<String>) {
        self.toast = Some((text.into(), Instant::now()));
    }
}

/// Visuals shared by the player window and the popups.
pub fn setup_style(ctx: &egui::Context, popup: bool) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = Color32::TRANSPARENT;
    v.window_fill = Color32::from_rgba_premultiplied(18, 18, 22, 236);
    v.window_stroke = Stroke::new(1.0, Color32::from_white_alpha(22));
    v.window_corner_radius = CornerRadius::same(12);
    v.window_shadow = egui::Shadow { offset: [0, 10], blur: 28, spread: 0, color: Color32::from_black_alpha(110) };
    v.popup_shadow = egui::Shadow { offset: [0, 6], blur: 18, spread: 0, color: Color32::from_black_alpha(100) };
    v.menu_corner_radius = CornerRadius::same(10);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.85);
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = ACCENT;
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = CornerRadius::same(7);
    }
    v.widgets.inactive.weak_bg_fill = Color32::from_white_alpha(if popup { 14 } else { 0 });
    v.widgets.inactive.bg_fill = Color32::from_white_alpha(22);
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.widgets.hovered.weak_bg_fill = Color32::from_white_alpha(30);
    v.widgets.hovered.bg_fill = Color32::from_white_alpha(36);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(40));
    v.widgets.active.weak_bg_fill = ACCENT.gamma_multiply(0.45);
    v.widgets.active.bg_fill = ACCENT.gamma_multiply(0.6);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(18));
    v.extreme_bg_color = Color32::from_black_alpha(90);
    ctx.set_visuals(v);
    ctx.all_styles_mut(|s| {
        s.spacing.button_padding = Vec2::new(9.0, 4.0);
        s.spacing.item_spacing = Vec2::new(6.0, 6.0);
        s.spacing.slider_rail_height = 4.0;
        s.spacing.combo_height = 320.0;
        s.interaction.tooltip_delay = 0.35;
    });
}

/// Vertical gradient (premultiplied colours).
fn vgradient(painter: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut m = egui::Mesh::default();
    m.colored_vertex(rect.left_top(), top);
    m.colored_vertex(rect.right_top(), top);
    m.colored_vertex(rect.left_bottom(), bottom);
    m.colored_vertex(rect.right_bottom(), bottom);
    m.add_triangle(0, 1, 2);
    m.add_triangle(1, 3, 2);
    painter.add(egui::Shape::mesh(m));
}

/// Horizontal gradient.
fn hgradient(painter: &egui::Painter, rect: Rect, left: Color32, right: Color32) {
    let mut m = egui::Mesh::default();
    m.colored_vertex(rect.left_top(), left);
    m.colored_vertex(rect.right_top(), right);
    m.colored_vertex(rect.left_bottom(), left);
    m.colored_vertex(rect.right_bottom(), right);
    m.add_triangle(0, 1, 2);
    m.add_triangle(1, 3, 2);
    painter.add(egui::Shape::mesh(m));
}

/// Soft glow: concentric translucent circles.
fn glow(painter: &egui::Painter, c: Pos2, r: f32, color: Color32, strength: f32) {
    for i in 1..=4 {
        let k = i as f32;
        painter.circle_filled(c, r + k * 2.5, color.gamma_multiply(0.11 * strength / k));
    }
}

fn button_frame(ui: &mut egui::Ui, tip: &str, active: bool) -> (Rect, egui::Response, Color32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(32.0, 30.0), Sense::click());
    let hover = ui.ctx().animate_bool_with_time(resp.id.with("h"), resp.hovered(), 0.12);
    if hover > 0.0 {
        ui.painter().rect_filled(rect.shrink(1.0), 8.0, Color32::from_white_alpha((20.0 * hover) as u8));
    }
    if active {
        ui.painter().rect_filled(rect.shrink(1.0), 8.0, ACCENT.gamma_multiply(0.16));
    }
    let color = if active {
        ACCENT
    } else if resp.hovered() {
        Color32::WHITE
    } else {
        Color32::from_gray(205)
    };
    (rect, resp.on_hover_text(tip), color)
}

fn paint_icon(ui: &egui::Ui, icons: &Icons, name: &str, rect: Rect, size: f32, color: Color32) {
    match icons.get(name) {
        Some(t) => {
            let r = Rect::from_center_size(rect.center(), Vec2::splat(size));
            let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
            ui.painter().image(t.id(), r, uv, color);
        }
        None => {
            ui.painter().text(rect.center(), Align2::CENTER_CENTER, name, FontId::proportional(14.0), color);
        }
    }
}

/// Button showing one of the icons from /icons, tinted (accent when active).
fn icon_button(ui: &mut egui::Ui, icons: &Icons, name: &str, tip: &str, active: bool) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, active);
    let press = if resp.is_pointer_button_down_on() { 0.9 } else { 1.0 };
    paint_icon(ui, icons, name, rect, 19.0 * press, color);
    resp
}

/// Small text button in the same style (In / Out markers).
fn text_button(ui: &mut egui::Ui, text: &str, tip: &str) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, false);
    ui.painter().text(rect.center(), Align2::CENTER_CENTER, text, FontId::monospace(16.0), if resp.hovered() { HANDLE } else { color });
    resp
}

fn lock_button(ui: &mut egui::Ui, locked: bool, tip: &str) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, locked);
    icons::paint_lock(ui.painter(), Rect::from_center_size(rect.center(), Vec2::splat(18.0)), locked, color);
    resp
}

/// The big round play/pause button with its glow. The symbols are drawn as shapes
/// (centred exactly; the PNG icons have uneven margins).
fn play_button(ui: &mut egui::Ui, playing: bool, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(44.0, 44.0), Sense::click());
    let ctx = ui.ctx();
    let hover = ctx.animate_bool_with_time(resp.id.with("h"), resp.hovered(), 0.15);
    let r = 18.0 + 1.5 * hover - if resp.is_pointer_button_down_on() { 1.5 } else { 0.0 };
    // One centre for every part, on a whole physical pixel (no half-pixel drift).
    let ppp = ctx.pixels_per_point();
    let c = pos2((rect.center().x * ppp).round() / ppp, (rect.center().y * ppp).round() / ppp);
    let p = ui.painter();
    glow(p, c, r, ACCENT, 0.6 + hover);
    if playing {
        // A slow breathing ring while playing.
        let t = ctx.input(|i| i.time) as f32;
        let k = (t * 1.6).sin() * 0.5 + 0.5;
        p.circle_stroke(c, r + 3.0 + 2.0 * k, Stroke::new(1.0, ACCENT.gamma_multiply(0.25 * (1.0 - k) + 0.05)));
    }
    // Accent disc lit from above: one concentric disc with a vertical gradient.
    let top = ACCENT.lerp_to_gamma(Color32::WHITE, 0.14 + 0.08 * hover);
    let bottom = ACCENT_2.lerp_to_gamma(ACCENT, 0.45);
    gradient_disc(p, c, r, top, bottom);
    p.circle_stroke(c, r, Stroke::new(1.0, Color32::from_white_alpha(70)));
    let k = r / 18.0;
    if playing {
        for dx in [-3.6, 3.6] {
            let bar = Rect::from_center_size(c + vec2(dx * k, 0.0), vec2(3.6 * k, 13.0 * k));
            p.rect_filled(bar, 1.2, Color32::WHITE);
        }
    } else {
        // Triangle placed by its centroid (optical centre) on the disc's centre.
        let (w, h) = (12.0 * k, 14.0 * k);
        let x0 = c.x - w / 3.0;
        let pts = vec![pos2(x0, c.y - h / 2.0), pos2(x0 + w, c.y), pos2(x0, c.y + h / 2.0)];
        p.add(egui::Shape::convex_polygon(pts, Color32::WHITE, Stroke::NONE));
    }
    resp.on_hover_text(tip)
}

/// Decoded frames on the timeline.
const CACHE_GREEN: Color32 = Color32::from_rgb(0x3d, 0xc4, 0x6a);

/// One cache bar on the timeline: the pixels any of `frames` covers (`span` gives a
/// frame's x range), drawn as runs. Frames are bucketed per pixel, not sorted (there
/// can be a hundred thousand kept packets).
fn cache_bar(painter: &egui::Painter, track: Rect, frames: &[i64], span: &dyn Fn(i64) -> (f32, f32), y: std::ops::RangeInclusive<f32>, color: Color32) {
    let w = track.width().max(1.0).ceil() as usize;
    let mut cols = vec![false; w + 1];
    let mut any = false;
    for &f in frames {
        let (x0, x1) = span(f);
        if x1 <= track.left() || x0 >= track.right() {
            continue;
        }
        let a = ((x0 - track.left()).max(0.0)) as usize;
        let b = (((x1 - track.left()).ceil()) as usize).clamp(a + 1, w);
        for c in &mut cols[a..b] {
            *c = true;
        }
        any = true;
    }
    if !any {
        return;
    }
    let mut i = 0;
    while i < w {
        if !cols[i] {
            i += 1;
            continue;
        }
        let a = i;
        while i < w && cols[i] {
            i += 1;
        }
        let r = Rect::from_x_y_ranges(track.left() + a as f32..=track.left() + i as f32, y.clone());
        painter.rect_filled(r, 1.0, color);
    }
}

/// A small on/off switch (pill with a knob). Flips `on` when clicked.
pub(crate) fn on_off_switch(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let (rect, mut resp) = ui.allocate_exact_size(vec2(38.0, 20.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(resp.id, *on);
    let bg = Color32::from_gray(70).lerp_to_gamma(ACCENT, t);
    let bg = if resp.hovered() { bg.gamma_multiply(1.15) } else { bg };
    ui.painter().rect_filled(rect, rect.height() / 2.0, bg);
    let r = rect.height() / 2.0 - 2.5;
    let x = egui::lerp((rect.left() + r + 2.5)..=(rect.right() - r - 2.5), t);
    ui.painter().circle_filled(pos2(x, rect.center().y), r, Color32::WHITE);
    resp
}

/// A button whose icon is drawn by `paint` (centre, colour).
fn painted_button(ui: &mut egui::Ui, tip: &str, active: bool, paint: fn(&egui::Painter, Pos2, Color32)) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, active);
    paint(ui.painter(), rect.center(), color);
    resp
}

/// Color: three overlapping circles.
fn paint_color_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    for (dx, dy) in [(-3.2, -2.0), (3.2, -2.0), (0.0, 3.4)] {
        p.circle_stroke(c + vec2(dx, dy), 5.0, Stroke::new(1.5, color));
    }
}

/// Info: an "i" in a circle.
fn paint_info_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    p.circle_stroke(c, 8.5, Stroke::new(1.6, color));
    p.circle_filled(c + vec2(0.0, -4.0), 1.3, color);
    p.line_segment([c + vec2(0.0, -1.2), c + vec2(0.0, 4.6)], Stroke::new(1.8, color));
}

/// Effects: a magic wand with sparkles.
fn paint_effects_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    p.line_segment([c + vec2(-7.0, 7.0), c + vec2(2.0, -2.0)], Stroke::new(2.0, color));
    // A four-point sparkle: two crossed thin diamonds.
    let sparkle = |at: Pos2, a: f32, b: f32| {
        p.add(egui::Shape::convex_polygon(vec![at + vec2(0.0, -a), at + vec2(b, 0.0), at + vec2(0.0, a), at + vec2(-b, 0.0)], color, Stroke::NONE));
        p.add(egui::Shape::convex_polygon(vec![at + vec2(-a, 0.0), at + vec2(0.0, -b), at + vec2(a, 0.0), at + vec2(0.0, b)], color, Stroke::NONE));
    };
    sparkle(c + vec2(4.0, -4.5), 5.0, 1.4);
    sparkle(c + vec2(-3.5, -5.5), 2.6, 0.8);
    sparkle(c + vec2(6.5, 3.5), 2.6, 0.8);
}

/// Export: an arrow out of a tray.
fn paint_export_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let s = Stroke::new(1.8, color);
    p.line_segment([c + vec2(0.0, 3.0), c + vec2(0.0, -7.0)], s);
    p.line_segment([c + vec2(-3.5, -3.5), c + vec2(0.0, -7.0)], s);
    p.line_segment([c + vec2(3.5, -3.5), c + vec2(0.0, -7.0)], s);
    p.add(egui::Shape::line(vec![c + vec2(-7.0, 0.0), c + vec2(-7.0, 7.0), c + vec2(7.0, 7.0), c + vec2(7.0, 0.0)], s));
}

/// Batch export: a stack of three frames.
fn paint_batch_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let s = Stroke::new(1.6, color);
    for (i, k) in [(0, 1.0), (1, 0.75), (2, 0.5)] {
        let o = vec2(-3.0 + 3.0 * i as f32, 3.0 - 3.0 * i as f32);
        let r = Rect::from_center_size(c + o, vec2(11.0, 8.0));
        if i == 0 {
            p.rect_stroke(r, 1.5, s, StrokeKind::Middle);
        } else {
            p.line_segment([r.left_top(), r.right_top()], Stroke::new(1.6, color.gamma_multiply(k)));
            p.line_segment([r.right_top(), r.right_bottom()], Stroke::new(1.6, color.gamma_multiply(k)));
        }
    }
}

/// The app mark in the title bar: a small play triangle on an accent disc.
fn paint_logo(p: &egui::Painter, c: Pos2) {
    glow(p, c, 7.0, ACCENT, 0.8);
    p.circle_filled(c, 8.0, ACCENT_2.lerp_to_gamma(ACCENT, 0.5));
    let pts = vec![c + vec2(-2.2, -4.0), c + vec2(4.4, 0.0), c + vec2(-2.2, 4.0)];
    p.add(egui::Shape::convex_polygon(pts, Color32::WHITE, Stroke::NONE));
}

#[derive(Clone, Copy, PartialEq)]
enum WinButton {
    Minimize,
    Maximize,
    Restore,
    Close,
    ExitFullscreen,
}

/// Minimise / maximise / close at the right end of the title bar.
fn window_button(ui: &mut egui::Ui, kind: WinButton, tip: &str) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(44.0, TITLE_H), Sense::click());
    let hover = ui.ctx().animate_bool_with_time(resp.id.with("h"), resp.hovered(), 0.1);
    let p = ui.painter();
    if hover > 0.0 {
        let fill = if kind == WinButton::Close { OUTSIDE.gamma_multiply(0.95 * hover) } else { Color32::from_white_alpha((22.0 * hover) as u8) };
        p.rect_filled(rect, 0.0, fill);
    }
    let col = if hover > 0.5 { Color32::WHITE } else { Color32::from_gray(200) };
    let st = Stroke::new(1.2, col);
    let c = rect.center();
    match kind {
        WinButton::Minimize => {
            p.line_segment([c + vec2(-5.0, 0.0), c + vec2(5.0, 0.0)], st);
        }
        WinButton::Maximize => {
            p.rect_stroke(Rect::from_center_size(c, Vec2::splat(10.0)), 1.5, st, StrokeKind::Inside);
        }
        WinButton::Restore => {
            p.rect_stroke(Rect::from_center_size(c + vec2(-1.5, 1.5), Vec2::splat(8.0)), 1.5, st, StrokeKind::Inside);
            p.line_segment([c + vec2(-3.0, -4.5), c + vec2(4.5, -4.5)], st);
            p.line_segment([c + vec2(4.5, -4.5), c + vec2(4.5, 3.0)], st);
        }
        WinButton::Close => {
            p.line_segment([c + vec2(-5.0, -5.0), c + vec2(5.0, 5.0)], st);
            p.line_segment([c + vec2(-5.0, 5.0), c + vec2(5.0, -5.0)], st);
        }
        WinButton::ExitFullscreen => {
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = c + vec2(2.0 * sx, 2.0 * sy);
                p.line_segment([corner, corner + vec2(4.0 * sx, 0.0)], st);
                p.line_segment([corner, corner + vec2(0.0, 4.0 * sy)], st);
            }
        }
    }
    resp.on_hover_text(tip)
}

/// Resizing a window without system borders: near an edge or corner the pointer shows
/// a resize cursor, and a press there starts the system resize.
fn resize_edges(ctx: &egui::Context, screen: Rect, actions: &mut Vec<Action>) {
    use winit::window::ResizeDirection as D;
    const EDGE: f32 = 6.0;
    let Some(pos) = ctx.input(|i| i.pointer.latest_pos()) else { return };
    let (l, r) = (pos.x - screen.left() < EDGE, screen.right() - pos.x < EDGE);
    let (t, b) = (pos.y - screen.top() < EDGE, screen.bottom() - pos.y < EDGE);
    let dir = match (l, r, t, b) {
        (true, _, true, _) => D::NorthWest,
        (_, true, true, _) => D::NorthEast,
        (true, _, _, true) => D::SouthWest,
        (_, true, _, true) => D::SouthEast,
        (true, ..) => D::West,
        (_, true, ..) => D::East,
        (_, _, true, _) => D::North,
        (_, _, _, true) => D::South,
        _ => return,
    };
    ctx.set_cursor_icon(match dir {
        D::NorthWest | D::SouthEast => egui::CursorIcon::ResizeNwSe,
        D::NorthEast | D::SouthWest => egui::CursorIcon::ResizeNeSw,
        D::West | D::East => egui::CursorIcon::ResizeHorizontal,
        D::North | D::South => egui::CursorIcon::ResizeVertical,
    });
    if ctx.input(|i| i.pointer.primary_pressed()) {
        actions.push(Action::Resize(dir));
    }
}

/// The footage at a glance, next to the file name: size, rate and scan, codec, pixel
/// format, dynamic range, alpha, audio, length.
fn footage_chips(ui: &mut egui::Ui, p: &Player, s: &UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    let Some(src) = &p.source else { return };
    let i = &src.info;
    // Chips that no longer fit are left out (the rest is in Info).
    let mut full = false;
    let mut chip = |ui: &mut egui::Ui, text: String, color: Color32, tip: &str| {
        let w = ui.painter().layout_no_wrap(text.clone(), FontId::proportional(11.5), Color32::WHITE).size().x + 20.0;
        if full || ui.available_width() < w {
            full = true;
            return;
        }
        let r = Frame::new()
            .fill(color.gamma_multiply(0.14))
            .stroke(Stroke::new(1.0, color.gamma_multiply(0.38)))
            .corner_radius(6.0)
            .inner_margin(Margin::symmetric(6, 1))
            .show(ui, |ui| ui.label(RichText::new(text).size(11.5).color(color.lerp_to_gamma(Color32::WHITE, 0.55))))
            .response;
        if !tip.is_empty() {
            r.on_hover_text(tip);
        }
    };
    let gray = Color32::from_gray(150);
    ui.spacing_mut().item_spacing.x = 5.0;
    // EXR with several layers: the one shown, click for the others.
    if i.layers.len() > 1 {
        let accent = Color32::from_rgb(0x8a, 0xb4, 0xf8);
        let label = i.layers.get(i.layer).map_or(String::new(), |(_, label)| label.clone());
        let r = Frame::new()
            .fill(accent.gamma_multiply(0.14))
            .stroke(Stroke::new(1.0, accent.gamma_multiply(0.45)))
            .corner_radius(6.0)
            .inner_margin(Margin::symmetric(6, 1))
            .show(ui, |ui| {
                let c = accent.lerp_to_gamma(Color32::WHITE, 0.55);
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(RichText::new(label).size(11.5).color(c));
                // A small down arrow (the font has no ▾).
                let (r, _) = ui.allocate_exact_size(Vec2::new(7.0, 11.0), egui::Sense::hover());
                let m = r.center();
                ui.painter().line(vec![Pos2::new(m.x - 3.0, m.y - 1.5), Pos2::new(m.x, m.y + 1.5), Pos2::new(m.x + 3.0, m.y - 1.5)], Stroke::new(1.3, c));
            })
            .response
            .interact(egui::Sense::click())
            .on_hover_text(l.t("exr_layer"))
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        egui::Popup::menu(&r).show(|ui| {
            ui.set_min_width(180.0);
            for (k, (name, label)) in i.layers.iter().enumerate() {
                if ui.selectable_label(k == i.layer, label).clicked() && k != i.layer {
                    actions.push(Action::SetLayer(name.clone()));
                    ui.close();
                }
            }
        });
    }
    chip(ui, format!("{} × {}", i.width, i.height), gray, l.t("resolution"));
    // Scan: interlaced (with field order) or progressive, as detected.
    let (scan, scan_tip, scan_col) = match (src.interlaced(), p.shown.as_ref().map(|f| f.tff)) {
        (Some(true), tff) => {
            let order = match tff {
                Some(true) => " TFF",
                Some(false) => " BFF",
                None => "",
            };
            (format!("{:.3}i{order}", i.frame_rate), l.t("interlaced"), HANDLE)
        }
        (Some(false), _) => (format!("{:.3}p", i.frame_rate), l.t("progressive"), gray),
        (None, _) => (format!("{:.3} fps", i.frame_rate), l.t("unknown"), gray),
    };
    chip(ui, scan, scan_col, scan_tip);
    chip(ui, i.codec.clone(), gray, l.t("codec"));
    chip(ui, format!("{} · {}-bit", i.pix_fmt, i.bit_depth), gray, l.t("pixel_format"));
    if i.color.is_hdr() {
        chip(ui, format!("HDR {:?}", i.color.transfer), Color32::from_rgb(0xf0, 0x9a, 0x3e), l.t("hdr"));
    } else if i.color.transfer == brp_media::Transfer::Linear {
        chip(ui, l.t("linear_light").into(), Color32::from_rgb(0xf0, 0x9a, 0x3e), l.t("linear_tip"));
    } else {
        chip(ui, "SDR".into(), gray, l.t("hdr"));
    }
    if i.has_alpha {
        chip(ui, l.t("transparent").into(), Color32::from_rgb(0x4c, 0xd9, 0xa0), l.t("alpha"));
    } else {
        chip(ui, l.t("opaque").into(), gray, l.t("alpha"));
    }
    let audio = if i.audio.is_some() { l.t("audio") } else { l.t("no_audio_short") };
    chip(ui, audio.into(), gray, i.audio.as_deref().unwrap_or(""));
    let secs = src.frame_count() as f64 / i.frame_rate.max(1.0);
    let dur = if secs >= 3600.0 {
        format!("{}:{:02}:{:02}", (secs / 3600.0) as i64, (secs / 60.0) as i64 % 60, secs as i64 % 60)
    } else if secs >= 60.0 {
        format!("{}:{:02}", (secs / 60.0) as i64, secs as i64 % 60)
    } else {
        format!("{secs:.2} s")
    };
    chip(ui, format!("{dur} · {} {}", src.frame_count(), l.t("frames_short")), gray, l.t("duration"));
    if !src.index_ready() {
        ui.add(egui::Spinner::new().size(11.0));
        ui.label(RichText::new(l.t("indexing")).small().weak());
    }
}

/// Output frames per second (slow motion and fields add frames).
fn out_fps(p: &Player) -> f64 {
    p.native_fps().max(1.0) * p.per_frame() as f64 / p.slow() as f64
}

/// Position of an output frame as shown everywhere (frame number or timecode; no
/// hours for clips under an hour).
fn fmt_pos(p: &Player, s: &UiState, out: i64) -> String {
    if s.settings.show_frames {
        // Zero-padded to the digits of the last frame, so every position has one width.
        let digits = (p.out_count() - 1).max(1).to_string().len();
        return format!("{out:0digits$}");
    }
    let tc = p.out_timecode(out);
    let long = p.out_count() as f64 / out_fps(p) >= 3600.0;
    match tc.strip_prefix("00:") {
        Some(rest) if !long => rest.to_string(),
        _ => tc,
    }
}

/// Spacing of labelled ruler ticks, in output frames, for `px_per_frame`.
fn tick_step(fps: f64, frames_mode: bool, px_per_frame: f64) -> f64 {
    const MIN_PX: f64 = 74.0;
    if frames_mode {
        let mut step = 1.0;
        loop {
            for m in [1.0, 2.0, 5.0] {
                if step * m * px_per_frame >= MIN_PX {
                    return step * m;
                }
            }
            step *= 10.0;
        }
    }
    for f in [1.0, 2.0, 5.0, 10.0] {
        if f < fps * 0.9 && f * px_per_frame >= MIN_PX {
            return f;
        }
    }
    for secs in [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0, 7200.0, 14400.0] {
        let f = (secs * fps).round();
        if f * px_per_frame >= MIN_PX {
            return f;
        }
    }
    (14400.0 * fps).round()
}

/// Ruler label: as precise as the tick spacing needs.
fn fmt_tick(out: f64, step: f64, fps: f64, long: bool, frames_mode: bool) -> String {
    if frames_mode {
        return format!("{}", out.round() as i64);
    }
    let secs = out / fps;
    let whole = secs.floor();
    let (h, m, s) = ((whole / 3600.0) as i64, (whole / 60.0) as i64 % 60, whole as i64 % 60);
    let base = if long { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") };
    if step < fps * 0.99 {
        let f = (out - (whole * fps).round()).round().max(0.0) as i64;
        format!("{base}:{f:02}")
    } else {
        base
    }
}

/// Draws the player window's UI. `px` is the window size in physical pixels.
pub fn draw(root: &mut egui::Ui, p: &mut Player, s: &mut UiState) -> Vec<Action> {
    let ctx = &root.ctx().clone();
    let mut actions = Vec::new();
    let l = s.lang;
    let ppp = ctx.pixels_per_point();

    // Windowed: the bars sit above and below the picture and stay. Fullscreen: they
    // float over it and fade out during playback.
    let overlay = s.fullscreen;
    let pointer = ctx.input(|i| i.pointer.latest_pos());
    if pointer != s.last_pointer || ctx.input(|i| !i.events.is_empty() && i.pointer.any_down()) {
        s.last_activity = Instant::now();
        s.last_pointer = pointer;
    }
    let idle = s.last_activity.elapsed() > HIDE_AFTER;
    let show_controls = !overlay || !(p.playing && idle) || p.source.is_none() || s.tl.drag.is_some();
    let fade = ctx.animate_bool_with_time(Id::new("controls"), show_controls, 0.22);
    ctx.set_cursor_icon(if show_controls { egui::CursorIcon::Default } else { egui::CursorIcon::None });

    keyboard(ctx, p, s, &mut actions);
    s.export.poll(p, ctx);
    let screen = ctx.content_rect();
    if !overlay && !s.maximized {
        resize_edges(ctx, screen, &mut actions);
    }

    // ---- Title bar ----
    if fade > 0.0 {
        let frame = if overlay { Frame::new() } else { Frame::new().fill(BAR_BG) };
        egui::Panel::top("top")
            .frame(frame.inner_margin(Margin { left: 10, right: 0, top: 0, bottom: 0 }))
            .exact_size(TITLE_H)
            .show_separator_line(false)
            .show(root, |ui| {
                let bar = ui.max_rect();
                let bg = ui.ctx().layer_painter(ui.layer_id());
                if overlay {
                    let r = Rect::from_min_max(screen.min, pos2(screen.max.x, bar.bottom() + 34.0));
                    vgradient(&bg, r, Color32::from_black_alpha((185.0 * fade) as u8), Color32::TRANSPARENT);
                } else {
                    // A faint accent line under the title bar.
                    let line = Rect::from_x_y_ranges(screen.left()..=screen.right(), bar.bottom() - 1.0..=bar.bottom());
                    hgradient(&bg, line, ACCENT.gamma_multiply(0.0), ACCENT.gamma_multiply(0.35));
                }
                ui.set_opacity(fade);
                // Empty parts of the bar move the window (double-click: maximise).
                let drag = ui.interact(Rect::from_min_max(screen.min, pos2(screen.max.x, bar.bottom())), Id::new("titlebar"), Sense::click_and_drag());
                if !overlay {
                    if drag.double_clicked() {
                        actions.push(Action::ToggleMaximize);
                    } else if drag.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_pressed()) {
                        actions.push(Action::DragWindow);
                    }
                }
                ui.horizontal_centered(|ui| {
                    // App mark.
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::hover());
                    match &s.icons.app {
                        Some(t) => {
                            let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                            ui.painter().image(t.id(), r, uv, Color32::WHITE);
                        }
                        None => paint_logo(ui.painter(), r.center()),
                    }
                    ui.add_space(4.0);
                    let title = p.source.as_ref().map(|s| s.info.file_name.clone()).unwrap_or_else(|| "BRP Player".into());
                    ui.label(RichText::new(title).strong().size(13.5).color(Color32::from_gray(238)));
                    ui.add_space(6.0);
                    // Chips only where they fit before the buttons on the right.
                    let room = (ui.available_width() - if overlay { 270.0 } else { 370.0 }).max(0.0);
                    ui.horizontal(|ui| {
                        ui.set_max_width(room);
                        footage_chips(ui, p, s, &mut actions);
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        if overlay {
                            if window_button(ui, WinButton::ExitFullscreen, l.t("exit_fullscreen")).clicked() {
                                actions.push(Action::ExitFullscreen);
                            }
                        } else {
                            if window_button(ui, WinButton::Close, l.t("close_window")).clicked() {
                                actions.push(Action::Close);
                            }
                            let kind = if s.maximized { WinButton::Restore } else { WinButton::Maximize };
                            if window_button(ui, kind, l.t("maximize")).clicked() {
                                actions.push(Action::ToggleMaximize);
                            }
                            if window_button(ui, WinButton::Minimize, l.t("minimize")).clicked() {
                                actions.push(Action::Minimize);
                            }
                        }
                        ui.add_space(8.0);
                        if icon_button(ui, &s.icons, "folder_expanded", l.t("open"), false).clicked() {
                            actions.push(Action::Open);
                        }
                        // Batch export, then Export (right to left), each with a ring of
                        // progress while it runs.
                        let ring = |ui: &egui::Ui, r: &egui::Response, f: Option<f32>| {
                            let Some(f) = f else { return };
                            let c = r.rect.center();
                            let n = 40;
                            let pts: Vec<Pos2> = (0..=((n as f32 * f) as usize).max(1))
                                .map(|i| {
                                    let a = -std::f32::consts::FRAC_PI_2 + i as f32 / n as f32 * std::f32::consts::TAU;
                                    c + vec2(a.cos(), a.sin()) * 12.5
                                })
                                .collect();
                            ui.painter().add(egui::Shape::line(pts, Stroke::new(2.0, ACCENT)));
                        };
                        let bf = s.export.batch_fraction();
                        let r = painted_button(ui, l.t("batch_title_tip"), s.show_batch || bf.is_some(), paint_batch_icon);
                        ring(ui, &r, bf);
                        if r.clicked() {
                            s.toggle(PopupKind::Batch);
                        }
                        let ef = s.export.export_fraction();
                        let r = painted_button(ui, l.t("ex_title_tip"), s.show_export || ef.is_some(), paint_export_icon);
                        ring(ui, &r, ef);
                        if r.clicked() {
                            s.toggle(PopupKind::Export);
                        }
                        if painted_button(ui, l.t("color_tip"), s.show_color || s.ocio.enabled, paint_color_icon).clicked() {
                            s.toggle(PopupKind::Color);
                        }
                        let fx_on = p.fx.any_enabled();
                        if painted_button(ui, l.t("effects"), s.show_fx || fx_on, paint_effects_icon).clicked() {
                            s.toggle(PopupKind::Effects);
                        }
                        if icon_button(ui, &s.icons, "config", l.t("settings"), s.show_settings).clicked() {
                            s.toggle(PopupKind::Settings);
                        }
                        if painted_button(ui, l.t("info"), s.show_info, paint_info_icon).clicked() {
                            s.toggle(PopupKind::Info);
                        }
                        let tip = if s.locked { l.t("lock") } else { l.t("unlock") };
                        if lock_button(ui, s.locked, tip).clicked() {
                            s.locked = !s.locked;
                        }
                    });
                });
            });
    }

    // ---- Bottom bar ----
    if fade > 0.0 && p.source.is_some() {
        let frame = if overlay { Frame::new() } else { Frame::new().fill(BAR_BG) };
        egui::Panel::bottom("bottom")
            .frame(frame.inner_margin(Margin { left: 16, right: 16, top: 6, bottom: 8 }))
            .show_separator_line(false)
            .show(root, |ui| {
                let bg = ui.ctx().layer_painter(ui.layer_id());
                let top = ui.max_rect().top() - 6.0;
                if overlay {
                    vgradient(
                        &bg,
                        Rect::from_min_max(pos2(screen.min.x, top - 54.0), screen.max),
                        Color32::TRANSPARENT,
                        Color32::from_black_alpha((215.0 * fade) as u8),
                    );
                } else {
                    let line = Rect::from_x_y_ranges(screen.left()..=screen.right(), top..=top + 1.0);
                    bg.rect_filled(line, 0.0, Color32::from_white_alpha(14));
                }
                ui.set_opacity(fade);
                timeline_row(ui, p, s);
                controls_row(ui, p, s, &mut actions);
            });
    }

    // ---- Viewer (everything not covered by bars) ----
    egui::CentralPanel::default().frame(Frame::NONE).show(root, |ui| {
        let rect = ui.max_rect();
        // Where the picture goes: between the bars in a window, everywhere in fullscreen.
        let area = if overlay { screen } else { rect };
        s.area = [area.left() * ppp, area.top() * ppp, area.right() * ppp, area.bottom() * ppp];
        let resp = ui.interact(rect, Id::new("viewer"), Sense::click_and_drag());
        if p.source.is_none() {
            empty_state(ui, rect, p, s, &resp);
            if resp.double_clicked() || resp.clicked() {
                actions.push(Action::Open);
            }
            return;
        }
        // Single click = play/pause (delayed so a double click can cancel it).
        if resp.double_clicked() {
            s.pending_click = None;
            actions.push(Action::ToggleFullscreen);
        } else if resp.clicked() {
            s.pending_click = Some(Instant::now());
        }
        if let Some(t) = s.pending_click {
            if t.elapsed() >= CLICK_DELAY {
                s.pending_click = None;
                p.toggle();
                s.toast(if p.playing { l.t("playing") } else { l.t("paused") });
            }
        }
        // Drag (left or middle) pans, wheel zooms at the cursor.
        if resp.dragged() {
            let d = resp.drag_delta() * ppp;
            p.view.pan[0] += d.x;
            p.view.pan[1] += d.y;
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(pos) = resp.hover_pos() {
                    p.zoom_at((scroll / 240.0).clamp(-1.0, 1.0), [pos.x * ppp, pos.y * ppp], s.area);
                }
            }
        }
    });

    toast(ctx, s);
    update_banner(ctx, s, &mut actions);

    // Popups inside this window (only when they could not get windows of their own).
    if !s.detached {
        for k in PopupKind::ALL {
            if !s.shown(k) {
                continue;
            }
            let mut open = true;
            egui::Window::new(l.t(k.title_key()))
                .id(Id::new(("popup", k)))
                .open(&mut open)
                .default_pos(match k {
                    PopupKind::Effects => pos2(12.0, 52.0),
                    _ => pos2(screen.right() - k.width() - 40.0, 52.0),
                })
                .resizable(false)
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.set_width(k.width() - 28.0);
                    popup_contents(ui, k, p, s, &mut actions);
                });
            if !open {
                s.set_shown(k, false);
            }
        }
    }
    actions
}

/// No file yet: a soft, slowly breathing invitation to drop one.
fn empty_state(ui: &mut egui::Ui, rect: Rect, p: &Player, s: &UiState, resp: &egui::Response) {
    let l = s.lang;
    let painter = ui.painter();
    let c = rect.center() - vec2(0.0, 24.0);
    let t = ui.input(|i| i.time) as f32;
    let k = (t * 1.2).sin() * 0.5 + 0.5;
    let hover = ui.ctx().animate_bool_with_time(Id::new("empty_hover"), resp.hovered(), 0.2);
    // Rings.
    for (i, a) in [(0, 26.0), (1, 14.0), (2, 7.0)] {
        let r = 46.0 + i as f32 * 16.0 + k * 4.0;
        painter.circle_stroke(c, r, Stroke::new(1.0, ACCENT.gamma_multiply((a + 20.0 * hover) / 255.0 * 4.0)));
    }
    glow(painter, c, 34.0, ACCENT, 0.8 + hover);
    painter.circle_filled(c, 34.0, Color32::from_rgba_premultiplied(18, 22, 34, 230));
    painter.circle_stroke(c, 34.0, Stroke::new(1.5, ACCENT.gamma_multiply(0.7 + 0.3 * hover)));
    paint_icon(ui, &s.icons, "folder_expanded", Rect::from_center_size(c, Vec2::splat(30.0)), 28.0, Color32::WHITE);
    let text = p.error.as_deref().map(|e| format!("{}\n{e}", l.t("error_open"))).unwrap_or_else(|| l.t("drop_hint").to_string());
    painter.text(c + vec2(0.0, 82.0), Align2::CENTER_CENTER, text, FontId::proportional(18.0), Color32::from_gray(200));
    painter.text(c + vec2(0.0, 108.0), Align2::CENTER_CENTER, l.t("drop_sub"), FontId::proportional(13.0), Color32::from_gray(120));
    ui.ctx().request_repaint_after(Duration::from_millis(33));
}

/// Updates and component downloads: a small card in the top right corner.
fn update_banner(ctx: &egui::Context, s: &mut UiState, actions: &mut Vec<Action>) {
    use crate::update::Phase;
    if !s.update.enabled() {
        return;
    }
    let l = s.lang;
    let (phase, comps, comp_err, dismissed) = {
        let st = s.update.state.lock();
        (st.phase.clone(), st.components.clone(), st.component_error.clone(), st.dismissed)
    };
    let busy = !comps.is_empty() || matches!(phase, Phase::Downloading(..));
    if !busy && (dismissed || (matches!(phase, Phase::Idle) && comp_err.is_none())) {
        return;
    }
    let mb = |b: u64| format!("{:.0} MB", b as f64 / 1e6);
    let bar = |ui: &mut egui::Ui, done: u64, total: u64| {
        let f = if total > 0 { done as f32 / total as f32 } else { 0.0 };
        ui.add(egui::ProgressBar::new(f).desired_width(260.0).text(if total > 0 { format!("{} / {}", mb(done), mb(total)) } else { mb(done) }));
    };
    let screen = ctx.content_rect();
    egui::Area::new(Id::new("update_banner")).order(Order::Foreground).fixed_pos(pos2(screen.right() - 300.0, screen.top() + 48.0)).show(ctx, |ui| {
        Frame::new()
            .fill(Color32::from_rgba_premultiplied(16, 18, 26, 235))
            .stroke(Stroke::new(1.0, ACCENT.gamma_multiply(0.55)))
            .corner_radius(CornerRadius::same(12))
            .inner_margin(Margin::same(12))
            .show(ui, |ui| {
                ui.set_width(270.0);
                if !comps.is_empty() {
                    ui.label(RichText::new(l.t("upd_components")).strong());
                    let (done, total) = comps.iter().fold((0, 0), |a, c| (a.0 + c.1, a.1 + c.2));
                    if let Some(c) = comps.iter().find(|c| c.1 < c.2) {
                        ui.label(RichText::new(&c.0).small().color(Color32::from_gray(160)));
                    }
                    bar(ui, done, total);
                }
                match &phase {
                    Phase::Available(r) => {
                        ui.label(RichText::new(format!("{} · {}", l.t("upd_available"), r.version)).strong());
                        ui.horizontal(|ui| {
                            if ui.button(l.t("upd_install")).clicked() {
                                s.update.install();
                            }
                            if ui.button(l.t("upd_later")).clicked() {
                                s.update.state.lock().dismissed = true;
                            }
                        });
                    }
                    Phase::Downloading(what, d, t) => {
                        ui.label(RichText::new(format!("{} {what}", l.t("upd_downloading"))).strong());
                        bar(ui, *d, *t);
                    }
                    Phase::Ready => {
                        ui.label(RichText::new(l.t("upd_ready")).strong());
                        if ui.button(l.t("upd_restart")).clicked() {
                            actions.push(Action::RestartForUpdate);
                        }
                    }
                    Phase::Failed(e) => {
                        ui.label(RichText::new(l.t("upd_failed")).strong());
                        ui.label(RichText::new(e).small().color(Color32::from_gray(170)));
                        if ui.button("OK").clicked() {
                            s.update.state.lock().dismissed = true;
                        }
                    }
                    Phase::Idle => {}
                }
                if let Some(e) = comp_err.as_ref().filter(|_| comps.is_empty()) {
                    ui.label(RichText::new(l.t("upd_comp_failed")).strong());
                    ui.label(RichText::new(e).small().color(Color32::from_gray(170)));
                    if ui.button("OK").clicked() {
                        let mut st = s.update.state.lock();
                        st.component_error = None;
                        st.dismissed = true;
                    }
                }
            });
    });
    if busy {
        ctx.request_repaint_after(Duration::from_millis(250));
    }
}

/// The short message over the picture (fades out).
fn toast(ctx: &egui::Context, s: &mut UiState) {
    let Some((text, at)) = &s.toast else { return };
    let age = at.elapsed();
    if age > TOAST_FOR {
        s.toast = None;
        return;
    }
    let a = age.as_secs_f32();
    let alpha = (a / 0.12).min(1.0) * ((TOAST_FOR.as_secs_f32() - a) / 0.35).min(1.0);
    let rise = (1.0 - (a / 0.18).min(1.0)) * 8.0;
    let painter = ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("toast")));
    let screen = ctx.content_rect();
    let galley = painter.layout_no_wrap(text.clone(), FontId::proportional(17.0), Color32::WHITE);
    let size = galley.size() + vec2(36.0, 18.0);
    let rect = Rect::from_center_size(pos2(screen.center().x, screen.top() + 84.0 + rise), size);
    painter.rect_filled(rect.translate(vec2(0.0, 4.0)), 20.0, Color32::from_black_alpha((70.0 * alpha) as u8));
    painter.rect_filled(rect, 20.0, Color32::from_rgba_premultiplied(16, 18, 26, (225.0 * alpha) as u8));
    painter.rect_stroke(rect, 20.0, Stroke::new(1.0, ACCENT.gamma_multiply(0.55 * alpha)), StrokeKind::Inside);
    painter.galley(rect.center() - galley.size() / 2.0, galley, Color32::WHITE.gamma_multiply(alpha));
    ctx.request_repaint();
}

fn keyboard(ctx: &egui::Context, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    let l = s.lang;
    let mut msg: Option<String> = None;
    ctx.input(|i| {
        if i.modifiers.command && i.key_pressed(Key::Z) && !i.modifiers.shift {
            p.fx.undo();
        }
        if i.modifiers.command && (i.key_pressed(Key::Y) || (i.key_pressed(Key::Z) && i.modifiers.shift)) {
            p.fx.redo();
        }
        let shift = i.modifiers.shift;
        let big = if shift { 10 } else { 1 };
        if i.key_pressed(Key::Space) || i.key_pressed(Key::K) {
            if i.key_pressed(Key::K) { p.pause() } else { p.toggle() }
        }
        if i.key_pressed(Key::ArrowRight) {
            p.step(big);
        }
        if i.key_pressed(Key::ArrowLeft) {
            p.step(-big);
        }
        if i.key_pressed(Key::L) {
            p.shuttle(1);
        }
        if i.key_pressed(Key::J) {
            p.shuttle(-1);
        }
        if i.key_pressed(Key::Home) {
            p.seek(p.in_point);
        }
        if i.key_pressed(Key::End) {
            p.seek(p.out_frame());
        }
        if i.key_pressed(Key::I) {
            p.set_in();
            msg = Some(format!("{}  {}", l.t("toast_in"), fmt_pos(p, s, p.out_of(p.in_point))));
        }
        if i.key_pressed(Key::O) && !i.modifiers.command {
            p.set_out();
            msg = Some(format!("{}  {}", l.t("toast_out"), fmt_pos(p, s, p.out_of(p.out_frame()))));
        }
        if i.key_pressed(Key::O) && i.modifiers.command {
            actions.push(Action::Open);
        }
        if i.key_pressed(Key::M) {
            p.set_muted(!p.muted);
            msg = Some(if p.muted { l.t("muted").to_string() } else { format!("{} {:.0}%", l.t("volume"), p.volume * 100.0) });
        }
        if i.key_pressed(Key::ArrowUp) {
            p.set_volume(p.volume + 0.05);
            msg = Some(format!("{} {:.0}%", l.t("volume"), p.volume * 100.0));
        }
        if i.key_pressed(Key::ArrowDown) {
            p.set_volume(p.volume - 0.05);
            msg = Some(format!("{} {:.0}%", l.t("volume"), p.volume * 100.0));
        }
        if i.key_pressed(Key::G) {
            p.toggle_one_to_one();
        }
        if i.key_pressed(Key::E) {
            p.fx.set_bypass(!p.fx.bypass);
            msg = Some(l.t(if p.fx.bypass { "fx_all_off" } else { "fx_all_on" }).to_string());
        }
        if i.key_pressed(Key::R) {
            p.reset_view();
        }
        if i.key_pressed(Key::F) || i.key_pressed(Key::F11) {
            actions.push(Action::ToggleFullscreen);
        }
        if i.key_pressed(Key::Escape) && s.fullscreen {
            actions.push(Action::ExitFullscreen);
        }
        // Timeline zoom: + / - around the playhead, \ shows the whole clip.
        let n = p.out_count().max(1) as f64;
        let cur = p.out_index() as f64;
        let zoom = |s: &mut UiState, k: f64| {
            let len = if s.tl.len <= 0.0 { n } else { s.tl.len };
            let new = (len * k).clamp(8.0_f64.min(n), n);
            s.tl.len = if new >= n { 0.0 } else { new };
            s.tl.start = (cur - (cur - s.tl.start.clamp(0.0, n)) * new / len).clamp(0.0, (n - new).max(0.0));
            if len >= n {
                s.tl.start = (cur - new / 2.0).clamp(0.0, (n - new).max(0.0));
            }
        };
        if i.key_pressed(Key::Plus) || i.key_pressed(Key::Equals) {
            zoom(s, 0.5);
        }
        if i.key_pressed(Key::Minus) {
            zoom(s, 2.0);
        }
        if i.key_pressed(Key::Backslash) {
            s.tl.len = 0.0;
        }
    });
    if let Some(m) = msg {
        s.toast(m);
    }
}

/// In value, timeline, Out value.
fn timeline_row(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        let in_out = p.out_of(p.in_point);
        // The end labels as wide as their text needs (frame numbers are much narrower
        // than timecode), both the same width.
        let label_w = {
            let pt = ui.painter();
            let width = |t: String, font: FontId| pt.layout_no_wrap(t, font, Color32::WHITE).size().x;
            let f = width(fmt_pos(p, s, p.out_count() - 1), FontId::monospace(13.5))
                .max(width(s.lang.t("in_label").into(), FontId::proportional(10.5)))
                .max(width(s.lang.t("out_label").into(), FontId::proportional(10.5)));
            (f + 14.0).ceil()
        };
        if end_label(ui, s, s.lang.t("in_label"), fmt_pos(p, s, in_out), true, p.has_range(), label_w) {
            p.seek(p.in_point);
        }
        let w = (ui.available_width() - label_w - 10.0).max(60.0);
        timeline(ui, p, s, w);
        let out = p.out_of(p.out_frame());
        if end_label(ui, s, s.lang.t("out_label"), fmt_pos(p, s, out), false, p.has_range(), label_w) {
            p.seek(p.out_frame());
        }
    });
    // Double-click on either end clears the range (checked here: needs &mut p).
    if ui.ctx().memory(|m| m.data.get_temp::<bool>(Id::new("clear_range"))).unwrap_or(false) {
        ui.ctx().memory_mut(|m| m.data.remove::<bool>(Id::new("clear_range")));
        p.clear_range();
        s.toast(s.lang.t("range_cleared"));
    }
}

/// "IN / 00:01:02" at one end of the timeline. Returns true when clicked.
fn end_label(ui: &mut egui::Ui, s: &UiState, caption: &str, value: String, left: bool, range: bool, w: f32) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 40.0), Sense::click());
    let resp = resp.on_hover_text(s.lang.t("in_out_hint"));
    let hover = ui.ctx().animate_bool_with_time(resp.id.with("h"), resp.hovered(), 0.12);
    let p = ui.painter();
    if hover > 0.0 {
        p.rect_filled(rect, 8.0, Color32::from_white_alpha((14.0 * hover) as u8));
    }
    let (align, x) = if left { (Align2::LEFT_CENTER, rect.left() + 6.0) } else { (Align2::RIGHT_CENTER, rect.right() - 6.0) };
    let cap_color = if range { HANDLE } else { Color32::from_gray(120) };
    p.text(pos2(x, rect.top() + 11.0), align, caption, FontId::proportional(10.5), cap_color);
    let val_color = if range { Color32::from_gray(240) } else { Color32::from_gray(175) };
    p.text(pos2(x, rect.top() + 27.0), align, value, FontId::monospace(13.5), val_color);
    if resp.double_clicked() {
        ui.ctx().memory_mut(|m| m.data.insert_temp(Id::new("clear_range"), true));
    }
    resp.clicked()
}

/// The zoomable timeline: ruler with self-adjusting labels, cached frames, the In/Out
/// range (outside it in red) with draggable handles, playhead, and an overview of the
/// visible part when zoomed in.
fn timeline(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, w: f32) {
    let n = p.out_count().max(1) as f64;
    let pf = p.per_frame();
    let fps = out_fps(p);
    let frames_mode = s.settings.show_frames;
    let long = n / fps >= 3600.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, 44.0), Sense::click_and_drag());
    let resp = resp.on_hover_text(s.lang.t("timeline_hint"));
    let hover_t = ui.ctx().animate_bool_with_time(resp.id.with("h"), resp.hovered() || s.tl.drag.is_some(), 0.15);
    let ruler = Rect::from_min_max(rect.min, pos2(rect.max.x, rect.min.y + 15.0));
    let th = 14.0 + 6.0 * hover_t;
    let track = Rect::from_min_max(pos2(rect.min.x, rect.min.y + 18.0), pos2(rect.max.x, rect.min.y + 18.0 + th));
    let overview = Rect::from_min_max(pos2(rect.min.x, rect.max.y - 3.0), rect.max);

    // View (zoom) state.
    let min_len = 8.0_f64.min(n);
    let mut len = if s.tl.len <= 0.0 || s.tl.len >= n { n } else { s.tl.len.max(min_len) };
    let cur = p.out_index() as f64;
    let zoomed = len < n;
    if zoomed && p.playing && s.tl.drag.is_none() && (cur < s.tl.start || cur >= s.tl.start + len) {
        // Follow the playhead a page at a time.
        s.tl.start = cur - len * 0.08;
    }
    let mut start = if zoomed { s.tl.start.clamp(0.0, n - len) } else { 0.0 };

    // Wheel: zoom around the pointer; Shift or horizontal: scroll.
    if resp.hovered() {
        let (d, shift) = ui.input(|i| (i.smooth_scroll_delta, i.modifiers.shift));
        if let Some(pos) = resp.hover_pos() {
            let at = start + ((pos.x - track.left()) / track.width()) as f64 * len;
            if d.x != 0.0 || (shift && d.y != 0.0) {
                let dx = if d.x != 0.0 { d.x } else { d.y };
                start -= dx as f64 / track.width() as f64 * len;
            } else if d.y != 0.0 {
                let new = (len * (-d.y as f64 / 160.0).exp()).clamp(min_len, n);
                start = at - (at - start) * new / len;
                len = new;
            }
        }
    }
    // Press: pick what the drag does (handles first).
    let x_of = |f: f64, start: f64, len: f64| track.left() + ((f - start) / len) as f32 * track.width();
    let in_x = x_of(p.out_of(p.in_point) as f64, start, len);
    let out_x = x_of((p.out_of(p.out_frame()) + pf) as f64, start, len);
    let (pressed, middle, released, latest) = ui.input(|i| {
        (i.pointer.primary_pressed(), i.pointer.button_pressed(PointerButton::Middle), i.pointer.any_released(), i.pointer.latest_pos())
    });
    if (pressed || middle) && resp.hovered() && s.tl.drag.is_none() {
        if let Some(pos) = resp.hover_pos() {
            let on_track = pos.y >= track.top() - 4.0;
            s.tl.drag = Some(if middle {
                TlDrag::Pan(start, pos.x)
            } else if on_track && (pos.x - out_x).abs() <= 7.0 && (pos.x - out_x).abs() <= (pos.x - in_x).abs() {
                TlDrag::Out
            } else if on_track && (pos.x - in_x).abs() <= 7.0 {
                TlDrag::In
            } else {
                TlDrag::Scrub
            });
            if !middle && s.scrub_resume.is_none() {
                s.scrub_resume = Some(p.playing);
                p.pause();
            }
        }
    }
    if let (Some(d), Some(pos)) = (s.tl.drag, latest) {
        let f = start + ((pos.x - track.left()) / track.width()) as f64 * len;
        let last = p.last_frame();
        match d {
            TlDrag::Scrub => p.seek_out(f.floor() as i64),
            TlDrag::In => {
                let fr = ((f / pf as f64).round() as i64).clamp(0, (p.out_frame() - 1).max(0));
                if fr != p.in_point {
                    p.in_point = fr;
                    p.seek(fr);
                }
            }
            TlDrag::Out => {
                let fr = ((f / pf as f64).round() as i64 - 1).clamp((p.in_point + 1).min(last), last);
                let want = if fr >= last { i64::MAX } else { fr };
                if want != p.out_point {
                    p.out_point = want;
                    p.seek(fr);
                }
            }
            TlDrag::Pan(s0, x0) => start = s0 - ((pos.x - x0) / track.width()) as f64 * len,
        }
        ui.ctx().request_repaint();
    }
    if released && s.tl.drag.is_some() {
        s.tl.drag = None;
        if let Some(resume) = s.scrub_resume.take() {
            if resume {
                p.play();
            }
        }
    }
    if resp.double_clicked() && resp.hover_pos().is_some_and(|q| q.y < track.top()) {
        len = n;
    }
    start = start.clamp(0.0, (n - len).max(0.0));
    s.tl.start = start;
    s.tl.len = if len >= n { 0.0 } else { len };
    let x_of = |f: f64| x_of(f, start, len);

    let painter = ui.painter_at(rect.expand2(vec2(8.0, 4.0)));
    let px_per_frame = track.width() as f64 / len;

    // Track.
    painter.rect_filled(track, 5.0, Color32::from_white_alpha(16));
    vgradient(&painter, Rect::from_min_max(track.min, pos2(track.max.x, track.center().y)), Color32::from_white_alpha(10), Color32::TRANSPARENT);
    // Cache bars along the bottom of the track: green = decoded, ready to show (in
    // video memory for GPU-decoded files); thin grey above it = held in system RAM
    // (compressed packets kept for re-decoding, FFmpeg's decoded frames, processed
    // effect frames).
    if let Some(src) = &p.source {
        let map = src.cache_map();
        let span = |f: i64| (x_of(p.out_of(f) as f64), x_of(p.out_of(f + 1) as f64));
        let ram = map.ram;
        cache_bar(&painter, track, &ram, &span, track.bottom() - 5.0..=track.bottom() - 3.5, Color32::from_gray(150).gamma_multiply(0.75));
        cache_bar(&painter, track, &map.decoded, &span, track.bottom() - 3.0..=track.bottom() - 1.0, CACHE_GREEN);
        // Effect pictures of the current stack kept in RAM (not computed again).
        let fx = p.ai.ram_cached_frames();
        cache_bar(&painter, track, &fx, &span, track.bottom() - 7.5..=track.bottom() - 6.0, ACCENT.gamma_multiply(0.85));
    }
    // In/Out range: inside lit, outside red.
    let (in_x, out_x) = (x_of(p.out_of(p.in_point) as f64), x_of((p.out_of(p.out_frame()) + pf) as f64));
    if p.has_range() {
        let red = OUTSIDE.gamma_multiply(0.42);
        let left = Rect::from_x_y_ranges(track.left()..=in_x.clamp(track.left(), track.right()), track.y_range());
        let right = Rect::from_x_y_ranges(out_x.clamp(track.left(), track.right())..=track.right(), track.y_range());
        for r in [left, right] {
            if r.width() > 0.0 {
                painter.rect_filled(r, 5.0, red);
                // Diagonal hatching reads as "excluded" even at a glance.
                let step = 7.0;
                let mut x = (r.left() / step).floor() * step - th;
                while x < r.right() {
                    let a = pos2(x, r.bottom());
                    let b = pos2(x + th, r.top());
                    let seg = clip_segment(a, b, r);
                    if let Some((a, b)) = seg {
                        painter.line_segment([a, b], Stroke::new(1.0, OUTSIDE.gamma_multiply(0.35)));
                    }
                    x += step;
                }
            }
        }
        let inside = Rect::from_x_y_ranges(in_x.max(track.left())..=out_x.min(track.right()), track.y_range());
        if inside.width() > 0.0 {
            hgradient(&painter, inside, ACCENT.gamma_multiply(0.22), ACCENT_2.gamma_multiply(0.22));
        }
    }

    // Ruler: labelled ticks every `step`, minor ticks between.
    let step = tick_step(fps, frames_mode, px_per_frame);
    let minor = if step >= 10.0 && step * px_per_frame >= 50.0 { step / 5.0 } else if step >= 2.0 { step / 2.0 } else { 0.0 };
    if minor > 0.0 && minor * px_per_frame >= 5.0 {
        let mut f = (start / minor).floor() * minor;
        while f <= start + len {
            let x = x_of(f);
            painter.line_segment([pos2(x, ruler.bottom() - 3.0), pos2(x, ruler.bottom())], Stroke::new(1.0, Color32::from_white_alpha(40)));
            f += minor;
        }
    }
    let mut f = (start / step).floor() * step;
    while f <= start + len + step {
        let x = x_of(f);
        if x >= track.left() - 1.0 && x <= track.right() + 1.0 {
            painter.line_segment([pos2(x, ruler.bottom() - 7.0), pos2(x, ruler.bottom())], Stroke::new(1.0, Color32::from_white_alpha(95)));
            let label = fmt_tick(f, step, fps, long, frames_mode);
            if x + 3.0 + 50.0 < track.right() + 30.0 {
                painter.text(pos2(x + 3.0, ruler.top() + 4.0), Align2::LEFT_CENTER, label, FontId::monospace(10.0), Color32::from_gray(150));
            }
        }
        f += step;
    }

    // Frame under the playhead, when frames are wide enough to see.
    let px = x_of(cur);
    if px_per_frame >= 3.0 {
        let fr = Rect::from_x_y_ranges(px..=x_of(cur + 1.0), track.y_range());
        painter.rect_filled(fr, 1.0, Color32::from_white_alpha(46));
    }

    // In/Out handles.
    for (x, is_in) in [(in_x, true), (out_x, false)] {
        if x < track.left() - 8.0 || x > track.right() + 8.0 {
            continue;
        }
        let active = matches!((s.tl.drag, is_in), (Some(TlDrag::In), true) | (Some(TlDrag::Out), false));
        let near = resp.hover_pos().is_some_and(|q| (q.x - x).abs() <= 7.0 && q.y >= track.top() - 4.0);
        let col = if p.has_range() || near || active { HANDLE } else { HANDLE.gamma_multiply(0.45) };
        let top = track.top() - 3.0;
        let bot = track.bottom() + 3.0;
        let dir = if is_in { 1.0 } else { -1.0 };
        let wdt = if near || active { 3.0 } else { 2.0 };
        if near || active {
            painter.rect_filled(Rect::from_x_y_ranges(x - 4.0..=x + 4.0, top..=bot), 3.0, HANDLE.gamma_multiply(0.18));
        }
        painter.line_segment([pos2(x, top), pos2(x, bot)], Stroke::new(wdt, col));
        painter.line_segment([pos2(x, top), pos2(x + 6.0 * dir, top)], Stroke::new(wdt, col));
        painter.line_segment([pos2(x, bot), pos2(x + 6.0 * dir, bot)], Stroke::new(wdt, col));
    }

    // Playhead: a glowing line and a cap on the ruler.
    if px >= track.left() - 1.0 && px <= track.right() + 1.0 {
        for (w, a) in [(7.0, 18u8), (4.0, 40u8)] {
            painter.line_segment([pos2(px, ruler.top() + 2.0), pos2(px, track.bottom() + 2.0)], Stroke::new(w, Color32::from_white_alpha(a)));
        }
        painter.line_segment([pos2(px, ruler.top() + 2.0), pos2(px, track.bottom() + 2.0)], Stroke::new(2.0, Color32::WHITE));
        let cap = [pos2(px - 5.5, ruler.top()), pos2(px + 5.5, ruler.top()), pos2(px + 5.5, ruler.top() + 6.0), pos2(px, ruler.top() + 11.0), pos2(px - 5.5, ruler.top() + 6.0)];
        painter.add(egui::Shape::convex_polygon(cap.to_vec(), Color32::WHITE, Stroke::NONE));
    }

    // Overview of the visible part when zoomed in.
    if len < n {
        painter.rect_filled(overview, 1.5, Color32::from_white_alpha(14));
        let a = overview.left() + (start / n) as f32 * overview.width();
        let b = overview.left() + ((start + len) / n) as f32 * overview.width();
        let thumb = Rect::from_x_y_ranges(a..=b.max(a + 3.0), overview.y_range());
        hgradient(&painter, thumb, ACCENT, ACCENT_2);
        let c = overview.left() + (cur / n) as f32 * overview.width();
        painter.circle_filled(pos2(c, overview.center().y), 2.0, Color32::WHITE);
    }

    // Hover: a ghost line and the time under the pointer.
    if let Some(pos) = resp.hover_pos().filter(|_| s.tl.drag.is_none() || matches!(s.tl.drag, Some(TlDrag::Pan(..)))) {
        let f = (start + ((pos.x - track.left()) / track.width()) as f64 * len).floor().clamp(0.0, n - 1.0);
        let gx = x_of(f);
        painter.line_segment([pos2(gx, ruler.top() + 2.0), pos2(gx, track.bottom())], Stroke::new(1.0, Color32::from_white_alpha(80)));
        let tip = ui.ctx().layer_painter(LayerId::new(Order::Tooltip, Id::new("tl_hover")));
        let text = fmt_pos(p, s, f as i64);
        let g = tip.layout_no_wrap(text, FontId::monospace(12.0), Color32::WHITE);
        let r = Rect::from_center_size(pos2(gx, rect.top() - 14.0), g.size() + vec2(14.0, 8.0));
        tip.rect_filled(r, 7.0, Color32::from_rgba_premultiplied(14, 16, 24, 235));
        tip.rect_stroke(r, 7.0, Stroke::new(1.0, Color32::from_white_alpha(30)), StrokeKind::Inside);
        tip.galley(r.center() - g.size() / 2.0, g, Color32::WHITE);
    }
}

/// The part of segment a-b inside `r` (Liang–Barsky).
fn clip_segment(a: Pos2, b: Pos2, r: Rect) -> Option<(Pos2, Pos2)> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - r.left()), (d.x, r.right() - a.x), (-d.y, a.y - r.top()), (d.y, r.bottom() - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 <= t1).then(|| (a + d * t0, a + d * t1))
}

/// Typed position: a frame number, or a time (m:ss, h:mm:ss, h:mm:ss:ff, seconds with
/// a decimal point). Times in the file's own timecode are understood too.
fn parse_pos(p: &Player, text: &str) -> Option<i64> {
    let t = text.trim();
    if let Ok(f) = t.parse::<i64>() {
        return Some(f);
    }
    let fps = out_fps(p);
    let to_frames = |t: &str| -> Option<f64> {
        let parts: Vec<&str> = t.split([':', ';']).collect();
        let nums: Vec<f64> = parts.iter().map(|x| x.replace(',', ".").parse::<f64>()).collect::<Result<_, _>>().ok()?;
        let (secs, frames) = match nums.as_slice() {
            [s] => (*s, 0.0),
            [m, s] => (m * 60.0 + s, 0.0),
            [h, m, s] => (h * 3600.0 + m * 60.0 + s, 0.0),
            [h, m, s, f] => (h * 3600.0 + m * 60.0 + s, *f),
            _ => return None,
        };
        let nominal = fps.round().max(1.0);
        Some(secs * fps + frames * fps / nominal)
    };
    let f = to_frames(t)?;
    // The file's timecode may not start at zero.
    let origin = to_frames(&p.out_timecode(0)).unwrap_or(0.0);
    Some(if origin > 0.0 && f >= origin { f - origin } else { f }.round() as i64)
}

fn controls_row(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    const CENTER_W: f32 = 6.0 * 32.0 + 44.0 + 6.0 * 4.0;
    let (row, _) = ui.allocate_exact_size(vec2(ui.available_width(), 46.0), Sense::hover());
    let center = Rect::from_center_size(row.center(), vec2(CENTER_W, row.height()));
    let left = Rect::from_min_max(row.min, pos2(center.left() - 10.0, row.max.y));
    let right = Rect::from_min_max(pos2(center.right() + 10.0, row.min.y), row.max);

    // Centre: transport.
    ui.scope_builder(UiBuilder::new().max_rect(center).layout(Layout::left_to_right(Align::Center)), |ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if text_button(ui, "[", l.t("set_in")).clicked() {
            p.set_in();
            s.toast(format!("{}  {}", l.t("toast_in"), fmt_pos(p, s, p.out_of(p.in_point))));
        }
        if icon_button(ui, &s.icons, "toStart", l.t("to_in"), false).clicked() {
            p.seek(p.in_point);
        }
        if icon_button(ui, &s.icons, "step_back", l.t("prev_frame"), false).clicked() {
            p.step(-1);
        }
        let tip = if p.playing { l.t("pause") } else { l.t("play") };
        if play_button(ui, p.playing, tip).clicked() {
            p.toggle();
        }
        if icon_button(ui, &s.icons, "step_foward", l.t("next_frame"), false).clicked() {
            p.step(1);
        }
        if icon_button(ui, &s.icons, "toEnd", l.t("to_out"), false).clicked() {
            p.seek(p.out_frame());
        }
        if text_button(ui, "]", l.t("set_out")).clicked() {
            p.set_out();
            s.toast(format!("{}  {}", l.t("toast_out"), fmt_pos(p, s, p.out_of(p.out_frame()))));
        }
    });

    // Left: position (editable), length, display mode, deinterlacing.
    ui.scope_builder(UiBuilder::new().max_rect(left).layout(Layout::left_to_right(Align::Center)), |ui| {
        ui.set_clip_rect(left.intersect(ui.clip_rect()));
        let id = Id::new("pos_edit");
        let editing = ui.memory(|m| m.has_focus(id));
        if !editing {
            s.pos_text = fmt_pos(p, s, p.out_index());
        }
        let chars = s.pos_text.chars().count().max(3) as f32;
        let mut edit = egui::TextEdit::singleline(&mut s.pos_text)
            .id(id)
            .desired_width(chars * 9.6 + 6.0)
            .font(FontId::monospace(16.0))
            .text_color(Color32::WHITE);
        if !editing {
            edit = edit.frame(Frame::NONE);
        }
        let r = ui.add(edit).on_hover_text(l.t("pos_hint"));
        if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
            if let Some(f) = parse_pos(p, &s.pos_text) {
                p.seek_out(f);
            }
        }
        ui.label(RichText::new(format!("/ {}", fmt_pos(p, s, p.out_count() - 1))).monospace().color(Color32::from_gray(120)));
        let (mode, tip) = if s.settings.show_frames { ("F#", l.t("show_timecode")) } else { ("TC", l.t("show_frames")) };
        let chip = egui::Button::new(RichText::new(mode).small().strong().color(ACCENT)).corner_radius(9.0).stroke(Stroke::new(1.0, ACCENT.gamma_multiply(0.45)));
        if ui.add(chip).on_hover_text(tip).clicked() {
            s.settings.show_frames = !s.settings.show_frames;
            s.settings_dirty = true;
        }
        ui.add_space(8.0);
        devices(ui, p, s, actions);
        ui.add_space(4.0);
        if let Some(m) = p.deint_method() {
            let name = match m {
                brp_render::DeintMethod::NnediAdaptive => "NNEDI3+",
                brp_render::DeintMethod::Nnedi => "NNEDI3",
                brp_render::DeintMethod::Bwdif => "bwdif",
            };
            let fps = p.native_fps() * if p.field_rate { 2.0 } else { 1.0 };
            let mut text = format!("{name} {fps:.2}p");
            if p.field_rate_active() {
                text += &format!(" · {} {}", l.t("field_short"), p.field + 1);
            }
            let chip = egui::Button::new(RichText::new(text).small().color(ACCENT)).corner_radius(9.0).stroke(Stroke::new(1.0, ACCENT.gamma_multiply(0.5)));
            if ui.add(chip).on_hover_text(l.t("deint_hint")).clicked() {
                s.set_shown(PopupKind::Settings, true);
            }
        }
        shaders_note(ui, l);
    });

    // Right: loop, volume, speed, view.
    ui.scope_builder(UiBuilder::new().max_rect(right).layout(Layout::right_to_left(Align::Center)), |ui| {
        ui.set_clip_rect(right.intersect(ui.clip_rect()));
        ui.spacing_mut().item_spacing.x = 4.0;
        if icon_button(ui, &s.icons, "fullscreen_icon", l.t("fullscreen"), s.fullscreen).clicked() {
            actions.push(Action::ToggleFullscreen);
        }
        if icon_button(ui, &s.icons, "originalSize", l.t("one_to_one"), p.view.one_to_one).clicked() {
            p.toggle_one_to_one();
        }
        if icon_button(ui, &s.icons, "restoreZoom", l.t("fit"), false).clicked() {
            p.reset_view();
        }
        ui.add_space(6.0);
        let fps_id = Id::new("fps_edit");
        if !ui.memory(|m| m.has_focus(fps_id)) {
            s.fps_text = format!("{:.3}", p.native_fps() * p.speed);
        }
        ui.label(RichText::new("fps").small().color(Color32::from_gray(130)));
        let r = ui
            .add(egui::TextEdit::singleline(&mut s.fps_text).id(fps_id).desired_width(54.0).font(egui::TextStyle::Monospace))
            .on_hover_text(l.t("fps"));
        if r.lost_focus() {
            if let Ok(v) = s.fps_text.trim().replace(',', ".").parse::<f64>() {
                p.set_fps(v);
            }
        }
        if p.has_audio() {
            ui.add_space(6.0);
            let mut v = p.volume;
            ui.spacing_mut().slider_width = 84.0;
            if ui
                .add(egui::Slider::new(&mut v, 0.0..=1.0).show_value(false).trailing_fill(true))
                .on_hover_text(format!("{} {:.0}%", l.t("volume"), v * 100.0))
                .changed()
            {
                p.set_volume(v);
                if p.muted && v > 0.0 {
                    p.set_muted(false);
                }
            }
            let (icon, tip) = if p.muted || p.volume == 0.0 { ("muted", l.t("unmute")) } else { ("audio", l.t("mute")) };
            if icon_button(ui, &s.icons, icon, tip, false).clicked() {
                p.set_muted(!p.muted);
            }
        }
        ui.add_space(6.0);
        if icon_button(ui, &s.icons, "repeat_icon", l.t("loop"), p.loop_on).clicked() {
            p.loop_on = !p.loop_on;
            s.toast(if p.loop_on { l.t("loop_on") } else { l.t("loop_off") });
        }
        if p.has_range() && icon_button(ui, &s.icons, "clear", l.t("clear_range"), false).clicked() {
            p.clear_range();
            s.toast(l.t("range_cleared"));
        }
    });
}

/// "Compiling shaders… n/m" while start-up builds them (see `warm`), redrawn as it
/// goes.
fn shaders_note(ui: &mut egui::Ui, l: Lang) {
    if let Some((done, total)) = crate::warm::progress() {
        ui.add_space(4.0);
        let frac = done as f32 / total.max(1) as f32;
        ui.add(egui::ProgressBar::new(frac).desired_width(150.0).desired_height(14.0).corner_radius(7.0).fill(ACCENT).text(RichText::new(format!("{} {done}/{total}", l.t("shaders_building"))).small()))
            .on_hover_text(l.t("shaders_building_tip"));
        ui.ctx().request_repaint_after(Duration::from_millis(100));
    }
}

// ---------------------------------------------------------------------------------
// Popups

/// What a popup window's frame asks of its window this frame.
#[derive(Default)]
pub struct PopupResponse {
    pub close: bool,
    /// Start moving the window (title bar pressed).
    pub drag: bool,
    /// Size the content needs, in points.
    pub size: Vec2,
}

/// A popup window's whole UI: minimal title bar (drag to move) and the content.
/// `appear` animates the content in (0 → 1).
pub fn popup_ui(root: &mut egui::Ui, k: PopupKind, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>, appear: f32) -> PopupResponse {
    let mut out = PopupResponse::default();
    let l = s.lang;
    egui::CentralPanel::default().frame(Frame::NONE).show(root, |ui| {
        let full = ui.max_rect();
        let bar = Rect::from_min_size(full.min, vec2(full.width(), 36.0));
        // Close button.
        let close_rect = Rect::from_center_size(pos2(bar.right() - 20.0, bar.center().y), Vec2::splat(22.0));
        // The bar first, the close button after it: the later one is on top for clicks.
        let drag = ui.interact(bar, Id::new("popup_title"), Sense::click_and_drag());
        let close = ui.interact(close_rect, Id::new("popup_close"), Sense::click());
        let on_close = ui.input(|i| i.pointer.interact_pos()).is_some_and(|p| close_rect.expand(2.0).contains(p));
        if !on_close && drag.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_pressed()) {
            out.drag = true;
        }
        let painter = ui.painter();
        let ch = ui.ctx().animate_bool_with_time(close.id, close.hovered(), 0.12);
        if ch > 0.0 {
            painter.circle_filled(close_rect.center(), 11.0, OUTSIDE.gamma_multiply(0.9 * ch));
        }
        let xc = close_rect.center();
        let xs = 4.0;
        let xcol = Color32::from_gray((170.0 + 85.0 * ch) as u8);
        painter.line_segment([xc - vec2(xs, xs), xc + vec2(xs, xs)], Stroke::new(1.6, xcol));
        painter.line_segment([xc + vec2(-xs, xs), xc + vec2(xs, -xs)], Stroke::new(1.6, xcol));
        if close.on_hover_text(l.t("close")).clicked() {
            out.close = true;
        }
        // Title: a little accent orb and the name.
        let orb = pos2(bar.left() + 18.0, bar.center().y);
        glow(painter, orb, 3.5, ACCENT, 1.0);
        painter.circle_filled(orb, 3.5, ACCENT.lerp_to_gamma(Color32::WHITE, 0.25));
        painter.text(pos2(bar.left() + 30.0, bar.center().y), Align2::LEFT_CENTER, l.t(k.title_key()), FontId::proportional(13.5), Color32::from_gray(232));
        // Hairline under the bar, fading at both ends.
        let line = Rect::from_x_y_ranges(bar.left() + 12.0..=bar.right() - 12.0, bar.bottom()..=bar.bottom() + 1.0);
        let mid = line.center().x;
        hgradient(painter, Rect::from_x_y_ranges(line.left()..=mid, line.y_range()), Color32::TRANSPARENT, Color32::from_white_alpha(34));
        hgradient(painter, Rect::from_x_y_ranges(mid..=line.right(), line.y_range()), Color32::from_white_alpha(34), Color32::TRANSPARENT);

        // Content, sliding up a little as it appears.
        let top = bar.bottom() + 10.0 + (1.0 - appear) * 6.0;
        let inner = Rect::from_min_size(pos2(full.left() + 14.0, top), vec2(k.width() - 28.0, 4000.0));
        let mut child = ui.new_child(UiBuilder::new().max_rect(inner).layout(Layout::top_down(Align::Min)));
        child.set_opacity(appear);
        child.set_width(k.width() - 28.0);
        popup_contents(&mut child, k, p, s, actions);
        let used = child.min_rect();
        out.size = vec2(k.width(), (used.bottom() - full.top() + 14.0).ceil());
    });
    if root.ctx().input(|i| i.key_pressed(Key::Escape)) && !root.ctx().egui_wants_keyboard_input() {
        out.close = true;
    }
    // Undo and redo for the effect stack also work while a popup has the keyboard.
    if !root.ctx().egui_wants_keyboard_input() {
        let (undo, redo) = root.ctx().input(|i| {
            let z = i.modifiers.command && i.key_pressed(Key::Z);
            (z && !i.modifiers.shift, (i.modifiers.command && i.key_pressed(Key::Y)) || (z && i.modifiers.shift))
        });
        if undo {
            p.fx.undo();
        }
        if redo {
            p.fx.redo();
        }
    }
    out
}

fn popup_contents(ui: &mut egui::Ui, k: PopupKind, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    match k {
        PopupKind::Info => info_contents(ui, p, s),
        PopupKind::Settings => settings_contents(ui, p, s, actions),
        PopupKind::Effects => effects_contents(ui, p, s, actions),
        PopupKind::Export => crate::export_ui::export_contents(ui, p, s, actions),
        PopupKind::Batch => crate::export_ui::batch_contents(ui, p, s, actions),
        PopupKind::Color => crate::color_ui::color_contents(ui, p, s, actions),
    }
}

/// A small caps section heading.
pub(crate) fn section(ui: &mut egui::Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(RichText::new(text.to_uppercase()).size(10.5).strong().color(ACCENT.lerp_to_gamma(Color32::WHITE, 0.35)));
}

fn info_contents(ui: &mut egui::Ui, p: &Player, s: &UiState) {
    let l = s.lang;
    let mut rows: Vec<(&str, String)> = Vec::new();
    if let Some(src) = &p.source {
        let i = &src.info;
        let yn = |b: bool| if b { l.t("yes") } else { l.t("no") }.to_string();
        rows.push((l.t("file"), i.file_name.clone()));
        rows.push((l.t("container"), i.container.clone()));
        rows.push((l.t("codec"), i.codec.clone()));
        rows.push((l.t("resolution"), format!("{} × {}", i.width, i.height)));
        if let Some((w, h)) = p.fx_output {
            rows.push((l.t("fx_resolution"), format!("{w} × {h}")));
        }
        rows.push((l.t("frame_rate"), format!("{:.3}", i.frame_rate)));
        rows.push((l.t("frames"), format!("{}", src.frame_count())));
        rows.push((l.t("duration"), format!("{:.3} s", src.frame_count() as f64 / i.frame_rate.max(1.0))));
        rows.push((l.t("pixel_format"), format!("{} ({}-bit)", i.pix_fmt, i.bit_depth)));
        rows.push((l.t("color"), format!("{:?} / {:?} / {}", i.color.matrix, i.color.transfer, if i.color.full_range { "full" } else { "limited" })));
        rows.push((l.t("hdr"), yn(i.color.is_hdr())));
        rows.push((
            l.t("field"),
            match (src.interlaced(), p.shown.as_ref().map(|f| f.tff)) {
                (Some(true), Some(tff)) => format!("{} ({})", l.t("interlaced"), if tff { "TFF" } else { "BFF" }),
                (Some(true), None) => l.t("interlaced").to_string(),
                (Some(false), _) => l.t("progressive").to_string(),
                (None, _) => l.t("unknown").to_string(),
            },
        ));
        rows.push((l.t("alpha"), yn(i.has_alpha)));
        rows.push((l.t("audio"), i.audio.clone().unwrap_or_else(|| l.t("no_audio").to_string())));
        if let Some(tc) = &i.timecode {
            rows.push((l.t("timecode"), tc.clone()));
        }
        let map = src.cache_map();
        rows.push((l.t("frames_cached"), format!("{}", map.decoded.len())));
        let mut ram = map.ram;
        ram.sort_unstable();
        ram.dedup();
        rows.push((l.t("frames_ram"), format!("{}", ram.len())));
    } else {
        ui.label(RichText::new(l.t("no_file")).weak());
    }
    rows.push((l.t("gpu"), s.gpu_name.clone()));
    egui::Grid::new("info").num_columns(2).spacing(Vec2::new(16.0, 5.0)).striped(false).show(ui, |ui| {
        for (k, v) in rows {
            ui.label(RichText::new(k).color(Color32::from_gray(185)));
            ui.add(egui::Label::new(RichText::new(v).color(Color32::from_gray(235))).wrap());
            ui.end_row();
        }
    });
}

fn settings_contents(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    const COMBO_W: f32 = 270.0;
    section(ui, l.t("sec_general"));
    egui::Grid::new("settings_general").num_columns(2).spacing(Vec2::new(14.0, 8.0)).show(ui, |ui| {
        ui.label(RichText::new(l.t("language")).color(Color32::from_gray(185)));
        ui.horizontal(|ui| {
            for lang in [Lang::En, Lang::Pt] {
                if ui.selectable_value(&mut s.lang, lang, lang.label()).changed() {
                    s.settings.lang = lang;
                    s.settings_dirty = true;
                }
            }
        });
        ui.end_row();
        ui.label(RichText::new(l.t("positions")).color(Color32::from_gray(185)));
        ui.horizontal(|ui| {
            let mut frames = s.settings.show_frames;
            ui.selectable_value(&mut frames, false, l.t("timecode"));
            ui.selectable_value(&mut frames, true, l.t("frames"));
            if frames != s.settings.show_frames {
                s.settings.show_frames = frames;
                s.settings_dirty = true;
            }
        });
        ui.end_row();
    });
    ui.add_space(6.0);

    crate::color_ui::settings_section(ui, s);
    ui.add_space(6.0);

    section(ui, l.t("sec_devices"));
    egui::Grid::new("settings_devices").num_columns(2).spacing(Vec2::new(14.0, 8.0)).show(ui, |ui| {
        // GPU.
        ui.label(RichText::new(l.t("gpu")).color(Color32::from_gray(185))).on_hover_text(l.t("gpu_hint"));
        let current = s.settings.gpu.clone();
        let label = current.clone().unwrap_or_else(|| format!("{} — {}", l.t("gpu_auto"), s.gpu_name));
        let mut choice: Option<Option<String>> = None;
        egui::ComboBox::from_id_salt("gpu_dev").selected_text(label).width(COMBO_W).show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), l.t("gpu_auto")).clicked() {
                choice = Some(None);
            }
            for (name, discrete) in &s.gpus {
                let kind = if *discrete { l.t("discrete") } else { l.t("integrated") };
                let text = format!("{name}  ·  {kind}");
                if ui.selectable_label(current.as_deref() == Some(name.as_str()), text).clicked() {
                    choice = Some(Some(name.clone()));
                }
            }
        });
        if let Some(c) = choice {
            if c != current {
                actions.push(Action::SetGpu(c));
            }
        }
        ui.end_row();

        // Audio.
        ui.label(RichText::new(l.t("audio_device")).color(Color32::from_gray(185)));
        let current = p.audio_device.clone();
        let label = current.clone().unwrap_or_else(|| l.t("default_device").to_string());
        let mut choice: Option<Option<String>> = None;
        egui::ComboBox::from_id_salt("audio_dev").selected_text(label).width(COMBO_W).show_ui(ui, |ui| {
            if ui.selectable_label(current.is_none(), l.t("default_device")).clicked() {
                choice = Some(None);
            }
            for d in &s.audio_devices {
                if ui.selectable_label(current.as_deref() == Some(d.as_str()), d).clicked() {
                    choice = Some(Some(d.clone()));
                }
            }
        });
        if let Some(c) = choice {
            if c != current {
                p.set_audio_device(c.clone());
                s.settings.audio_device = c;
                s.settings_dirty = true;
            }
        }
        ui.end_row();
    });
    ui.add_space(6.0);

    section(ui, l.t("sec_video"));
    egui::Grid::new("settings_video").num_columns(2).spacing(Vec2::new(14.0, 8.0)).show(ui, |ui| {
        ui.label(RichText::new(l.t("deinterlace")).color(Color32::from_gray(185)));
        let modes = [
            (DeintMode::Auto, l.t("deint_auto")),
            (DeintMode::Off, l.t("deint_off")),
            (DeintMode::Best, l.t("deint_best")),
            (DeintMode::Nnedi, "NNEDI3"),
            (DeintMode::Bwdif, "bwdif"),
        ];
        let current = modes.iter().find(|m| m.0 == p.deint_mode).map(|m| m.1).unwrap_or("");
        egui::ComboBox::from_id_salt("deint").selected_text(current).width(COMBO_W).show_ui(ui, |ui| {
            for (m, label) in modes {
                ui.selectable_value(&mut p.deint_mode, m, label);
            }
        });
        ui.end_row();
        ui.label(RichText::new(l.t("deint_output")).color(Color32::from_gray(185)));
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.field_rate, true, l.t("field_rate"));
            ui.selectable_value(&mut p.field_rate, false, l.t("frame_rate_out"));
        });
        ui.end_row();

        // Frame cache.
        ui.label(RichText::new(l.t("frame_cache")).color(Color32::from_gray(185))).on_hover_text(l.t("frame_cache_hint"));
        ui.horizontal(|ui| {
            let total = brp_media::mem::total_ram().unwrap_or(8 << 30) as f64;
            let gb = |b: f64| b / (1u64 << 30) as f64;
            let mut v = gb(brp_media::mem::cache_budget() as f64);
            let max = (gb(total) * 0.9).floor().max(1.0);
            ui.spacing_mut().slider_width = 150.0;
            let r = ui.add(egui::Slider::new(&mut v, 0.5..=max).step_by(0.5).suffix(" GB").max_decimals(1)).on_hover_text(l.t("frame_cache_hint"));
            if r.changed() {
                let bytes = (v * (1u64 << 30) as f64) as u64;
                brp_media::mem::set_cache_budget(bytes as usize);
                s.settings.cache_bytes = Some(bytes);
                s.settings_dirty = true;
            }
            ui.label(RichText::new(format!("{:.0}% {}", v / gb(total) * 100.0, l.t("of_ram"))).small().color(Color32::from_gray(130)));
            if s.settings.cache_bytes.is_some() && ui.small_button(l.t("default")).clicked() {
                brp_media::mem::set_cache_budget(brp_media::mem::default_cache_budget());
                s.settings.cache_bytes = None;
                s.settings_dirty = true;
            }
        });
        ui.end_row();
    });
    // About (the license asks forks to keep this attribution).
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(concat!("BRP Player ", env!("CARGO_PKG_VERSION"), " · brurpo")).small().color(Color32::from_gray(130)));
        ui.hyperlink_to(RichText::new("github.com/brurpo/brp-player-2").small(), "https://github.com/brurpo/brp-player-2");
    });
    // License terms (LICENSE.md): free under US$1M a year in revenue.
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(l.t("license_free")).small().color(Color32::from_gray(130)));
        ui.hyperlink_to(RichText::new("contact@brpapp.com").small(), "mailto:contact@brpapp.com");
    });
}

fn effects_contents(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    use crate::effects::{clean_desc, param_views, ParamView};
    use brp_render::usershader::ParamKind;
    let l = s.lang;
    // Toolbar.
    ui.horizontal(|ui| {
        // All effects on/off (E): the stack stays as it is.
        let mut on = !p.fx.bypass;
        if on_off_switch(ui, &mut on).on_hover_text(l.t("fx_all_tip")).clicked() {
            p.fx.set_bypass(!on);
        }
        // "+ Add effect" opens a list right here in the window (not a dropdown: the
        // window grows with it like with any other content).
        let open = s.fx_add_open;
        let label = RichText::new(format!("+  {}", l.t("fx_add"))).color(Color32::WHITE);
        let r = ui.add(egui::Button::new(label).selected(open).min_size(vec2(170.0, 0.0)));
        let chevron = pos2(r.rect.right() - 14.0, r.rect.center().y);
        if open { paint_up_icon(ui.painter(), chevron, Color32::WHITE) } else { paint_down_icon(ui.painter(), chevron, Color32::from_gray(200)) }
        if r.clicked() {
            s.fx_add_open = !open;
        }
        ui.menu_button(l.t("fx_presets"), |ui| {
            if ui.button(l.t("fx_save")).clicked() {
                actions.push(Action::FxSavePreset);
                ui.close();
            }
            if ui.button(l.t("fx_load")).clicked() {
                actions.push(Action::FxLoadPreset);
                ui.close();
            }
        });
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.add_enabled(!p.fx.list.is_empty(), egui::Button::new(l.t("fx_clear"))).clicked() {
                p.fx.clear();
            }
            let (r, _) = ui.allocate_exact_size(vec2(28.0, 26.0), Sense::hover());
            if row_button(ui, r, Id::new("fx_redo"), &format!("{} (Ctrl+Y)", l.t("redo")), p.fx.can_redo(), paint_redo_icon).clicked() {
                p.fx.redo();
            }
            let (r, _) = ui.allocate_exact_size(vec2(28.0, 26.0), Sense::hover());
            if row_button(ui, r, Id::new("fx_undo"), &format!("{} (Ctrl+Z)", l.t("undo")), p.fx.can_undo(), paint_undo_icon).clicked() {
                p.fx.undo();
            }
        });
    });
    ui.add_space(4.0);

    if s.fx_add_open {
        add_effect_list(ui, p, s, actions);
        ui.add_space(4.0);
    }

    if crate::warm::progress().is_some() {
        ui.horizontal(|ui| shaders_note(ui, l));
    }
    if p.fx.list.is_empty() {
        ui.add_space(6.0);
        ui.label(RichText::new(l.t("fx_empty")).color(Color32::from_gray(140)));
        ui.add_space(4.0);
        return;
    }
    // Chain. action: (id, -1 up / +1 down / 0 remove / 2 toggle)
    let mut action: Option<(u64, i32)> = None;
    // Entries up to the last enabled AI effect are processed in the background:
    // their status is the background work's.
    let background = if p.fx.bypass { 0 } else { p.fx.list.iter().rposition(|e| e.ai.is_some() && e.spec.enabled).map_or(0, |i| i + 1) };
    if p.fx.bypass {
        ui.label(RichText::new(l.t("fx_all_off")).small().color(Color32::from_rgb(0xe8, 0xa3, 0x3d)));
        ui.add_space(2.0);
    }
    let ai_missing = brp_ai::check_models().err().map(|e| format!("{}\n{e:#}", l.t("ai_missing")));
    let n_fx = p.fx.list.len();
    let mut rows: Vec<(u64, Rect)> = Vec::new();
    for (pos, e) in p.fx.list.iter().enumerate() {
        let id = e.spec.id;
        let mut status = s.fx_status.iter().find(|(i, _)| *i == id).map(|(_, st)| st.clone());
        if pos < background && e.spec.enabled {
            status = match (&ai_missing, &p.ai.error) {
                (Some(m), _) if e.ai.is_some() => Some(brp_render::EffectStatus::Error(m.clone())),
                (_, Some(err)) => Some(brp_render::EffectStatus::Error(err.clone())),
                _ if p.ai.busy() => Some(brp_render::EffectStatus::Compiling),
                _ => None,
            };
        }
        let selected = p.fx.selected == Some(id);
        let dragged = s.fx_drag == Some(id);
        // Each effect as a card of fixed layout: grip, on/off, name, status, then the
        // up, down and remove buttons at the right (the name is cut to fit).
        let (row, _) = ui.allocate_exact_size(vec2(ui.available_width(), 30.0), Sense::hover());
        rows.push((id, row));
        let fill = if selected { ACCENT.gamma_multiply(0.14) } else { Color32::from_white_alpha(8) };
        let stroke = if selected { Stroke::new(1.0, ACCENT.gamma_multiply(0.6)) } else { Stroke::new(1.0, Color32::from_white_alpha(14)) };
        let card = row.shrink2(vec2(0.0, 1.5));
        ui.painter().rect(card, 8.0, if dragged { fill.gamma_multiply(0.4) } else { fill }, stroke, StrokeKind::Inside);

        // Grip: drag to reorder.
        let grip = Rect::from_min_size(card.min, vec2(20.0, card.height()));
        let g = ui.interact(grip, Id::new(("fx_grip", id)), Sense::drag()).on_hover_text(l.t("fx_drag"));
        if g.hovered() || g.dragged() {
            ui.ctx().set_cursor_icon(if g.dragged() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
        }
        if g.drag_started() {
            s.fx_drag = Some(id);
        }
        let gc = if g.hovered() || dragged { Color32::from_gray(220) } else { Color32::from_gray(120) };
        for (dx, dy) in [(-2.5, -5.0), (2.5, -5.0), (-2.5, 0.0), (2.5, 0.0), (-2.5, 5.0), (2.5, 5.0)] {
            ui.painter().circle_filled(grip.center() + vec2(dx, dy), 1.3, gc);
        }

        // Right side: remove, down, up (drawn icons), then the status.
        let bw = 24.0;
        let btn = |k: f32| Rect::from_center_size(pos2(card.right() - 4.0 - bw * (k + 0.5), card.center().y), vec2(bw - 2.0, 22.0));
        let last = pos + 1 == n_fx;
        if row_button(ui, btn(0.0), Id::new(("fx_rm", id)), l.t("fx_remove"), true, paint_x_icon).clicked() {
            action = Some((id, 0));
        }
        if row_button(ui, btn(1.0), Id::new(("fx_dn", id)), l.t("fx_down"), !last, paint_down_icon).clicked() {
            action = Some((id, 1));
        }
        if row_button(ui, btn(2.0), Id::new(("fx_up", id)), l.t("fx_up"), pos > 0, paint_up_icon).clicked() {
            action = Some((id, -1));
        }
        let status_rect = Rect::from_center_size(pos2(card.right() - 4.0 - bw * 3.0 - 12.0, card.center().y), vec2(20.0, 20.0));
        match &status {
            Some(brp_render::EffectStatus::Compiling) => {
                ui.put(status_rect, egui::Spinner::new().size(12.0)).on_hover_text(if pos < background { l.t("ai_working") } else { l.t("fx_compiling") });
            }
            Some(brp_render::EffectStatus::Error(msg)) => {
                let r = ui.interact(status_rect, Id::new(("fx_err", id)), Sense::hover());
                paint_warning_icon(ui.painter(), status_rect.center(), OUTSIDE);
                r.on_hover_text(msg);
            }
            _ => {}
        }

        // On/off and the name.
        let check = Rect::from_min_size(pos2(grip.right(), card.top()), vec2(24.0, card.height()));
        let mut en = e.spec.enabled;
        if ui.put(check, egui::Checkbox::without_text(&mut en)).changed() {
            action = Some((id, 2));
        }
        let name = Rect::from_min_max(pos2(check.right() + 2.0, card.top() + 3.0), pos2(status_rect.left() - 4.0, card.bottom() - 3.0));
        let mut text = RichText::new(e.title(l));
        if !e.spec.enabled || p.fx.bypass {
            text = text.weak();
        } else if e.ai.is_some() {
            text = text.color(ACCENT.lerp_to_gamma(Color32::WHITE, 0.45));
        }
        // The name, left-aligned and cut with an ellipsis; click selects it.
        let nr = ui.interact(name, Id::new(("fx_name", id)), Sense::click());
        if selected {
            ui.painter().rect_filled(name, 6.0, ACCENT.gamma_multiply(0.8));
        } else if nr.hovered() {
            ui.painter().rect_filled(name, 6.0, Color32::from_white_alpha(16));
        }
        let galley = egui::WidgetText::from(text).into_galley(ui, Some(egui::TextWrapMode::Truncate), (name.width() - 16.0).max(10.0), egui::TextStyle::Button);
        let fallback = if selected { Color32::WHITE } else { Color32::from_gray(220) };
        ui.painter().galley(pos2(name.left() + 8.0, name.center().y - galley.size().y / 2.0), galley, fallback);
        if nr.clicked() {
            p.fx.selected = Some(id);
        }
    }
    // Reordering: where the dragged effect would land, and the drop.
    if let Some(drag_id) = s.fx_drag {
        let (pointer, released) = ui.input(|i| (i.pointer.latest_pos(), i.pointer.any_released()));
        if let Some(py) = pointer.map(|p| p.y) {
            let to = rows.iter().filter(|(_, r)| r.center().y < py).count();
            let y = match rows.get(to) {
                Some((_, r)) => r.top(),
                None => rows.last().map_or(0.0, |(_, r)| r.bottom()),
            };
            if let Some((_, r)) = rows.first() {
                let line = Rect::from_x_y_ranges(r.left() + 4.0..=r.right() - 4.0, y - 1.0..=y + 1.0);
                ui.painter().rect_filled(line, 1.0, ACCENT);
                ui.painter().circle_filled(pos2(line.left(), y), 3.0, ACCENT);
            }
            if released {
                p.fx.move_to(drag_id, to);
            }
        }
        if released || !ui.input(|i| i.pointer.any_down()) {
            s.fx_drag = None;
        }
        ui.ctx().request_repaint();
    }
    match action {
        Some((id, 0)) => p.fx.remove(id),
        Some((id, 2)) => {
            p.fx.checkpoint();
            if let Some(e) = p.fx.list.iter_mut().find(|e| e.spec.id == id) {
                e.spec.enabled = !e.spec.enabled;
            }
            p.fx.dirty = true;
        }
        Some((id, d)) => p.fx.move_by(id, d as isize),
        None => {}
    }

    // Parameters of the selected effect.
    let Some(sel) = p.fx.selected else { return };
    let Some(idx) = p.fx.list.iter().position(|e| e.spec.id == sel) else { return };
    let file = p.fx.list[idx].spec.file.clone();
    if file.params.is_empty() {
        return;
    }
    ui.add_space(6.0);
    section(ui, &p.fx.list[idx].title(l));
    let before = p.fx.list.clone();
    let mut changed = false;
    let mut started = false;
    // A fixed identity for the parameters: widgets get theirs from what was drawn
    // before them, and the effects list above changes (a busy spinner comes and goes
    // as a paused frame is computed). A slider whose identity changes is let go of
    // in the middle of a drag.
    ui.scope_builder(egui::UiBuilder::new().id(Id::new(("fx_params_area", sel))), |ui| {
        egui::ScrollArea::vertical().id_salt("fx_params").max_height(420.0).show(ui, |ui| {
            let ai_kind = p.fx.list[idx].ai;
            let text = |i: usize| crate::effects::param_text(&file, ai_kind, i, l);
            for (cat, views) in param_views(&file, &text) {
                let mut body = |ui: &mut egui::Ui| {
                    egui::Grid::new(("fx_params", sel, cat.clone())).num_columns(2).spacing(Vec2::new(10.0, 6.0)).show(ui, |ui| {
                        let ai = p.fx.list[idx].ai;
                        for v in &views {
                            let values = &mut p.fx.list[idx].spec.params;
                            if let (Some(k), ParamView::Single(i)) = (ai, v) {
                                if !k.param_shown(*i, values) {
                                    continue;
                                }
                            }
                            match v {
                                ParamView::Single(i) => {
                                    let pd = &file.params[*i];
                                    let (label, desc) = text(*i);
                                    ui.label(RichText::new(&label).color(Color32::from_gray(170))).on_hover_text(clean_desc(&desc));
                                    let (lo, hi) = (pd.min.unwrap_or(0.0), pd.max.unwrap_or(1.0));
                                    let options = crate::effects::param_options(&desc);
                                    let r = match pd.kind {
                                        _ if options.is_some() => {
                                            let options = options.unwrap();
                                            let mut n = (values[*i].round() as usize).min(options.len().saturating_sub(1));
                                            let r = egui::ComboBox::from_id_salt(("fx_opt", sel, *i))
                                                .selected_text(options.get(n).cloned().unwrap_or_default())
                                                .width(200.0)
                                                .show_ui(ui, |ui| {
                                                    for (k, o) in options.iter().enumerate() {
                                                        ui.selectable_value(&mut n, k, o);
                                                    }
                                                })
                                                .response;
                                            let picked = n as f64 != values[*i];
                                            values[*i] = n as f64;
                                            if picked {
                                                started = true;
                                                changed = true;
                                            }
                                            r
                                        }
                                        ParamKind::Float => ui.add(egui::Slider::new(&mut values[*i], lo..=hi).max_decimals(3)),
                                        _ if lo == 0.0 && hi == 1.0 => {
                                            let mut b = values[*i] >= 0.5;
                                            let r = ui.checkbox(&mut b, "");
                                            values[*i] = b as i32 as f64;
                                            r
                                        }
                                        _ => {
                                            let mut n = values[*i].round() as i64;
                                            let r = ui.add(egui::Slider::new(&mut n, lo as i64..=hi as i64));
                                            values[*i] = n as f64;
                                            r
                                        }
                                    };
                                    let r = r.on_hover_text(clean_desc(&desc));
                                    started |= r.drag_started() || (r.changed() && !r.dragged());
                                    changed |= r.changed();
                                }
                                ParamView::Color { label, idx: [r, g, b] } => {
                                    ui.label(RichText::new(label).color(Color32::from_gray(170))).on_hover_text(clean_desc(&text(*r).1));
                                    let mut c = [values[*r] as f32, values[*g] as f32, values[*b] as f32];
                                    let resp = ui.color_edit_button_rgb(&mut c);
                                    if resp.changed() {
                                        values[*r] = c[0] as f64;
                                        values[*g] = c[1] as f64;
                                        values[*b] = c[2] as f64;
                                        changed = true;
                                    }
                                    started |= resp.clicked();
                                }
                            }
                            ui.end_row();
                        }
                    });
                };
                match &cat {
                    Some(c) => {
                        egui::CollapsingHeader::new(c).id_salt(("fx_cat", sel, c.clone())).default_open(true).show(ui, &mut body);
                    }
                    None => body(ui),
                }
            }
            ui.add_space(4.0);
            if ui.button(l.t("fx_reset")).clicked() {
                p.fx.list[idx].spec.params = file.params.iter().map(|q| q.default).collect();
                started = true;
                changed = true;
            }
        });
    });
    if started {
        // One undo step per drag / click, holding the state before the change.
        let after = std::mem::replace(&mut p.fx.list, before);
        p.fx.checkpoint();
        p.fx.list = after;
    }
    if changed {
        p.fx.dirty = true;
    }
}

/// GPU and audio output pickers in the bottom bar: a chip each, a menu on click.
fn devices(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    let gpu_short = short_gpu_name(&s.gpu_name);
    let audio_name = p.audio_device.clone().unwrap_or_else(|| l.t("default_device").to_string());
    let audio_short = shorten(&audio_name, 22);
    // Names when both fit, icons only otherwise.
    let need = [&gpu_short, &audio_short].iter().map(|t| ui.painter().layout_no_wrap(t.to_string(), FontId::proportional(12.0), Color32::WHITE).size().x + 40.0).sum::<f32>();
    let compact = ui.available_width() < need + 8.0;
    let gpu = device_chip(ui, paint_gpu_icon, &gpu_short, &format!("{}: {}", l.t("gpu"), s.gpu_name), compact);
    egui::Popup::menu(&gpu).show(|ui| {
        ui.set_min_width(240.0);
        ui.label(RichText::new(l.t("gpu")).small().color(Color32::from_gray(140)));
        let current = s.settings.gpu.clone();
        if ui.selectable_label(current.is_none(), format!("{} — {}", l.t("gpu_auto"), short_gpu_name(&s.gpu_name))).clicked() && current.is_some() {
            actions.push(Action::SetGpu(None));
        }
        for (name, discrete) in &s.gpus {
            let kind = if *discrete { l.t("discrete") } else { l.t("integrated") };
            if ui.selectable_label(current.as_deref() == Some(name.as_str()), format!("{name}  ·  {kind}")).clicked() && current.as_deref() != Some(name.as_str()) {
                actions.push(Action::SetGpu(Some(name.clone())));
            }
        }
    });

    let audio = device_chip(ui, paint_speaker_icon, &audio_short, &format!("{}: {audio_name}", l.t("audio_device")), compact);
    if audio.clicked() {
        // Listed when the menu opens (devices come and go).
        s.audio_devices = brp_media::output_devices();
    }
    egui::Popup::menu(&audio).show(|ui| {
        ui.set_min_width(240.0);
        ui.label(RichText::new(l.t("audio_device")).small().color(Color32::from_gray(140)));
        let current = p.audio_device.clone();
        let mut choice: Option<Option<String>> = None;
        if ui.selectable_label(current.is_none(), l.t("default_device")).clicked() {
            choice = Some(None);
        }
        for d in &s.audio_devices {
            if ui.selectable_label(current.as_deref() == Some(d.as_str()), d).clicked() {
                choice = Some(Some(d.clone()));
            }
        }
        if let Some(c) = choice.filter(|c| *c != current) {
            p.set_audio_device(c.clone());
            s.settings.audio_device = c;
            s.settings_dirty = true;
        }
    });
}

/// "NVIDIA GeForce RTX 5070 Ti" -> "RTX 5070 Ti".
fn short_gpu_name(name: &str) -> String {
    let mut n = name.to_string();
    for p in ["NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel(R) ", "Intel "] {
        if let Some(rest) = n.strip_prefix(p) {
            n = rest.to_string();
            break;
        }
    }
    shorten(&n, 22)
}

fn shorten(s: &str, max: usize) -> String {
    if s.chars().count() <= max { s.to_string() } else { format!("{}…", s.chars().take(max - 1).collect::<String>()) }
}

/// A small rounded chip with an icon and a name (opens a menu).
fn device_chip(ui: &mut egui::Ui, icon: fn(&egui::Painter, Pos2, Color32), text: &str, tip: &str, compact: bool) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(if compact { String::new() } else { text.to_string() }, FontId::proportional(12.0), Color32::WHITE);
    let size = if compact { vec2(28.0, 26.0) } else { vec2(galley.size().x + 34.0, 26.0) };
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let hover = ui.ctx().animate_bool_with_time(resp.id.with("h"), resp.hovered(), 0.12);
    let p = ui.painter();
    p.rect_filled(rect, 8.0, Color32::from_white_alpha((10.0 + 16.0 * hover) as u8));
    p.rect_stroke(rect, 8.0, Stroke::new(1.0, Color32::from_white_alpha((18.0 + 20.0 * hover) as u8)), StrokeKind::Inside);
    let color = if hover > 0.5 { Color32::WHITE } else { Color32::from_gray(205) };
    icon(p, if compact { rect.center() } else { pos2(rect.left() + 14.0, rect.center().y) }, color);
    p.galley(pos2(rect.left() + 26.0, rect.center().y - galley.size().y / 2.0), galley, color);
    resp.on_hover_text(tip)
}

/// A chip with pins.
fn paint_gpu_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let body = Rect::from_center_size(c, vec2(9.0, 9.0));
    p.rect_stroke(body, 1.5, Stroke::new(1.3, color), StrokeKind::Inside);
    p.rect_filled(Rect::from_center_size(c, vec2(3.0, 3.0)), 0.5, color);
    for k in [-2.5, 0.0, 2.5] {
        p.line_segment([c + vec2(k, -4.5), c + vec2(k, -6.5)], Stroke::new(1.0, color));
        p.line_segment([c + vec2(k, 4.5), c + vec2(k, 6.5)], Stroke::new(1.0, color));
        p.line_segment([c + vec2(-4.5, k), c + vec2(-6.5, k)], Stroke::new(1.0, color));
        p.line_segment([c + vec2(4.5, k), c + vec2(6.5, k)], Stroke::new(1.0, color));
    }
}

/// A loudspeaker with one wave.
fn paint_speaker_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let pts = vec![c + vec2(-6.0, -2.2), c + vec2(-3.0, -2.2), c + vec2(1.0, -5.5), c + vec2(1.0, 5.5), c + vec2(-3.0, 2.2), c + vec2(-6.0, 2.2)];
    p.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
    let arc: Vec<Pos2> = (0..=8).map(|i| {
        let a = -0.9 + 1.8 * i as f32 / 8.0;
        c + vec2(2.0 + 4.0 * a.cos(), 4.0 * a.sin())
    }).collect();
    p.add(egui::Shape::line(arc, Stroke::new(1.3, color)));
}

/// A filled circle whose colour runs from `top` to `bottom`: a triangle fan with
/// per-vertex colours, plus a one-pixel ring fading to transparent at the edge (a
/// hand-built mesh gets none of egui's anti-aliasing).
fn gradient_disc(p: &egui::Painter, c: Pos2, r: f32, top: Color32, bottom: Color32) {
    let px = 1.0 / p.ctx().pixels_per_point();
    let at = |y: f32| top.lerp_to_gamma(bottom, ((y - (c.y - r)) / (2.0 * r)).clamp(0.0, 1.0));
    let (inner, outer) = (r - px * 0.5, r + px * 0.5);
    let n = 96u32;
    let mut m = egui::Mesh::default();
    m.colored_vertex(c, at(c.y));
    for i in 0..n {
        let a = std::f32::consts::TAU * i as f32 / n as f32;
        let d = vec2(a.cos(), a.sin());
        let q = c + d * inner;
        m.colored_vertex(q, at(q.y));
        m.colored_vertex(c + d * outer, Color32::TRANSPARENT);
    }
    for i in 0..n {
        let (a_in, a_out) = (1 + 2 * i, 2 + 2 * i);
        let (b_in, b_out) = (1 + 2 * ((i + 1) % n), 2 + 2 * ((i + 1) % n));
        m.add_triangle(0, a_in, b_in);
        m.add_triangle(a_in, a_out, b_out);
        m.add_triangle(a_in, b_out, b_in);
    }
    p.add(egui::Shape::mesh(m));
}

/// A small icon button at `rect` (dimmed and inert when disabled).
fn row_button(ui: &mut egui::Ui, rect: Rect, id: Id, tip: &str, enabled: bool, icon: fn(&egui::Painter, Pos2, Color32)) -> egui::Response {
    let resp = ui.interact(rect, id, if enabled { Sense::click() } else { Sense::hover() });
    let hover = enabled && resp.hovered();
    if hover {
        ui.painter().rect_filled(rect, 6.0, Color32::from_white_alpha(24));
    }
    let color = if !enabled {
        Color32::from_gray(80)
    } else if hover {
        Color32::WHITE
    } else {
        Color32::from_gray(190)
    };
    icon(ui.painter(), rect.center(), color);
    let resp = resp.on_hover_text(tip);
    if enabled { resp } else { resp.on_disabled_hover_text(tip) }
}

fn paint_up_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let s = Stroke::new(1.8, color);
    p.line_segment([c + vec2(-4.5, 2.5), c + vec2(0.0, -2.5)], s);
    p.line_segment([c + vec2(0.0, -2.5), c + vec2(4.5, 2.5)], s);
}

fn paint_down_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let s = Stroke::new(1.8, color);
    p.line_segment([c + vec2(-4.5, -2.5), c + vec2(0.0, 2.5)], s);
    p.line_segment([c + vec2(0.0, 2.5), c + vec2(4.5, -2.5)], s);
}

fn paint_x_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let s = Stroke::new(1.6, color);
    p.line_segment([c + vec2(-4.0, -4.0), c + vec2(4.0, 4.0)], s);
    p.line_segment([c + vec2(-4.0, 4.0), c + vec2(4.0, -4.0)], s);
}

fn paint_warning_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    let pts = vec![c + vec2(0.0, -7.0), c + vec2(7.5, 6.0), c + vec2(-7.5, 6.0)];
    p.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
    p.line_segment([c + vec2(0.0, -2.5), c + vec2(0.0, 1.8)], Stroke::new(1.6, Color32::from_gray(20)));
    p.circle_filled(c + vec2(0.0, 3.9), 0.9, Color32::from_gray(20));
}

/// A curved arrow: anticlockwise (undo) or clockwise (redo).
fn paint_turn_arrow(p: &egui::Painter, c: Pos2, color: Color32, redo: bool) {
    let k = if redo { -1.0 } else { 1.0 };
    let r = 5.5;
    let centre = c + vec2(0.0, 1.5);
    // Arc over the top, from the right side round to the left (mirrored for redo).
    let pts: Vec<Pos2> = (0..=12)
        .map(|i| {
            let a = std::f32::consts::PI * (i as f32 / 12.0);
            centre + vec2(k * r * a.cos(), -r * a.sin())
        })
        .collect();
    let end = *pts.last().unwrap();
    p.add(egui::Shape::line(pts, Stroke::new(1.7, color)));
    // Arrowhead at the end, pointing down.
    let head = vec![end + vec2(-3.2, -0.5), end + vec2(3.2, -0.5), end + vec2(0.0, 3.8)];
    p.add(egui::Shape::convex_polygon(head, color, Stroke::NONE));
}

fn paint_undo_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    paint_turn_arrow(p, c, color, false);
}

fn paint_redo_icon(p: &egui::Painter, c: Pos2, color: Color32) {
    paint_turn_arrow(p, c, color, true);
}

/// A small-caps group heading inside a menu.
fn menu_heading(ui: &mut egui::Ui, text: &str, tip: &str) {
    ui.add_space(2.0);
    ui.label(RichText::new(text.to_uppercase()).size(10.5).strong().color(Color32::from_gray(150))).on_hover_text(tip);
}

/// A four-point sparkle with a small companion (marks AI effects).
fn paint_sparkle(p: &egui::Painter, c: Pos2, color: Color32) {
    let star = |at: Pos2, a: f32, b: f32| {
        p.add(egui::Shape::convex_polygon(vec![at + vec2(0.0, -a), at + vec2(b, 0.0), at + vec2(0.0, a), at + vec2(-b, 0.0)], color, Stroke::NONE));
        p.add(egui::Shape::convex_polygon(vec![at + vec2(-a, 0.0), at + vec2(0.0, -b), at + vec2(a, 0.0), at + vec2(0.0, b)], color, Stroke::NONE));
    };
    star(c + vec2(-1.0, 1.0), 5.5, 1.5);
    star(c + vec2(4.5, -4.0), 2.5, 0.8);
}

/// The list "+ Add effect" opens inside the Effects window: AI effects, then shader
/// effects, each group labelled; picking one adds it and closes the list.
fn add_effect_list(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    let mut ais: Vec<(String, crate::ai::AiKind)> = crate::ai::AiKind::ALL.into_iter().map(|k| (l.t(k.title_key()).to_string(), k)).collect();
    ais.sort_by_key(|(n, _)| n.to_lowercase());
    let mut shaders: Vec<(String, std::path::PathBuf)> = s.fx_library.iter().map(|f| (p.fx.title_of(f, l), f.clone())).collect();
    shaders.sort_by_key(|(n, _)| n.to_lowercase());
    let mut add: Option<std::path::PathBuf> = None;
    let mut add_ai: Option<crate::ai::AiKind> = None;
    Frame::new().fill(Color32::from_black_alpha(70)).stroke(Stroke::new(1.0, Color32::from_white_alpha(18))).corner_radius(8.0).inner_margin(Margin::same(6)).show(ui, |ui| {
        egui::ScrollArea::vertical().max_height(340.0).auto_shrink([false, true]).show(ui, |ui| {
            menu_heading(ui, l.t("fx_group_ai"), l.t("fx_ai_tip"));
            for (name, k) in ais {
                let r = ui.add(egui::Button::selectable(false, format!("      {name}")).min_size(vec2(ui.available_width(), 0.0))).on_hover_text(l.t("fx_ai_tip"));
                // A sparkle in front of the name marks the AI effects.
                paint_sparkle(ui.painter(), pos2(r.rect.left() + 13.0, r.rect.center().y), ACCENT.lerp_to_gamma(Color32::WHITE, 0.35));
                if r.clicked() {
                    add_ai = Some(k);
                }
            }
            ui.add_space(4.0);
            menu_heading(ui, l.t("fx_group_shaders"), l.t("fx_shader_tip"));
            for (name, path) in shaders {
                if ui.add(egui::Button::selectable(false, name).min_size(vec2(ui.available_width(), 0.0))).on_hover_text(l.t("fx_shader_tip")).clicked() {
                    add = Some(path);
                }
            }
            ui.separator();
            if ui.add(egui::Button::selectable(false, l.t("fx_browse")).min_size(vec2(ui.available_width(), 0.0))).clicked() {
                actions.push(Action::FxBrowse);
                s.fx_add_open = false;
            }
        });
    });
    if let Some(path) = add {
        if let Err(e) = p.fx.add(&path) {
            log::error!("add effect: {e:#}");
        }
        s.fx_add_open = false;
    }
    if let Some(k) = add_ai {
        p.fx.add_ai(k);
        s.fx_add_open = false;
    }
}

impl UiState {
    /// What an export takes of OpenColorIO (None: off, or its configs not loaded yet).
    pub fn ocio_job(&self) -> Option<crate::export::OcioJob> {
        if !self.ocio.enabled {
            return None;
        }
        let lib = self.ocio.loaded()?;
        Some(crate::export::OcioJob { lib, choices: self.ocio.choices(), settings: self.settings.clone() })
    }
}
