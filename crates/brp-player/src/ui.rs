//! The interface: a full-window viewer with overlay controls that fade during playback.

use crate::i18n::Lang;
use crate::icons::{self, Icons};
use crate::player::{DeintMode, Player};
use egui::{Align, Align2, Color32, CornerRadius, Frame, Key, Layout, Margin, RichText, Sense, Stroke, Vec2};
use std::time::{Duration, Instant};

pub const ACCENT: Color32 = Color32::from_rgb(0x4f, 0x8c, 0xff);
const BAR_BG: Color32 = Color32::from_rgba_premultiplied(14, 14, 16, 215);
const HIDE_AFTER: Duration = Duration::from_millis(2200);
const CLICK_DELAY: Duration = Duration::from_millis(220);

pub enum Action {
    Open,
    ToggleFullscreen,
    ExitFullscreen,
}

pub struct UiState {
    pub lang: Lang,
    pub show_info: bool,
    pub show_settings: bool,
    /// Output devices, listed when Settings opens.
    audio_devices: Vec<String>,
    pub locked: bool,
    pub last_activity: Instant,
    pub frame_text: String,
    pub fps_text: String,
    pub gpu_name: String,
    pub fullscreen: bool,
    scrub_resume: Option<bool>,
    pending_click: Option<Instant>,
    last_pointer: Option<egui::Pos2>,
    pub icons: Icons,
}

impl UiState {
    pub fn new(gpu_name: String, ctx: &egui::Context) -> Self {
        UiState {
            icons: Icons::load(ctx),
            lang: Lang::default(),
            show_info: false,
            show_settings: false,
            audio_devices: Vec::new(),
            locked: false,
            last_activity: Instant::now(),
            frame_text: String::new(),
            fps_text: String::new(),
            gpu_name,
            fullscreen: false,
            scrub_resume: None,
            pending_click: None,
            last_pointer: None,
        }
    }

    /// True when a delayed single click is pending (needs a repaint to resolve).
    pub fn has_pending_click(&self) -> bool {
        self.pending_click.is_some()
    }
}

pub fn setup_style(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = BAR_BG;
    v.window_fill = Color32::from_rgba_premultiplied(20, 20, 23, 235);
    v.window_stroke = Stroke::new(1.0, Color32::from_white_alpha(18));
    v.window_corner_radius = CornerRadius::same(10);
    v.selection.bg_fill = ACCENT;
    v.widgets.hovered.corner_radius = CornerRadius::same(6);
    v.widgets.inactive.corner_radius = CornerRadius::same(6);
    v.widgets.active.corner_radius = CornerRadius::same(6);
    v.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    ctx.set_visuals(v);
    ctx.all_styles_mut(|s| {
        s.spacing.button_padding = Vec2::new(8.0, 4.0);
        s.spacing.item_spacing = Vec2::new(6.0, 6.0);
    });
}

fn button_frame(ui: &mut egui::Ui, tip: &str, active: bool) -> (egui::Rect, egui::Response, Color32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(32.0, 28.0), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(rect, 6.0, Color32::from_white_alpha(16));
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

/// Button showing one of the icons from /icons, tinted (accent when active).
fn icon_button(ui: &mut egui::Ui, icons: &Icons, name: &str, tip: &str, active: bool) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, active);
    match icons.get(name) {
        Some(t) => {
            let r = egui::Rect::from_center_size(rect.center(), Vec2::splat(19.0));
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(t.id(), r, uv, color);
        }
        None => {
            ui.painter().text(rect.center(), Align2::CENTER_CENTER, name, egui::FontId::proportional(14.0), color);
        }
    }
    resp
}

/// Small text button in the same style (In / Out markers).
fn text_button(ui: &mut egui::Ui, text: &str, tip: &str) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, false);
    ui.painter().text(rect.center(), Align2::CENTER_CENTER, text, egui::FontId::proportional(15.0), color);
    resp
}

fn lock_button(ui: &mut egui::Ui, locked: bool, tip: &str) -> egui::Response {
    let (rect, resp, color) = button_frame(ui, tip, locked);
    icons::paint_lock(ui.painter(), egui::Rect::from_center_size(rect.center(), Vec2::splat(18.0)), locked, color);
    resp
}

/// Draws the UI. `px` is the window size in physical pixels.
pub fn draw(root: &mut egui::Ui, p: &mut Player, s: &mut UiState, px: [f32; 2]) -> Vec<Action> {
    let ctx = &root.ctx().clone();
    let mut actions = Vec::new();
    let l = s.lang;
    let ppp = ctx.pixels_per_point();

    // Activity tracking for auto-hide.
    let pointer = ctx.input(|i| i.pointer.latest_pos());
    if pointer != s.last_pointer || ctx.input(|i| !i.events.is_empty() && i.pointer.any_down()) {
        s.last_activity = Instant::now();
        s.last_pointer = pointer;
    }
    let idle = s.last_activity.elapsed() > HIDE_AFTER;
    let show_controls = !(p.playing && idle) || p.source.is_none();
    let fade = ctx.animate_bool_with_time(egui::Id::new("controls"), show_controls, 0.18);
    ctx.set_cursor_icon(if show_controls { egui::CursorIcon::Default } else { egui::CursorIcon::None });

    keyboard(ctx, p, s, &mut actions);

    // ---- Top bar ----
    if fade > 0.0 {
        egui::Panel::top("top")
            .frame(Frame::new().fill(BAR_BG.gamma_multiply(fade)).inner_margin(Margin::symmetric(10, 6)))
            .show_separator_line(false)
            .show(root, |ui| {
                ui.set_opacity(fade);
                ui.horizontal(|ui| {
                    let title = p.source.as_ref().map(|s| s.info.file_name.clone()).unwrap_or_else(|| "BRP Player".into());
                    ui.label(RichText::new(title).strong().color(Color32::from_gray(235)));
                    if let Some(src) = &p.source {
                        if !src.index_ready() {
                            ui.label(RichText::new(l.t("indexing")).small().weak());
                        }
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icon_button(ui, &s.icons, "folder_expanded", l.t("open"), false).clicked() {
                            actions.push(Action::Open);
                        }
                        if icon_button(ui, &s.icons, "config", l.t("settings"), s.show_settings).clicked() {
                            s.show_settings = !s.show_settings;
                            if s.show_settings {
                                s.audio_devices = brp_media::output_devices();
                            }
                        }
                        if icon_button(ui, &s.icons, "video", l.t("info"), s.show_info).clicked() {
                            s.show_info = !s.show_info;
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
        egui::Panel::bottom("bottom")
            .frame(Frame::new().fill(BAR_BG.gamma_multiply(fade)).inner_margin(Margin::symmetric(12, 8)))
            .show_separator_line(false)
            .show(root, |ui| {
                ui.set_opacity(fade);
                timeline(ui, p, s);
                ui.add_space(2.0);
                controls_row(ui, p, s, &mut actions);
            });
    }

    // ---- Viewer (everything not covered by bars) ----
    egui::CentralPanel::default().frame(Frame::NONE).show(root, |ui| {
        let rect = ui.max_rect();
        let resp = ui.interact(rect, egui::Id::new("viewer"), Sense::click_and_drag());
        if p.source.is_none() {
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                p.error.as_deref().map(|e| format!("{}\n{e}", l.t("error_open"))).unwrap_or_else(|| l.t("drop_hint").to_string()),
                egui::FontId::proportional(18.0),
                Color32::from_gray(150),
            );
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
                    p.zoom_at((scroll / 240.0).clamp(-1.0, 1.0), [pos.x * ppp, pos.y * ppp], px[0], px[1]);
                }
            }
        }
    });

    if s.show_settings {
        settings_window(ctx, p, s);
    }
    if s.show_info {
        info_window(ctx, p, s);
    }
    actions
}

fn keyboard(ctx: &egui::Context, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    if ctx.egui_wants_keyboard_input() {
        return;
    }
    ctx.input(|i| {
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
        }
        if i.key_pressed(Key::O) && !i.modifiers.command {
            p.set_out();
        }
        if i.key_pressed(Key::O) && i.modifiers.command {
            actions.push(Action::Open);
        }
        if i.key_pressed(Key::M) {
            p.set_muted(!p.muted);
        }
        if i.key_pressed(Key::ArrowUp) {
            p.set_volume(p.volume + 0.05);
        }
        if i.key_pressed(Key::ArrowDown) {
            p.set_volume(p.volume - 0.05);
        }
        if i.key_pressed(Key::G) {
            p.toggle_one_to_one();
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
    });
}

fn timeline(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState) {
    let n = p.frame_count().max(1);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 18.0), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let track = egui::Rect::from_center_size(rect.center(), Vec2::new(rect.width(), 4.0));
    let x_of = |f: i64| track.left() + track.width() * (f as f32 + 0.5) / n as f32;
    painter.rect_filled(track, 2.0, Color32::from_white_alpha(30));

    // Decoded frames (cache) as a faint fill.
    if let Some(src) = &p.source {
        let mut cached = src.cached_frames();
        cached.sort_unstable();
        let mut i = 0;
        while i < cached.len() {
            let start = cached[i];
            let mut end = start;
            while i + 1 < cached.len() && cached[i + 1] == end + 1 {
                i += 1;
                end = cached[i];
            }
            let r = egui::Rect::from_x_y_ranges(
                track.left() + track.width() * start as f32 / n as f32..=track.left() + track.width() * (end + 1) as f32 / n as f32,
                track.y_range(),
            );
            painter.rect_filled(r, 0.0, Color32::from_white_alpha(55));
            i += 1;
        }
    }
    // In/Out range.
    if p.has_range() {
        let r = egui::Rect::from_x_y_ranges(
            track.left() + track.width() * p.in_point as f32 / n as f32..=track.left() + track.width() * (p.out_frame() + 1) as f32 / n as f32,
            track.y_range(),
        );
        painter.rect_filled(r, 2.0, ACCENT.gamma_multiply(0.55));
        for x in [r.left(), r.right()] {
            painter.line_segment([egui::pos2(x, rect.top() + 2.0), egui::pos2(x, rect.bottom() - 2.0)], Stroke::new(2.0, ACCENT));
        }
    }
    // Playhead.
    let x = x_of(p.current);
    painter.circle_filled(egui::pos2(x, rect.center().y), if resp.hovered() || resp.dragged() { 7.0 } else { 5.5 }, Color32::WHITE);

    if let Some(pos) = resp.interact_pointer_pos() {
        let f = (((pos.x - track.left()) / track.width()) * n as f32).floor() as i64;
        if resp.drag_started() || resp.clicked() {
            if s.scrub_resume.is_none() {
                s.scrub_resume = Some(p.playing);
            }
            p.pause();
        }
        p.seek(f);
    }
    if resp.drag_stopped() || resp.clicked() {
        if let Some(resume) = s.scrub_resume.take() {
            if resume {
                p.play();
            }
        }
    }
    if let Some(pos) = resp.hover_pos() {
        let f = (((pos.x - track.left()) / track.width()) * n as f32).floor().clamp(0.0, (n - 1) as f32) as i64;
        resp.on_hover_text_at_pointer(format!("{f}  ·  {}", p.timecode(f)));
    }
}

fn controls_row(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    ui.horizontal(|ui| {
        if icon_button(ui, &s.icons, "toStart", l.t("to_in"), false).clicked() {
            p.seek(p.in_point);
        }
        if icon_button(ui, &s.icons, "step_back", l.t("prev_frame"), false).clicked() {
            p.step(-1);
        }
        let (icon, tip) = if p.playing { ("pause", l.t("pause")) } else { ("play", l.t("play")) };
        if icon_button(ui, &s.icons, icon, tip, false).clicked() {
            p.toggle();
        }
        if icon_button(ui, &s.icons, "step_foward", l.t("next_frame"), false).clicked() {
            p.step(1);
        }
        if icon_button(ui, &s.icons, "toEnd", l.t("to_out"), false).clicked() {
            p.seek(p.out_frame());
        }
        ui.separator();

        // Frame number (editable) / total.
        let focused_id = egui::Id::new("frame_edit");
        let editing = ui.memory(|m| m.has_focus(focused_id));
        if !editing {
            s.frame_text = p.current.to_string();
        }
        let r = ui.add(
            egui::TextEdit::singleline(&mut s.frame_text)
                .id(focused_id)
                .desired_width(56.0)
                .horizontal_align(Align::Max)
                .font(egui::TextStyle::Monospace),
        );
        if r.lost_focus() {
            if let Ok(f) = s.frame_text.trim().parse::<i64>() {
                p.seek(f);
            }
        }
        ui.label(RichText::new(format!("/ {}", p.last_frame())).monospace().weak());
        ui.label(RichText::new(p.timecode(p.current)).monospace().color(Color32::from_gray(230)));
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
            let chip = egui::Button::new(RichText::new(text).small().color(ACCENT)).corner_radius(8.0).stroke(Stroke::new(1.0, ACCENT.gamma_multiply(0.5)));
            if ui.add(chip).on_hover_text(l.t("deint_hint")).clicked() {
                s.show_settings = true;
                s.audio_devices = brp_media::output_devices();
            }
        }
        ui.separator();

        if text_button(ui, "[", l.t("set_in")).clicked() {
            p.set_in();
        }
        if text_button(ui, "]", l.t("set_out")).clicked() {
            p.set_out();
        }
        if p.has_range() && icon_button(ui, &s.icons, "clear", l.t("clear_range"), false).clicked() {
            p.clear_range();
        }
        if icon_button(ui, &s.icons, "repeat_icon", l.t("loop"), p.loop_on).clicked() {
            p.loop_on = !p.loop_on;
        }

        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if icon_button(ui, &s.icons, "fullscreen_icon", l.t("fullscreen"), s.fullscreen).clicked() {
                actions.push(Action::ToggleFullscreen);
            }
            if icon_button(ui, &s.icons, "originalSize", l.t("one_to_one"), p.view.one_to_one).clicked() {
                p.toggle_one_to_one();
            }
            if icon_button(ui, &s.icons, "restoreZoom", l.t("fit"), false).clicked() {
                p.reset_view();
            }
            ui.separator();
            let fps_id = egui::Id::new("fps_edit");
            if !ui.memory(|m| m.has_focus(fps_id)) {
                s.fps_text = format!("{:.3}", p.native_fps() * p.speed);
            }
            let r = ui
                .add(egui::TextEdit::singleline(&mut s.fps_text).id(fps_id).desired_width(54.0).font(egui::TextStyle::Monospace))
                .on_hover_text(l.t("fps"));
            if r.lost_focus() {
                if let Ok(v) = s.fps_text.trim().replace(',', ".").parse::<f64>() {
                    p.set_fps(v);
                }
            }
            ui.label(RichText::new("fps").weak());
            if p.has_audio() {
                ui.separator();
                let mut v = p.volume;
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
        });
    });
}

fn info_window(ctx: &egui::Context, p: &Player, s: &mut UiState) {
    let l = s.lang;
    let mut open = true;
    egui::Window::new(l.t("info"))
        .open(&mut open)
        .anchor(Align2::RIGHT_TOP, Vec2::new(-12.0, 48.0))
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            egui::Grid::new("info").num_columns(2).spacing(Vec2::new(14.0, 4.0)).show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(RichText::new(k).weak());
                    ui.label(v);
                    ui.end_row();
                };
                if let Some(src) = &p.source {
                    let i = &src.info;
                    let yn = |b: bool| if b { l.t("yes") } else { l.t("no") }.to_string();
                    row(l.t("file"), i.file_name.clone());
                    row(l.t("container"), i.container.clone());
                    row(l.t("codec"), i.codec.clone());
                    row(l.t("resolution"), format!("{} × {}", i.width, i.height));
                    row(l.t("frame_rate"), format!("{:.3}", i.frame_rate));
                    row(l.t("frames"), format!("{}", src.frame_count()));
                    row(l.t("pixel_format"), format!("{} ({}-bit)", i.pix_fmt, i.bit_depth));
                    row(
                        l.t("color"),
                        format!(
                            "{:?} / {:?} / {}",
                            i.color.matrix,
                            i.color.transfer,
                            if i.color.full_range { "full" } else { "limited" }
                        ),
                    );
                    row(l.t("hdr"), yn(i.color.is_hdr()));
                    row(
                        l.t("field"),
                        match (src.interlaced(), p.shown.as_ref().map(|f| f.tff)) {
                            (Some(true), Some(tff)) => format!("{} ({})", l.t("interlaced"), if tff { "TFF" } else { "BFF" }),
                            (Some(true), None) => l.t("interlaced").to_string(),
                            (Some(false), _) => l.t("progressive").to_string(),
                            (None, _) => l.t("unknown").to_string(),
                        },
                    );
                    row(l.t("alpha"), yn(i.has_alpha));
                    row(l.t("audio"), i.audio.clone().unwrap_or_else(|| l.t("no_audio").to_string()));
                    if let Some(tc) = &i.timecode {
                        row(l.t("timecode"), tc.clone());
                    }
                }
                row(l.t("gpu"), s.gpu_name.clone());
            });
        });
    if !open {
        s.show_info = false;
    }
}

fn settings_window(ctx: &egui::Context, p: &mut Player, s: &mut UiState) {
    let l = s.lang;
    let mut open = true;
    egui::Window::new(l.t("settings"))
        .open(&mut open)
        .anchor(Align2::RIGHT_TOP, Vec2::new(-12.0, 48.0))
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            egui::Grid::new("settings").num_columns(2).spacing(Vec2::new(14.0, 8.0)).show(ui, |ui| {
                ui.label(RichText::new(l.t("language")).weak());
                ui.horizontal(|ui| {
                    for lang in [Lang::En, Lang::Pt] {
                        ui.selectable_value(&mut s.lang, lang, lang.label());
                    }
                });
                ui.end_row();

                ui.label(RichText::new(l.t("audio_device")).weak());
                let current = p.audio_device.clone();
                let label = current.clone().unwrap_or_else(|| l.t("default_device").to_string());
                let mut choice: Option<Option<String>> = None;
                egui::ComboBox::from_id_salt("audio_dev").selected_text(label).width(260.0).show_ui(ui, |ui| {
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
                        p.set_audio_device(c);
                    }
                }
                ui.end_row();

                ui.label(RichText::new(l.t("deinterlace")).weak());
                let modes = [
                    (DeintMode::Auto, l.t("deint_auto")),
                    (DeintMode::Off, l.t("deint_off")),
                    (DeintMode::Best, l.t("deint_best")),
                    (DeintMode::Nnedi, "NNEDI3"),
                    (DeintMode::Bwdif, "bwdif"),
                ];
                let current = modes.iter().find(|m| m.0 == p.deint_mode).map(|m| m.1).unwrap_or("");
                egui::ComboBox::from_id_salt("deint").selected_text(current).width(260.0).show_ui(ui, |ui| {
                    for (m, label) in modes {
                        ui.selectable_value(&mut p.deint_mode, m, label);
                    }
                });
                ui.end_row();
                ui.label(RichText::new(l.t("deint_output")).weak());
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut p.field_rate, true, l.t("field_rate"));
                    ui.selectable_value(&mut p.field_rate, false, l.t("frame_rate_out"));
                });
                ui.end_row();

                ui.label(RichText::new(l.t("gpu")).weak());
                ui.label(&s.gpu_name).on_hover_text(l.t("gpu_hint"));
                ui.end_row();
            });
        });
    if !open {
        s.show_settings = false;
    }
}
