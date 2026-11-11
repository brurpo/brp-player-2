//! The Export window and the Batch export window (glass popups, see `ui` and
//! `popup`).
//!
//! Both use the same settings panel (`settings_rows` and `advanced`): format,
//! quality, toggles that show only where the codec supports them (alpha, HDR, GPU
//! acceleration, audio), size, frame rate (progressive or interlaced, and how to
//! make the pictures the new rate needs), and a collapsed Advanced section with the
//! technical details. Batch export is its own workflow: the files on the left, the
//! panel on the right editing the selected files (or, with none selected, the
//! settings new files get), as the original app's batch dialog had it.

use crate::export::{self, Conform, ExportJob, ExportSettings, Fit, ItemState, Progress, Running, Size, State, OTHER};
use crate::i18n::Lang;
use crate::player::Player;
use crate::ui::{section, Action, UiState, ACCENT};
use egui::{Color32, RichText, Vec2};
use std::collections::HashSet;
use std::time::Duration;

/// Export state of a player window.
#[derive(Default)]
pub struct ExportUi {
    pub settings: ExportSettings,
    /// Presets saved by the user.
    pub presets: Vec<ExportSettings>,
    /// "Save preset" is open, with the name typed so far.
    saving: Option<String>,
    pub output: String,
    /// The source the output name was made for.
    output_for: String,
    /// Export only In..Out.
    pub use_range: bool,
    pub running: Option<Running>,
    /// How the last export ended.
    pub last: Option<Progress>,
    pub batch: export::Batch,
    /// Batch files selected (their ids); the panel edits these.
    batch_sel: HashSet<u64>,
    /// Last file clicked (shift-click selects up to it).
    batch_anchor: Option<u64>,
    /// Batch: apply the current effects.
    pub batch_effects: bool,
    /// Every encoder of FFmpeg (Advanced, "Other" codec), listed when first needed.
    all_encoders: Vec<brp_media::encode::EncoderInfo>,
    loaded: bool,
}

const SIZES: &[(u32, u32, &str)] = &[
    (3840, 2160, "3840 × 2160 (4K)"),
    (2560, 1440, "2560 × 1440"),
    (1920, 1080, "1920 × 1080 (HD)"),
    (1280, 720, "1280 × 720"),
    (720, 576, "720 × 576 (PAL)"),
    (720, 480, "720 × 480 (NTSC)"),
    (1080, 1920, "1080 × 1920 (vertical)"),
    (1080, 1080, "1080 × 1080 (square)"),
];

/// Frame rates offered: (frames per second, interlaced). "59.94i" is 29.97 frames of
/// two fields.
const RATES: &[((i32, i32), bool)] = &[
    ((24000, 1001), false),
    ((24, 1), false),
    ((25, 1), false),
    ((30000, 1001), false),
    ((30, 1), false),
    ((50, 1), false),
    ((60000, 1001), false),
    ((60, 1), false),
    ((25, 1), true),
    ((30000, 1001), true),
    ((30, 1), true),
    ((15, 1), false),
];

const AUDIO: &[(&str, &str)] = &[("aac", "AAC"), ("pcm_s16le", "PCM 16-bit"), ("pcm_s24le", "PCM 24-bit"), ("flac", "FLAC"), ("libopus", "Opus"), ("alac", "ALAC"), ("libmp3lame", "MP3")];

impl ExportUi {
    /// Settings and presets from the saved settings (once).
    pub fn load(&mut self, s: &crate::settings::Settings) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        self.presets = s.export_presets.iter().map(ExportSettings::from_json).collect();
        if let Some(v) = &s.export_last {
            self.settings = ExportSettings::from_json(v);
        }
        self.batch.out_dir = s.batch_dir.clone().map(Into::into);
        self.batch.defaults = s.batch_defaults.as_ref().map(ExportSettings::from_json).unwrap_or_default();
        self.batch_effects = s.batch_effects;
    }

    /// Progress of the export running from the Export window (0..1).
    pub fn export_fraction(&self) -> Option<f32> {
        self.running.as_ref().filter(|r| !r.ended()).map(|r| r.snapshot().fraction())
    }

    /// Progress of the batch (0..1) while it runs.
    pub fn batch_fraction(&self) -> Option<f32> {
        if !(self.batch.active || self.batch.running().is_some()) {
            return None;
        }
        let n = self.batch.items.len().max(1) as f32;
        let done = self.batch.items.iter().filter(|i| !matches!(i.state, ItemState::Waiting | ItemState::Running)).count() as f32;
        let cur = self.batch.running().map_or(0.0, |(_, p)| p.fraction());
        Some(((done + cur) / n).clamp(0.0, 1.0))
    }

    pub fn busy(&self) -> bool {
        self.export_fraction().is_some() || self.batch_fraction().is_some()
    }

    /// Called every frame of the player window: moves the batch on, collects a
    /// finished export.
    pub fn poll(&mut self, p: &Player, ctx: &egui::Context) {
        self.batch.poll(p.ai.ctx.as_ref());
        if self.running.as_ref().is_some_and(|r| r.ended()) {
            self.last = self.running.take().map(|r| r.snapshot());
        }
        if self.busy() {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
}

fn all_presets(e: &ExportUi) -> Vec<ExportSettings> {
    e.presets.iter().cloned().chain(export::builtin_presets()).collect()
}

fn label(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).color(Color32::from_gray(185)));
}

fn weak(text: impl Into<String>) -> RichText {
    RichText::new(text.into()).small().color(Color32::from_gray(140))
}

fn fmt_secs(s: f64) -> String {
    let s = s.max(0.0);
    if s >= 3600.0 {
        format!("{}:{:02}:{:02}", (s / 3600.0) as u64, (s / 60.0) as u64 % 60, s as u64 % 60)
    } else {
        format!("{}:{:02}", (s / 60.0) as u64, s as u64 % 60)
    }
}

/// An on/off chip.
fn toggle(ui: &mut egui::Ui, on: &mut bool, text: &str, tip: &str) -> bool {
    let r = ui.add(egui::Button::selectable(*on, RichText::new(if *on { format!("✔ {text}") } else { text.to_string() })).corner_radius(11.0)).on_hover_text(tip);
    if r.clicked() {
        *on = !*on;
        return true;
    }
    false
}

fn progress_line(ui: &mut egui::Ui, l: Lang, p: &Progress) {
    ui.add(egui::ProgressBar::new(p.fraction()).desired_height(14.0).corner_radius(7.0).fill(ACCENT).text(RichText::new(format!("{:.0}%", p.fraction() * 100.0)).small()));
    let state = match p.state {
        State::Starting | State::Indexing => l.t("ex_preparing").to_string(),
        State::Finishing => l.t("ex_finishing").to_string(),
        _ => {
            let eta = p.eta().map_or(String::new(), |e| format!(" · {} {}", fmt_secs(e), l.t("ex_left")));
            format!("{} / {} · {:.1} fps{eta}", p.done, p.total, p.fps)
        }
    };
    ui.label(weak(state));
}

/// What the panel knows about the source(s) it sets up.
struct PanelSrc {
    /// The source is HDR / has audio (None: several files, not known: shown where
    /// the codec supports it, applied where the file has it).
    hdr: Option<bool>,
    audio: Option<bool>,
    /// The effects' picture size, the stack's own frame rate and slow-motion factor
    /// (None: per file).
    size: Option<(u32, u32)>,
    stack_rate: Option<((i32, i32), bool)>,
    slow: i64,
}

impl PanelSrc {
    fn of_player(p: &Player) -> PanelSrc {
        let src = p.source.as_ref();
        let stages = p.fx.all_stages();
        let deint = p.ai_source_deint();
        let fields = if deint.is_some_and(|d| d.fields) { 2 } else { 1 };
        PanelSrc {
            hdr: src.map(|s| s.info.color.is_hdr()),
            audio: src.map(|s| s.info.audio.is_some()),
            size: p.fx_output.or(src.map(|s| (s.info.width, s.info.height))),
            stack_rate: src.map(|s| {
                let q = export::standard_rate(s.info.frame_rate_q);
                ((q.0 * fields, q.1), false)
            }),
            slow: crate::ai::slowdown_of(&stages),
        }
    }

    fn batch() -> PanelSrc {
        PanelSrc { hdr: None, audio: None, size: None, stack_rate: None, slow: 1 }
    }
}

/// Pictures per second the output needs: the frame rate, or the field rate.
fn picture_rate(r: (i32, i32), interlaced: bool) -> f64 {
    export::rate_f(r) * if interlaced { 2.0 } else { 1.0 }
}

/// What the export will make: size · rate · frames · length (· speed).
fn summary(p: &Player, e: &ExportUi) -> Option<String> {
    let src = p.source.as_ref()?;
    let s = &e.settings;
    let stages = p.fx.all_stages();
    let deint = p.ai_source_deint();
    let fields = if deint.is_some_and(|d| d.fields) { 2 } else { 1 };
    let subs = crate::ai::subs_of(&stages, deint);
    let slow = crate::ai::slowdown_of(&stages);
    let q = export::standard_rate(src.info.frame_rate_q);
    let stack_rate = (q.0 * fields, q.1);
    let (a, b) = if e.use_range && p.has_range() { (p.in_point, p.out_frame()) } else { (0, p.last_frame()) };
    let rate = s.fps.unwrap_or(stack_rate);
    let interlaced = s.interlaced && s.fps.is_some();
    let pr = picture_rate(rate, interlaced);
    let same = (pr - export::rate_f(stack_rate)).abs() < 1e-6;
    let (fw, fh) = p.fx_output.unwrap_or((src.info.width, src.info.height));
    let align = s.family().map_or((2, 2), |f| (f.align, if f.id == "cfhd" { 1 } else { f.align }));
    let g = export::geometry(s, fw, fh, src.info.dar, align);
    let src_secs = src.frame_end_time(b) - src.frame_time(a);
    let (pictures, secs) = if same || s.conform == Conform::Retime {
        let n = (b - a + 1) * subs;
        (n, n as f64 / pr)
    } else {
        let d = src_secs * slow as f64;
        ((d * pr).round() as i64, d)
    };
    let frames = if interlaced { (pictures + 1) / 2 } else { pictures };
    let speed = src_secs / secs.max(1e-9) * 100.0;
    let sp = if (speed - 100.0).abs() > 0.5 { format!(" · {speed:.0}%") } else { String::new() };
    Some(format!("{} × {} · {} · {} · {}{sp}", g.w, g.h, export::rate_spec_label(rate, interlaced), frames, fmt_secs(secs)))
}

fn swap_ext(path: &str, ext: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    std::path::Path::new(path).with_extension(ext).to_string_lossy().into_owned()
}

/// Switch to codec family `id` (with the first container it goes in, unless the
/// current one takes it too).
fn set_codec(st: &mut ExportSettings, id: &str) {
    if st.codec == id {
        return;
    }
    st.codec = id.to_string();
    if let Some(f) = export::family(id) {
        if !f.containers.contains(&st.container.as_str()) {
            st.container = f.containers[0].to_string();
        }
    }
    let gpu = st.encoder == "gpu";
    st.encoder = if gpu { "gpu".into() } else { String::new() };
    let had_audio = !st.audio.is_empty();
    st.apply_family_defaults();
    if !had_audio {
        st.audio.clear();
    }
    if !st.interlace_possible() {
        st.interlaced = false;
    }
}

fn format_label(id: &str, l: Lang) -> String {
    match export::family(id) {
        Some(f) => format!("{}  ·  .{}", f.label, export::container(f.containers[0]).ext),
        None => l.t("ex_other_codec").to_string(),
    }
}

/// The preset choice (with Save… and delete) for `st`. True when the saved presets
/// changed.
#[allow(clippy::too_many_arguments)]
fn preset_row(ui: &mut egui::Ui, l: Lang, id: &str, st: &mut ExportSettings, presets: &[ExportSettings], user: &mut Vec<ExportSettings>, saving: &mut Option<String>, w: f32) -> bool {
    let mut saved = false;
    label(ui, l.t("ex_preset"));
    ui.horizontal(|ui| {
        let current = presets.iter().find(|x| x.name == st.name && *x == st).map(|x| x.name.clone()).unwrap_or_else(|| l.t("ex_custom").to_string());
        egui::ComboBox::from_id_salt((id, "preset")).selected_text(current).width(w).height(360.0).show_ui(ui, |ui| {
            let n_user = user.len();
            for (i, pr) in presets.iter().enumerate() {
                if i == n_user && n_user > 0 {
                    ui.separator();
                }
                if ui.selectable_label(pr.name == st.name, &pr.name).clicked() {
                    *st = pr.clone();
                }
            }
        });
        if ui.small_button(format!("{}…", l.t("ex_save_ok"))).on_hover_text(l.t("ex_save_tip")).clicked() {
            *saving = Some(st.name.clone());
        }
    });
    ui.end_row();
    if let Some(name) = saving {
        label(ui, "");
        let mut done = false;
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(name).hint_text(l.t("ex_preset_name")).desired_width(w - 70.0));
            let n = name.trim().to_string();
            if ui.add_enabled(!n.is_empty(), egui::Button::new(l.t("ex_save_ok"))).clicked() {
                let mut s = st.clone();
                s.name = n.clone();
                user.retain(|x| x.name != n);
                user.insert(0, s.clone());
                *st = s;
                done = true;
            }
            if user.iter().any(|x| x.name == n) && ui.small_button(l.t("ex_delete_preset")).on_hover_text(l.t("ex_delete_tip")).clicked() {
                user.retain(|x| x.name != n);
                done = true;
            }
        });
        if done {
            *saving = None;
            saved = true;
        }
        ui.end_row();
    }
    saved
}

/// Every main export choice for `st`, as rows of a 2-column grid that is open.
fn settings_rows(ui: &mut egui::Ui, l: Lang, id: &str, st: &mut ExportSettings, src: &PanelSrc, w: f32) {
    // Format: one choice for codec and container.
    label(ui, l.t("ex_format_label"));
    egui::ComboBox::from_id_salt((id, "format")).selected_text(format_label(&st.codec, l)).width(w).height(400.0).show_ui(ui, |ui| {
        for f in export::FAMILIES.iter().filter(|f| f.software.iter().chain(f.hardware).any(|x| brp_media::encode::encoder(x).is_some())) {
            if ui.selectable_label(st.codec == f.id, format_label(f.id, l)).clicked() {
                set_codec(st, f.id);
            }
        }
        ui.separator();
        if ui.selectable_label(st.codec == OTHER, l.t("ex_other_codec")).on_hover_text(l.t("ex_other_tip")).clicked() && st.codec != OTHER {
            st.codec = OTHER.into();
            st.encoder.clear();
            st.quality = None;
            st.pix_fmt.clear();
            if st.container != "mkv" && st.container != "mov" {
                st.container = "mkv".into();
            }
        }
    });
    ui.end_row();

    // Quality: named levels, or the codec's profiles.
    let levels = st.quality_levels();
    let profiles = st.family().map(|f| f.profiles).unwrap_or(&[]);
    if !levels.is_empty() {
        label(ui, l.t("ex_quality"));
        ui.horizontal(|ui| {
            for (key, v) in &levels {
                let on = st.quality == Some(*v);
                if ui.add(egui::Button::selectable(on, l.t(key)).corner_radius(11.0)).on_hover_text(l.t(&format!("{key}_tip"))).clicked() {
                    st.quality = Some(*v);
                }
            }
        });
        ui.end_row();
    } else if !profiles.is_empty() {
        label(ui, l.t("ex_quality"));
        let cur = profiles.iter().find(|x| x.0 == st.profile).map_or(profiles[0].1, |x| x.1);
        egui::ComboBox::from_id_salt((id, "profile")).selected_text(cur).width(w).show_ui(ui, |ui| {
            for (v, name) in profiles {
                ui.selectable_value(&mut st.profile, v.to_string(), *name);
            }
        });
        ui.end_row();
    } else if st.family().is_some_and(|f| matches!(f.id, "ffv1" | "utvideo" | "png" | "tiff" | "exr" | "dpx" | "qtrle")) {
        label(ui, l.t("ex_quality"));
        ui.label(weak(l.t("ex_lossless")));
        ui.end_row();
    }

    // Toggles, only where the codec (and the source) supports them.
    let mux = export::container(&st.container).muxer;
    let can_audio = src.audio != Some(false) && !brp_media::encode::is_image_sequence(mux) && st.container != "gif";
    let (a, h, g) = (st.alpha_possible(), src.hdr != Some(false) && st.hdr_possible(), st.gpu_possible() && (!st.interlaced || st.gpu_is_ours()));
    if a || h || g || can_audio {
        label(ui, l.t("ex_options"));
        ui.horizontal_wrapped(|ui| {
            if a {
                toggle(ui, &mut st.alpha, l.t("alpha"), l.t("ex_alpha_tip"));
            }
            if h {
                let mut hdr = !st.tonemap;
                if toggle(ui, &mut hdr, l.t("hdr"), l.t("ex_hdr_tip")) {
                    st.tonemap = !hdr;
                }
            }
            if g {
                let mut on = !st.encoder.is_empty();
                if toggle(ui, &mut on, l.t("ex_gpu"), l.t(if st.gpu_is_ours() { "ex_gpu_ours_tip" } else { "ex_gpu_tip" })) {
                    st.encoder = if on { "gpu".into() } else { String::new() };
                }
            }
            if can_audio {
                let mut on = !st.audio.is_empty();
                if toggle(ui, &mut on, l.t("audio"), l.t("ex_audio_tip")) {
                    st.audio = if on { export::container(&st.container).audio.to_string() } else { String::new() };
                }
            }
        });
        ui.end_row();
    }

    // Size.
    label(ui, l.t("resolution"));
    let same = match src.size {
        Some((fw, fh)) => format!("{} ({fw} × {fh})", l.t("ex_same_size")),
        None => l.t("ex_same_size_files").to_string(),
    };
    let cur = match &st.size {
        Size::Source => same.clone(),
        Size::Fixed(a, b) => SIZES.iter().find(|s| (s.0, s.1) == (*a, *b)).map_or_else(|| format!("{a} × {b}"), |s| s.2.to_string()),
        Size::Scale(pc) => format!("{pc:.0}%"),
    };
    egui::ComboBox::from_id_salt((id, "size")).selected_text(cur).width(w).show_ui(ui, |ui| {
        if ui.selectable_label(st.size == Size::Source, &same).clicked() {
            st.size = Size::Source;
        }
        for (a, b, name) in SIZES {
            if ui.selectable_label(st.size == Size::Fixed(*a, *b), *name).clicked() {
                st.size = Size::Fixed(*a, *b);
            }
        }
        if ui.selectable_label(st.size == Size::Scale(50.0), "50%").clicked() {
            st.size = Size::Scale(50.0);
        }
        let custom = matches!(st.size, Size::Fixed(a, b) if !SIZES.iter().any(|s| (s.0, s.1) == (a, b)));
        if ui.selectable_label(custom, l.t("ex_custom")).clicked() && !custom {
            st.size = Size::Fixed(src.size.map_or(1920, |s| s.0), 0);
        }
    });
    ui.end_row();
    if let Size::Fixed(a, b) = &mut st.size {
        if !SIZES.iter().any(|s| (s.0, s.1) == (*a, *b)) {
            label(ui, "");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(a).range(0..=16384).speed(2.0));
                ui.label("×");
                ui.add(egui::DragValue::new(b).range(0..=16384).speed(2.0));
                ui.label(weak(l.t("ex_zero_auto")));
            });
            ui.end_row();
        }
    }
    if matches!(st.size, Size::Fixed(a, b) if a > 0 && b > 0) {
        label(ui, "");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut st.fit, Fit::Fit, l.t("ex_fit")).on_hover_text(l.t("ex_fit_tip"));
            ui.selectable_value(&mut st.fit, Fit::Fill, l.t("ex_fill")).on_hover_text(l.t("ex_fill_tip"));
            ui.selectable_value(&mut st.fit, Fit::Stretch, l.t("ex_stretch")).on_hover_text(l.t("ex_stretch_tip"));
        });
        ui.end_row();
    }

    // Frame rate: progressive or interlaced.
    label(ui, l.t("frame_rate"));
    let same = match src.stack_rate {
        Some((r, i)) => format!("{} ({})", l.t("ex_same_rate"), export::rate_spec_label(r, i)),
        None => l.t("ex_same_rate_files").to_string(),
    };
    let cur = st.fps.map_or(same.clone(), |r| export::rate_spec_label(r, st.interlaced));
    egui::ComboBox::from_id_salt((id, "rate")).selected_text(cur).width(w).height(360.0).show_ui(ui, |ui| {
        if ui.selectable_label(st.fps.is_none(), &same).clicked() {
            st.fps = None;
            st.interlaced = false;
        }
        for (r, i) in RATES {
            if *i && !st.interlace_possible() {
                continue;
            }
            let tip = if *i { l.t("ex_interlaced_tip") } else { l.t("ex_progressive_tip") };
            if ui.selectable_label(st.fps == Some(*r) && st.interlaced == *i, export::rate_spec_label(*r, *i)).on_hover_text(tip).clicked() {
                st.fps = Some(*r);
                st.interlaced = *i;
                if *i && st.encoder == "gpu" && !st.gpu_is_ours() {
                    st.encoder.clear();
                }
            }
        }
    });
    ui.end_row();
    // A rate whose pictures the stack does not make: how to make them.
    if let Some(r) = st.fps {
        let need = picture_rate(r, st.interlaced);
        let differs = src.stack_rate.is_none_or(|(s, _)| (need - export::rate_f(s)).abs() > 1e-6);
        if differs {
            label(ui, l.t("ex_conform"));
            let name = |c: Conform| match c {
                Conform::Motion => (l.t("ex_motion"), l.t("ex_motion_tip")),
                Conform::Nearest => (l.t("ex_nearest"), l.t("ex_nearest_tip")),
                Conform::Blend => (l.t("ex_blend"), l.t("ex_blend_tip")),
                Conform::Retime => (l.t("ex_retime"), l.t("ex_retime_tip")),
            };
            egui::ComboBox::from_id_salt((id, "conform")).selected_text(name(st.conform).0).width(w).show_ui(ui, |ui| {
                for c in [Conform::Motion, Conform::Nearest, Conform::Blend, Conform::Retime] {
                    let (n, tip) = name(c);
                    ui.selectable_value(&mut st.conform, c, n).on_hover_text(tip);
                }
            });
            ui.end_row();
        }
    }
    if src.slow > 1 {
        label(ui, "");
        ui.label(weak(format!("{} {}×", l.t("ex_slow_note"), src.slow)));
        ui.end_row();
    }
}

/// The technical choices, collapsed by default.
fn advanced(ui: &mut egui::Ui, l: Lang, id: &str, st: &mut ExportSettings, all_encoders: &mut Vec<brp_media::encode::EncoderInfo>, slow: i64, w: f32) {
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new(l.t("ex_advanced")).color(Color32::from_gray(200))).id_salt((id, "advanced")).default_open(std::env::var_os("BRP_TEST_EXPORT_ADVANCED").is_some()).show(ui, |ui| {
        egui::Grid::new((id, "adv")).num_columns(2).spacing(Vec2::new(12.0, 7.0)).min_col_width(66.0).show(ui, |ui| {
            // Container (those the codec goes in).
            label(ui, l.t("container"));
            let allowed: Vec<&export::Container> = match st.family() {
                Some(f) => export::CONTAINERS.iter().filter(|c| f.containers.contains(&c.id)).collect(),
                None => export::CONTAINERS.iter().filter(|c| c.muxer != "image2" && c.id != "gif").collect(),
            };
            egui::ComboBox::from_id_salt((id, "container")).selected_text(export::container(&st.container).label).width(w).show_ui(ui, |ui| {
                for c in allowed {
                    if ui.selectable_label(st.container == c.id, c.label).clicked() && st.container != c.id {
                        st.container = c.id.to_string();
                        if !st.audio.is_empty() {
                            st.audio = c.audio.to_string();
                        }
                    }
                }
            });
            ui.end_row();

            // Encoder.
            label(ui, l.t("ex_encoder"));
            if st.codec == OTHER {
                if all_encoders.is_empty() {
                    *all_encoders = brp_media::encode::video_encoders();
                    all_encoders.sort_by(|a, b| a.name.cmp(&b.name));
                }
                let mux = export::container(&st.container).muxer;
                let cur = if st.encoder.is_empty() { "—".to_string() } else { st.encoder.clone() };
                egui::ComboBox::from_id_salt((id, "any_enc")).selected_text(cur).width(w).height(320.0).show_ui(ui, |ui| {
                    for x in all_encoders.iter().filter(|x| brp_media::encode::muxer_accepts(mux, &x.name) != Some(false)) {
                        let hw = if x.hardware { " (GPU)" } else { "" };
                        if ui.selectable_label(st.encoder == x.name, format!("{}{hw} — {}", x.name, x.long_name)).clicked() {
                            st.encoder = x.name.clone();
                            st.pix_fmt.clear();
                        }
                    }
                });
            } else if let Some(f) = st.family() {
                let ok = export::hardware_ok();
                let sw = f.software.iter().find(|x| brp_media::encode::encoder(x).is_some()).copied().unwrap_or("—");
                let name_of = |v: &str| match v {
                    "" => format!("{} ({sw})", l.t("ex_software")),
                    "gpu" => l.t("ex_gpu_auto").to_string(),
                    n if export::is_ours(n) => format!("{n} (BRP, GPU)"),
                    n => n.to_string(),
                };
                egui::ComboBox::from_id_salt((id, "encoder")).selected_text(name_of(&st.encoder)).width(w).show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.encoder, String::new(), name_of(""));
                    for x in f.software.iter().skip(1).filter(|x| brp_media::encode::encoder(x).is_some()) {
                        ui.selectable_value(&mut st.encoder, x.to_string(), *x);
                    }
                    if f.hardware.iter().any(|h| ok.iter().any(|o| o == h)) && (!st.interlaced || st.gpu_is_ours()) {
                        ui.selectable_value(&mut st.encoder, "gpu".to_string(), name_of("gpu"));
                        for h in f.hardware.iter().filter(|h| ok.iter().any(|o| o == *h)) {
                            ui.selectable_value(&mut st.encoder, h.to_string(), name_of(h));
                        }
                    }
                    if !export::hardware_probed() && !f.hardware.is_empty() {
                        ui.label(weak(l.t("ex_probing")));
                    }
                });
            }
            ui.end_row();

            // Field order (interlaced).
            if st.interlaced && st.fps.is_some() {
                label(ui, l.t("ex_field_order"));
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut st.tff, true, l.t("ex_tff")).on_hover_text(l.t("ex_tff_tip"));
                    ui.selectable_value(&mut st.tff, false, l.t("ex_bff")).on_hover_text(l.t("ex_bff_tip"));
                });
                ui.end_row();
            }

            // Exact quality or bit rate.
            let fam = st.family();
            let q = fam.and_then(|f| f.quality);
            let br = fam.and_then(|f| f.bitrate);
            if q.is_some() || br.is_some() || st.codec == OTHER {
                label(ui, l.t("ex_rate_control"));
                ui.horizontal(|ui| {
                    if q.is_some() || st.codec == OTHER {
                        let on = st.quality.is_some();
                        if ui.selectable_label(on, l.t("ex_constant_q")).clicked() && !on {
                            st.quality = Some(q.map_or(20.0, |q| q.2));
                        }
                    }
                    if br.is_some() || st.codec == OTHER {
                        let on = st.quality.is_none();
                        if ui.selectable_label(on, l.t("ex_bitrate")).clicked() {
                            st.quality = None;
                        }
                    }
                    match st.quality.as_mut() {
                        Some(v) => {
                            let (lo, hi, _) = q.unwrap_or((0.0, 63.0, 20.0));
                            ui.add(egui::DragValue::new(v).range(lo..=hi).speed(0.2).max_decimals(0)).on_hover_text(l.t("ex_quality_tip"));
                        }
                        None => {
                            ui.add(egui::DragValue::new(&mut st.kbps).range(100..=2_000_000).speed(100.0).suffix(" kbit/s"));
                        }
                    }
                });
                ui.end_row();
            }

            // Pixel format.
            label(ui, l.t("pixel_format"));
            let enc_name = match st.codec.as_str() {
                OTHER => st.encoder.clone(),
                _ => st.family().and_then(|f| f.software.iter().chain(f.hardware).find(|x| brp_media::encode::encoder(x).is_some()).map(|s| s.to_string())).unwrap_or_default(),
            };
            let fmts = brp_media::encode::encoder(&enc_name).map(|x| x.pix_fmts).unwrap_or_default();
            let cur = if st.pix_fmt.is_empty() { l.t("ex_auto_pix").to_string() } else { st.pix_fmt.clone() };
            egui::ComboBox::from_id_salt((id, "pix")).selected_text(cur).width(w).height(300.0).show_ui(ui, |ui| {
                ui.selectable_value(&mut st.pix_fmt, String::new(), l.t("ex_auto_pix"));
                for f in fmts.iter().filter(|f| brp_media::encode::pix_fmt_info(f).is_some_and(|i| !i.hw)) {
                    ui.selectable_value(&mut st.pix_fmt, f.clone(), f);
                }
            });
            ui.end_row();

            // Audio codec.
            let mux = export::container(&st.container).muxer;
            if !st.audio.is_empty() {
                label(ui, l.t("ex_audio_codec"));
                ui.horizontal(|ui| {
                    let cur = AUDIO.iter().find(|a| a.0 == st.audio).map_or_else(|| st.audio.clone(), |a| a.1.to_string());
                    egui::ComboBox::from_id_salt((id, "audio")).selected_text(cur).width(120.0).show_ui(ui, |ui| {
                        for (aid, name) in AUDIO {
                            if brp_media::encode::audio_encoder_exists(aid) && brp_media::encode::muxer_accepts(mux, aid) != Some(false) {
                                ui.selectable_value(&mut st.audio, aid.to_string(), *name);
                            }
                        }
                    });
                    if matches!(st.audio.as_str(), "aac" | "libopus" | "libmp3lame") {
                        ui.add(egui::DragValue::new(&mut st.audio_kbps).range(32..=512).speed(4.0).suffix(" kbit/s"));
                    }
                });
                ui.end_row();
                if slow > 1 || st.conform == Conform::Retime {
                    label(ui, l.t("ex_speed_audio"));
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut st.keep_pitch, true, l.t("ex_keep_pitch"));
                        ui.selectable_value(&mut st.keep_pitch, false, l.t("ex_shift_pitch"));
                    });
                    ui.end_row();
                }
            }

            // Anything else, as on the FFmpeg command line.
            label(ui, l.t("ex_extra"));
            ui.add(egui::TextEdit::singleline(&mut st.extra).hint_text("preset=slow tune=film").desired_width(w)).on_hover_text(l.t("ex_extra_tip"));
            ui.end_row();
        });
    });
}

// ---- Export --------------------------------------------------------------------------------

pub fn export_contents(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    const W: f32 = 236.0;
    s.export.load(&s.settings);
    export::probe_hardware();
    let Some(src_path) = p.source.as_ref().map(|s| s.info.path.clone()) else {
        ui.label(RichText::new(l.t("no_file")).weak());
        return;
    };
    let src = PanelSrc::of_player(p);
    let UiState { export: e, settings: cfg, settings_dirty: dirty, .. } = s;
    if e.output_for != src_path || e.output.is_empty() {
        e.output = export::default_output(&src_path, None, &e.settings);
        e.output_for = src_path.clone();
        e.use_range = p.has_range();
    }
    let before = e.settings.clone();
    let presets = all_presets(e);

    egui::Grid::new("ex_main").num_columns(2).spacing(Vec2::new(12.0, 8.0)).min_col_width(78.0).show(ui, |ui| {
        if preset_row(ui, l, "ex", &mut e.settings, &presets, &mut e.presets, &mut e.saving, W) {
            cfg.export_presets = e.presets.iter().map(ExportSettings::to_json).collect();
            *dirty = true;
        }
        settings_rows(ui, l, "ex", &mut e.settings, &src, W);
        // Range (only with In/Out set).
        if p.has_range() {
            label(ui, l.t("ex_range"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut e.use_range, false, l.t("ex_whole"));
                ui.selectable_value(&mut e.use_range, true, format!("{} {}–{}", l.t("ex_in_out"), p.in_point, p.out_frame()));
            });
            ui.end_row();
        }
        // File.
        label(ui, l.t("file"));
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut e.output).desired_width(W - 64.0));
            if ui.button(l.t("ex_browse")).clicked() {
                actions.push(Action::ExportBrowse);
            }
        });
        ui.end_row();
    });
    advanced(ui, l, "ex", &mut e.settings, &mut e.all_encoders, src.slow, W);

    if e.settings != before {
        if e.settings.container != before.container {
            e.output = swap_ext(&e.output, export::container(&e.settings.container).ext);
        }
        let v = Some(e.settings.to_json());
        if cfg.export_last != v {
            cfg.export_last = v;
            *dirty = true;
        }
    }

    // Summary and Export.
    ui.add_space(6.0);
    if let Some(sum) = summary(p, e) {
        ui.label(RichText::new(sum).color(Color32::from_gray(225)));
    }
    let n_fx = p.fx.active_count();
    if n_fx > 0 {
        ui.label(weak(format!("{} {n_fx}", l.t("ex_with_effects"))));
    }
    ui.add_space(4.0);
    let running = e.running.as_ref().filter(|r| !r.ended()).map(|r| r.snapshot());
    if let Some(pr) = &running {
        progress_line(ui, l, pr);
        if ui.button(RichText::new(l.t("ex_cancel")).color(Color32::from_rgb(255, 140, 140))).clicked() {
            if let Some(r) = &e.running {
                r.cancel();
            }
        }
        ui.ctx().request_repaint_after(Duration::from_millis(200));
    } else {
        let can = p.ai.ctx.is_some() && !e.output.trim().is_empty();
        let b = egui::Button::new(RichText::new(l.t("ex_export")).strong().color(Color32::WHITE)).fill(ACCENT).min_size(Vec2::new(120.0, 28.0)).corner_radius(14.0);
        if ui.add_enabled(can, b).clicked() {
            let job = ExportJob {
                input: src_path.clone(),
                output: e.output.trim().to_string(),
                settings: e.settings.clone(),
                stack: p.fx.all_stages(),
                range: (e.use_range && p.has_range()).then(|| (p.in_point, p.out_frame())),
                deint: export::DeintChoice::Fixed(p.ai_source_deint()),
                layer: p.source.as_ref().and_then(|s| s.info.layers.get(s.info.layer).map(|(n, _)| n.clone())),
            };
            e.last = None;
            e.running = Some(export::start(job, p.ai.ctx.clone().unwrap()));
        }
        if let Some(last) = &e.last {
            match &last.state {
                State::Done => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("✔ {} ({})", l.t("ex_done"), fmt_secs(last.elapsed))).color(Color32::from_rgb(120, 220, 140)));
                        if ui.small_button(l.t("ex_show")).clicked() {
                            actions.push(Action::Reveal(last.output.clone()));
                        }
                    });
                }
                State::Failed(err) => {
                    ui.add(egui::Label::new(RichText::new(format!("{}: {err}", l.t("ex_failed"))).color(Color32::from_rgb(255, 130, 130))).wrap());
                }
                State::Cancelled => {
                    ui.label(weak(l.t("ex_cancelled")));
                }
                _ => {}
            }
        }
    }
}

// ---- Batch -------------------------------------------------------------------------------------

/// Width of the file list (the settings panel takes the rest).
const LIST_W: f32 = 330.0;

pub fn batch_contents(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<Action>) {
    let l = s.lang;
    s.export.load(&s.settings);
    export::probe_hardware();
    let presets = all_presets(&s.export);
    let UiState { export: e, settings: cfg, settings_dirty: dirty, .. } = s;
    let busy = e.batch.active || e.batch.running().is_some();
    let items = &e.batch.items;
    e.batch_sel.retain(|id| items.iter().any(|i| i.id == *id && i.state == ItemState::Waiting));

    let panes = ui.horizontal_top(|ui| {
        // ---- Left: the files. ----
        ui.vertical(|ui| {
            ui.set_width(LIST_W);
            section(ui, l.t("batch_files"));
            ui.horizontal(|ui| {
                if ui.button(l.t("batch_add")).clicked() {
                    actions.push(Action::BatchAddFiles);
                }
                if let Some(path) = p.source.as_ref().map(|s| s.info.path.clone()) {
                    if ui.button(l.t("batch_add_current")).clicked() {
                        e.batch.add(path);
                    }
                }
            });
            ui.add_space(4.0);
            let mut remove: Option<usize> = None;
            let mut clicked: Option<(u64, egui::Modifiers)> = None;
            let running = e.batch.running();
            egui::ScrollArea::vertical().id_salt("batch_list").max_height(440.0).auto_shrink([false, true]).show(ui, |ui| {
                if e.batch.items.is_empty() {
                    ui.add(egui::Label::new(weak(l.t("batch_empty"))).wrap());
                }
                for (i, it) in e.batch.items.iter().enumerate() {
                    let sel = e.batch_sel.contains(&it.id);
                    let fill = if sel { ACCENT.gamma_multiply(0.22) } else { Color32::from_white_alpha(8) };
                    let stroke = if sel { egui::Stroke::new(1.0, ACCENT.gamma_multiply(0.7)) } else { egui::Stroke::NONE };
                    let resp = egui::Frame::new()
                        .fill(fill)
                        .stroke(stroke)
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.horizontal(|ui| {
                                let mut name = std::path::Path::new(&it.path).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                                if name.chars().count() > 26 {
                                    name = format!("{}…", name.chars().take(25).collect::<String>());
                                }
                                ui.label(RichText::new(name).color(Color32::from_gray(232))).on_hover_text(&it.path);
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if it.state != ItemState::Running && ui.small_button("×").on_hover_text(l.t("fx_remove")).clicked() {
                                        remove = Some(i);
                                    }
                                    let (text, color) = match &it.state {
                                        ItemState::Waiting => (l.t("batch_waiting").to_string(), Color32::from_gray(150)),
                                        ItemState::Running => (format!("{:.0}%", it.progress * 100.0), ACCENT),
                                        ItemState::Done => (format!("✔ {}", it.took.map_or(String::new(), fmt_secs)), Color32::from_rgb(120, 220, 140)),
                                        ItemState::Failed(_) => (l.t("ex_failed").to_string(), Color32::from_rgb(255, 130, 130)),
                                        ItemState::Cancelled => (l.t("ex_cancelled").to_string(), Color32::from_gray(150)),
                                    };
                                    let r = ui.label(RichText::new(text).small().color(color));
                                    if let ItemState::Failed(err) = &it.state {
                                        r.on_hover_text(err);
                                    }
                                    if it.state == ItemState::Done {
                                        if let Some(out) = &it.output {
                                            if ui.small_button(l.t("ex_show")).clicked() {
                                                actions.push(Action::Reveal(out.clone()));
                                            }
                                        }
                                    }
                                });
                            });
                            // What it becomes, in one line.
                            let st = &it.settings;
                            let rate = st.fps.map_or_else(|| l.t("batch_same_rate").to_string(), |r| export::rate_spec_label(r, st.interlaced));
                            let size = match &st.size {
                                Size::Source => l.t("batch_same_size").to_string(),
                                Size::Fixed(a, 0) => format!("{a} × auto"),
                                Size::Fixed(0, b) => format!("auto × {b}"),
                                Size::Fixed(a, b) => format!("{a} × {b}"),
                                Size::Scale(pc) => format!("{pc:.0}%"),
                            };
                            let fmt = st.family().map_or_else(|| l.t("ex_other_codec").to_string(), |f| f.label.to_string());
                            ui.label(weak(format!("{fmt} · {rate} · {size}")));
                            if it.state == ItemState::Running {
                                if let Some((_, pr)) = running.as_ref().filter(|(ri, _)| *ri == i) {
                                    progress_line(ui, l, pr);
                                }
                            }
                            if let ItemState::Failed(err) = &it.state {
                                ui.add(egui::Label::new(weak(err.clone())).wrap());
                            }
                        })
                        .response;
                    let r = ui.interact(resp.rect, egui::Id::new(("batch_item", it.id)), egui::Sense::click());
                    if r.clicked() && it.state == ItemState::Waiting {
                        clicked = Some((it.id, ui.input(|i| i.modifiers)));
                    }
                    ui.add_space(3.0);
                }
            });
            // Selection: click selects one, Ctrl adds or removes, Shift a run.
            if let Some((id, m)) = clicked {
                if m.command {
                    if !e.batch_sel.remove(&id) {
                        e.batch_sel.insert(id);
                    }
                } else if m.shift && e.batch_anchor.is_some() {
                    let pos = |x: u64| e.batch.items.iter().position(|i| i.id == x);
                    if let (Some(a), Some(b)) = (pos(e.batch_anchor.unwrap()), pos(id)) {
                        for it in &e.batch.items[a.min(b)..=a.max(b)] {
                            if it.state == ItemState::Waiting {
                                e.batch_sel.insert(it.id);
                            }
                        }
                    }
                } else if e.batch_sel.len() == 1 && e.batch_sel.contains(&id) {
                    e.batch_sel.clear();
                } else {
                    e.batch_sel.clear();
                    e.batch_sel.insert(id);
                }
                e.batch_anchor = Some(id);
            }
            if let Some(i) = remove {
                e.batch.remove(i);
            }
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if !e.batch.items.is_empty() && ui.small_button(l.t("batch_select_all")).clicked() {
                    e.batch_sel = e.batch.items.iter().filter(|i| i.state == ItemState::Waiting).map(|i| i.id).collect();
                }
                if !e.batch_sel.is_empty() && ui.small_button(l.t("batch_select_none")).clicked() {
                    e.batch_sel.clear();
                }
                if e.batch.items.iter().any(|i| !matches!(i.state, ItemState::Waiting | ItemState::Running)) && ui.small_button(l.t("batch_clear_done")).clicked() {
                    e.batch.remove_finished();
                }
            });
        });

        // Room for the divider, drawn once the height is known.
        ui.add_space(22.0);

        // ---- Right: the settings of the selected files, or of all of them. ----
        ui.vertical(|ui| {
            const W: f32 = 230.0;
            let n_sel = e.batch_sel.len();
            let waiting = e.batch.items.iter().filter(|i| i.state == ItemState::Waiting).count();
            // No selection: the panel edits every waiting file (and files added
            // later), so a change always reaches the files in the list.
            let heading = match (n_sel, waiting) {
                (0, 0) => l.t("batch_for_new").to_string(),
                (0, n) => format!("{} ({n})", l.t("batch_for_all")),
                (1, _) => l.t("batch_for_one").to_string(),
                (n, _) => format!("{} ({n})", l.t("batch_for_sel")),
            };
            section(ui, &heading);
            let hint = match (n_sel, waiting) {
                (0, 0) => l.t("batch_sel_hint_empty"),
                (0, _) => l.t("batch_sel_hint"),
                _ => l.t("batch_sel_edit"),
            };
            ui.add(egui::Label::new(weak(hint)).wrap());
            if e.batch.panel_mixed(&e.batch_sel) {
                ui.add(egui::Label::new(RichText::new(l.t("batch_mixed")).small().color(Color32::from_rgb(255, 196, 90))).wrap());
            }
            ui.add_space(4.0);
            let mut st = e.batch.panel_settings(&e.batch_sel);
            let before = st.clone();
            let src = PanelSrc::batch();
            egui::Grid::new("batch_panel").num_columns(2).spacing(Vec2::new(12.0, 8.0)).min_col_width(78.0).show(ui, |ui| {
                if preset_row(ui, l, "bp", &mut st, &presets, &mut e.presets, &mut e.saving, W) {
                    cfg.export_presets = e.presets.iter().map(ExportSettings::to_json).collect();
                    *dirty = true;
                }
                settings_rows(ui, l, "bp", &mut st, &src, W);
            });
            advanced(ui, l, "bp", &mut st, &mut e.all_encoders, 1, W);
            if st != before {
                e.batch.apply_panel(&e.batch_sel, &st);
                if n_sel == 0 {
                    cfg.batch_defaults = Some(st.to_json());
                    *dirty = true;
                }
            }
            if n_sel > 0 && waiting > 0 {
                ui.add_space(2.0);
                if ui.small_button(l.t("batch_apply_all")).on_hover_text(l.t("batch_apply_all_tip")).clicked() {
                    for it in e.batch.items.iter_mut().filter(|i| i.state == ItemState::Waiting) {
                        it.settings = st.clone();
                    }
                    e.batch.defaults = st.clone();
                    cfg.batch_defaults = Some(st.to_json());
                    *dirty = true;
                }
            }
            ui.add_space(6.0);

            // For every file: effects and output folder.
            section(ui, l.t("batch_all_files"));
            egui::Grid::new("batch_common").num_columns(2).spacing(Vec2::new(12.0, 8.0)).min_col_width(78.0).show(ui, |ui| {
                label(ui, l.t("effects"));
                let n_fx = p.fx.active_count();
                ui.add_enabled_ui(!busy && n_fx > 0, |ui| {
                    let text = if n_fx > 0 { format!("{} ({n_fx})", l.t("batch_effects")) } else { l.t("batch_no_effects").to_string() };
                    if ui.checkbox(&mut e.batch_effects, text).on_hover_text(l.t("batch_effects_tip")).changed() {
                        cfg.batch_effects = e.batch_effects;
                        *dirty = true;
                    }
                });
                ui.end_row();
                label(ui, l.t("batch_folder"));
                ui.horizontal(|ui| {
                    let full = e.batch.out_dir.as_ref().map(|d| d.to_string_lossy().into_owned());
                    let cur = full.as_deref().map_or(l.t("batch_next_to").to_string(), |d| {
                        let n = d.chars().count();
                        if n > 24 { format!("…{}", d.chars().skip(n - 23).collect::<String>()) } else { d.to_string() }
                    });
                    let r = ui.label(RichText::new(cur).color(Color32::from_gray(225)));
                    if let Some(f) = &full {
                        r.on_hover_text(f);
                    }
                    if ui.small_button(l.t("ex_browse")).clicked() {
                        actions.push(Action::BatchFolder);
                    }
                    if e.batch.out_dir.is_some() && ui.small_button("×").on_hover_text(l.t("batch_reset_folder")).clicked() {
                        e.batch.out_dir = None;
                        cfg.batch_dir = None;
                        *dirty = true;
                    }
                });
                ui.end_row();
            });
        });
    });
    // The divider between the list and the panel.
    let r = panes.response.rect;
    let x = r.left() + LIST_W + 11.0;
    ui.painter().line_segment([egui::pos2(x, r.top() + 4.0), egui::pos2(x, r.bottom() - 4.0)], egui::Stroke::new(1.0, Color32::from_white_alpha(28)));

    // ---- Below: overall progress and Start / Stop. ----
    ui.add_space(8.0);
    if let Some(f) = e.batch_fraction() {
        ui.add(egui::ProgressBar::new(f).desired_height(10.0).corner_radius(5.0).fill(ACCENT));
    }
    ui.horizontal(|ui| {
        if busy {
            if ui.button(RichText::new(l.t("batch_stop")).color(Color32::from_rgb(255, 140, 140))).clicked() {
                e.batch.cancel();
            }
            if ui.button(l.t("batch_skip")).on_hover_text(l.t("batch_skip_tip")).clicked() {
                e.batch.skip();
            }
        } else {
            let waiting = e.batch.items.iter().filter(|i| matches!(i.state, ItemState::Waiting | ItemState::Cancelled)).count();
            let text = if waiting > 0 { format!("{} ({waiting})", l.t("batch_start")) } else { l.t("batch_start").to_string() };
            let b = egui::Button::new(RichText::new(text).strong().color(Color32::WHITE)).fill(ACCENT).min_size(Vec2::new(130.0, 28.0)).corner_radius(14.0);
            if ui.add_enabled(waiting > 0 && p.ai.ctx.is_some(), b).on_hover_text(l.t("batch_start_tip")).clicked() {
                let stack = if e.batch_effects { p.fx.all_stages() } else { Vec::new() };
                e.batch.start(stack, p.deint_mode, p.field_rate);
                e.batch_sel.clear();
            }
        }
        let done = e.batch.items.iter().filter(|i| i.state == ItemState::Done).count();
        let failed = e.batch.items.iter().filter(|i| matches!(i.state, ItemState::Failed(_))).count();
        if done + failed > 0 {
            ui.label(weak(format!("{done} {} · {failed} {}", l.t("batch_n_done"), l.t("batch_n_failed"))));
        }
    });
    if e.busy() {
        ui.ctx().request_repaint_after(Duration::from_millis(250));
    }
}

impl ExportUi {
    /// Select the first `n` waiting batch files (layout checks: `BRP_TEST_BATCH_SELECT`).
    pub fn select_first(&mut self, n: usize) {
        self.batch_sel = self.batch.items.iter().filter(|i| i.state == ItemState::Waiting).take(n).map(|i| i.id).collect();
    }
}
