//! Choices kept between runs: `%APPDATA%\BRP Player\settings.json`.

use crate::i18n::Lang;
use crate::player::DeintMode;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Where the player window was: outer position and inner size (physical pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowPlace {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub maximized: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub lang: Lang,
    /// GPU by name (None: the best one).
    pub gpu: Option<String>,
    /// Audio output by name (None: system default).
    pub audio_device: Option<String>,
    /// Frame cache per file in bytes (None: 70% of RAM).
    pub cache_bytes: Option<u64>,
    /// Show positions as frame numbers instead of timecode.
    pub show_frames: bool,
    pub window: Option<WindowPlace>,
    pub volume: f32,
    pub muted: bool,
    pub loop_on: bool,
    pub deint_mode: DeintMode,
    /// Deinterlace to field rate (else frame rate).
    pub field_rate: bool,
    /// Export presets saved by the user, and the last export settings used.
    pub export_presets: Vec<Value>,
    pub export_last: Option<Value>,
    /// Batch export output folder (None: next to each source).
    pub batch_dir: Option<String>,
    /// Batch export: the preset used for every file, and whether the current
    /// effects are applied.
    pub batch_preset: Option<String>,
    pub batch_effects: bool,
    /// Settings new batch files get.
    pub batch_defaults: Option<Value>,
    /// OpenColorIO (DESIGN.md §7d): on or off (off unless turned on), and the defaults
    /// the Color panel's "(Default)" choices use: entry ids ("maya:ACEScg"), a view name,
    /// and the output as "display:<id>" or "space:<id>". None: no default (Auto).
    pub ocio_enabled: bool,
    pub ocio_input: Option<String>,
    pub ocio_display: Option<String>,
    pub ocio_view: Option<String>,
    pub ocio_output: Option<String>,
    /// A config file of the user's own (its entries say "(custom)").
    pub ocio_custom: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            lang: Lang::En,
            gpu: None,
            audio_device: None,
            cache_bytes: None,
            show_frames: true,
            window: None,
            volume: 1.0,
            muted: false,
            loop_on: true,
            deint_mode: DeintMode::Auto,
            field_rate: true,
            export_presets: Vec::new(),
            export_last: None,
            batch_dir: None,
            batch_preset: None,
            batch_effects: true,
            batch_defaults: None,
            ocio_enabled: false,
            ocio_input: None,
            ocio_display: None,
            ocio_view: None,
            ocio_output: None,
            ocio_custom: None,
        }
    }
}

fn path() -> PathBuf {
    let base = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    base.join("BRP Player").join("settings.json")
}

fn deint_name(m: DeintMode) -> &'static str {
    match m {
        DeintMode::Auto => "auto",
        DeintMode::Off => "off",
        DeintMode::Best => "best",
        DeintMode::Nnedi => "nnedi",
        DeintMode::Bwdif => "bwdif",
    }
}

impl Settings {
    pub fn load() -> Settings {
        let d = Settings::default();
        let Ok(text) = std::fs::read_to_string(path()) else { return d };
        let Ok(v) = serde_json::from_str::<Value>(&text) else { return d };
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string).filter(|s| !s.is_empty());
        let b = |k: &str, def: bool| v.get(k).and_then(Value::as_bool).unwrap_or(def);
        let window = v.get("window").and_then(|w| {
            let i = |k: &str| w.get(k).and_then(Value::as_i64);
            Some(WindowPlace {
                x: i("x")? as i32,
                y: i("y")? as i32,
                w: i("w")?.max(200) as u32,
                h: i("h")?.max(150) as u32,
                maximized: w.get("maximized").and_then(Value::as_bool).unwrap_or(false),
            })
        });
        let deint_mode = match s("deinterlace").as_deref() {
            Some("off") => DeintMode::Off,
            Some("best") => DeintMode::Best,
            Some("nnedi") => DeintMode::Nnedi,
            Some("bwdif") => DeintMode::Bwdif,
            _ => DeintMode::Auto,
        };
        Settings {
            lang: if s("lang").as_deref() == Some("pt") { Lang::Pt } else { Lang::En },
            gpu: s("gpu"),
            audio_device: s("audio_device"),
            cache_bytes: v.get("cache_mb").and_then(Value::as_u64).map(|mb| mb << 20),
            // Frame numbers unless timecode was chosen.
            show_frames: v.get("time_display").and_then(Value::as_str) != Some("timecode"),
            window,
            volume: v.get("volume").and_then(Value::as_f64).map_or(d.volume, |x| x.clamp(0.0, 1.0) as f32),
            muted: b("muted", d.muted),
            loop_on: b("loop", d.loop_on),
            deint_mode,
            field_rate: b("field_rate", d.field_rate),
            export_presets: v.get("export_presets").and_then(Value::as_array).cloned().unwrap_or_default(),
            export_last: v.get("export_last").cloned().filter(|x| x.is_object()),
            batch_dir: s("batch_dir"),
            batch_preset: s("batch_preset"),
            batch_effects: b("batch_effects", true),
            batch_defaults: v.get("batch_defaults").cloned().filter(|x| x.is_object()),
            ocio_enabled: b("ocio_enabled", false),
            ocio_input: s("ocio_input"),
            ocio_display: s("ocio_display"),
            ocio_view: s("ocio_view"),
            ocio_output: s("ocio_output"),
            ocio_custom: s("ocio_custom"),
        }
    }

    pub fn save(&self) {
        let v = json!({
            "lang": if self.lang == Lang::Pt { "pt" } else { "en" },
            "gpu": self.gpu,
            "audio_device": self.audio_device,
            "cache_mb": self.cache_bytes.map(|b| b >> 20),
            "time_display": if self.show_frames { "frames" } else { "timecode" },
            "window": self.window.map(|w| json!({ "x": w.x, "y": w.y, "w": w.w, "h": w.h, "maximized": w.maximized })),
            "volume": self.volume,
            "muted": self.muted,
            "loop": self.loop_on,
            "deinterlace": deint_name(self.deint_mode),
            "field_rate": self.field_rate,
            "export_presets": self.export_presets,
            "export_last": self.export_last,
            "batch_dir": self.batch_dir,
            "batch_preset": self.batch_preset,
            "batch_effects": self.batch_effects,
            "batch_defaults": self.batch_defaults,
            "ocio_enabled": self.ocio_enabled,
            "ocio_input": self.ocio_input,
            "ocio_display": self.ocio_display,
            "ocio_view": self.ocio_view,
            "ocio_output": self.ocio_output,
            "ocio_custom": self.ocio_custom,
        });
        let p = path();
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Err(e) = std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap_or_default()) {
            log::warn!("saving settings: {e}");
        }
    }
}
