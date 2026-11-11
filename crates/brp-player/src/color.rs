//! OpenColorIO in the player (DESIGN.md §7d): the four choices (input, display, view,
//! output), each Auto, Default (Settings) or picked; the configs, loaded on first use;
//! and the window's conversion, built off the interface thread.

use brp_media::{ColorInfo, MediaInfo, Transfer};
use brp_ocio::auto::{self, Media, OutputTarget, Screen};
use brp_ocio::{Entry, Library, Target};
use crossbeam_channel::Receiver;
use std::sync::Arc;

/// A choice in the Color panel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pick<T> {
    /// From the file, the screen, the export format.
    Auto,
    /// The default set in Settings (Auto when none is set).
    Default,
    Manual(T),
}

/// What the window and the export use, resolved.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resolved {
    pub input: Entry,
    pub display: Entry,
    pub view: String,
}

pub struct Ocio {
    pub enabled: bool,
    pub input: Pick<Entry>,
    pub display: Pick<Entry>,
    pub view: Pick<String>,
    pub output: Pick<OutputTarget>,
    lib: Option<Arc<Library>>,
    /// The user's config file (loaded with the others).
    custom: Option<std::path::PathBuf>,
    loading: Option<Receiver<Library>>,
    /// The conversion shown, and one being built.
    shown: Option<u64>,
    building: Option<(u64, Receiver<anyhow::Result<brp_render::OcioSetup>>)>,
    pub error: Option<String>,
}

/// Settings' default for an output: "display:<entry id>" or "space:<entry id>".
pub fn output_id(t: &OutputTarget) -> String {
    match t {
        OutputTarget::Display(e) => format!("display:{}", e.id()),
        OutputTarget::DisplayView(e, v) => format!("view:{}|{v}", e.id()),
        OutputTarget::Space(e) => format!("space:{}", e.id()),
    }
}

/// From `output_id` ("view:<display id>|<view>" for a display with its own view).
pub fn output_from_id(s: &str) -> Option<OutputTarget> {
    if let Some(r) = s.strip_prefix("display:") {
        return Entry::from_id(r).map(OutputTarget::Display);
    }
    if let Some(r) = s.strip_prefix("view:") {
        let (d, v) = r.rsplit_once('|')?;
        return Entry::from_id(d).map(|e| OutputTarget::DisplayView(e, v.to_string()));
    }
    s.strip_prefix("space:").and_then(Entry::from_id).map(OutputTarget::Space)
}

/// How a file's colour reads for OCIO.
pub fn media_of(info: &MediaInfo) -> Media {
    let c: &ColorInfo = &info.color;
    match c.transfer {
        Transfer::Pq => Media::Pq,
        Transfer::Hlg => Media::Hlg,
        Transfer::Linear => Media::Linear { bt2020: c.primaries == brp_media::Primaries::Bt2020 },
        Transfer::Sdr if c.matrix == brp_media::Matrix::Rgb || info.container == "image sequence" || info.frame_count_hint <= 1 => Media::SrgbImage,
        Transfer::Sdr => Media::SdrVideo,
    }
}

/// Where OCIO's configs are: `ocio/` next to the executable.
pub fn config_dir() -> std::path::PathBuf {
    std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("ocio"))).unwrap_or_else(|| "ocio".into())
}

impl Ocio {
    /// From the settings: on or off, each choice Default when a default is set.
    pub fn new(s: &crate::settings::Settings) -> Ocio {
        Ocio {
            enabled: s.ocio_enabled,
            input: if s.ocio_input.is_some() { Pick::Default } else { Pick::Auto },
            display: if s.ocio_display.is_some() { Pick::Default } else { Pick::Auto },
            view: if s.ocio_view.is_some() { Pick::Default } else { Pick::Auto },
            output: if s.ocio_output.is_some() { Pick::Default } else { Pick::Auto },
            lib: None,
            custom: s.ocio_custom.as_ref().map(std::path::PathBuf::from),
            loading: None,
            shown: None,
            building: None,
            error: None,
        }
    }

    /// The configs, once loaded (loading starts on the first call).
    pub fn library(&mut self) -> Option<Arc<Library>> {
        if self.lib.is_none() {
            match &self.loading {
                None => {
                    let custom = self.custom.clone();
                    let (tx, rx) = crossbeam_channel::bounded(1);
                    let _ = std::thread::Builder::new().name("brp-ocio-load".into()).spawn(move || {
                        let _ = tx.send(Library::load(&config_dir(), custom.as_deref()));
                    });
                    self.loading = Some(rx);
                }
                Some(rx) => {
                    if let Ok(l) = rx.try_recv() {
                        self.lib = Some(Arc::new(l));
                        self.loading = None;
                    }
                }
            }
        }
        self.lib.clone()
    }

    /// Use the config file `path` (None: none) besides the shipped ones: the configs
    /// load again, and the window's conversion is made again.
    pub fn set_custom(&mut self, path: Option<std::path::PathBuf>) {
        self.custom = path;
        self.lib = None;
        self.loading = None;
        self.shown = None;
        self.building = None;
        self.error = None;
    }

    /// The window dropped its conversion (OCIO off): the next `update` with OCIO on
    /// builds and sends it again.
    pub fn window_off(&mut self) {
        self.shown = None;
    }

    /// Still loading the configs or building the conversion.
    pub fn busy(&self) -> bool {
        self.loading.is_some() || self.building.is_some()
    }

    /// The input, display and view for `media` on `screen`.
    pub fn resolve(&self, lib: &Library, s: &crate::settings::Settings, media: Media, screen: Screen) -> Option<Resolved> {
        let def = |o: &Option<String>| o.as_deref().and_then(Entry::from_id);
        let input = match &self.input {
            Pick::Manual(e) if lib.has_space(e) => Some(e.clone()),
            Pick::Default => def(&s.ocio_input).filter(|e| lib.has_space(e)),
            _ => None,
        }
        .or_else(|| auto::input(lib, media))?;
        let display = match &self.display {
            Pick::Manual(e) if lib.has_display(e) => Some(e.clone()),
            Pick::Default => def(&s.ocio_display).filter(|e| lib.has_display(e)),
            _ => None,
        }
        .or_else(|| auto::display(lib, screen))?;
        let view = match &self.view {
            Pick::Manual(v) => auto::view_named(lib, &display, v, media),
            Pick::Default => match &s.ocio_view {
                Some(v) => auto::view_named(lib, &display, v, media),
                None => auto::view(lib, &display, media),
            },
            Pick::Auto => auto::view(lib, &display, media),
        }?;
        Some(Resolved { input, display, view })
    }

    /// The output target for an export of kind `kind`.
    pub fn resolve_output(&self, lib: &Library, s: &crate::settings::Settings, kind: auto::Output) -> Option<OutputTarget> {
        let ok = |t: &OutputTarget| match t {
            OutputTarget::Display(e) => lib.has_display(e),
            OutputTarget::DisplayView(e, v) => lib.views(e).contains(v),
            OutputTarget::Space(e) => lib.has_space(e),
        };
        match &self.output {
            Pick::Manual(t) if ok(t) => Some(t.clone()),
            Pick::Default => s.ocio_output.as_deref().and_then(output_from_id).filter(ok),
            _ => None,
        }
        .or_else(|| auto::output(lib, kind))
    }

    /// Keep the window's conversion up to date (each frame): None when off or not
    /// ready yet.
    pub fn update(&mut self, s: &crate::settings::Settings, info: Option<&MediaInfo>, screen: Screen) -> Option<brp_render::OcioSetup> {
        if !self.enabled {
            self.shown = None;
            return None;
        }
        let lib = self.library()?;
        let media = info.map(media_of).unwrap_or(Media::SdrVideo);
        let Some(r) = self.resolve(&lib, s, media, screen) else {
            self.error = Some("no colour space for this file in the configs".into());
            return None;
        };
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };
        if let Some((k, rx)) = &self.building {
            if let Ok(res) = rx.try_recv() {
                let k = *k;
                self.building = None;
                match res {
                    Ok(setup) => {
                        self.shown = Some(k);
                        self.error = None;
                        return Some(setup);
                    }
                    Err(e) => {
                        log::error!("OCIO: {e:#}");
                        self.error = Some(format!("{:#}", friendly(e)));
                        self.shown = Some(k);
                    }
                }
            }
        }
        let pending = self.building.as_ref().map(|(k, _)| *k);
        if self.shown != Some(key) && pending != Some(key) {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let lib2 = lib.clone();
            let r2 = r.clone();
            let _ = std::thread::Builder::new().name("brp-ocio-build".into()).spawn(move || {
                let t = std::time::Instant::now();
                let res = build_setup(&lib2, &r2, key);
                log::info!("OCIO {} -> {} / {}: built in {:?}", r2.input.label(), r2.display.label(), r2.view, t.elapsed());
                let _ = tx.send(res);
            });
            self.building = Some((key, rx));
        }
        None
    }
}

/// OCIO's GLSL and textures as the renderer takes them.
pub fn to_render(s: brp_ocio::Shader) -> brp_render::OcioShader {
    brp_render::OcioShader {
        glsl: s.glsl,
        textures: s
            .textures
            .into_iter()
            .map(|t| brp_render::OcioTexture { binding: t.binding, dims: t.dims, w: t.w, h: t.h, d: t.d, channels: t.channels, linear: t.linear, data: t.data })
            .collect(),
    }
}

fn build_setup(lib: &Library, r: &Resolved, key: u64) -> anyhow::Result<brp_render::OcioSetup> {
    let src = lib.get(r.input.source).ok_or_else(|| anyhow::anyhow!("config not loaded"))?;
    let dst = lib.get(r.display.source).ok_or_else(|| anyhow::anyhow!("config not loaded"))?;
    let target = Target::Display { display: &r.display.name, view: &r.view };
    let direct = brp_ocio::shader(src, &r.input.name, dst, target, false, brp_render::ocio::OCIO_SET, 1)?;
    let xyz = brp_ocio::shader(src, &r.input.name, dst, target, true, brp_render::ocio::OCIO_SET, 1)?;
    anyhow::ensure!(direct.uniforms == 0 && xyz.uniforms == 0, "this conversion has dynamic parameters (not supported)");
    let hdr_display = matches!(auto::encoding(lib, &r.display, true), auto::Encoding::Pq | auto::Encoding::Hlg);
    Ok(brp_render::OcioSetup { direct: to_render(direct), xyz: to_render(xyz), hdr_display, key })
}

/// The export's OCIO (DESIGN.md §7d): the file's colour space through the output
/// target (and the view, for a display), and how the output file is tagged.
#[derive(Clone)]
pub struct ExportOcio {
    pub shader: brp_render::OcioShader,
    /// The same on the CPU (`--verify`).
    pub cpu: Arc<brp_ocio::Cpu>,
    pub encoding: auto::Encoding,
    pub label: String,
}

/// What an export of kind `kind` writes with OCIO on: the target and its conversion.
#[allow(clippy::too_many_arguments)]
pub fn export_ocio(lib: &Library, o: &OcioChoices, s: &crate::settings::Settings, info: &MediaInfo, kind: auto::Output) -> anyhow::Result<ExportOcio> {
    let media = media_of(info);
    let probe = Ocio { enabled: true, input: o.input.clone(), display: o.display.clone(), view: o.view.clone(), output: o.output.clone(), lib: None, custom: None, loading: None, shown: None, building: None, error: None };
    // The view is the window's (resolved on the output display when it is one).
    let r = probe.resolve(lib, s, media, Screen::Sdr).ok_or_else(|| anyhow::anyhow!("no OCIO colour space for this file"))?;
    let target = probe.resolve_output(lib, s, kind).ok_or_else(|| anyhow::anyhow!("no OCIO output for this format"))?;
    let src = lib.get(r.input.source).ok_or_else(|| anyhow::anyhow!("config not loaded"))?;
    let (shader, cpu, encoding, label) = match &target {
        OutputTarget::Display(d) | OutputTarget::DisplayView(d, _) => {
            let view = match (&target, &o.view) {
                (OutputTarget::DisplayView(_, v), _) => Some(v.clone()),
                (_, Pick::Auto) => auto::view(lib, d, media),
                _ => auto::view_named(lib, d, &r.view, media),
            }
            .ok_or_else(|| anyhow::anyhow!("display {} has no views", d.label()))?;
            let dst = lib.get(d.source).ok_or_else(|| anyhow::anyhow!("config not loaded"))?;
            let t = Target::Display { display: &d.name, view: &view };
            let sh = brp_ocio::shader(src, &r.input.name, dst, t, false, 1, 1)?;
            let cpu = brp_ocio::Cpu::new(src, &r.input.name, dst, t, false)?;
            (sh, cpu, auto::encoding(lib, d, true), format!("{} → {} / {view}", r.input.label(), d.label()))
        }
        OutputTarget::Space(c) => {
            let dst = lib.get(c.source).ok_or_else(|| anyhow::anyhow!("config not loaded"))?;
            let sh = brp_ocio::shader(src, &r.input.name, dst, Target::Space(&c.name), false, 1, 1)?;
            let cpu = brp_ocio::Cpu::new(src, &r.input.name, dst, Target::Space(&c.name), false)?;
            (sh, cpu, auto::encoding(lib, c, false), format!("{} → {}", r.input.label(), c.label()))
        }
    };
    anyhow::ensure!(shader.uniforms == 0, "this conversion has dynamic parameters (not supported)");
    Ok(ExportOcio { shader: to_render(shader), cpu: Arc::new(cpu), encoding, label })
}

/// The Color panel's choices, as an export takes them.
#[derive(Clone, Debug)]
pub struct OcioChoices {
    pub input: Pick<Entry>,
    pub display: Pick<Entry>,
    pub view: Pick<String>,
    pub output: Pick<OutputTarget>,
}

impl Ocio {
    pub fn choices(&self) -> OcioChoices {
        OcioChoices { input: self.input.clone(), display: self.display.clone(), view: self.view.clone(), output: self.output.clone() }
    }

    /// The configs if already loaded (does not start loading).
    pub fn loaded(&self) -> Option<Arc<Library>> {
        self.lib.clone()
    }
}

/// OCIO's errors as the panel shows them: the one about mixing configs explained.
pub fn friendly(e: anyhow::Error) -> anyhow::Error {
    let m = format!("{e:#}");
    if m.contains("interchange") {
        return anyhow::anyhow!("These two configs cannot be mixed: one has no interchange roles (aces_interchange, cie_xyz_d65_interchange), as older configs often do. Pick the input, display and output from the same config. (OCIO: {m})");
    }
    e
}
