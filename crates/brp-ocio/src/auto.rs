//! Automatic choices (DESIGN.md §7d): the input from the file, the display from the
//! screen, the output from the export format. All from OCIO's built-in config, found by
//! name or alias, so they hold across its versions.

use crate::{Entry, Library, Source};

/// What a file holds, as far as colour goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Media {
    /// Video (Rec.709, gamma 2.4).
    SdrVideo,
    /// Still images and RGB files (sRGB).
    SrgbImage,
    Pq,
    Hlg,
    /// Linear light (EXR, renders).
    Linear { bt2020: bool },
}

impl Media {
    /// Display-referred: the picture as a screen shows it (video, images), not light
    /// from a scene (renders).
    pub fn display_referred(self) -> bool {
        !matches!(self, Media::Linear { .. })
    }
}

/// The screen the window is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Sdr,
    /// HDR (PQ swapchain).
    Hdr,
}

/// What an export writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// Video for SDR (Rec.709): DNxHD/HR, ProRes, H.264…
    SdrVideo,
    /// Still images (sRGB): PNG, JPEG, TIFF, GIF…
    Image,
    /// HDR video kept HDR.
    Pq,
    Hlg,
    /// Linear light (EXR).
    Linear,
}

/// Where an output goes: a display through the view, or a plain colour space.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OutputTarget {
    /// A display through the window's view.
    Display(Entry),
    /// A display through a view of its own (AgX, Filmic…).
    DisplayView(Entry, String),
    Space(Entry),
}

fn builtin_space(lib: &Library, candidates: &[&str]) -> Option<Entry> {
    lib.get(Source::Builtin)?.find_space(candidates).map(|n| Entry::new(Source::Builtin, n))
}

fn builtin_display(lib: &Library, candidates: &[&str]) -> Option<Entry> {
    lib.get(Source::Builtin)?.find_display(candidates).map(|n| Entry::new(Source::Builtin, n))
}

/// The input colour space for a file.
pub fn input(lib: &Library, m: Media) -> Option<Entry> {
    match m {
        Media::SdrVideo => builtin_space(lib, &["Rec.1886 Rec.709 - Display", "rec1886_rec709_display"]),
        Media::SrgbImage => builtin_space(lib, &["sRGB - Display", "srgb_display"]),
        Media::Pq => builtin_space(lib, &["Rec.2100-PQ - Display", "rec2100_pq_display"]),
        Media::Hlg => builtin_space(lib, &["Rec.2100-HLG - Display", "rec2100_hlg_display"]),
        Media::Linear { bt2020: false } => builtin_space(lib, &["Linear Rec.709 (sRGB)", "lin_rec709_srgb", "lin_rec709"]),
        Media::Linear { bt2020: true } => builtin_space(lib, &["Linear Rec.2020", "lin_rec2020"]),
    }
}

/// The display for the screen.
pub fn display(lib: &Library, s: Screen) -> Option<Entry> {
    match s {
        Screen::Sdr => builtin_display(lib, &["sRGB - Display"]),
        Screen::Hdr => builtin_display(lib, &["Rec.2100-PQ - Display"]),
    }
}

/// The view on `display` for a file: video and images shown as they are ("Video
/// (colorimetric)", or the like), renders through the display's own default (a tone
/// map: ACES, AgX…).
pub fn view(lib: &Library, display: &Entry, m: Media) -> Option<String> {
    let d = lib.get(display.source)?.display(&display.name)?;
    if m.display_referred() {
        for want in ["Video (colorimetric)", "Un-tone-mapped", "Standard"] {
            if let Some(v) = d.views.iter().find(|v| v.name == want) {
                return Some(v.name.clone());
            }
        }
    }
    Some(d.default_view.clone())
}

/// The view called `name` on `display`, else what `view` would choose.
pub fn view_named(lib: &Library, display: &Entry, name: &str, m: Media) -> Option<String> {
    let d = lib.get(display.source)?.display(&display.name)?;
    if d.views.iter().any(|v| v.name == name) {
        return Some(name.to_string());
    }
    view(lib, display, m)
}

/// Where an export of kind `o` goes.
pub fn output(lib: &Library, o: Output) -> Option<OutputTarget> {
    match o {
        Output::SdrVideo => builtin_display(lib, &["Rec.1886 Rec.709 - Display"]).map(OutputTarget::Display),
        Output::Image => builtin_display(lib, &["sRGB - Display"]).map(OutputTarget::Display),
        Output::Pq => builtin_display(lib, &["Rec.2100-PQ - Display"]).map(OutputTarget::Display),
        Output::Hlg => builtin_display(lib, &["Rec.2100-HLG - Display"]).map(OutputTarget::Display),
        Output::Linear => builtin_space(lib, &["Linear Rec.709 (sRGB)", "lin_rec709_srgb"]).map(OutputTarget::Space),
    }
}

/// How a display or colour space encodes light, to write it into the swapchain or tag
/// an exported file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Sdr,
    Pq,
    Hlg,
    Linear,
}

/// The encoding of a display (its default view's colour space) or a colour space,
/// from OCIO's `encoding` attribute, else its name.
pub fn encoding(lib: &Library, e: &Entry, is_display: bool) -> Encoding {
    let Some(c) = lib.get(e.source) else { return Encoding::Sdr };
    let (enc, name) = if is_display {
        match c.display(&e.name) {
            Some(d) => {
                let v = d.views.iter().find(|v| v.encoding != "data").or(d.views.first());
                (v.map(|v| v.encoding.clone()).unwrap_or_default(), v.map(|v| v.colorspace.clone()).unwrap_or_else(|| e.name.clone()))
            }
            None => (String::new(), e.name.clone()),
        }
    } else {
        (c.space(&e.name).map(|s| s.encoding.clone()).unwrap_or_default(), e.name.clone())
    };
    let n = name.to_ascii_lowercase();
    if n.contains("hlg") {
        Encoding::Hlg
    } else if n.contains("pq") || n.contains("2084") {
        Encoding::Pq
    } else if enc == "scene-linear" || enc == "display-linear" || n.contains("linear") || n.starts_with("acescg") || n.starts_with("aces2065") {
        Encoding::Linear
    } else {
        Encoding::Sdr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_auto_choices() {
        let lib = Library::load(std::path::Path::new("nonexistent"), None);
        let e = input(&lib, Media::SdrVideo).unwrap();
        assert_eq!(e.name, "Rec.1886 Rec.709 - Display");
        assert_eq!(input(&lib, Media::Linear { bt2020: false }).unwrap().name, "Linear Rec.709 (sRGB)");
        let d = display(&lib, Screen::Hdr).unwrap();
        assert_eq!(encoding(&lib, &d, true), Encoding::Pq);
        assert_eq!(view(&lib, &d, Media::SdrVideo).unwrap(), "Video (colorimetric)");
        assert!(view(&lib, &display(&lib, Screen::Sdr).unwrap(), Media::Linear { bt2020: false }).unwrap().starts_with("ACES"));
        assert!(matches!(output(&lib, Output::Linear), Some(OutputTarget::Space(_))));
        assert_eq!(Entry::from_id(&Entry::new(Source::Blender, "AgX Log").id()), Some(Entry::new(Source::Blender, "AgX Log")));
        assert_eq!(Entry::new(Source::Maya, "ACEScg").label(), "ACEScg (maya)");
    }
}
