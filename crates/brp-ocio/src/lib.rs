//! OpenColorIO for the player (DESIGN.md §7d): configs (OCIO's built-in studio config,
//! and the Maya and Blender configs shipped next to the executable), what they offer,
//! and conversions as GPU shaders (Vulkan GLSL + LUT textures) or CPU processors.

pub mod auto;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::Arc;

unsafe extern "C" {
    fn brp_ocio_free(s: *mut c_char);
    fn brp_ocio_version() -> *const c_char;
    fn brp_ocio_load(path: *const c_char, err: *mut *mut c_char) -> *mut c_void;
    fn brp_ocio_release(c: *mut c_void);
    fn brp_ocio_describe(c: *mut c_void) -> *mut c_char;
    fn brp_ocio_shader(
        src: *mut c_void,
        src_cs: *const c_char,
        dst: *mut c_void,
        display: *const c_char,
        view: *const c_char,
        dst_cs: *const c_char,
        to_xyz: i32,
        part: i32,
        set: u32,
        binding_start: u32,
        err: *mut *mut c_char,
    ) -> *mut c_void;
    fn brp_ocio_shader_release(s: *mut c_void);
    fn brp_ocio_shader_text(s: *mut c_void) -> *const c_char;
    fn brp_ocio_shader_uniforms(s: *mut c_void) -> u32;
    fn brp_ocio_shader_textures(s: *mut c_void) -> u32;
    fn brp_ocio_shader_textures3d(s: *mut c_void) -> u32;
    fn brp_ocio_shader_texture(
        s: *mut c_void,
        i: u32,
        name: *mut *const c_char,
        w: *mut u32,
        h: *mut u32,
        channels: *mut u32,
        dims: *mut u32,
        linear: *mut i32,
        binding: *mut u32,
        values: *mut *const f32,
    );
    fn brp_ocio_shader_texture3d(s: *mut c_void, i: u32, name: *mut *const c_char, edge: *mut u32, linear: *mut i32, binding: *mut u32, values: *mut *const f32);
    fn brp_ocio_cpu(
        src: *mut c_void,
        src_cs: *const c_char,
        dst: *mut c_void,
        display: *const c_char,
        view: *const c_char,
        dst_cs: *const c_char,
        to_xyz: i32,
        part: i32,
        err: *mut *mut c_char,
    ) -> *mut c_void;
    fn brp_ocio_cpu_release(c: *mut c_void);
    fn brp_ocio_cpu_apply(c: *mut c_void, rgb: *mut f32, n: i64);
}

/// OpenColorIO's version.
pub fn version() -> String {
    unsafe { CStr::from_ptr(brp_ocio_version()).to_string_lossy().into_owned() }
}

fn take_err(e: *mut c_char) -> anyhow::Error {
    if e.is_null() {
        return anyhow!("OpenColorIO error");
    }
    let m = unsafe { CStr::from_ptr(e).to_string_lossy().into_owned() };
    unsafe { brp_ocio_free(e) };
    anyhow!("{m}")
}

fn cs(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

/// A colour space as the interface lists it.
#[derive(Clone, Debug)]
pub struct Space {
    pub name: String,
    pub family: String,
    /// OCIO's encoding: scene-linear, sdr-video, hdr-video, log, data… (may be empty).
    pub encoding: String,
    pub data: bool,
    /// Display-referred (a display's colour space, e.g. "sRGB - Display").
    pub display: bool,
    pub aliases: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct View {
    pub name: String,
    /// The colour space the view ends in on its display.
    pub colorspace: String,
    pub encoding: String,
}

#[derive(Clone, Debug)]
pub struct Display {
    pub name: String,
    pub default_view: String,
    pub views: Vec<View>,
}

/// A loaded OCIO config.
pub struct Config {
    ptr: *mut c_void,
    pub name: String,
    pub description: String,
    pub spaces: Vec<Space>,
    pub displays: Vec<Display>,
    pub default_display: String,
    pub roles: HashMap<String, String>,
}

// OCIO configs are immutable once loaded and safe to share between threads.
unsafe impl Send for Config {}
unsafe impl Sync for Config {}

impl Drop for Config {
    fn drop(&mut self) {
        unsafe { brp_ocio_release(self.ptr) };
    }
}

impl Config {
    /// A config file, or a built-in config ("ocio://studio-config-latest").
    pub fn load(path: &str) -> Result<Config> {
        let mut err = std::ptr::null_mut();
        let ptr = unsafe { brp_ocio_load(cs(path).as_ptr(), &mut err) };
        if ptr.is_null() {
            return Err(take_err(err));
        }
        let text = unsafe {
            let p = brp_ocio_describe(ptr);
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            brp_ocio_free(p);
            s
        };
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let s = |v: &serde_json::Value, k: &str| v[k].as_str().unwrap_or_default().to_string();
        let spaces = v["colorspaces"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| Space {
                        name: s(c, "name"),
                        family: s(c, "family"),
                        encoding: s(c, "encoding"),
                        data: c["data"].as_bool().unwrap_or(false),
                        display: c["display"].as_bool().unwrap_or(false),
                        aliases: c["aliases"].as_array().map(|x| x.iter().filter_map(|a| a.as_str().map(str::to_string)).collect()).unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let displays = v["displays"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|d| Display {
                        name: s(d, "name"),
                        default_view: s(d, "default_view"),
                        views: d["views"].as_array().map(|x| x.iter().map(|w| View { name: s(w, "name"), colorspace: s(w, "colorspace"), encoding: s(w, "encoding") }).collect()).unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let roles = v["roles"].as_object().map(|o| o.iter().map(|(k, x)| (k.clone(), x.as_str().unwrap_or_default().to_string())).collect()).unwrap_or_default();
        Ok(Config { ptr, name: s(&v, "name"), description: s(&v, "description"), spaces, displays, default_display: s(&v, "default_display"), roles })
    }

    /// The first of `candidates` that names a colour space here (by name or alias, case
    /// does not matter), as the space's own name.
    pub fn find_space(&self, candidates: &[&str]) -> Option<String> {
        candidates.iter().find_map(|c| {
            self.spaces.iter().find(|s| s.name.eq_ignore_ascii_case(c) || s.aliases.iter().any(|a| a.eq_ignore_ascii_case(c))).map(|s| s.name.clone())
        })
    }

    /// The first of `candidates` that names a display here.
    pub fn find_display(&self, candidates: &[&str]) -> Option<String> {
        candidates.iter().find_map(|c| self.displays.iter().find(|d| d.name.eq_ignore_ascii_case(c)).map(|d| d.name.clone()))
    }

    pub fn space(&self, name: &str) -> Option<&Space> {
        self.spaces.iter().find(|s| s.name == name)
    }

    pub fn display(&self, name: &str) -> Option<&Display> {
        self.displays.iter().find(|d| d.name == name)
    }

    pub fn role(&self, role: &str) -> Option<&str> {
        self.roles.get(role).map(String::as_str)
    }
}

/// Where a conversion ends.
#[derive(Clone, Copy, Debug)]
pub enum Target<'a> {
    /// A colour space (no view: a plain conversion).
    Space(&'a str),
    /// A display through one of its views.
    Display { display: &'a str, view: &'a str },
}

/// Which part of a conversion (DESIGN.md §7d): all of it, or one side of the effects'
/// working space (linear light, Rec.2020 primaries, 1.0 = SDR white). `Out` after
/// `In` is `Whole`: both meet at an interchange colour space of the input's config
/// (display-referred CIE-XYZ D65 for display inputs, no view or tone curve; ACES2065-1
/// for scene-referred ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Whole,
    /// The input to the working space (the target is not used).
    In,
    /// The working space to the target.
    Out,
}

impl Part {
    fn id(self) -> i32 {
        match self {
            Part::Whole => 0,
            Part::In => 1,
            Part::Out => 2,
        }
    }
}

/// One LUT texture of a GPU shader.
#[derive(Clone, Debug)]
pub struct Texture {
    pub name: String,
    /// Binding in the shader's descriptor set.
    pub binding: u32,
    /// 1, 2 or 3 dimensions.
    pub dims: u8,
    pub w: u32,
    pub h: u32,
    pub d: u32,
    /// 1 (red) or 3 (RGB) channels per texel.
    pub channels: u8,
    pub linear: bool,
    pub data: Vec<f32>,
}

/// A conversion on the GPU: GLSL (Vulkan 4.6) defining `vec4 OCIOMain(vec4)`, and the
/// textures it samples.
#[derive(Clone, Debug)]
pub struct Shader {
    pub glsl: String,
    pub textures: Vec<Texture>,
    /// Dynamic parameters (a uniform block); none for the conversions we make.
    pub uniforms: u32,
}

fn ptrs(t: Target<'_>) -> (CString, CString, CString) {
    match t {
        Target::Space(s) => (cs(""), cs(""), cs(s)),
        Target::Display { display, view } => (cs(display), cs(view), cs("")),
    }
}

/// `src_cs` (of `src`) to `target` (of `dst`) on the GPU; `to_xyz`: then on to the
/// display-referred CIE-XYZ D65 of `dst` (1.0 = the display's white, or 100 nits for HDR
/// displays). Textures go in descriptor set `set` from `binding_start`.
pub fn shader(src: &Config, src_cs: &str, dst: &Config, target: Target<'_>, to_xyz: bool, set: u32, binding_start: u32) -> Result<Shader> {
    shader_part(src, src_cs, dst, target, to_xyz, Part::Whole, set, binding_start)
}

/// `part` of the conversion `shader` makes.
#[allow(clippy::too_many_arguments)]
pub fn shader_part(src: &Config, src_cs: &str, dst: &Config, target: Target<'_>, to_xyz: bool, part: Part, set: u32, binding_start: u32) -> Result<Shader> {
    let (d, v, c) = ptrs(target);
    let mut err = std::ptr::null_mut();
    let s = unsafe { brp_ocio_shader(src.ptr, cs(src_cs).as_ptr(), dst.ptr, d.as_ptr(), v.as_ptr(), c.as_ptr(), to_xyz as i32, part.id(), set, binding_start, &mut err) };
    if s.is_null() {
        return Err(take_err(err));
    }
    let out = unsafe {
        let glsl = CStr::from_ptr(brp_ocio_shader_text(s)).to_string_lossy().into_owned();
        let mut textures = Vec::new();
        for i in 0..brp_ocio_shader_textures(s) {
            let (mut name, mut w, mut h, mut ch, mut dims, mut lin, mut b, mut vals) = (std::ptr::null(), 0, 0, 0, 0, 0, 0, std::ptr::null());
            brp_ocio_shader_texture(s, i, &mut name, &mut w, &mut h, &mut ch, &mut dims, &mut lin, &mut b, &mut vals);
            let n = (w * h.max(1) * ch) as usize;
            textures.push(Texture {
                name: CStr::from_ptr(name).to_string_lossy().into_owned(),
                binding: b,
                dims: dims as u8,
                w,
                h: h.max(1),
                d: 1,
                channels: ch as u8,
                linear: lin != 0,
                data: std::slice::from_raw_parts(vals, n).to_vec(),
            });
        }
        for i in 0..brp_ocio_shader_textures3d(s) {
            let (mut name, mut e, mut lin, mut b, mut vals) = (std::ptr::null(), 0, 0, 0, std::ptr::null());
            brp_ocio_shader_texture3d(s, i, &mut name, &mut e, &mut lin, &mut b, &mut vals);
            let n = (e * e * e * 3) as usize;
            textures.push(Texture {
                name: CStr::from_ptr(name).to_string_lossy().into_owned(),
                binding: b,
                dims: 3,
                w: e,
                h: e,
                d: e,
                channels: 3,
                linear: lin != 0,
                data: std::slice::from_raw_parts(vals, n).to_vec(),
            });
        }
        Shader { glsl, textures, uniforms: brp_ocio_shader_uniforms(s) }
    };
    unsafe { brp_ocio_shader_release(s) };
    Ok(out)
}

/// The same conversion on the CPU (checks).
pub struct Cpu(*mut c_void);

unsafe impl Send for Cpu {}
unsafe impl Sync for Cpu {}

impl Drop for Cpu {
    fn drop(&mut self) {
        unsafe { brp_ocio_cpu_release(self.0) };
    }
}

impl Cpu {
    pub fn new(src: &Config, src_cs: &str, dst: &Config, target: Target<'_>, to_xyz: bool) -> Result<Cpu> {
        Cpu::part(src, src_cs, dst, target, to_xyz, Part::Whole)
    }

    /// `part` of the conversion.
    pub fn part(src: &Config, src_cs: &str, dst: &Config, target: Target<'_>, to_xyz: bool, part: Part) -> Result<Cpu> {
        let (d, v, c) = ptrs(target);
        let mut err = std::ptr::null_mut();
        let p = unsafe { brp_ocio_cpu(src.ptr, cs(src_cs).as_ptr(), dst.ptr, d.as_ptr(), v.as_ptr(), c.as_ptr(), to_xyz as i32, part.id(), &mut err) };
        if p.is_null() {
            return Err(take_err(err));
        }
        Ok(Cpu(p))
    }

    /// Packed RGB pixels, in place.
    pub fn apply(&self, rgb: &mut [f32]) {
        unsafe { brp_ocio_cpu_apply(self.0, rgb.as_mut_ptr(), (rgb.len() / 3) as i64) };
    }
}

// ---- the player's configs ----------------------------------------------------------

/// Which config an entry comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// OCIO's own (built-in studio config): names shown as they are.
    Builtin,
    Maya,
    Blender,
    /// A config file the user loaded.
    Custom,
}

impl Source {
    pub const ALL: [Source; 4] = [Source::Builtin, Source::Maya, Source::Blender, Source::Custom];

    /// Appended to entry names in the interface.
    pub fn suffix(self) -> &'static str {
        match self {
            Source::Builtin => "",
            Source::Maya => " (maya)",
            Source::Blender => " (blender)",
            Source::Custom => " (custom)",
        }
    }

    fn id(self) -> &'static str {
        match self {
            Source::Builtin => "ocio",
            Source::Maya => "maya",
            Source::Blender => "blender",
            Source::Custom => "custom",
        }
    }
}

/// A colour space or display of one of the configs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Entry {
    pub source: Source,
    pub name: String,
}

impl Entry {
    pub fn new(source: Source, name: impl Into<String>) -> Entry {
        Entry { source, name: name.into() }
    }

    /// As the interface shows it: "ACEScg", "ACEScg (maya)", "AgX Log (blender)".
    pub fn label(&self) -> String {
        format!("{}{}", self.name, self.source.suffix())
    }

    /// For settings: "maya:ACEScg".
    pub fn id(&self) -> String {
        format!("{}:{}", self.source.id(), self.name)
    }

    pub fn from_id(s: &str) -> Option<Entry> {
        let (src, name) = s.split_once(':')?;
        let source = Source::ALL.into_iter().find(|x| x.id() == src)?;
        Some(Entry::new(source, name))
    }
}

/// The built-in config used for OCIO's own entries.
pub const BUILTIN: &str = "ocio://studio-config-latest";

/// The configs the player offers: OCIO's, and Maya's and Blender's when shipped
/// (`<dir>/maya/config.ocio`, `<dir>/blender/config.ocio`), and a config file of the
/// user's own.
pub struct Library {
    configs: Vec<(Source, Arc<Config>)>,
    /// The user's config: its name in the interface, or why it did not load.
    pub custom_name: Option<String>,
    pub custom_error: Option<String>,
}

impl Library {
    /// The shipped configs from `dir`, and `custom` (a `.ocio` file) if given.
    pub fn load(dir: &Path, custom: Option<&Path>) -> Library {
        let mut configs = Vec::new();
        let (mut custom_name, mut custom_error) = (None, None);
        let mut list = vec![
            (Source::Builtin, BUILTIN.to_string()),
            (Source::Maya, dir.join("maya/config.ocio").to_string_lossy().into_owned()),
            (Source::Blender, dir.join("blender/config.ocio").to_string_lossy().into_owned()),
        ];
        if let Some(p) = custom {
            list.push((Source::Custom, p.to_string_lossy().into_owned()));
        }
        for (source, path) in list {
            let t = std::time::Instant::now();
            match Config::load(&path) {
                Ok(c) => {
                    log::info!("OCIO config {:?} ({}): {} spaces, {} displays in {:?}", c.name, source.id(), c.spaces.len(), c.displays.len(), t.elapsed());
                    if source == Source::Custom {
                        // Its own name, else the file's folder ("my_show/config.ocio": "my_show").
                        let p = Path::new(&path);
                        let folder = p.parent().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned());
                        let stem = p.file_stem().map(|n| n.to_string_lossy().into_owned()).filter(|s| s != "config");
                        custom_name = Some(Some(c.name.clone()).filter(|n| !n.is_empty()).or(stem).or(folder).unwrap_or_else(|| "Custom".into()));
                    }
                    configs.push((source, Arc::new(c)));
                }
                Err(e) => {
                    log::warn!("OCIO config {path}: {e:#}");
                    if source == Source::Custom {
                        custom_error = Some(format!("{e:#}"));
                    }
                }
            }
        }
        Library { configs, custom_name, custom_error }
    }

    pub fn get(&self, s: Source) -> Option<&Arc<Config>> {
        self.configs.iter().find(|(x, _)| *x == s).map(|(_, c)| c)
    }

    pub fn sources(&self) -> impl Iterator<Item = (Source, &Arc<Config>)> {
        self.configs.iter().map(|(s, c)| (*s, c))
    }

    /// Every active colour space, config by config.
    pub fn spaces(&self) -> Vec<Entry> {
        self.configs.iter().flat_map(|(s, c)| c.spaces.iter().map(move |x| Entry::new(*s, x.name.clone()))).collect()
    }

    /// Every display, config by config.
    pub fn displays(&self) -> Vec<Entry> {
        self.configs.iter().flat_map(|(s, c)| c.displays.iter().map(move |x| Entry::new(*s, x.name.clone()))).collect()
    }

    /// `e` names a colour space here.
    pub fn has_space(&self, e: &Entry) -> bool {
        self.get(e.source).is_some_and(|c| c.space(&e.name).is_some())
    }

    pub fn has_display(&self, e: &Entry) -> bool {
        self.get(e.source).is_some_and(|c| c.display(&e.name).is_some())
    }

    /// The views of display `e`.
    pub fn views(&self, e: &Entry) -> Vec<String> {
        self.get(e.source).and_then(|c| c.display(&e.name)).map(|d| d.views.iter().map(|v| v.name.clone()).collect()).unwrap_or_default()
    }
}

impl Library {
    /// The name a config's entries are listed under.
    pub fn group_name(&self, s: Source) -> String {
        match s {
            Source::Builtin => "OCIO".into(),
            Source::Maya => "Maya".into(),
            Source::Blender => "Blender".into(),
            Source::Custom => self.custom_name.clone().unwrap_or_else(|| "Custom".into()),
        }
    }
}

#[cfg(test)]
mod split_tests {
    //! The effects' working space (DESIGN.md §7d): `Part::Out` after `Part::In` is the
    //! whole conversion, and `Part::In` gives the player's own linear light where the
    //! player can decode the file itself.
    use super::*;

    fn lib() -> Library {
        Library::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/ocio-configs"), None)
    }

    /// Pixels over `lo..hi` on each channel (a 9³ grid).
    fn grid(lo: f32, hi: f32) -> Vec<f32> {
        let mut v = Vec::new();
        for r in 0..9 {
            for g in 0..9 {
                for b in 0..9 {
                    v.extend([r, g, b].map(|i| lo + (hi - lo) * i as f32 / 8.0));
                }
            }
        }
        v
    }

    fn apply(c: &Cpu, px: &[f32]) -> Vec<f32> {
        let mut v = px.to_vec();
        c.apply(&mut v);
        v
    }

    /// Largest difference relative to max(1, |value|).
    fn worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(1.0)).fold(0.0, f32::max)
    }

    #[test]
    fn out_after_in_is_whole() {
        let lib = lib();
        let b = lib.get(Source::Builtin).unwrap();
        let cases: Vec<(Source, &str, (f32, f32))> = vec![
            (Source::Builtin, "Rec.1886 Rec.709 - Display", (0.0, 1.0)),
            (Source::Builtin, "sRGB - Display", (0.0, 1.0)),
            (Source::Builtin, "Rec.2100-PQ - Display", (0.0, 0.9)),
            (Source::Builtin, "Rec.2100-HLG - Display", (0.0, 1.0)),
            (Source::Builtin, "ARRI LogC4", (0.05, 0.9)),
            (Source::Builtin, "ARRI LogC3 (EI800)", (0.05, 0.9)),
            (Source::Builtin, "ACEScg", (0.0, 8.0)),
            (Source::Builtin, "Linear Rec.709 (sRGB)", (0.0, 8.0)),
            (Source::Maya, "ACEScg", (0.0, 8.0)),
            (Source::Blender, "AgX Log", (0.0, 1.0)),
            (Source::Blender, "Linear Rec.709", (0.0, 8.0)),
        ];
        let targets: Vec<(Source, Target)> = vec![
            (Source::Builtin, Target::Display { display: "sRGB - Display", view: "Video (colorimetric)" }),
            (Source::Builtin, Target::Display { display: "sRGB - Display", view: "ACES 2.0 - SDR 100 nits (Rec.709)" }),
            (Source::Builtin, Target::Space("Rec.1886 Rec.709 - Display")),
            (Source::Builtin, Target::Space("ACEScg")),
            (Source::Blender, Target::Display { display: "sRGB", view: "AgX" }),
        ];
        for (s, name, (lo, hi)) in &cases {
            let src = lib.get(*s).unwrap();
            let px = grid(*lo, *hi);
            let pre = Cpu::part(src, name, b, Target::Space("ACEScg"), false, Part::In).unwrap();
            let lin = apply(&pre, &px);
            for (d, t) in &targets {
                let dst = lib.get(*d).unwrap();
                let Ok(whole) = Cpu::new(src, name, dst, *t, false) else { continue };
                let post = Cpu::part(src, name, dst, *t, false, Part::Out).unwrap();
                let (a, z) = (apply(&whole, &px), apply(&post, &lin));
                let e = worst(&a, &z);
                if e > 5e-4 {
                    let k = (0..a.len()).max_by(|&i, &j| ((a[i] - z[i]).abs() / a[i].abs().max(1.0)).total_cmp(&((a[j] - z[j]).abs() / a[j].abs().max(1.0)))).unwrap() / 3;
                    println!("  worst at input {:?}: whole {:?}, split {:?}", &px[k * 3..k * 3 + 3], &a[k * 3..k * 3 + 3], &z[k * 3..k * 3 + 3]);
                }
                println!("{name} ({s:?}) -> {t:?}: worst {e:.2e} ({:.3} 8-bit levels)", e * 255.0);
                // Largest where a channel is exactly 0 and the target has a steep gamma: the matrix
                // round trip leaves ~1e-8 of light, gamma 1/2.4 makes it ~0.001.
                assert!(e < 4e-3, "{name} -> {t:?}: {e}");
            }
        }
    }

    #[test]
    fn in_is_the_players_linear_light() {
        let lib = lib();
        let b = lib.get(Source::Builtin).unwrap();
        let to2020 = |c: [f32; 3]| -> [f32; 3] {
            let m = [[0.6274039, 0.3292830, 0.0433131], [0.0690973, 0.9195404, 0.0113623], [0.0163914, 0.0880133, 0.8955953]];
            m.map(|r| r[0] * c[0] + r[1] * c[1] + r[2] * c[2])
        };
        let pq = |e: f32| {
            let (m1, m2, c1, c2, c3) = (0.1593017578125f32, 78.84375f32, 0.8359375f32, 18.8515625f32, 18.6875f32);
            let p = e.powf(1.0 / m2);
            ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1) * 10000.0 / 203.0
        };
        let px = grid(0.0, 1.0);
        let check = |name: &str, f: &dyn Fn([f32; 3]) -> [f32; 3], tol: f32| {
            let pre = Cpu::part(b, name, b, Target::Space("ACEScg"), false, Part::In).unwrap();
            let got = apply(&pre, &px);
            let want: Vec<f32> = px.chunks(3).flat_map(|c| f([c[0], c[1], c[2]])).collect();
            let e = worst(&want, &got);
            println!("{name}: in vs the player's linear: worst {e:.2e}");
            assert!(e < tol, "{name}: {e}");
        };
        check("Rec.1886 Rec.709 - Display", &|c| to2020(c.map(|x| x.powf(2.4))), 1e-4);
        check("Rec.2100-PQ - Display", &|c| c.map(pq), 1e-3);
        check("Linear Rec.709 (sRGB)", &|c| to2020(c), 1e-4);
        check("Linear Rec.2020", &|c| c, 1e-4);
    }
}
