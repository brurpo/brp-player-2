//! Effect chain state per window: library of shader files, the chain, undo/redo, presets.

use anyhow::{Context, Result};
use brp_render::usershader::{self, ParamKind, ShaderFile};
use brp_render::EffectSpec;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ai::{AiKind, Stage};

#[derive(Clone)]
pub struct EffectEntry {
    pub spec: EffectSpec,
    pub path: PathBuf,
    pub name: String,
    /// AI effect (its parameters live in `spec` like a shader's).
    pub ai: Option<AiKind>,
}

impl EffectEntry {
    /// Name shown in the stack.
    pub fn title(&self, lang: crate::i18n::Lang) -> String {
        match self.ai {
            Some(k) => lang.t(k.title_key()).to_string(),
            None => self.spec.file.title(lang.code()).map_or_else(|| self.name.clone(), str::to_string),
        }
    }
}

/// Parameter `i`'s label and description as the panel shows them, in `lang`: AI
/// effects from `i18n` (`Lang::ai_param`), shader effects from the file's
/// translations; English when there is none. Tags ("(cat: …)", "(options: …)") the
/// translation leaves out come from the English description.
pub fn param_text(file: &ShaderFile, ai: Option<AiKind>, i: usize, lang: crate::i18n::Lang) -> (String, String) {
    let p = &file.params[i];
    match ai {
        Some(k) => match lang.ai_param(k.id(), &p.name) {
            Some((label, desc)) => (label.to_string(), usershader::parse::with_tags(desc, &p.desc)),
            None => (p.name.clone(), p.desc.clone()),
        },
        None => (file.param_label(i, lang.code()).to_string(), file.param_desc(i, lang.code())),
    }
}

#[derive(Clone, Default)]
struct Snapshot(Vec<EffectEntry>);

#[derive(Default)]
pub struct Effects {
    pub list: Vec<EffectEntry>,
    /// The renderer must be told about changes.
    pub dirty: bool,
    pub selected: Option<u64>,
    /// All effects off at once (the Effects panel switch, E): the stack and its
    /// settings stay as they are and come back unchanged when it is turned on.
    pub bypass: bool,
    next_id: u64,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// Parsed files by path (shared between instances of the same file).
    cache: Vec<(PathBuf, std::time::SystemTime, Arc<ShaderFile>)>,
}

/// Folders searched for .glsl effects: next to the executable, then the user's folder.
pub fn shader_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("shaders"))) {
        v.push(dir);
    }
    v.push(user_shader_dir());
    v
}

pub fn user_shader_dir() -> PathBuf {
    let base = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let d = base.join("BRP Player").join("shaders");
    let _ = std::fs::create_dir_all(&d);
    d
}

/// All .glsl files in the shader folders, sorted by name.
pub fn library() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = shader_dirs()
        .into_iter()
        .filter_map(|d| std::fs::read_dir(d).ok())
        .flat_map(|it| it.flatten().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("glsl")))
        .collect();
    v.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));
    v.dedup_by_key(|p| p.file_name().map(|n| n.to_os_string()));
    v
}

fn display_name(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().replace('_', " ")).unwrap_or_default()
}

impl Effects {
    fn file(&mut self, path: &Path) -> Result<Arc<ShaderFile>> {
        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
        if let Some((_, t, f)) = self.cache.iter().find(|(p, _, _)| p == path) {
            if *t == mtime {
                return Ok(f.clone());
            }
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let f = Arc::new(usershader::parse::parse(&name, &text)?);
        self.cache.retain(|(p, _, _)| p != path);
        self.cache.push((path.to_path_buf(), mtime, f.clone()));
        Ok(f)
    }

    /// Name of a library file as the "Add effect" list shows it: the file's own
    /// translation for `lang`, else its file name.
    pub fn title_of(&mut self, path: &Path, lang: crate::i18n::Lang) -> String {
        match self.file(path) {
            Ok(f) => f.title(lang.code()).map_or_else(|| display_name(path), str::to_string),
            Err(_) => display_name(path),
        }
    }

    pub fn checkpoint(&mut self) {
        self.undo.push(Snapshot(self.list.clone()));
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    pub fn undo(&mut self) {
        if let Some(s) = self.undo.pop() {
            self.redo.push(Snapshot(std::mem::replace(&mut self.list, s.0)));
            self.dirty = true;
        }
    }

    pub fn redo(&mut self) {
        if let Some(s) = self.redo.pop() {
            self.undo.push(Snapshot(std::mem::replace(&mut self.list, s.0)));
            self.dirty = true;
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn add(&mut self, path: &Path) -> Result<()> {
        let file = self.file(path)?;
        self.checkpoint();
        self.next_id += 1;
        let params = file.params.iter().map(|p| p.default).collect();
        let id = self.next_id;
        self.list.push(EffectEntry { spec: EffectSpec { id, file, params, enabled: true }, path: path.to_path_buf(), name: display_name(path), ai: None });
        self.selected = Some(id);
        self.dirty = true;
        Ok(())
    }

    pub fn add_ai(&mut self, kind: AiKind) -> u64 {
        self.checkpoint();
        self.next_id += 1;
        let file = kind.file();
        let params = file.params.iter().map(|p| p.default).collect();
        let id = self.next_id;
        self.list.push(EffectEntry { spec: EffectSpec { id, file, params, enabled: true }, path: PathBuf::new(), name: kind.id().into(), ai: Some(kind) });
        self.selected = Some(id);
        self.dirty = true;
        id
    }

    /// Split the stack at its last enabled AI effect: the stages up to it (run in the
    /// background) and the shader effects after it (run live by the renderer).
    pub fn split(&self) -> (Vec<Stage>, Vec<EffectSpec>) {
        if self.bypass {
            // The live shaders stay in the chain, switched off: they keep their
            // compiled pipelines and come back at once.
            let off = self.list.iter().filter(|e| e.ai.is_none()).map(|e| EffectSpec { enabled: false, ..e.spec.clone() }).collect();
            return (Vec::new(), off);
        }
        // AI entries never go to the live shader chain (they are not shaders: a
        // disabled one there showed as compiling).
        let shaders = |l: &[EffectEntry]| -> Vec<EffectSpec> { l.iter().filter(|e| e.ai.is_none()).map(|e| e.spec.clone()).collect() };
        let Some(last_ai) = self.list.iter().rposition(|e| e.ai.is_some() && e.spec.enabled) else {
            return (Vec::new(), shaders(&self.list));
        };
        let mut stages: Vec<Stage> = Vec::new();
        for e in self.list[..=last_ai].iter().filter(|e| e.spec.enabled) {
            match e.ai {
                Some(kind) => stages.push(Stage::Ai { kind, params: e.spec.params.clone() }),
                None => match stages.last_mut() {
                    Some(Stage::Shaders(v)) => v.push(e.spec.clone()),
                    _ => stages.push(Stage::Shaders(vec![e.spec.clone()])),
                },
            }
        }
        (stages, shaders(&self.list[last_ai + 1..]))
    }

    /// `--fx=A.glsl,ai:upscale:Quality=1,B.glsl:Strength=0.5`: shader files from the
    /// library, AI effects as ai:<name>, parameters after colons.
    pub fn add_from_arg(&mut self, list: &str) {
        let lib = library();
        for item in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let (name, params) = match item.strip_prefix("ai:") {
                Some(rest) => rest.split_once(':').map_or((rest, ""), |(a, b)| (a, b)),
                None => item.split_once(':').map_or((item, ""), |(a, b)| (a, b)),
            };
            let added = if item.starts_with("ai:") {
                match AiKind::from_id(name) {
                    Some(k) => Some(self.add_ai(k)),
                    None => {
                        log::error!("--fx: unknown AI effect {name}");
                        None
                    }
                }
            } else {
                // A file in the library, or a path to a .glsl file (relative paths:
                // a Windows drive letter's colon would read as a parameter).
                let path = lib.iter().find(|p| p.file_name().is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(name))).cloned().or_else(|| Some(PathBuf::from(name)).filter(|p| p.is_file()));
                match path.as_ref() {
                    Some(p) => match self.add(p) {
                        Ok(()) => self.selected,
                        Err(e) => {
                            log::error!("--fx {name}: {e:#}");
                            None
                        }
                    },
                    None => {
                        log::error!("--fx: {name} not found in {:?}", shader_dirs());
                        None
                    }
                }
            };
            let Some(e) = added.and_then(|id| self.list.iter_mut().find(|e| e.spec.id == id)) else { continue };
            for (k, v) in params.split(':').filter_map(|kv| kv.split_once('=')) {
                match (e.spec.file.params.iter().position(|p| p.name.eq_ignore_ascii_case(k)), v.parse::<f64>()) {
                    (Some(i), Ok(v)) => e.spec.params[i] = v,
                    _ => log::error!("--fx {name}: bad parameter {k}={v}"),
                }
            }
        }
    }

    /// The whole enabled stack as stages (export runs everything in one place,
    /// shader effects after the last AI one included).
    pub fn all_stages(&self) -> Vec<Stage> {
        if self.bypass {
            return Vec::new();
        }
        let mut stages: Vec<Stage> = Vec::new();
        for e in self.list.iter().filter(|e| e.spec.enabled) {
            match e.ai {
                Some(kind) => stages.push(Stage::Ai { kind, params: e.spec.params.clone() }),
                None => match stages.last_mut() {
                    Some(Stage::Shaders(v)) => v.push(e.spec.clone()),
                    _ => stages.push(Stage::Shaders(vec![e.spec.clone()])),
                },
            }
        }
        stages
    }

    pub fn remove(&mut self, id: u64) {
        self.checkpoint();
        self.list.retain(|e| e.spec.id != id);
        if self.selected == Some(id) {
            self.selected = self.list.last().map(|e| e.spec.id);
        }
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        if !self.list.is_empty() {
            self.checkpoint();
            self.list.clear();
            self.selected = None;
            self.dirty = true;
        }
    }

    pub fn move_by(&mut self, id: u64, delta: isize) {
        let Some(i) = self.list.iter().position(|e| e.spec.id == id) else { return };
        let j = i as isize + delta;
        if j < 0 || j as usize >= self.list.len() {
            return;
        }
        self.checkpoint();
        self.list.swap(i, j as usize);
        self.dirty = true;
    }

    /// Move an entry to position `to` (counted before it is taken out): one undo step.
    pub fn move_to(&mut self, id: u64, to: usize) {
        let Some(i) = self.list.iter().position(|e| e.spec.id == id) else { return };
        let to = if to > i { to - 1 } else { to }.min(self.list.len() - 1);
        if to == i {
            return;
        }
        self.checkpoint();
        let e = self.list.remove(i);
        self.list.insert(to, e);
        self.dirty = true;
    }


    pub fn any_enabled(&self) -> bool {
        !self.bypass && self.list.iter().any(|e| e.spec.enabled)
    }

    /// Effects that run: enabled, and not all switched off.
    pub fn active_count(&self) -> usize {
        if self.bypass { 0 } else { self.list.iter().filter(|e| e.spec.enabled).count() }
    }

    /// Switch all effects off or back on (not an undo step: nothing in the stack changes).
    pub fn set_bypass(&mut self, on: bool) {
        if self.bypass != on {
            self.bypass = on;
            self.dirty = true;
        }
    }

    /// Copy a dropped/browsed shader into the user folder and add it.
    pub fn import(&mut self, src: &Path) -> Result<PathBuf> {
        let dst = user_shader_dir().join(src.file_name().context("no file name")?);
        if dst != src {
            std::fs::copy(src, &dst).with_context(|| format!("copying to {}", dst.display()))?;
        }
        self.add(&dst)?;
        Ok(dst)
    }

    pub fn save_preset(&self, path: &Path) -> Result<()> {
        let effects: Vec<serde_json::Value> = self
            .list
            .iter()
            .map(|e| {
                let params: serde_json::Map<String, serde_json::Value> = e
                    .spec
                    .file
                    .params
                    .iter()
                    .zip(&e.spec.params)
                    .map(|(p, v)| (p.name.clone(), serde_json::json!(v)))
                    .collect();
                if let Some(k) = e.ai {
                    return serde_json::json!({ "ai": k.id(), "enabled": e.spec.enabled, "params": params });
                }
                serde_json::json!({
                    "file": e.path.file_name().map(|n| n.to_string_lossy().into_owned()),
                    "path": e.path.to_string_lossy(),
                    "enabled": e.spec.enabled,
                    "params": params,
                })
            })
            .collect();
        let doc = serde_json::json!({ "brp_effects": 1, "effects": effects });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        Ok(())
    }

    /// Load a preset. Files are found by path, or by name in the shader folders.
    /// Unknown parameter names are ignored (shader edited since the preset was saved).
    pub fn load_preset(&mut self, path: &Path) -> Result<Vec<String>> {
        let doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        let mut missing = Vec::new();
        let mut list = Vec::new();
        for e in doc["effects"].as_array().context("not a BRP effects preset")? {
            if let Some(id) = e["ai"].as_str() {
                let Some(kind) = AiKind::from_id(id) else {
                    missing.push(id.to_string());
                    continue;
                };
                let file = kind.file();
                let params = file.params.iter().map(|p| e["params"].get(&p.name).and_then(|v| v.as_f64()).unwrap_or(p.default)).collect();
                self.next_id += 1;
                list.push(EffectEntry {
                    spec: EffectSpec { id: self.next_id, file, params, enabled: e["enabled"].as_bool().unwrap_or(true) },
                    name: id.to_string(),
                    path: PathBuf::new(),
                    ai: Some(kind),
                });
                continue;
            }
            let by_path = e["path"].as_str().map(PathBuf::from).filter(|p| p.exists());
            let by_name = e["file"].as_str().and_then(|n| library().into_iter().find(|p| p.file_name().is_some_and(|f| f.to_string_lossy() == n)));
            let Some(fpath) = by_path.or(by_name) else {
                missing.push(e["file"].as_str().unwrap_or("?").to_string());
                continue;
            };
            let file = self.file(&fpath)?;
            let params = file
                .params
                .iter()
                .map(|p| e["params"].get(&p.name).and_then(|v| v.as_f64()).unwrap_or(p.default))
                .collect();
            self.next_id += 1;
            list.push(EffectEntry {
                spec: EffectSpec { id: self.next_id, file, params, enabled: e["enabled"].as_bool().unwrap_or(true) },
                name: display_name(&fpath),
                path: fpath,
                ai: None,
            });
        }
        self.checkpoint();
        self.list = list;
        self.selected = self.list.first().map(|e| e.spec.id);
        self.dirty = true;
        Ok(missing)
    }
}

/// A parameter as the panel shows it: plain, or an R/G/B triple shown as a colour.
pub enum ParamView {
    Single(usize),
    Color { label: String, idx: [usize; 3] },
}

/// Group parameters: `xxxR/xxxG/xxxB` float triples become one colour control.
/// Returns (category, views) in file order; categories come from "(cat: Name)" in DESC.
/// `text(i)` is parameter i's (label, description) as shown (see `param_text`): a
/// colour takes its R parameter's label when that is translated.
pub fn param_views(file: &ShaderFile, text: &dyn Fn(usize) -> (String, String)) -> Vec<(Option<String>, Vec<ParamView>)> {
    let mut out: Vec<(Option<String>, Vec<ParamView>)> = Vec::new();
    let mut used = vec![false; file.params.len()];
    let mut category: Option<String> = None;
    for (i, p) in file.params.iter().enumerate() {
        let (shown_label, desc) = text(i);
        if let Some(c) = desc.split("(cat:").nth(1).and_then(|s| s.split(')').next()) {
            category = Some(c.trim().to_string());
        }
        if used[i] {
            continue;
        }
        let mut view = ParamView::Single(i);
        if p.kind == ParamKind::Float {
            if let Some(prefix) = p.name.strip_suffix('R').or_else(|| p.name.strip_suffix('r')) {
                let find = |s: &str| file.params.iter().position(|q| q.kind == ParamKind::Float && (q.name == format!("{prefix}{s}") || q.name == format!("{prefix}{}", s.to_lowercase())));
                if let (Some(g), Some(b)) = (find("G"), find("B")) {
                    used[g] = true;
                    used[b] = true;
                    let label = if prefix.is_empty() { "Color".to_string() } else { prefix.trim_end_matches("Color").trim_end_matches('_').to_string() };
                    let label = if shown_label != p.name { shown_label } else if label.is_empty() { "Color".into() } else { label };
                    view = ParamView::Color { label, idx: [i, g, b] };
                }
            }
        }
        used[i] = true;
        match out.last_mut() {
            Some((c, v)) if *c == category => v.push(view),
            _ => out.push((category.clone(), vec![view])),
        }
    }
    out
}

/// DESC without the "(cat: …)" and "(options: …)" tags.
pub fn clean_desc(desc: &str) -> String {
    if let Some(i) = desc.find("(options:") {
        let end = matching_paren(&desc[i + 1..]).map(|j| i + 2 + j).unwrap_or(desc.len());
        return clean_desc(&format!("{}{}", &desc[..i], &desc[end..]));
    }
    match desc.find("(cat:") {
        Some(i) => {
            let end = desc[i..].find(')').map(|j| i + j + 1).unwrap_or(desc.len());
            format!("{}{}", &desc[..i], &desc[end..]).trim().to_string()
        }
        None => desc.trim().to_string(),
    }
}

/// Choices of an int parameter whose description has "(options: a|b|c)".
pub fn param_options(desc: &str) -> Option<Vec<String>> {
    let rest = desc.split("(options:").nth(1)?;
    let end = matching_paren(rest)?;
    Some(rest[..end].split('|').map(|s| s.trim().to_string()).collect())
}

/// Index of the ")" closing an already opened "(" (options may contain parentheses).
fn matching_paren(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth == 0 => return Some(i),
            ')' => depth -= 1,
            _ => {}
        }
    }
    None
}
