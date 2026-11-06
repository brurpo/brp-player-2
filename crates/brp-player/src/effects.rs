//! Effect chain state per window: library of shader files, the chain, undo/redo, presets.

use anyhow::{Context, Result};
use brp_render::usershader::{self, ParamKind, ShaderFile};
use brp_render::EffectSpec;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct EffectEntry {
    pub spec: EffectSpec,
    pub path: PathBuf,
    pub name: String,
}

#[derive(Clone, Default)]
struct Snapshot(Vec<EffectEntry>);

#[derive(Default)]
pub struct Effects {
    pub list: Vec<EffectEntry>,
    /// The renderer must be told about changes.
    pub dirty: bool,
    pub selected: Option<u64>,
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
        self.list.push(EffectEntry { spec: EffectSpec { id, file, params, enabled: true }, path: path.to_path_buf(), name: display_name(path) });
        self.selected = Some(id);
        self.dirty = true;
        Ok(())
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

    pub fn specs(&self) -> Vec<EffectSpec> {
        self.list.iter().map(|e| e.spec.clone()).collect()
    }

    pub fn any_enabled(&self) -> bool {
        self.list.iter().any(|e| e.spec.enabled)
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
pub fn param_views(file: &ShaderFile) -> Vec<(Option<String>, Vec<ParamView>)> {
    let mut out: Vec<(Option<String>, Vec<ParamView>)> = Vec::new();
    let mut used = vec![false; file.params.len()];
    let mut category: Option<String> = None;
    for (i, p) in file.params.iter().enumerate() {
        if let Some(c) = p.desc.split("(cat:").nth(1).and_then(|s| s.split(')').next()) {
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
                    view = ParamView::Color { label: if label.is_empty() { "Color".into() } else { label }, idx: [i, g, b] };
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

/// DESC without the "(cat: …)" tag.
pub fn clean_desc(desc: &str) -> String {
    match desc.find("(cat:") {
        Some(i) => {
            let end = desc[i..].find(')').map(|j| i + j + 1).unwrap_or(desc.len());
            format!("{}{}", &desc[..i], &desc[end..]).trim().to_string()
        }
        None => desc.trim().to_string(),
    }
}
