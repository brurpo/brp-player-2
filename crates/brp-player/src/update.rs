//! Updates and downloaded components.
//!
//! A portable install has `manifest.json` next to the executable (development builds
//! have none, and skip all of this). It names the app's version, its GitHub
//! repository, and the components that are downloaded rather than shipped in the
//! zip: the GPL `ffmpeg.exe` (x264/x265 export) and the AI models, hosted as zips in
//! versioned folders on Google Drive.
//!
//! - At start: files left by the last update (`*.old`) are removed; missing or
//!   outdated components are downloaded in the background (checked by SHA-256),
//!   each into its folder with a `.component.json` marker; GitHub's latest release
//!   is checked.
//! - A newer release: a banner offers it. Installing downloads its zip, renames the
//!   files it replaces to `*.old` (Windows lets running files be renamed, not
//!   overwritten), moves the new ones in and offers a restart. Files the new release
//!   no longer has stay, except bundled effects: a `.glsl` in `shaders/` that the
//!   installed release shipped and the new one does not is deleted (it would stay in
//!   the effect list otherwise); files the user put there are never touched. The
//!   same is done at the first start after an update, from the previous release's
//!   manifest (still there as `manifest.json.old`): the update is carried out by
//!   the old version, and versions up to 0.2.3 do not know this step.

use anyhow::{bail, Context as _, Result};
use parking_lot::Mutex;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const UA: &str = concat!("brp-player/", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    /// Folder relative to the executable.
    pub dir: String,
    pub version: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Release {
    pub version: String,
    pub zip_url: String,
    pub zip_size: u64,
}

#[derive(Clone, Debug, Default)]
pub enum Phase {
    #[default]
    Idle,
    /// A newer release exists.
    Available(Release),
    /// Downloading: what, bytes done, bytes total.
    Downloading(String, u64, u64),
    /// Installed: restart to use it.
    Ready,
    Failed(String),
}

#[derive(Default)]
pub struct State {
    pub phase: Phase,
    /// Components being fetched: (name, done, total).
    pub components: Vec<(String, u64, u64)>,
    pub component_error: Option<String>,
    /// The user closed the banner.
    pub dismissed: bool,
}

#[derive(Clone)]
pub struct Updater {
    pub state: Arc<Mutex<State>>,
    dir: Option<PathBuf>,
    repo: String,
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(|p| p.to_path_buf())
}

impl Updater {
    /// Start the checks (nothing happens without `manifest.json`).
    pub fn start() -> Updater {
        let dir = exe_dir().filter(|d| d.join("manifest.json").is_file());
        let repo = dir.as_ref().and_then(|d| read_manifest(d).ok()).and_then(|m| m["repo"].as_str().map(String::from)).unwrap_or_default();
        let u = Updater { state: Arc::default(), dir, repo };
        if let Some(dir) = u.dir.clone() {
            let me = u.clone();
            let _ = std::thread::Builder::new().name("update".into()).spawn(move || {
                // Before the last update's leftovers go: the old manifest is one of them.
                let gone = remove_shaders_dropped_by_last_update(&dir);
                if !gone.is_empty() {
                    log::info!("update: effects no longer shipped removed: {}", gone.join(", "));
                }
                remove_old(&dir);
                if let Err(e) = me.fetch_components(&dir) {
                    log::warn!("components: {e:#}");
                    me.state.lock().component_error = Some(format!("{e:#}"));
                }
                match me.latest() {
                    Ok(Some(r)) => {
                        log::info!("update: {} available", r.version);
                        me.state.lock().phase = Phase::Available(r);
                        // Testing: install without the click.
                        if std::env::var_os("BRP_UPDATE_AUTO").is_some() {
                            me.install();
                        }
                    }
                    Ok(None) => {}
                    Err(e) => log::info!("update check: {e:#}"),
                }
            });
        }
        u
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// GitHub's latest release, when newer than this build.
    fn latest(&self) -> Result<Option<Release>> {
        if self.repo.is_empty() {
            return Ok(None);
        }
        let url = format!("https://api.github.com/repos/{}/releases/latest", self.repo);
        let v: Value = serde_json::from_reader(get(&url)?.into_reader())?;
        let tag = v["tag_name"].as_str().unwrap_or_default().trim_start_matches('v').to_string();
        // Testing: act as an older version.
        let mine = std::env::var("BRP_UPDATE_PRETEND_VERSION").unwrap_or_else(|_| env!("CARGO_PKG_VERSION").into());
        if !newer(&tag, &mine) {
            return Ok(None);
        }
        let asset = v["assets"].as_array().into_iter().flatten().find(|a| a["name"].as_str().is_some_and(|n| n.ends_with("-windows-x64.zip")));
        let Some(a) = asset else { return Ok(None) };
        Ok(Some(Release {
            version: tag,
            zip_url: a["browser_download_url"].as_str().unwrap_or_default().to_string(),
            zip_size: a["size"].as_u64().unwrap_or(0),
        }))
    }

    /// Download what the manifest lists and is missing or outdated here.
    fn fetch_components(&self, dir: &Path) -> Result<()> {
        let want: Vec<Component> = components(&read_manifest(dir)?).into_iter().filter(|c| !installed(dir, c)).collect();
        if want.is_empty() {
            return Ok(());
        }
        self.state.lock().components = want.iter().map(|c| (c.name.clone(), 0, c.size)).collect();
        let mut errors = Vec::new();
        for (i, c) in want.iter().enumerate() {
            let st = self.state.clone();
            let r = install_component(dir, c, &move |done, total| {
                if let Some(e) = st.lock().components.get_mut(i) {
                    e.1 = done;
                    e.2 = total.max(e.2);
                }
            });
            if let Err(e) = r {
                errors.push(format!("{}: {e:#}", c.name));
            }
            self.state.lock().components[i].1 = c.size.max(1);
        }
        self.state.lock().components.clear();
        if !errors.is_empty() {
            bail!("{}", errors.join("; "));
        }
        Ok(())
    }

    /// Download and put in place the offered release (in the background).
    pub fn install(&self) {
        let Some(dir) = self.dir.clone() else { return };
        let Phase::Available(r) = self.state.lock().phase.clone() else { return };
        let me = self.clone();
        let _ = std::thread::Builder::new().name("update-install".into()).spawn(move || {
            let st = me.state.clone();
            let label = format!("BRP Player {}", r.version);
            st.lock().phase = Phase::Downloading(label.clone(), 0, r.zip_size);
            let res = (|| -> Result<()> {
                let work = dir.join(".update");
                let _ = std::fs::remove_dir_all(&work);
                std::fs::create_dir_all(&work)?;
                let zip = work.join("app.zip");
                let s2 = st.clone();
                download(&r.zip_url, &zip, &move |d, t| s2.lock().phase = Phase::Downloading(label.clone(), d, t))?;
                let new = work.join("new");
                unzip(&zip, &new)?;
                // The zip holds one top folder (brp-player-<version>/): its contents go in.
                let root = single_subdir(&new).unwrap_or(new.clone());
                // Before the new files move in: what this install shipped (its
                // manifest is about to be replaced) and what the new release ships.
                let shipped = shipped_shaders(read_manifest(&dir).ok().as_ref());
                let ships = glsl_names(&root.join("shaders"));
                replace_tree(&root, &dir)?;
                let gone = remove_dropped_shaders(&dir.join("shaders"), &shipped, &ships);
                if !gone.is_empty() {
                    log::info!("update: effects no longer shipped removed: {}", gone.join(", "));
                }
                let _ = std::fs::remove_dir_all(&work);
                Ok(())
            })();
            match &res {
                Ok(()) => log::info!("update: {} installed", r.version),
                Err(e) => log::warn!("update failed: {e:#}"),
            }
            st.lock().phase = match res {
                Ok(()) => Phase::Ready,
                Err(e) => Phase::Failed(format!("{e:#}")),
            };
        });
    }
}

/// Start the new executable with the same arguments; the caller then closes.
pub fn restart() {
    if let Ok(exe) = std::env::current_exe() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let _ = std::process::Command::new(exe).args(args).spawn();
    }
}

fn read_manifest(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?)
}

fn components(m: &Value) -> Vec<Component> {
    m["components"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| {
            Some(Component {
                name: c["name"].as_str()?.into(),
                dir: c["dir"].as_str()?.into(),
                version: c["version"].as_str()?.into(),
                url: c["url"].as_str()?.into(),
                sha256: c["sha256"].as_str()?.to_lowercase(),
                size: c["size"].as_u64().unwrap_or(0),
            })
        })
        .collect()
}

fn installed(dir: &Path, c: &Component) -> bool {
    let Ok(b) = std::fs::read(dir.join(&c.dir).join(".component.json")) else { return false };
    let Ok(v) = serde_json::from_slice::<Value>(&b) else { return false };
    v["version"].as_str() == Some(c.version.as_str()) && v["sha256"].as_str() == Some(c.sha256.as_str())
}

fn install_component(dir: &Path, c: &Component, progress: &dyn Fn(u64, u64)) -> Result<()> {
    let work = dir.join(".update");
    std::fs::create_dir_all(&work)?;
    let zip = work.join(format!("{}.zip", c.name.replace(['/', '\\'], "_")));
    download(&c.url, &zip, progress)?;
    let got = sha256_file(&zip)?;
    if got != c.sha256 {
        let _ = std::fs::remove_file(&zip);
        bail!("checksum mismatch (got {got})");
    }
    let target = dir.join(&c.dir);
    let tmp = work.join(format!("{}.new", c.name.replace(['/', '\\'], "_")));
    let _ = std::fs::remove_dir_all(&tmp);
    unzip(&zip, &tmp)?;
    if target.exists() {
        replace_tree(&tmp, &target)?;
    } else {
        if let Some(p) = target.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::rename(&tmp, &target)?;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_file(&zip);
    let marker = serde_json::json!({ "name": c.name, "version": c.version, "sha256": c.sha256 });
    std::fs::write(target.join(".component.json"), serde_json::to_vec_pretty(&marker)?)?;
    Ok(())
}

fn get(url: &str) -> Result<ureq::Response> {
    let agent = ureq::AgentBuilder::new().user_agent(UA).timeout_connect(std::time::Duration::from_secs(15)).timeout_read(std::time::Duration::from_secs(60)).build();
    agent.get(url).call().with_context(|| format!("GET {url}"))
}

/// Download `url` to `path` (via a .part file), reporting (done, total) bytes.
fn download(url: &str, path: &Path, progress: &dyn Fn(u64, u64)) -> Result<()> {
    let resp = get(url)?;
    let ct = resp.header("Content-Type").unwrap_or_default().to_string();
    if ct.starts_with("text/html") {
        bail!("{url} returned a web page instead of the file (is it shared publicly?)");
    }
    let total = resp.header("Content-Length").and_then(|s| s.parse().ok()).unwrap_or(0);
    let part = path.with_extension("part");
    let mut out = std::fs::File::create(&part)?;
    let mut r = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    out.flush()?;
    drop(out);
    if total > 0 && done != total {
        bail!("download cut short ({done} of {total} bytes)");
    }
    std::fs::rename(&part, path)?;
    Ok(())
}

fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn unzip(zip: &Path, to: &Path) -> Result<()> {
    let mut a = zip::ZipArchive::new(std::fs::File::open(zip)?)?;
    std::fs::create_dir_all(to)?;
    for i in 0..a.len() {
        let mut f = a.by_index(i)?;
        let Some(rel) = f.enclosed_name() else { bail!("unsafe path in {}", zip.display()) };
        let out = to.join(rel);
        if f.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(p) = out.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut w = std::fs::File::create(&out)?;
        std::io::copy(&mut f, &mut w)?;
    }
    Ok(())
}

fn single_subdir(d: &Path) -> Option<PathBuf> {
    let mut it = std::fs::read_dir(d).ok()?.flatten();
    let first = it.next()?;
    (it.next().is_none() && first.path().is_dir()).then(|| first.path())
}

/// Move every file of `src` to the same place under `dst`; a file already there is
/// renamed to `<name>.old` first (removed at the next start).
fn replace_tree(src: &Path, dst: &Path) -> Result<()> {
    for e in std::fs::read_dir(src)?.flatten() {
        let from = e.path();
        let to = dst.join(e.file_name());
        if from.is_dir() {
            std::fs::create_dir_all(&to)?;
            replace_tree(&from, &to)?;
            continue;
        }
        if to.exists() {
            let old = old_name(&to);
            let _ = std::fs::remove_file(&old);
            std::fs::rename(&to, &old).with_context(|| format!("rename {}", to.display()))?;
        }
        std::fs::rename(&from, &to).or_else(|_| std::fs::copy(&from, &to).map(|_| ())).with_context(|| format!("put {}", to.display()))?;
    }
    Ok(())
}

/// Effect files shipped by every release up to 0.2.3 (read from their zips), whose
/// manifests did not list them yet.
const SHADERS_UNTIL_0_2_3: [&str; 24] = [
    "AMatte_BackgroundColor.glsl",
    "AMatte_Keyer.glsl",
    "Builtin_ColorCorrection.glsl",
    "Builtin_Deblur.glsl",
    "Builtin_Edges.glsl",
    "Builtin_Film_Grain.glsl",
    "Builtin_GaussianBlur.glsl",
    "Builtin_Glitch.glsl",
    "Builtin_Glow.glsl",
    "Builtin_Glow_Fast.glsl",
    "Builtin_Glow_Slow.glsl",
    "Builtin_Invert.glsl",
    "Builtin_Keying.glsl",
    "Builtin_LensDistortion.glsl",
    "Builtin_LensFlare.glsl",
    "Builtin_Noise.glsl",
    "Builtin_Painterly.glsl",
    "Builtin_SpatialDenoiser.glsl",
    "Builtin_Temporal_Denoiser.glsl",
    "FSRCNN_upscale4x.glsl",
    "MB.glsl",
    "example.glsl",
    "temporalNL.glsl",
    "upscale4x.glsl",
];

/// A `.glsl` file's place in `shaders/`: its name, or folders and name joined by "/"
/// ("Damage/Builtin_Grain.glsl"). Nothing that could lead out of the folder: the only
/// form taken from a manifest.
fn is_glsl_path(n: &str) -> bool {
    let parts: Vec<&str> = n.split('/').collect();
    parts.len() <= 6 && parts.iter().all(|p| !p.is_empty() && *p != "." && *p != ".." && !p.contains(['\\', ':'])) && n.len() > 5 && n.to_ascii_lowercase().ends_with(".glsl")
}

/// The effect files an install's release shipped in `shaders/` (see `is_glsl_path`):
/// its manifest's "shaders" list (written by tools/release/package.py), or, for a
/// manifest without one (releases up to 0.2.3), the files those releases shipped.
fn shipped_shaders(manifest: Option<&Value>) -> Vec<String> {
    match manifest.and_then(|m| m["shaders"].as_array()) {
        Some(list) => list.iter().filter_map(|v| v.as_str()).filter(|n| is_glsl_path(n)).map(String::from).collect(),
        None => SHADERS_UNTIL_0_2_3.iter().map(|s| s.to_string()).collect(),
    }
}

/// The `.glsl` files in `dir` and its subfolders, as places in it (see `is_glsl_path`).
fn glsl_names(dir: &Path) -> Vec<String> {
    let files = brp_render::usershader::glsl_files(dir);
    files.iter().filter_map(|p| p.strip_prefix(dir).ok()).map(|r| r.iter().map(|c| c.to_string_lossy()).collect::<Vec<_>>().join("/")).filter(|n| is_glsl_path(n)).collect()
}

/// After an update: delete from `dir` (the install's `shaders/`) the effect files
/// the old release shipped (`shipped`) and the new one does not (`ships`: the files
/// in the new release's `shaders/`), each named by its place there, so an effect the
/// new release keeps in another folder is deleted from where it was. Anything the old release did not ship is the
/// user's and stays. Nothing is deleted when the new release has no effect files at
/// all (not a release as we package them). Returns the names deleted.
fn remove_dropped_shaders(dir: &Path, shipped: &[String], ships: &[String]) -> Vec<String> {
    if ships.is_empty() {
        return Vec::new();
    }
    let mut gone = Vec::new();
    for name in shipped.iter().filter(|n| is_glsl_path(n) && !ships.iter().any(|s| s.eq_ignore_ascii_case(n))) {
        let p = dir.join(name);
        if !p.is_file() {
            continue;
        }
        match std::fs::remove_file(&p) {
            Ok(()) => gone.push(name.clone()),
            Err(e) => log::warn!("update: cannot remove {}: {e}", p.display()),
        }
        // Its folders too, when nothing is left in them (remove_dir refuses otherwise).
        let mut folder = p.parent();
        while let Some(f) = folder.filter(|f| *f != dir && f.starts_with(dir)) {
            if std::fs::remove_dir(f).is_err() {
                break;
            }
            folder = f.parent();
        }
    }
    gone
}

/// First start after an update (the previous release's manifest is still there as
/// `manifest.json.old`): delete the bundled effects that release shipped and this one
/// does not. `install` does it when the version doing the update knows how; versions
/// up to 0.2.3 do not, and one of them does the update that brings this version in.
/// Nothing happens unless this release's manifest lists its effects.
fn remove_shaders_dropped_by_last_update(dir: &Path) -> Vec<String> {
    let previous = std::fs::read(old_name(&dir.join("manifest.json"))).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok());
    let (Some(previous), Ok(current)) = (previous, read_manifest(dir)) else { return Vec::new() };
    if current["shaders"].as_array().is_none() {
        return Vec::new();
    }
    remove_dropped_shaders(&dir.join("shaders"), &shipped_shaders(Some(&previous)), &shipped_shaders(Some(&current)))
}

fn old_name(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

/// Files renamed by the last update.
fn remove_old(dir: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if !e.file_name().to_string_lossy().starts_with('.') {
                remove_old(&p);
            }
        } else if p.extension().is_some_and(|x| x == "old") {
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// `a` is a newer version than `b` (dotted numbers).
fn newer(a: &str, b: &str) -> bool {
    let n = |s: &str| s.split(['.', '-']).map(|x| x.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    let (x, y) = (n(a), n(b));
    for i in 0..x.len().max(y.len()) {
        let (p, q) = (x.get(i).copied().unwrap_or(0), y.get(i).copied().unwrap_or(0));
        if p != q {
            return p > q;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch install or release tree: (place under it, text) files.
    fn tree(root: &Path, files: &[(&str, &str)]) {
        for (name, text) in files {
            let p = root.join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        }
    }

    fn names(root: &Path) -> Vec<String> {
        let mut v = glsl_names(&root.join("shaders"));
        v.sort();
        v
    }

    /// What `install` does with an unpacked release.
    fn update(dir: &Path, root: &Path) -> Vec<String> {
        let shipped = shipped_shaders(read_manifest(dir).ok().as_ref());
        let ships = glsl_names(&root.join("shaders"));
        replace_tree(root, dir).unwrap();
        let mut gone = remove_dropped_shaders(&dir.join("shaders"), &shipped, &ships);
        gone.sort();
        gone
    }

    /// An update deletes the effects the old release shipped and the new one does
    /// not have in the same place, and nothing else: the user's files stay.
    #[test]
    fn dropped_shaders_removed() {
        let base = std::env::temp_dir().join(format!("brp-update-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (old, new, newer) = (base.join("install"), base.join("new"), base.join("newer"));
        // An install of a release without a "shaders" list (effects right in
        // shaders/): two shipped effects, the user's own effects (one in a folder of
        // theirs), and a file of the user's that is not an effect.
        tree(&old, &[
            ("manifest.json", r#"{"version": "0.2.3", "repo": "x/y", "components": []}"#),
            ("shaders/Builtin_Glow.glsl", "old glow"),
            ("shaders/Builtin_Noise.glsl", "old noise"),
            ("shaders/my_look.glsl", "mine"),
            ("shaders/Mine/look2.glsl", "mine"),
            ("shaders/notes.txt", "mine too"),
        ]);
        // The new release keeps its effects in folders: Glow moves into one, Noise is
        // gone, Grain is new; its manifest lists them.
        tree(&new, &[
            ("manifest.json", r#"{"version": "0.2.4", "repo": "x/y", "components": [], "shaders": ["Effects/Builtin_Glow.glsl", "Damage/Builtin_Grain.glsl"]}"#),
            ("shaders/Effects/Builtin_Glow.glsl", "new glow"),
            ("shaders/Damage/Builtin_Grain.glsl", "grain"),
        ]);
        assert_eq!(update(&old, &new), ["Builtin_Glow.glsl", "Builtin_Noise.glsl"]);
        assert_eq!(names(&old), ["Damage/Builtin_Grain.glsl", "Effects/Builtin_Glow.glsl", "Mine/look2.glsl", "my_look.glsl"]);
        assert_eq!(std::fs::read_to_string(old.join("shaders/Effects/Builtin_Glow.glsl")).unwrap(), "new glow");
        assert_eq!(std::fs::read_to_string(old.join("shaders/my_look.glsl")).unwrap(), "mine");
        assert!(old.join("shaders/notes.txt").is_file());

        // The next release drops Grain: now the installed manifest's own list says
        // what was shipped, and the emptied folder goes too. Files of the user's that
        // release did not ship stay: one named like an old effect, one in a shipped folder.
        tree(&old, &[("shaders/MB.glsl", "the user's"), ("shaders/Effects/mine.glsl", "the user's")]);
        tree(&newer, &[
            ("manifest.json", r#"{"version": "0.2.5", "repo": "x/y", "components": [], "shaders": ["Effects/Builtin_Glow.glsl"]}"#),
            ("shaders/Effects/Builtin_Glow.glsl", "newer glow"),
        ]);
        assert_eq!(update(&old, &newer), ["Damage/Builtin_Grain.glsl"]);
        assert_eq!(names(&old), ["Effects/Builtin_Glow.glsl", "Effects/mine.glsl", "MB.glsl", "Mine/look2.glsl", "my_look.glsl"]);
        assert!(!old.join("shaders/Damage").exists());

        // A new tree without effect files deletes nothing; places that could lead
        // out of shaders/ (a bad manifest) are ignored.
        assert!(remove_dropped_shaders(&old.join("shaders"), &shipped_shaders(None), &[]).is_empty());
        let bad = serde_json::json!({ "shaders": ["../manifest.json", "../x.glsl", "a/../x.glsl", "..\\x.glsl", "a\\x.glsl", "C:x.glsl", "/x.glsl", "a//x.glsl", ".glsl", "x.txt", "Effects/Builtin_Glow.glsl", "Builtin_Glow.glsl"] });
        assert_eq!(shipped_shaders(Some(&bad)), ["Effects/Builtin_Glow.glsl", "Builtin_Glow.glsl"]);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// An install of 0.2.3 (24 effects right in shaders/, no list in its manifest)
    /// updated to the effects in the repo's `shaders/` (in folders) loses all 24 old
    /// files: the twelve removed, and the old copies of the ones now in folders.
    #[test]
    fn update_from_0_2_3_drops_the_old_effects() {
        let dir = std::env::temp_dir().join(format!("brp-update-test-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for n in SHADERS_UNTIL_0_2_3.iter().chain(&["users_own.glsl"]) {
            std::fs::write(dir.join(n), "x").unwrap();
        }
        let ships = glsl_names(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders"));
        assert!(ships.len() > 10 && ships.iter().all(|n| n.contains('/')), "{ships:?}");
        // Every effect of the repo that 0.2.3 had is still shipped, under a folder.
        let file = |n: &str| n.rsplit('/').next().unwrap().to_string();
        let kept: Vec<String> = ships.iter().map(|n| file(n)).filter(|n| SHADERS_UNTIL_0_2_3.contains(&n.as_str())).collect();
        assert_eq!(kept.len(), 12);
        let gone = remove_dropped_shaders(&dir, &shipped_shaders(Some(&serde_json::json!({ "version": "0.2.3" }))), &ships);
        assert_eq!(gone.len(), 24);
        assert_eq!(glsl_names(&dir), ["users_own.glsl"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The update done by a version without the deletion step (up to 0.2.3): the
    /// new version cleans up at its first start, once, from the old manifest.
    #[test]
    fn first_start_after_update_by_an_old_version() {
        let base = std::env::temp_dir().join(format!("brp-update-test-start-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (dir, new) = (base.join("install"), base.join("new"));
        tree(&dir, &[
            ("manifest.json", r#"{"version": "0.2.3", "repo": "x/y", "components": []}"#),
            ("shaders/Builtin_Glow.glsl", "old"),
            ("shaders/Builtin_Noise.glsl", "old"),
            ("shaders/MB.glsl", "old"),
            ("shaders/my_look.glsl", "mine"),
        ]);
        tree(&new, &[
            ("manifest.json", r#"{"version": "0.2.4", "repo": "x/y", "components": [], "shaders": ["Effects/Builtin_Glow.glsl", "Damage/Builtin_Grain.glsl"]}"#),
            ("shaders/Effects/Builtin_Glow.glsl", "new"),
            ("shaders/Damage/Builtin_Grain.glsl", "new"),
        ]);
        // 0.2.3's update: files added and replaced, nothing deleted.
        replace_tree(&new, &dir).unwrap();
        assert_eq!(names(&dir), ["Builtin_Glow.glsl", "Builtin_Noise.glsl", "Damage/Builtin_Grain.glsl", "Effects/Builtin_Glow.glsl", "MB.glsl", "my_look.glsl"]);
        // The new version's first start.
        let mut gone = remove_shaders_dropped_by_last_update(&dir);
        gone.sort();
        assert_eq!(gone, ["Builtin_Glow.glsl", "Builtin_Noise.glsl", "MB.glsl"]);
        remove_old(&dir);
        assert_eq!(names(&dir), ["Damage/Builtin_Grain.glsl", "Effects/Builtin_Glow.glsl", "my_look.glsl"]);
        assert!(!dir.join("manifest.json.old").exists());
        // Later starts: no old manifest, so a file the user adds under an old name stays.
        tree(&dir, &[("shaders/MB.glsl", "the user's")]);
        assert!(remove_shaders_dropped_by_last_update(&dir).is_empty());
        assert!(dir.join("shaders/MB.glsl").is_file());
        // An install whose own manifest has no list (a version before this) does nothing.
        tree(&dir, &[("manifest.json.old", r#"{"version": "0.2.2"}"#), ("manifest.json", r#"{"version": "0.2.3"}"#)]);
        assert!(remove_shaders_dropped_by_last_update(&dir).is_empty());
        assert!(dir.join("shaders/MB.glsl").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn versions() {
        assert!(newer("0.2.0", "0.1.9"));
        assert!(newer("1.0", "0.9.9"));
        assert!(!newer("0.1.0", "0.1.0"));
        assert!(!newer("0.1.0", "0.1.1"));
        assert!(newer("0.1.10", "0.1.9"));
    }
}
