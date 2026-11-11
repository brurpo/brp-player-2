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
//!   overwritten), moves the new ones in and offers a restart.

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
                remove_old(&dir);
                if let Err(e) = me.fetch_components(&dir) {
                    log::warn!("components: {e:#}");
                    me.state.lock().component_error = Some(format!("{e:#}"));
                }
                match me.latest() {
                    Ok(Some(r)) => me.state.lock().phase = Phase::Available(r),
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
        if !newer(&tag, env!("CARGO_PKG_VERSION")) {
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
                replace_tree(&root, &dir)?;
                let _ = std::fs::remove_dir_all(&work);
                Ok(())
            })();
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
    use super::newer;
    #[test]
    fn versions() {
        assert!(newer("0.2.0", "0.1.9"));
        assert!(newer("1.0", "0.9.9"));
        assert!(!newer("0.1.0", "0.1.0"));
        assert!(!newer("0.1.0", "0.1.1"));
        assert!(newer("0.1.10", "0.1.9"));
    }
}
