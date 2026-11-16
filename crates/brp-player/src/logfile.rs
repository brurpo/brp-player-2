//! The log file, and what the user sees when the player cannot go on.
//!
//! A release build has no console, so what the player logs also goes to
//! `%LOCALAPPDATA%\BRP Player\brp-player.log` (the run before it is kept as
//! `brp-player.previous.log`). Each line is written at once: a run that ends badly
//! leaves its last steps there. `RUST_LOG` chooses what is logged, as on the console.
//!
//! A failed start, a panic and a crash (inside a graphics driver, say) are written
//! there too and shown in a message box that names the file: without it the window
//! just went away.

use crate::i18n::Lang;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// The file stops growing here (a player left open for days); failures still get in.
const MAX_BYTES: usize = 8 << 20;

static FILE: Mutex<Option<File>> = Mutex::new(None);
static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
static START: OnceLock<Instant> = OnceLock::new();
static WRITTEN: AtomicUsize = AtomicUsize::new(0);
/// The interface language is Portuguese (known once the settings are read).
static PT: AtomicBool = AtomicBool::new(false);
/// One message box per run: the first failure is the one that matters.
static BOX_SHOWN: AtomicBool = AtomicBool::new(false);

/// The log's folder: `%LOCALAPPDATA%\BRP Player` (else `$XDG_CACHE_HOME` or `~/.cache`
/// + `brp-player`), next to the shader cache.
fn dir() -> Option<PathBuf> {
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return Some(PathBuf::from(local).join("BRP Player"));
    }
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("brp-player"))
}

/// The log file of this run, if it could be made.
pub fn path() -> Option<PathBuf> {
    PATH.get().cloned().flatten()
}

/// Messages in this language from now on.
pub fn set_lang(lang: Lang) {
    PT.store(lang == Lang::Pt, Ordering::Relaxed);
}

fn lang() -> Lang {
    if PT.load(Ordering::Relaxed) { Lang::Pt } else { Lang::En }
}

/// One line into the file, with the seconds since start. `always`: past the size limit too.
fn write(file: &mut Option<File>, text: &str, always: bool) {
    let Some(f) = file else { return };
    if WRITTEN.fetch_add(text.len() + 12, Ordering::Relaxed) > MAX_BYTES && !always {
        return;
    }
    let t = START.get().map_or(0.0, |s| s.elapsed().as_secs_f64());
    // One write per line, unbuffered: it is in the file even if the program dies next.
    let _ = f.write_all(format!("{t:9.3} {text}\n").as_bytes());
}

/// A line about a failure. Waits only a moment for the file: the thread that failed
/// may be the one that was writing to it.
fn write_failure(text: &str) {
    for _ in 0..100 {
        match FILE.try_lock() {
            Ok(mut f) => return write(&mut f, text, true),
            Err(std::sync::TryLockError::Poisoned(e)) => return write(&mut e.into_inner(), text, true),
            Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}

/// The console logger as before, and every line it takes into the file as well.
struct Tee(env_logger::Logger);

impl log::Log for Tee {
    fn enabled(&self, m: &log::Metadata) -> bool {
        self.0.enabled(m)
    }

    fn log(&self, r: &log::Record) {
        if !self.0.matches(r) {
            return;
        }
        self.0.log(r);
        let thread = std::thread::current();
        let line = format!("{:5} [{}] {}: {}", r.level(), thread.name().unwrap_or("-"), r.target(), r.args());
        write(&mut FILE.lock().unwrap_or_else(|e| e.into_inner()), &line, false);
    }

    fn flush(&self) {}
}

/// Start logging (console and file), and report panics and crashes. `filters`: what is
/// logged unless `RUST_LOG` says otherwise.
pub fn init(filters: &str) {
    START.get_or_init(Instant::now);
    let path = PATH.get_or_init(|| {
        let d = dir()?;
        std::fs::create_dir_all(&d).ok()?;
        let p = d.join("brp-player.log");
        // The run before this one (the one that failed, when the user starts again).
        let _ = std::fs::rename(&p, d.join("brp-player.previous.log"));
        let mut f = File::create(&p).ok()?;
        // UTF-8 with its mark, so every editor shows accented names right.
        let _ = f.write_all(&[0xef, 0xbb, 0xbf]);
        *FILE.lock().unwrap() = Some(f);
        Some(p)
    });
    let console = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(filters)).build();
    let level = console.filter();
    if log::set_boxed_logger(Box::new(Tee(console))).is_ok() {
        log::set_max_level(level);
    }
    log::info!("BRP Player {} on {}", env!("CARGO_PKG_VERSION"), os_version());
    log::info!("started as: {}", std::env::args().collect::<Vec<_>>().join(" "));
    if let Some(p) = path {
        log::info!("log file: {}", p.display());
    }

    let before = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info.payload().downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| info.payload().downcast_ref::<String>().cloned()).unwrap_or_default();
        let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        let thread = std::thread::current();
        let text = format!("panic in thread {} at {at}: {msg}", thread.name().unwrap_or("-"));
        write_failure(&format!("ERROR {text}"));
        before(info);
        // A release build ends at a panic (`panic = "abort"`): say so before it goes.
        if cfg!(panic = "abort") {
            message_box("crashed", None, &text);
        }
    }));
    #[cfg(windows)]
    crash::install();
}

/// The player cannot go on: say why, in the log and in a message box. `headline` and
/// `hint` are interface strings (`i18n`); `detail` is the error itself.
pub fn fatal(headline: &str, hint: Option<&str>, detail: &str) {
    write_failure(&format!("ERROR {} {}", Lang::En.t(headline), detail.replace('\n', "; ")));
    message_box(headline, hint, detail);
}

/// The message box of a failure: what happened, what to do if known, the error, and
/// where the log is (Yes shows it in its folder).
fn message_box(headline: &str, hint: Option<&str>, detail: &str) {
    // BRP_TEST_NO_DIALOG: no message box (tests that only read the log).
    if BOX_SHOWN.swap(true, Ordering::SeqCst) || std::env::var_os("BRP_TEST_NO_DIALOG").is_some() {
        return;
    }
    let l = lang();
    let mut text = l.t(headline).to_string();
    if let Some(h) = hint {
        text += &format!("\n\n{}", l.t(h));
    }
    text += &format!("\n\n{detail}");
    let log = path();
    if let Some(p) = &log {
        text += &format!("\n\n{}\n{}\n\n{}", l.t("log_saved"), p.display(), l.t("log_show"));
    }
    // On a thread of its own: this one may be in the middle of the player window's
    // events, where Windows never gets to show the box (it waits for the thread to
    // have nothing else to do, and a window that is not being drawn always has).
    let ask = log.is_some();
    let shown = std::thread::Builder::new().name("brp-message".into()).spawn(move || ask_box(&text, ask));
    if let (Some(true), Some(p)) = (shown.ok().and_then(|h| h.join().ok()), log) {
        crate::reveal(&p.to_string_lossy());
    }
}

/// An error box with `text`; `ask`: with Yes and No. True if the answer is Yes.
#[cfg(windows)]
fn ask_box(text: &str, ask: bool) -> bool {
    #[link(name = "user32")]
    unsafe extern "system" {
        fn MessageBoxW(owner: *mut std::ffi::c_void, text: *const u16, caption: *const u16, kind: u32) -> i32;
    }
    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    // MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST (above a fullscreen player), and
    // MB_YESNO or MB_OK; the answer IDYES.
    let kind = 0x10 | 0x1_0000 | 0x4_0000 | if ask { 0x4 } else { 0 };
    unsafe { MessageBoxW(std::ptr::null_mut(), wide(text).as_ptr(), wide("BRP Player").as_ptr(), kind) == 6 }
}

#[cfg(not(windows))]
fn ask_box(text: &str, ask: bool) -> bool {
    let buttons = if ask { rfd::MessageButtons::YesNo } else { rfd::MessageButtons::Ok };
    rfd::MessageDialog::new().set_level(rfd::MessageLevel::Error).set_title("BRP Player").set_description(text).set_buttons(buttons).show() == rfd::MessageDialogResult::Yes
}

#[cfg(windows)]
fn os_version() -> String {
    #[repr(C)]
    struct OsVersionInfo {
        size: u32,
        major: u32,
        minor: u32,
        build: u32,
        platform: u32,
        csd: [u16; 128],
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn RtlGetVersion(info: *mut OsVersionInfo) -> i32;
    }
    let mut v = OsVersionInfo { size: std::mem::size_of::<OsVersionInfo>() as u32, major: 0, minor: 0, build: 0, platform: 0, csd: [0; 128] };
    unsafe { RtlGetVersion(&mut v) };
    // Windows 11 still says 10.0: its builds start at 22000.
    format!("Windows {}.{} build {}", v.major, v.minor, v.build)
}

#[cfg(not(windows))]
fn os_version() -> String {
    std::env::consts::OS.to_string()
}

/// Crashes that are not panics (an access violation inside a driver or a library):
/// Windows calls `filter` before it ends the program.
#[cfg(windows)]
mod crash {
    use std::ffi::c_void;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        parameters: u32,
        information: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut c_void,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetUnhandledExceptionFilter(filter: Option<unsafe extern "system" fn(*mut ExceptionPointers) -> i32>) -> *mut c_void;
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
    }

    pub fn install() {
        unsafe { SetUnhandledExceptionFilter(Some(filter)) };
    }

    /// The file (a .dll or the .exe) whose code is at `address`.
    fn module_of(address: *mut c_void) -> String {
        // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | ..._UNCHANGED_REFCOUNT
        let mut module = std::ptr::null_mut();
        let mut name = [0u16; 512];
        unsafe {
            if GetModuleHandleExW(0x4 | 0x2, address as *const u16, &mut module) == 0 {
                return "unknown code".into();
            }
            let n = GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) as usize;
            let path = String::from_utf16_lossy(&name[..n.min(name.len())]);
            format!("{} +{:#x}", path, address as usize - module as usize)
        }
    }

    unsafe extern "system" fn filter(p: *mut ExceptionPointers) -> i32 {
        let (code, address) = unsafe {
            let r = (*p).record;
            if r.is_null() { (0, std::ptr::null_mut()) } else { ((*r).code, (*r).address) }
        };
        let what = match code {
            0xC000_0005 => " (access violation)",
            0xC000_001D => " (illegal instruction: the processor lacks one the code uses)",
            0xC000_00FD => " (stack overflow)",
            0xC000_0094 => " (integer division by zero)",
            _ => "",
        };
        let thread = std::thread::current();
        let text = format!("crash in thread {}: exception {code:#010x}{what} in {}", thread.name().unwrap_or("-"), module_of(address));
        super::write_failure(&format!("ERROR {text}"));
        super::message_box("crashed", None, &text);
        // EXCEPTION_CONTINUE_SEARCH: Windows ends the program as it would have.
        0
    }
}
