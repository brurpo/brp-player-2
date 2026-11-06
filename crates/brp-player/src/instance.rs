//! Single instance: a second launch hands its file to the running player and exits.
//!
//! The running player owns a local socket (a named pipe on Windows) only while it is
//! open. No background process is left behind when the last window closes.

use interprocess::local_socket::{prelude::*, GenericFilePath, GenericNamespaced, ListenerOptions, Name, Stream};
use std::io::{BufRead, BufReader, Write};

#[derive(Debug, Clone)]
pub enum Msg {
    Open(String),
    Focus,
}

fn socket_name() -> std::io::Result<Name<'static>> {
    let user = std::env::var("USERNAME").or_else(|_| std::env::var("USER")).unwrap_or_default();
    let base = format!("brp-player-{}.sock", user.replace(|c: char| !c.is_ascii_alphanumeric(), "_"));
    if GenericNamespaced::is_supported() {
        base.to_ns_name::<GenericNamespaced>()
    } else {
        format!("/tmp/{base}").to_fs_name::<GenericFilePath>()
    }
}

/// Try to give `file` to an already running player. True if one took it.
pub fn hand_off(file: Option<&str>) -> bool {
    let Ok(name) = socket_name() else { return false };
    let Ok(conn) = Stream::connect(name) else { return false };
    let mut conn = BufReader::new(conn);
    let line = match file {
        Some(f) => format!("OPEN {}\n", absolute(f)),
        None => "FOCUS\n".to_string(),
    };
    if conn.get_mut().write_all(line.as_bytes()).is_err() {
        return false;
    }
    let mut reply = String::new();
    conn.read_line(&mut reply).is_ok() && reply.trim() == "OK"
}

/// The receiver resolves relative paths against its own directory, so send absolute ones.
fn absolute(f: &str) -> String {
    std::path::absolute(f).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| f.to_string())
}

/// Start listening for other launches. `on_msg` is called from the listener thread.
/// Returns false if another instance already owns the socket.
pub fn listen(on_msg: impl Fn(Msg) + Send + 'static) -> bool {
    let Ok(name) = socket_name() else { return false };
    let listener = match ListenerOptions::new().name(name).create_sync() {
        Ok(l) => l,
        Err(e) => {
            log::warn!("single-instance socket unavailable: {e}");
            return false;
        }
    };
    std::thread::Builder::new()
        .name("brp-instance".into())
        .spawn(move || {
            for conn in listener.incoming().filter_map(Result::ok) {
                let mut conn = BufReader::new(conn);
                let mut line = String::new();
                if conn.read_line(&mut line).is_err() {
                    continue;
                }
                let line = line.trim_end_matches(['\r', '\n']);
                let msg = if let Some(path) = line.strip_prefix("OPEN ") {
                    Msg::Open(path.to_string())
                } else if line == "FOCUS" {
                    Msg::Focus
                } else {
                    continue;
                };
                let _ = conn.get_mut().write_all(b"OK\n");
                on_msg(msg);
            }
        })
        .is_ok()
}
