//! Media layer: demux, frame-exact index, decode and decoded-frame cache.

mod ffutil;
pub mod audio;
pub mod detect;
pub mod frame;
pub mod index;
pub mod info;
pub mod source;

pub use frame::{FrameLayout, VideoFrame};
pub use info::{ColorInfo, FieldOrder, Matrix, MediaInfo, Primaries, Transfer};
pub use audio::{output_devices, Audio};
pub use detect::{FieldKind, Votes};
pub use source::{Request, Source};

/// Quiet FFmpeg's own logging (we log through `log`).
pub fn init() {
    unsafe { ffmpeg_sys_next::av_log_set_level(ffmpeg_sys_next::AV_LOG_ERROR) };
}
