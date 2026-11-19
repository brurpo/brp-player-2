//! The frame on screen saved as a picture, or copied to the clipboard (the viewer's
//! right-click menu). Made by the export, one frame long: at the source's full size,
//! with the whole effect stack, deinterlacing and OpenColorIO's output as an export
//! of a still image gets them, not the picture as scaled into the window. Written as
//! PNG into a folder of its own, then moved where the user chose or put on the
//! clipboard.

use crate::export::{self, DeintChoice, ExportJob, ExportSettings, Running, State};
use crate::player::Player;
use crate::ui::UiState;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub enum GrabTo {
    File(PathBuf),
    Clipboard,
}

pub struct Grab {
    run: Running,
    dir: PathBuf,
    to: GrabTo,
    /// Which of the frame's pictures is on screen (fields, slow motion's in-betweens:
    /// a source frame makes several).
    pick: usize,
}

impl Grab {
    /// Start making the frame on screen.
    pub fn start(p: &Player, s: &UiState, to: GrabTo) -> Result<Grab> {
        let src = p.source.as_ref().context("no video open")?;
        let ctx = p.ai.ctx.clone().context("the GPU is not ready")?;
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("brp-frame-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        // Alpha kept (a removed background stays transparent); opaque frames just
        // carry an opaque alpha channel.
        let settings = ExportSettings { container: "png".into(), codec: "png".into(), audio: String::new(), alpha: true, ..ExportSettings::default() };
        let job = ExportJob {
            input: src.info.path.clone(),
            output: dir.join("frame.png").to_string_lossy().into_owned(),
            settings,
            stack: p.fx.all_stages(),
            range: Some((p.current, p.current)),
            deint: DeintChoice::Fixed(p.ai_source_deint()),
            layer: src.info.layers.get(src.info.layer).map(|(n, _)| n.clone()),
            ocio: s.ocio_job(),
        };
        Ok(Grab { run: export::start(job, ctx), dir, to, pick: p.sub as usize })
    }

    /// None while it runs; then the message to show (Err: what went wrong).
    pub fn poll(&mut self) -> Option<Result<GrabDone>> {
        let pr = self.run.snapshot();
        let r = match pr.state {
            State::Done => self.finish(),
            State::Failed(e) => Err(anyhow::anyhow!(e)),
            State::Cancelled => Err(anyhow::anyhow!("cancelled")),
            _ => return None,
        };
        let _ = std::fs::remove_dir_all(&self.dir);
        Some(r)
    }

    fn finish(&self) -> Result<GrabDone> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png"))).collect();
        files.sort();
        let file = files.get(self.pick).or(files.last()).context("the export wrote no picture")?;
        match &self.to {
            GrabTo::File(to) => {
                if std::fs::rename(file, to).is_err() {
                    std::fs::copy(file, to).with_context(|| format!("writing {}", to.display()))?;
                }
                Ok(GrabDone::Saved(to.clone()))
            }
            GrabTo::Clipboard => {
                let (w, h) = to_clipboard(file)?;
                Ok(GrabDone::Copied(w, h))
            }
        }
    }
}

pub enum GrabDone {
    Saved(PathBuf),
    Copied(usize, usize),
}

/// A PNG on the clipboard as a picture (RGBA).
fn to_clipboard(png_file: &Path) -> Result<(usize, usize)> {
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(png_file)?));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width as usize, info.height as usize);
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..w * h * 4].to_vec(),
        png::ColorType::Rgb => buf[..w * h * 3].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..w * h * 2].chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf[..w * h].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        c => anyhow::bail!("unexpected PNG colour type {c:?}"),
    };
    let mut cb = arboard::Clipboard::new().context("clipboard")?;
    cb.set_image(arboard::ImageData { width: w, height: h, bytes: rgba.into() }).context("clipboard")?;
    Ok((w, h))
}
