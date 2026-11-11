//! Image sequences (DESIGN.md §7c): the numbered files a picked file belongs to.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// Still-image formats a sequence can be made of (decoded by FFmpeg; EXR by our own
/// reader, `exr_seq`).
pub const IMAGE_EXTS: &[&str] = &[
    "png", "jpg", "jpeg", "jpe", "tif", "tiff", "dpx", "cin", "bmp", "tga", "webp", "jp2", "j2k", "jxl", "psd", "hdr", "pfm", "sgi", "rgb", "rgba", "bw", "ppm", "pgm", "pbm", "pam", "qoi", "xwd", "sun", "ras", "pcx", "gif", "exr",
];

/// A still-image file by its extension.
pub fn is_image(p: &Path) -> bool {
    p.extension().is_some_and(|x| IMAGE_EXTS.iter().any(|e| x.eq_ignore_ascii_case(e)))
}

/// The last run of digits in a file stem: (before, digits, after).
pub fn split_number(stem: &str) -> Option<(&str, &str, &str)> {
    let end = stem.rfind(|c: char| c.is_ascii_digit())? + 1;
    let start = stem[..end].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    Some((&stem[..start], &stem[start..end], &stem[end..]))
}

/// (name before the number, name after it with the extension, the number).
fn key(p: &Path) -> Option<(String, String, i64)> {
    let stem = p.file_stem()?.to_string_lossy().to_string();
    let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let (a, n, b) = split_number(&stem)?;
    Some((a.to_string(), format!("{b}.{ext}"), n.parse().ok()?))
}

/// The files of the sequence `path` belongs to: same name and extension apart from the
/// last number (any padding, gaps allowed), in number order, with the first number.
/// A file without a number, or alone: just that file. A folder: its largest such set of
/// files that `member` accepts.
pub fn files(path: &Path, member: fn(&Path) -> bool) -> Result<(Vec<PathBuf>, Option<i64>)> {
    let (dir, pick) = if path.is_dir() { (path.to_path_buf(), None) } else { (path.parent().map(Path::to_path_buf).unwrap_or_default(), Some(path.to_path_buf())) };
    let dir_read = if dir.as_os_str().is_empty() { PathBuf::from(".") } else { dir.clone() };
    let all: Vec<PathBuf> = std::fs::read_dir(&dir_read)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && member(p)).collect();
    let target = match &pick {
        Some(p) => match key(p) {
            Some((a, b, _)) => (a, b),
            None => return Ok((vec![p.clone()], None)),
        },
        None => {
            let mut counts: std::collections::HashMap<(String, String), usize> = Default::default();
            for p in &all {
                if let Some((a, b, _)) = key(p) {
                    *counts.entry((a, b)).or_default() += 1;
                }
            }
            match counts.into_iter().max_by_key(|(k, n)| (*n, std::cmp::Reverse(k.clone()))) {
                Some((k, _)) => k,
                None => match all.first() {
                    Some(p) => return Ok((vec![p.clone()], None)),
                    None => bail!("no image files in {}", path.display()),
                },
            }
        }
    };
    let mut seq: Vec<(i64, PathBuf)> = all.into_iter().filter_map(|p| key(&p).filter(|(a, b, _)| (a, b) == (&target.0, &target.1)).map(|(_, _, n)| (n, p))).collect();
    seq.sort_by_key(|(n, _)| *n);
    seq.dedup_by_key(|(n, _)| *n);
    if seq.is_empty() {
        // The picked file itself (`member` may not cover it).
        return match pick {
            Some(p) => Ok((vec![p], None)),
            None => bail!("no image files in {}", path.display()),
        };
    }
    let first = seq[0].0;
    Ok((seq.into_iter().map(|(_, p)| p).collect(), Some(first)))
}

/// Name for a sequence: the first file's, with its number as #s (`chap_####.exr`).
pub fn display_name(files: &[PathBuf]) -> String {
    let first = &files[0];
    let name = first.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    if files.len() < 2 {
        return name;
    }
    let Some(stem) = first.file_stem().map(|s| s.to_string_lossy().to_string()) else { return name };
    let ext = first.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    match split_number(&stem) {
        Some((a, n, b)) => format!("{a}{}{b}{ext}", "#".repeat(n.len())),
        None => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_in_names() {
        assert_eq!(split_number("chap_0001"), Some(("chap_", "0001", "")));
        assert_eq!(split_number("shot2_v3.0042"), Some(("shot2_v3.", "0042", "")));
        assert_eq!(split_number("render0042_beauty"), Some(("render", "0042", "_beauty")));
        assert_eq!(split_number("plate"), None);
        assert_eq!(display_name(&[PathBuf::from("a/shot.0001.dpx"), PathBuf::from("a/shot.0002.dpx")]), "shot.####.dpx");
    }
}
