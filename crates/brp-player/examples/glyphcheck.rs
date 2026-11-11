//! Lists the non-ASCII characters used in the UI's string literals and whether egui's
//! fonts (the ones the player uses) can draw them.
//! cargo run --release -p brp-player --example glyphcheck

use std::collections::BTreeMap;

fn main() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut chars: BTreeMap<char, Vec<String>> = BTreeMap::new();
    for f in ["ui.rs", "i18n.rs", "popup.rs", "main.rs", "effects.rs", "ai.rs", "player.rs"] {
        let Ok(text) = std::fs::read_to_string(dir.join(f)) else { continue };
        for (n, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            // Characters inside string or char literals.
            let mut inside = false;
            for c in line.chars() {
                if c == '"' {
                    inside = !inside;
                } else if inside && !c.is_ascii() {
                    chars.entry(c).or_default().push(format!("{f}:{}", n + 1));
                }
            }
        }
    }
    let ctx = egui::Context::default();
    // One frame so the fonts are loaded.
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let mut missing = 0;
    for (c, at) in &chars {
        let ok = ctx.fonts_mut(|f| f.has_glyph(&egui::FontId::proportional(14.0), *c));
        if !ok {
            missing += 1;
        }
        let mut places = at.clone();
        places.dedup();
        let shown: Vec<_> = places.iter().take(4).cloned().collect();
        println!("{} U+{:04X} {:<7} {}", c, *c as u32, if ok { "ok" } else { "MISSING" }, shown.join(" "));
    }
    println!("{} characters, {missing} missing", chars.len());
}
