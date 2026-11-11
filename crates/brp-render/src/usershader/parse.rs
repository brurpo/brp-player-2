//! Parser for mpv / libplacebo user shaders ("//!HOOK" format).
//!
//! A file is a sequence of blocks. Each block starts with `//!` header lines and is
//! followed by a body: GLSL for a pass, the default value for a PARAM, hex data for
//! a TEXTURE.

use anyhow::{bail, Context, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Float,
    Int,
    Uint,
    /// `#define` — needs a recompile when changed.
    Define,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub desc: String,
    pub kind: ParamKind,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct TexDecl {
    pub name: String,
    pub w: u32,
    pub h: u32,
    /// GLSL image format qualifier, e.g. "rgba16f".
    pub format: String,
    pub storage: bool,
    /// Inline texel data (hex in the file), if any.
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Pass {
    pub hooks: Vec<String>,
    pub binds: Vec<String>,
    pub save: Option<String>,
    pub width: Option<Rpn>,
    pub height: Option<Rpn>,
    pub when: Option<Rpn>,
    pub components: Option<u32>,
    /// Block width/height and thread counts (`//!COMPUTE bw bh [tw th]`).
    pub compute: Option<[u32; 4]>,
    pub desc: String,
    pub body: String,
}

#[derive(Clone, Debug, Default)]
pub struct ShaderFile {
    pub name: String,
    pub params: Vec<Param>,
    pub textures: Vec<TexDecl>,
    pub passes: Vec<Pass>,
    /// Frames to replay after a jump so stateful (STORAGE) shaders see the right
    /// history (`// @brp temporal_warmup n`, our extension; `//!TEMPORAL_WARMUP n` too).
    pub warmup: Option<u32>,
    /// Translations (our extension, plain comments so mpv/libplacebo still accept the
    /// file; usually at the top, before the first `//!` line):
    ///   `// @brp pt.title Correção de cor`
    ///   `// @brp pt.param STRENGTH Intensidade | Quanto o efeito age (cat: Geral)`
    /// The language is a code ("pt"); the description after `|` is optional.
    pub l10n: Vec<Localized>,
    /// Working colour space of the file's MAIN/RGB passes (`// @brp space linear`, our
    /// extension; DESIGN.md §7a). None: the source's own encoding, as in mpv.
    pub space: Option<crate::workspace::Space>,
}

/// A translated name (or parameter label and description) for one language.
#[derive(Clone, Debug)]
pub struct Localized {
    pub lang: String,
    /// None: the effect's own name; Some: the parameter of that name.
    pub param: Option<String>,
    pub label: String,
    pub desc: Option<String>,
}

impl ShaderFile {
    pub fn is_stateful(&self) -> bool {
        self.textures.iter().any(|t| t.storage)
    }

    /// The effect's name in `lang`, if the file has one.
    pub fn title(&self, lang: &str) -> Option<&str> {
        self.l10n.iter().find(|l| l.param.is_none() && l.lang == lang).map(|l| l.label.as_str())
    }

    /// Parameter `i`'s label in `lang` (its name when not translated).
    pub fn param_label(&self, i: usize, lang: &str) -> &str {
        let p = &self.params[i];
        self.param_l10n(i, lang).map_or(p.name.as_str(), |l| l.label.as_str())
    }

    /// Parameter `i`'s description in `lang` (the file's DESC when not translated).
    /// "(cat: …)" and "(options: …)" tags the translation leaves out are taken from
    /// DESC, so grouping and choice lists stay as they are.
    pub fn param_desc(&self, i: usize, lang: &str) -> String {
        let p = &self.params[i];
        match self.param_l10n(i, lang).and_then(|l| l.desc.as_deref()) {
            Some(d) => with_tags(d, &p.desc),
            None => p.desc.clone(),
        }
    }

    fn param_l10n(&self, i: usize, lang: &str) -> Option<&Localized> {
        let name = &self.params.get(i)?.name;
        self.l10n.iter().find(|l| l.lang == lang && l.param.as_deref() == Some(name.as_str()))
    }
}

/// `text` with the "(cat: …)" and "(options: …)" tags of `from` it lacks appended.
pub fn with_tags(text: &str, from: &str) -> String {
    let mut out = text.trim().to_string();
    for tag in ["(options:", "(cat:"] {
        if out.contains(tag) {
            continue;
        }
        if let Some(i) = from.find(tag) {
            // To the ")" closing this tag (options may hold parentheses).
            let mut depth = 0;
            let end = from[i + 1..].char_indices().find_map(|(j, c)| match c {
                '(' => {
                    depth += 1;
                    None
                }
                ')' if depth == 0 => Some(i + 1 + j + 1),
                ')' => {
                    depth -= 1;
                    None
                }
                _ => None,
            });
            out.push(' ');
            out.push_str(&from[i..end.unwrap_or(from.len())]);
        }
    }
    out
}

/// `// @brp <lang>.title <name>` or `// @brp <lang>.param <NAME> <label> [| <desc>]`.
fn localized(rest: &str) -> Option<Localized> {
    let (tag, text) = rest.split_once(char::is_whitespace)?;
    let text = text.trim();
    if let Some(lang) = tag.strip_suffix(".title") {
        return Some(Localized { lang: lang.to_string(), param: None, label: text.to_string(), desc: None });
    }
    let lang = tag.strip_suffix(".param")?;
    let (name, text) = text.split_once(char::is_whitespace)?;
    let (label, desc) = match text.split_once('|') {
        Some((l, d)) => (l.trim(), Some(d.trim().to_string()).filter(|d| !d.is_empty())),
        None => (text.trim(), None),
    };
    Some(Localized { lang: lang.to_string(), param: Some(name.to_string()), label: label.to_string(), desc })
}

/// Reverse-polish expression from WIDTH / HEIGHT / WHEN.
#[derive(Clone, Debug)]
pub struct Rpn(pub Vec<String>);

impl Rpn {
    /// `size(name)` gives (w, h) of a texture by name (`OUTPUT`, `HOOKED`, `MAIN`, …);
    /// `param(name)` the value of a shader parameter.
    pub fn eval(&self, size: &dyn Fn(&str) -> Option<(f64, f64)>, param: &dyn Fn(&str) -> Option<f64>) -> Result<f64> {
        let mut st: Vec<f64> = Vec::new();
        for tok in &self.0 {
            let t = tok.as_str();
            let bin = |st: &mut Vec<f64>, f: fn(f64, f64) -> f64| -> Result<()> {
                let b = st.pop().context("RPN stack underflow")?;
                let a = st.pop().context("RPN stack underflow")?;
                st.push(f(a, b));
                Ok(())
            };
            match t {
                "+" => bin(&mut st, |a, b| a + b)?,
                "-" => bin(&mut st, |a, b| a - b)?,
                "*" => bin(&mut st, |a, b| a * b)?,
                "/" => bin(&mut st, |a, b| a / b)?,
                ">" => bin(&mut st, |a, b| (a > b) as i32 as f64)?,
                "<" => bin(&mut st, |a, b| (a < b) as i32 as f64)?,
                "=" => bin(&mut st, |a, b| (a == b) as i32 as f64)?,
                "!" => {
                    let a = st.pop().context("RPN stack underflow")?;
                    st.push((a == 0.0) as i32 as f64);
                }
                _ => {
                    if let Ok(v) = t.parse::<f64>() {
                        st.push(v);
                    } else if let Some((name, comp)) = t.rsplit_once('.') {
                        let (w, h) = size(name).with_context(|| format!("unknown texture '{name}' in expression"))?;
                        st.push(match comp {
                            "w" | "width" => w,
                            "h" | "height" => h,
                            _ => bail!("bad size component '{t}'"),
                        });
                    } else if let Some(v) = param(t) {
                        st.push(v);
                    } else {
                        bail!("bad expression token '{t}'");
                    }
                }
            }
        }
        st.pop().context("empty expression")
    }
}

fn rpn(s: &str) -> Rpn {
    Rpn(s.split_whitespace().map(str::to_string).collect())
}

enum Block {
    Pass(Pass),
    Param { name: String, desc: String, kind: ParamKind, min: Option<f64>, max: Option<f64>, body: String },
    Texture { decl: TexDecl, body: String },
}

pub fn parse(name: &str, text: &str) -> Result<ShaderFile> {
    let mut file = ShaderFile { name: name.to_string(), ..Default::default() };
    let mut blocks: Vec<Block> = Vec::new();
    let mut in_body = false;

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim_start();
        // Our extension as a plain comment, so mpv/libplacebo still accept the file:
        //   // @brp temporal_warmup 4
        if let Some(rest) = trimmed.strip_prefix("// @brp ") {
            if let Some(n) = rest.trim().strip_prefix("temporal_warmup") {
                file.warmup = n.trim().parse().ok();
            } else if let Some(s) = rest.trim().strip_prefix("space ") {
                match crate::workspace::Space::from_name(s) {
                    Some(sp) => file.space = Some(sp),
                    None => log::warn!("{name}:{}: unknown space '{}' (source, linear, display, pq)", lineno + 1, s.trim()),
                }
            } else if let Some(l) = localized(rest.trim()) {
                file.l10n.push(l);
            }
        }
        if let Some(header) = trimmed.strip_prefix("//!") {
            let (key, rest) = header.split_once(char::is_whitespace).unwrap_or((header, ""));
            let rest = rest.trim();
            let key = key.to_ascii_uppercase();
            if key == "TEMPORAL_WARMUP" {
                file.warmup = rest.parse().ok();
                continue;
            }
            // A new block starts at a block keyword or at any header after a body;
            // HOOK only starts one if we're not already in a pass header (mpv allows
            // DESC/BIND before HOOK, and several HOOK lines per pass).
            let in_pass_header = !in_body && matches!(blocks.last(), Some(Block::Pass(_)));
            let starts = match key.as_str() {
                "PARAM" | "TEXTURE" | "BUFFER" => true,
                "HOOK" => !in_pass_header,
                _ => in_body || blocks.is_empty(),
            };
            if starts {
                match key.as_str() {
                    "PARAM" => blocks.push(Block::Param {
                        name: rest.to_string(),
                        desc: String::new(),
                        kind: ParamKind::Float,
                        min: None,
                        max: None,
                        body: String::new(),
                    }),
                    "TEXTURE" => blocks.push(Block::Texture {
                        decl: TexDecl { name: rest.to_string(), w: 1, h: 1, format: "rgba8".into(), storage: false, data: Vec::new() },
                        body: String::new(),
                    }),
                    _ => blocks.push(Block::Pass(Pass {
                        hooks: Vec::new(),
                        binds: Vec::new(),
                        save: None,
                        width: None,
                        height: None,
                        when: None,
                        components: None,
                        compute: None,
                        desc: String::new(),
                        body: String::new(),
                    })),
                }
                in_body = false;
                if matches!(key.as_str(), "PARAM" | "TEXTURE") {
                    continue;
                }
            }
            let ctx = || format!("{name}:{}: {key}", lineno + 1);
            match blocks.last_mut().unwrap() {
                Block::Pass(p) => match key.as_str() {
                    "HOOK" => p.hooks.push(rest.to_string()),
                    "BIND" => p.binds.push(rest.to_string()),
                    "SAVE" => p.save = Some(rest.to_string()),
                    "WIDTH" => p.width = Some(rpn(rest)),
                    "HEIGHT" => p.height = Some(rpn(rest)),
                    "WHEN" => p.when = Some(rpn(rest)),
                    "COMPONENTS" => p.components = rest.parse().ok(),
                    "DESC" => p.desc = rest.to_string(),
                    "COMPUTE" => {
                        let v: Vec<u32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                        p.compute = match v.as_slice() {
                            [bw, bh] => Some([*bw, *bh, *bw, *bh]),
                            [bw, bh, tw, th] => Some([*bw, *bh, *tw, *th]),
                            _ => bail!("{}: expected 2 or 4 numbers", ctx()),
                        };
                    }
                    "OFFSET" | "ALIGN" => log::warn!("{}: not supported yet, ignored", ctx()),
                    _ => log::warn!("{}: unknown directive, ignored", ctx()),
                },
                Block::Param { desc, kind, min, max, .. } => match key.as_str() {
                    "DESC" => *desc = rest.to_string(),
                    "TYPE" => {
                        // "//!TYPE [DYNAMIC|CONSTANT] float"
                        let t = rest.split_whitespace().last().unwrap_or("float");
                        *kind = match t {
                            "float" => ParamKind::Float,
                            "int" => ParamKind::Int,
                            "uint" => ParamKind::Uint,
                            "DEFINE" | "define" => ParamKind::Define,
                            other => bail!("{}: unsupported type '{other}'", ctx()),
                        };
                    }
                    "MINIMUM" => *min = rest.parse().ok(),
                    "MAXIMUM" => *max = rest.parse().ok(),
                    _ => log::warn!("{}: unknown directive in PARAM, ignored", ctx()),
                },
                Block::Texture { decl, .. } => match key.as_str() {
                    "SIZE" => {
                        let v: Vec<u32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                        decl.w = *v.first().unwrap_or(&1);
                        decl.h = *v.get(1).unwrap_or(&1);
                    }
                    "FORMAT" => decl.format = rest.to_string(),
                    "STORAGE" => decl.storage = true,
                    "FILTER" | "BORDER" => {}
                    _ => log::warn!("{}: unknown directive in TEXTURE, ignored", ctx()),
                },
            }
        } else {
            if blocks.is_empty() {
                continue; // leading comments
            }
            in_body = true;
            match blocks.last_mut().unwrap() {
                Block::Pass(p) => {
                    p.body.push_str(line);
                    p.body.push('\n');
                }
                Block::Param { body, .. } | Block::Texture { body, .. } => {
                    body.push_str(line);
                    body.push('\n');
                }
            }
        }
    }

    for b in blocks {
        match b {
            Block::Pass(p) => {
                if p.hooks.is_empty() {
                    bail!("{name}: a pass without //!HOOK");
                }
                file.passes.push(p);
            }
            Block::Param { name: pname, desc, kind, min, max, body } => {
                let v = body.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("//")).unwrap_or("0");
                let default = v.trim_end_matches(['f', 'u', ';']).parse::<f64>().unwrap_or_else(|_| {
                    log::warn!("{name}: PARAM {pname}: cannot parse default '{v}', using 0");
                    0.0
                });
                file.params.push(Param { name: pname, desc, kind, default, min, max });
            }
            Block::Texture { mut decl, body } => {
                let hex: String = body.chars().filter(|c| c.is_ascii_hexdigit()).collect();
                decl.data = (0..hex.len() / 2).filter_map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()).collect();
                file.textures.push(decl);
            }
        }
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_space() {
        use crate::workspace::Space;
        let body = "//!HOOK MAIN\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";
        assert_eq!(parse("a.glsl", body).unwrap().space, None);
        let f = parse("b.glsl", &format!("// @brp space linear\n{body}")).unwrap();
        assert_eq!(f.space, Some(Space::Linear));
        // The MAIN pass works in the file's space; a LINEAR hook in libplacebo's.
        let lin = "//!HOOK LINEAR\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";
        let g = parse("c.glsl", &format!("{lin}{body}")).unwrap();
        let passes = crate::effects::main_passes(&g);
        assert_eq!(passes.iter().map(|p| (p.0, p.2)).collect::<Vec<_>>(), vec![(1, Space::Source), (0, Space::MpvLinear)]);
    }

    #[test]
    fn translations() {
        let src = "// @brp pt.title Inverter\n// @brp pt.param STRENGTH Intensidade | Quanto inverter\n// @brp pt.param MODE Modo\n//!PARAM STRENGTH\n//!DESC How much (cat: Main)\n//!TYPE float\n1.0\n//!PARAM MODE\n//!DESC Mode (options: A|B (fast))\n//!TYPE int\n0\n//!HOOK MAIN\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n";
        let f = parse("t.glsl", src).unwrap();
        assert_eq!(f.title("pt"), Some("Inverter"));
        assert_eq!(f.title("en"), None);
        assert_eq!(f.param_label(0, "pt"), "Intensidade");
        assert_eq!(f.param_label(0, "en"), "STRENGTH");
        assert_eq!(f.param_desc(0, "pt"), "Quanto inverter (cat: Main)");
        assert_eq!(f.param_desc(1, "pt"), "Mode (options: A|B (fast))");
        assert_eq!(f.param_label(1, "pt"), "Modo");
        assert_eq!(f.params[0].default, 1.0);
    }
}

#[cfg(test)]
mod bundled {
    use super::*;

    /// The shaders shipped in the repo's `shaders/` parse, name only parameters they
    /// have, and give every parameter the panel shows a Portuguese label (the G and B
    /// of a colour are shown with its R).
    #[test]
    fn bundled_shaders_translated() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
        let mut n = 0;
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "glsl") {
                continue;
            }
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let f = parse(&name, &std::fs::read_to_string(&p).unwrap()).unwrap();
            assert!(f.title("pt").is_some(), "{name}: no pt.title");
            for l in &f.l10n {
                if let Some(pn) = &l.param {
                    assert!(f.params.iter().any(|q| &q.name == pn), "{name}: pt.param {pn} is not a parameter");
                }
            }
            for (i, q) in f.params.iter().enumerate() {
                let colour_part = ["G", "B", "g", "b"].iter().any(|s| q.name.strip_suffix(s).is_some_and(|pre| f.params.iter().any(|r| r.name == format!("{pre}R") || r.name == format!("{pre}r"))));
                if !colour_part {
                    assert!(f.param_l10n(i, "pt").is_some(), "{name}: {} has no Portuguese", q.name);
                }
            }
            n += 1;
        }
        assert!(n > 10);
    }
}
