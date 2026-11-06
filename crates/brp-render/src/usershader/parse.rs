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
    /// history (`//!TEMPORAL_WARMUP n`, our extension).
    pub warmup: Option<u32>,
}

impl ShaderFile {
    pub fn is_stateful(&self) -> bool {
        self.textures.iter().any(|t| t.storage)
    }
}

/// Reverse-polish expression from WIDTH / HEIGHT / WHEN.
#[derive(Clone, Debug)]
pub struct Rpn(pub Vec<String>);

impl Rpn {
    /// `size(name)` gives (w, h) of a texture by name (`OUTPUT`, `HOOKED`, `MAIN`, …).
    pub fn eval(&self, size: &dyn Fn(&str) -> Option<(f64, f64)>) -> Result<f64> {
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
