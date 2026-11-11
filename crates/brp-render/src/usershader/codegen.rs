//! GLSL generation: wraps a user-shader pass in a compute shader that provides the
//! mpv / libplacebo environment (NAME_tex, NAME_pos, HOOKED_*, frame, …).

use super::parse::{ParamKind, ShaderFile};
use std::fmt::Write;

/// How a BIND name is bound.
#[derive(Clone, Debug, PartialEq)]
pub enum BindKind {
    /// Sampled texture (stage textures, SAVEd textures, inline-data TEXTUREs).
    Sampled,
    /// Read-write storage image with this GLSL format (STORAGE TEXTUREs).
    Storage(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamMode {
    /// Parameters in a uniform buffer (changing them needs no recompile).
    Uniform,
    /// Parameters as constants (for shaders that use them in constant expressions).
    Const,
}

/// Source transfer function, for linearize()/delinearize().
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Transfer {
    Sdr,
    Pq,
    Hlg,
    Linear,
}

pub struct Generated {
    pub glsl: String,
    /// Binding 0 is the parameter buffer when present.
    pub has_params: bool,
    /// One per BIND, in order, starting at binding 1.
    pub binds: Vec<(String, BindKind)>,
    /// Binding index of `out_image`.
    pub out_binding: u32,
    pub local_size: [u32; 2],
    /// Output pixels per workgroup.
    pub block: [u32; 2],
}

/// Values used for DEFINE params and in Const mode.
pub fn param_value_str(kind: ParamKind, v: f64) -> String {
    match kind {
        ParamKind::Float => {
            let s = format!("{v:?}");
            if s.contains('.') || s.contains('e') { s } else { format!("{s}.0") }
        }
        ParamKind::Int => format!("{}", v.round() as i64),
        ParamKind::Uint => format!("{}u", v.round().max(0.0) as u64),
        ParamKind::Define => {
            if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
        }
    }
}

fn ident(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

pub fn generate(
    file: &ShaderFile,
    pass_index: usize,
    hooked: &str,
    binds: &[(String, BindKind)],
    params: &[f64],
    mode: ParamMode,
    transfer: Transfer,
) -> Generated {
    let pass = &file.passes[pass_index];
    let compute = pass.compute;
    let (block, local) = match compute {
        Some([bw, bh, tw, th]) => ([bw, bh], [tw, th]),
        None => ([16, 16], [16, 16]),
    };
    let mut g = String::new();
    let _ = writeln!(g, "#version 450");
    let _ = writeln!(g, "// {} / pass {} ({})", file.name, pass_index, pass.desc);
    let _ = writeln!(g, "layout(local_size_x = {}, local_size_y = {}) in;", local[0], local[1]);
    let _ = writeln!(
        g,
        "layout(push_constant) uniform _brp_pc {{ vec2 _brp_out_size; vec2 input_size; vec2 target_size; vec2 tex_offset; int frame; float random; }};"
    );
    // Normalised position of the output pixel (libplacebo: the pass's texture coordinate).
    let _ = writeln!(g, "#define _brp_pos ((vec2(gl_GlobalInvocationID.xy) + 0.5) / _brp_out_size)");

    // Parameters.
    let uniform_params: Vec<_> = file.params.iter().enumerate().filter(|(_, p)| p.kind != ParamKind::Define).collect();
    let has_params = mode == ParamMode::Uniform && !uniform_params.is_empty();
    for (i, p) in file.params.iter().enumerate() {
        let v = params.get(i).copied().unwrap_or(p.default);
        if p.kind == ParamKind::Define {
            let _ = writeln!(g, "#define {} {}", p.name, param_value_str(p.kind, v));
        } else if mode == ParamMode::Const {
            let ty = match p.kind {
                ParamKind::Float => "float",
                ParamKind::Int => "int",
                _ => "uint",
            };
            let _ = writeln!(g, "const {ty} {} = {};", p.name, param_value_str(p.kind, v));
        }
    }
    if has_params {
        let _ = writeln!(g, "layout(std140, set = 0, binding = 0) uniform _brp_params {{");
        for (_, p) in &uniform_params {
            let ty = match p.kind {
                ParamKind::Float => "float",
                ParamKind::Int => "int",
                _ => "uint",
            };
            let _ = writeln!(g, "    {ty} {};", p.name);
        }
        let _ = writeln!(g, "}};");
    }

    // Bound textures.
    for (i, (name, kind)) in binds.iter().enumerate() {
        let b = i + 1;
        let id = ident(name);
        match kind {
            BindKind::Storage(fmt) => {
                let _ = writeln!(g, "layout(set = 0, binding = {b}, {fmt}) uniform image2D {id};");
            }
            BindKind::Sampled => {
                let _ = writeln!(g, "layout(set = 0, binding = {b}) uniform sampler2D _{id}_raw;");
                // Expressions, not variables: shaders use these in global
                // initializers, which run before main().
                let _ = writeln!(
                    g,
                    "#define _{id}_pos _brp_pos\n#define _{id}_size vec2(textureSize(_{id}_raw, 0))\n#define _{id}_pt (1.0 / _{id}_size)"
                );
                write_tex_macros(&mut g, &id, &id);
                // libplacebo also exposes the hooked texture under its stage name
                // (LUMA_*, CHROMA_*, MAIN_*); shaders test `defined(CHROMA_raw)`.
                if name == "HOOKED" && !binds.iter().any(|(n, _)| n == hooked) {
                    write_tex_macros(&mut g, &ident(hooked), &id);
                }
                if name == "HOOKED" || name == hooked {
                    // HOOKED_* always refers to the hooked texture.
                    if name != "HOOKED" && !binds.iter().any(|(n, _)| n == "HOOKED") {
                        write_tex_macros(&mut g, "HOOKED", &id);
                    }
                }
            }
        }
    }
    let out_binding = binds.len() as u32 + 1;
    let _ = writeln!(g, "layout(set = 0, binding = {out_binding}, rgba16f) uniform writeonly image2D out_image;");

    // linearize / delinearize for the source transfer (relative to SDR white).
    // Shaders may bring their own linearize/delinearize; only provide ours if not.
    let mut lin_fns = String::new();
    if !defines_fn(&pass.body, "linearize") {
        lin_fns += "vec4 linearize(vec4 c) { return vec4(_brp_lin(c.rgb), c.a); }
vec3 linearize(vec3 c) { return _brp_lin(c); }
";
    }
    if !defines_fn(&pass.body, "delinearize") {
        lin_fns += "vec4 delinearize(vec4 c) { return vec4(_brp_delin(c.rgb), c.a); }
vec3 delinearize(vec3 c) { return _brp_delin(c); }
";
    }
    let t = match transfer {
        Transfer::Sdr => 0,
        Transfer::Pq => 1,
        Transfer::Hlg => 2,
        Transfer::Linear => 3,
    };
    let _ = write!(
        g,
        r#"
#define _BRP_TRANSFER {t}
vec3 _brp_lin(vec3 c) {{
#if _BRP_TRANSFER == 0
    return pow(max(c, 0.0), vec3(2.4));
#elif _BRP_TRANSFER == 1
    vec3 p = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 78.84375));
    return pow(max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), vec3(1.0 / 0.1593017578125)) * (10000.0 / 203.0);
#elif _BRP_TRANSFER == 2
    vec3 lo = c * c / 3.0, hi = (exp((c - 0.55991073) / 0.17883277) + 0.28466892) / 12.0;
    return mix(lo, hi, step(0.5, c)) * (1000.0 / 203.0);
#else
    return c;
#endif
}}
vec3 _brp_delin(vec3 c) {{
#if _BRP_TRANSFER == 0
    return pow(max(c, 0.0), vec3(1.0 / 2.4));
#elif _BRP_TRANSFER == 1
    vec3 y = pow(max(c * (203.0 / 10000.0), 0.0), vec3(0.1593017578125));
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), vec3(78.84375));
#elif _BRP_TRANSFER == 2
    c *= 203.0 / 1000.0;
    vec3 lo = sqrt(3.0 * max(c, 0.0)), hi = 0.17883277 * log(max(12.0 * c - 0.28466892, 1e-6)) + 0.55991073;
    return mix(lo, hi, step(1.0 / 12.0, c));
#else
    return c;
#endif
}}
{lin_fns}
"#
    );

    let _ = writeln!(g, "#line 1");
    g.push_str(&pass.body);
    g.push('\n');
    if compute.is_some() {
        let _ = writeln!(g, "void main() {{\n    hook();\n}}");
    } else {
        let _ = writeln!(
            g,
            "void main() {{\n    ivec2 id = ivec2(gl_GlobalInvocationID.xy);\n    if (id.x >= int(_brp_out_size.x) || id.y >= int(_brp_out_size.y)) return;\n    imageStore(out_image, id, hook());\n}}"
        );
    }
    Generated { glsl: g, has_params, binds: binds.to_vec(), out_binding, local_size: local, block }
}

fn write_tex_macros(g: &mut String, alias: &str, id: &str) {
    let _ = writeln!(
        g,
        "#define {a}_raw _{id}_raw\n#define {a}_pos _{id}_pos\n#define {a}_size _{id}_size\n#define {a}_pt _{id}_pt\n\
         #define {a}_mul 1.0\n#define {a}_rot mat2(1.0, 0.0, 0.0, 1.0)\n#define {a}_off vec2(0.0)\n\
         #define {a}_tex(pos) (vec4(textureLod(_{id}_raw, (pos), 0.0)))\n\
         #define {a}_texOff(off) {a}_tex(_{id}_pos + _{id}_pt * vec2(off))\n\
         #define {a}_gather(pos, c) (textureGather(_{id}_raw, (pos), (c)))",
        a = alias
    );
}

/// Does `body` define a function called `name` (e.g. "vec3 linearize(...) {")?
fn defines_fn(body: &str, name: &str) -> bool {
    body.lines().any(|l| {
        let mut it = l.split_whitespace();
        let (Some(ty), Some(rest)) = (it.next(), it.next()) else { return false };
        matches!(ty, "vec2" | "vec3" | "vec4" | "float" | "void") && rest.split('(').next() == Some(name) && rest.contains('(')
    })
}
