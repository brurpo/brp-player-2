//! mpv / libplacebo user shaders: parsing, GLSL generation and compilation.

pub mod codegen;
pub mod parse;

pub use codegen::{BindKind, Generated, ParamMode, Transfer};
pub use parse::{Localized, Param, ParamKind, Pass, Rpn, ShaderFile, TexDecl};

use anyhow::{anyhow, Result};

thread_local! {
    /// One compiler per thread: compiles on several threads run at once (the start-up
    /// build of the effect shaders uses that).
    static COMPILER: shaderc::Compiler = shaderc::Compiler::new().expect("shaderc");
}

/// Appended to a kernel source: compile it without SPIR-V optimization.
pub const NO_OPT: &str = "\n// spirv: no-opt\n";

/// GLSL -> SPIR-V. Errors carry shaderc's messages (line numbers refer to the
/// pass body thanks to `#line 1`).
pub fn compile_glsl(glsl: &str, name: &str) -> Result<Vec<u32>> {
    // From the disk cache when this source was compiled before (see shader_cache).
    crate::shader_cache::spirv(glsl, "vulkan1.3 performance (no-opt if marked)", || compile_now(glsl, name))
}

fn compile_now(glsl: &str, name: &str) -> Result<Vec<u32>> {
    COMPILER.with(|c| compile_with(c, glsl, name, shaderc::ShaderKind::Compute))
}

/// A fragment shader, GLSL -> SPIR-V (cached like `compile_glsl`).
pub fn compile_fragment(glsl: &str, name: &str) -> Result<Vec<u32>> {
    crate::shader_cache::spirv(glsl, "fragment vulkan1.3 performance", || COMPILER.with(|c| compile_with(c, glsl, name, shaderc::ShaderKind::Fragment)))
}

fn compile_with(c: &shaderc::Compiler, glsl: &str, name: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
    let mut opts = shaderc::CompileOptions::new().map_err(|e| anyhow!("shaderc options: {e}"))?;
    opts.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    // Sources marked "// spirv: no-opt" skip spirv-opt: on fully unrolled code (the
    // denoiser's 8x8 DCTs) it took 15-22 s per kernel, for no speed (the driver
    // optimizes again).
    let level = if glsl.contains(NO_OPT) { shaderc::OptimizationLevel::Zero } else { shaderc::OptimizationLevel::Performance };
    opts.set_optimization_level(level);
    let art = c
        .compile_into_spirv(glsl, kind, name, "main", Some(&opts))
        .map_err(|e| anyhow!("{e}"))?;
    Ok(art.as_binary().to_vec())
}

/// Generate and compile one pass. Tries parameters as uniforms first; if the
/// shader uses them in constant expressions, falls back to constants.
pub fn build_pass(
    file: &ShaderFile,
    pass: usize,
    hooked: &str,
    binds: &[(String, BindKind)],
    params: &[f64],
    transfer: Transfer,
) -> Result<(Generated, Vec<u32>, ParamMode)> {
    let name = format!("{}#{}", file.name, pass);
    let g = codegen::generate(file, pass, hooked, binds, params, ParamMode::Uniform, transfer);
    match compile_glsl(&g.glsl, &name) {
        Ok(spv) => Ok((g, spv, ParamMode::Uniform)),
        Err(e1) => {
            let g2 = codegen::generate(file, pass, hooked, binds, params, ParamMode::Const, transfer);
            match compile_glsl(&g2.glsl, &name) {
                Ok(spv) => Ok((g2, spv, ParamMode::Const)),
                Err(_) => Err(e1),
            }
        }
    }
}

/// How each BIND of a pass is bound: storage images for STORAGE textures,
/// sampled textures for everything else.
pub fn bind_kinds(file: &ShaderFile, pass: &Pass) -> Vec<(String, BindKind)> {
    let mut names = pass.binds.clone();
    // Leniency: a declared STORAGE texture used in the code without BIND is bound
    // automatically (some hand-merged shaders rely on that).
    for t in file.textures.iter().filter(|t| t.storage) {
        if !names.contains(&t.name) && mentions(&pass.body, &t.name) {
            log::debug!("{}: auto-binding storage texture {}", file.name, t.name);
            names.push(t.name.clone());
        }
    }
    names
        .iter()
        .map(|b| {
            let kind = match file.textures.iter().find(|t| &t.name == b) {
                Some(t) if t.storage => BindKind::Storage(t.format.clone()),
                _ => BindKind::Sampled,
            };
            (b.clone(), kind)
        })
        .collect()
}

/// Whole-word occurrence of `word` in `text`.
fn mentions(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + word.len()..].chars().next();
        let is_id = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        !is_id(before) && !is_id(after)
    })
}
