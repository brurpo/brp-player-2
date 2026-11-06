//! mpv / libplacebo user shaders: parsing, GLSL generation and compilation.

pub mod codegen;
pub mod parse;

pub use codegen::{BindKind, Generated, ParamMode, Transfer};
pub use parse::{Param, ParamKind, Pass, Rpn, ShaderFile, TexDecl};

use anyhow::{anyhow, Result};
use std::sync::OnceLock;

fn compiler() -> &'static std::sync::Mutex<shaderc::Compiler> {
    static C: OnceLock<std::sync::Mutex<shaderc::Compiler>> = OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(shaderc::Compiler::new().expect("shaderc")))
}

/// GLSL -> SPIR-V. Errors carry shaderc's messages (line numbers refer to the
/// pass body thanks to `#line 1`).
pub fn compile_glsl(glsl: &str, name: &str) -> Result<Vec<u32>> {
    let c = compiler().lock().unwrap();
    let mut opts = shaderc::CompileOptions::new().map_err(|e| anyhow!("shaderc options: {e}"))?;
    opts.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);
    let art = c
        .compile_into_spirv(glsl, shaderc::ShaderKind::Compute, name, "main", Some(&opts))
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
