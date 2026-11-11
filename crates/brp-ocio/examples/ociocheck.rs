//! Lists the player's OCIO configs and makes a sample shader.
//!   cargo run -p brp-ocio --release --example ociocheck -- <ocio dir> [--spaces]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("OpenColorIO {}", brp_ocio::version());
    let lib = brp_ocio::Library::load(std::path::Path::new(&args[0]), args.get(1).filter(|a| !a.starts_with("--")).map(std::path::Path::new));
    for (s, c) in lib.sources() {
        println!("\n== {:?}: {} ({} spaces), default display {:?}", s, c.name, c.spaces.len(), c.default_display);
        for d in &c.displays {
            println!("  display {:?} (default view {:?})", d.name, d.default_view);
            for v in &d.views {
                println!("      view {:?} -> {:?} [{}]", v.name, v.colorspace, v.encoding);
            }
        }
        if args.iter().any(|a| a == "--spaces") {
            for sp in &c.spaces {
                println!("  space {:?} [{}] family {:?} display {} data {} aliases {:?}", sp.name, sp.encoding, sp.family, sp.display, sp.data, sp.aliases);
            }
        }
        println!("  roles {:?}", c.roles);
    }
    let b = lib.get(brp_ocio::Source::Builtin).unwrap();
    let bl = lib.get(brp_ocio::Source::Blender).unwrap();
    let t = std::time::Instant::now();
    let sh = brp_ocio::shader(b, "ACEScg", bl, brp_ocio::Target::Display { display: "sRGB", view: "AgX" }, false, 1, 1)?;
    println!("\nshader ACEScg -> blender sRGB/AgX in {:?}: {} chars, {} textures, {} uniforms", t.elapsed(), sh.glsl.len(), sh.textures.len(), sh.uniforms);
    for tx in &sh.textures {
        println!("  {} binding {} dims {} {}x{}x{} ch {} linear {}", tx.name, tx.binding, tx.dims, tx.w, tx.h, tx.d, tx.channels, tx.linear);
    }
    println!("{}", sh.glsl.lines().take(60).collect::<Vec<_>>().join("\n"));
    Ok(())
}
