// Builds the C++ shim over OpenColorIO (OCIO_DIR: an install made by
// tools/build_ocio.ps1) and links the library.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=OCIO_DIR");
    println!("cargo:rerun-if-changed=src/shim.cpp");
    let dir = PathBuf::from(std::env::var("OCIO_DIR").expect("OCIO_DIR must point to an OpenColorIO install (tools/build_ocio.ps1)"));
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/shim.cpp")
        .include(dir.join("include"))
        .flag_if_supported("/EHsc")
        .compile("brp_ocio_shim");
    println!("cargo:rustc-link-search=native={}", dir.join("lib").display());
    println!("cargo:rustc-link-lib=OpenColorIO");
}
