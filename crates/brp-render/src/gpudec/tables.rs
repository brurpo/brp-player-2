//! Bitstream tables of the codecs decoded on the GPU (values as in FFmpeg n7.1).

/// ProRes coefficient scan, progressive frames (position in the 8x8 block, raster).
pub const PRORES_PROGRESSIVE_SCAN: [u8; 64] = [
    0, 1, 8, 9, 2, 3, 10, 11, //
    16, 17, 24, 25, 18, 19, 26, 27, //
    4, 5, 12, 20, 13, 6, 7, 14, //
    21, 28, 29, 22, 15, 23, 30, 31, //
    32, 33, 40, 48, 41, 34, 35, 42, //
    49, 56, 57, 50, 43, 36, 37, 44, //
    51, 58, 59, 52, 45, 38, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// ProRes coefficient scan, interlaced frames.
pub const PRORES_INTERLACED_SCAN: [u8; 64] = [
    0, 8, 1, 9, 16, 24, 17, 25, //
    2, 10, 3, 11, 18, 26, 19, 27, //
    32, 40, 33, 34, 41, 48, 56, 49, //
    42, 35, 43, 50, 57, 58, 51, 59, //
    4, 12, 5, 6, 13, 20, 28, 21, //
    14, 7, 15, 22, 29, 36, 44, 37, //
    30, 23, 31, 38, 45, 52, 60, 53, //
    46, 39, 47, 54, 61, 62, 55, 63,
];

/// ProRes adaptive Rice / exp-Golomb codebooks: bits 0-1 switch, 2-4 exp-Golomb
/// order, 5-7 Rice order.
pub const PRORES_FIRST_DC_CB: u8 = 0xB8;
pub const PRORES_DC_CB: [u8; 7] = [0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70];
pub const PRORES_RUN_CB: [u8; 16] = [0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29, 0x29, 0x28, 0x28, 0x28, 0x28, 0x28, 0x28, 0x4C];
pub const PRORES_LEV_CB: [u8; 10] = [0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C];

/// A u8 table as a GLSL `const uint` array.
pub fn glsl_array(name: &str, t: &[u8]) -> String {
    let items: Vec<String> = t.iter().map(|v| format!("{v}u")).collect();
    format!("const uint {name}[{}] = uint[{}]({});\n", t.len(), t.len(), items.join(","))
}
