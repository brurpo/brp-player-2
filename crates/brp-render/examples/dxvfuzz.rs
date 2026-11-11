//! DXV GPU decoder vs FFmpeg on generated packets: DXT1 and DXT5 (the alpha
//! variant, which FFmpeg's encoder cannot write), LZ-packed and raw.
//!
//!   cargo run -p brp-render --release --example dxvfuzz -- [count] [WxH]
//!
//! Packets are made by walking the decoder's own state machine with random choices
//! (ops, back references, runs, literal blocks), so they are valid streams that use
//! every op; the texture blocks are random (every DXT colour and alpha mode).

use brp_media::gpudec::{self, codec, StreamDesc};
use brp_render::{GpuContext, GpuInstance};
use ffmpeg_sys_next as ff;
use std::sync::Arc;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

/// Builds a stream: bytes, plus the 2-bit op words patched in where the decoder
/// reads them.
struct Gen {
    out: Vec<u8>,
    op_at: usize,
    ops_left: u32,
}

impl Gen {
    fn op(&mut self, op: u32) {
        if self.ops_left == 0 {
            self.op_at = self.out.len();
            self.out.extend([0; 4]);
            self.ops_left = 16;
        }
        let shift = (16 - self.ops_left) * 2;
        let mut w = u32::from_le_bytes(self.out[self.op_at..self.op_at + 4].try_into().unwrap());
        w |= op << shift;
        self.out[self.op_at..self.op_at + 4].copy_from_slice(&w.to_le_bytes());
        self.ops_left -= 1;
    }
    fn u8(&mut self, v: u32) {
        self.out.push(v as u8);
    }
    fn u16(&mut self, v: u32) {
        self.out.extend((v as u16).to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.out.extend(v.to_le_bytes());
    }
}

/// Ops the generator may use (env `name`, e.g. "03"; default all).
fn allowed(name: &str) -> Vec<u32> {
    std::env::var(name).map_or(vec![0, 1, 2, 3], |s| s.chars().filter_map(|c| c.to_digit(10)).collect())
}

/// CHECKPOINT(x) with a random op valid at `pos`. Returns (op, idx).
fn checkpoint(g: &mut Gen, r: &mut Rng, x: usize, pos: usize) -> (u32, usize) {
    let allowed = allowed("DXV_CP");
    loop {
        let op = allowed[r.below(allowed.len() as u32) as usize];
        match op {
            0 => {
                g.op(0);
                return (0, 0);
            }
            1 if x <= pos => {
                g.op(1);
                return (1, x);
            }
            2 if 2 * x <= pos => {
                let b = r.below((pos / x - 2).min(255) as u32 + 1);
                g.op(2);
                g.u8(b);
                return (2, (b as usize + 2) * x);
            }
            3 if 0x102 * x <= pos => {
                let b = r.below((pos / x - 0x102).min(0xFFFF) as u32 + 1);
                g.op(3);
                g.u16(b);
                return (3, (b as usize + 0x102) * x);
            }
            _ => {}
        }
    }
}

fn gen_dxt1(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut g = Gen { out: Vec::new(), op_at: 0, ops_left: 0 };
    g.u32(r.next());
    g.u32(r.next());
    let mut pos = 2;
    while pos + 2 <= n {
        let (op, _) = checkpoint(&mut g, r, 2, pos);
        if op != 0 {
            pos += 2;
        } else {
            for _ in 0..2 {
                let (op, _) = checkpoint(&mut g, r, 2, pos);
                if op == 0 {
                    g.u32(r.next());
                }
                pos += 1;
            }
        }
    }
    g.out.extend([0; 8]);
    g.out
}

fn gen_dxt5(r: &mut Rng, n: usize) -> Vec<u8> {
    let mut g = Gen { out: Vec::new(), op_at: 0, ops_left: 0 };
    for _ in 0..4 {
        g.u32(r.next());
    }
    let (mut pos, mut run) = (4usize, 0u32);
    while pos + 2 <= n {
        if run > 0 {
            run -= 1;
            pos += 2;
        } else {
            let main = allowed("DXV_MAIN");
            let op = main[r.below(main.len() as u32) as usize];
            match op {
                0 => {
                    g.op(0);
                    let check = if r.below(8) == 0 {
                        g.u8(255);
                        let e = r.below(300);
                        g.u16(e);
                        256 + e
                    } else {
                        let c = r.below(4) + 1;
                        g.u8(c - 1);
                        c
                    };
                    let mut c = check;
                    while c > 0 && pos + 4 <= n {
                        pos += 4;
                        c -= 1;
                    }
                    continue;
                }
                1 => {
                    g.op(1);
                    // Sometimes the long form: 255, then 16-bit extensions (0xFFFF continues).
                    if r.below(8) == 0 {
                        g.u8(255);
                        run = 255;
                        if r.below(4) == 0 {
                            g.u16(0xFFFF);
                            run += 0xFFFF;
                        }
                        let e = r.below(300);
                        g.u16(e);
                        run += e;
                    } else {
                        run = r.below(4);
                        g.u8(run);
                    }
                    pos += 2;
                }
                2 if pos >= 8 => {
                    g.op(2);
                    let i = r.below(((pos - 8) / 4).min(0xFFFF) as u32 + 1);
                    g.u16(i);
                    pos += 2;
                }
                _ => {
                    g.op(3);
                    g.u32(r.next());
                    g.u32(r.next());
                    pos += 2;
                }
            }
        }
        if pos + 2 > n {
            break;
        }
        let (op, _) = checkpoint(&mut g, r, 4, pos);
        if op != 0 {
            pos += 2;
        } else {
            for _ in 0..2 {
                let (op, _) = checkpoint(&mut g, r, 4, pos);
                if op == 0 {
                    g.u32(r.next());
                }
                pos += 1;
            }
        }
    }
    g.out.extend([0; 8]);
    g.out
}

fn packet(tag: &[u8; 4], raw: bool, body: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend(tag.iter().rev()); // stored little-endian: "1TXD"
    p.extend([4, 0, raw as u8, 0]);
    p.extend((body.len() as u32).to_le_bytes());
    p.extend(body);
    p
}

/// FFmpeg's decode of one packet: RGBA rows (w*4 bytes each), or an error.
fn ffmpeg_decode(pkt: &[u8], w: u32, h: u32) -> Result<Vec<u8>, String> {
    unsafe {
        let codec = ff::avcodec_find_decoder(ff::AVCodecID::AV_CODEC_ID_DXV);
        let mut ctx = ff::avcodec_alloc_context3(codec);
        (*ctx).width = w as i32;
        (*ctx).height = h as i32;
        (*ctx).thread_count = 1;
        assert!(ff::avcodec_open2(ctx, codec, std::ptr::null_mut()) >= 0);
        let mut p = ff::av_packet_alloc();
        ff::av_new_packet(p, pkt.len() as i32);
        std::ptr::copy_nonoverlapping(pkt.as_ptr(), (*p).data, pkt.len());
        let mut f = ff::av_frame_alloc();
        let mut r = ff::avcodec_send_packet(ctx, p);
        if r >= 0 {
            r = ff::avcodec_receive_frame(ctx, f);
        }
        let out = if r < 0 {
            Err(format!("error {r}"))
        } else {
            let mut v = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h as usize {
                v.extend_from_slice(std::slice::from_raw_parts((*f).data[0].add(y * (*f).linesize[0] as usize), w as usize * 4));
            }
            Ok(v)
        };
        ff::av_frame_free(&mut f);
        ff::av_packet_free(&mut p);
        ff::avcodec_free_context(&mut ctx);
        out
    }
}

fn main() -> anyhow::Result<()> {
    unsafe { ff::av_log_set_level(ff::AV_LOG_QUIET) };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let count: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
    let (w, h) = args.get(1).and_then(|s| s.split_once('x')).map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap_or((1920u32, 1080u32));
    let ctx = Arc::new(GpuContext::headless(GpuInstance::new()?, std::env::var("BRP_GPU").ok().as_deref())?);
    println!("GPU: {}, {w}x{h}, {count} packets per kind", ctx.device_name);
    let factory = brp_render::gpudec::factory(&ctx);
    let desc = StreamDesc { codec_id: codec::DXV, codec_tag: 0, width: w, height: h, extradata: Vec::new() };
    let mut dec = factory.open(&desc).expect("DXV decoder");
    let blocks = (w.div_ceil(4) * h.div_ceil(4)) as usize;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut bad = 0;
    for kind in ["DXT1 lz", "DXT5 lz", "DXT1 raw", "DXT5 raw"] {
        let (mut same, mut differ, mut both_err) = (0, 0, 0);
        for _ in 0..count {
            let dxt5 = kind.starts_with("DXT5");
            let words = blocks * if dxt5 { 4 } else { 2 };
            let pkt = if kind.ends_with("raw") {
                let body: Vec<u8> = (0..words).flat_map(|_| rng.next().to_le_bytes()).collect();
                packet(if dxt5 { b"DXT5" } else { b"DXT1" }, true, &body)
            } else if dxt5 {
                packet(b"DXT5", false, &gen_dxt5(&mut rng, words))
            } else {
                packet(b"DXT1", false, &gen_dxt1(&mut rng, words))
            };
            let theirs = ffmpeg_decode(&pkt, w, h);
            let ours = dec.decode(&pkt);
            match (theirs, ours) {
                (Ok(a), Ok(pic)) => {
                    let bytes = pic.planes.download()?;
                    let (off, stride) = pic.planes.plane_offsets()[0];
                    let mut n = 0usize;
                    for y in 0..h as usize {
                        let row = &bytes[off as usize + y * stride as usize..][..w as usize * 4];
                        n += row.iter().zip(&a[y * w as usize * 4..][..w as usize * 4]).filter(|(x, y)| x != y).count();
                    }
                    if n > 0 && differ == 0 {
                        // First differing pixels: (x, y, ours, FFmpeg).
                        let mut shown = 0;
                        'o: for y in 0..h as usize {
                            for x in 0..w as usize {
                                let o = &bytes[off as usize + y * stride as usize + x * 4..][..4];
                                let t = &a[(y * w as usize + x) * 4..][..4];
                                if o != t {
                                    println!("    ({x},{y}) ours {o:?} FFmpeg {t:?}");
                                    shown += 1;
                                    if shown == 6 { break 'o; }
                                }
                            }
                        }
                    }
                    if n == 0 { same += 1 } else { differ += 1 }
                }
                (Err(_), Err(_)) => both_err += 1,
                (a, b) => {
                    println!("  {kind}: FFmpeg {:?}, GPU {:?}", a.err(), b.err().map(|e| e.to_string()));
                    differ += 1;
                }
            }
        }
        println!("{kind}: {same} identical, {differ} different, {both_err} refused by both");
        bad += differ;
    }
    let _ = gpudec::factory;
    println!("{}", if bad == 0 { "ALL EXACT" } else { "DIFFERENCES" });
    Ok(())
}
