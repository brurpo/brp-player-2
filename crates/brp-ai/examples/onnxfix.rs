//! Prepares an ONNX model for pnnx: fixes the graph input sizes and turns an input
//! into a constant (pnnx traces dynamic models with guessed sizes and random
//! values, which breaks RobustVideoMatting). No dependencies: edits the protobuf.
//!   onnxfix in.onnx out.onnx "1,3,1080,1920;1,16,135,240;..." [downsample_ratio]
use std::fs;
fn varint(b: &[u8], p: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut s = 0;
    loop {
        let c = b[*p];
        *p += 1;
        v |= ((c & 127) as u64) << s;
        if c < 128 {
            return v;
        }
        s += 7;
    }
}
fn put_varint(o: &mut Vec<u8>, mut v: u64) {
    loop {
        let c = (v & 127) as u8;
        v >>= 7;
        if v == 0 {
            o.push(c);
            return;
        }
        o.push(c | 128);
    }
}
// fields: (field, wire, payload start, end, raw start)
fn fields(b: &[u8]) -> Vec<(u64, u64, usize, usize, usize)> {
    let mut p = 0;
    let mut out = vec![];
    while p < b.len() {
        let st = p;
        let k = varint(b, &mut p);
        let (f, w) = (k >> 3, k & 7);
        match w {
            0 => {
                let s = p;
                varint(b, &mut p);
                out.push((f, w, s, p, st));
            }
            1 => {
                out.push((f, w, p, p + 8, st));
                p += 8;
            }
            5 => {
                out.push((f, w, p, p + 4, st));
                p += 4;
            }
            2 => {
                let l = varint(b, &mut p) as usize;
                out.push((f, w, p, p + l, st));
                p += l;
            }
            _ => panic!("wire {w}"),
        }
    }
    out
}
fn rebuild(b: &[u8], f: &mut dyn FnMut(u64, &[u8]) -> Option<Vec<u8>>) -> Vec<u8> {
    let mut o = vec![];
    for (fi, w, s, e, st) in fields(b) {
        if w == 2 {
            if let Some(n) = f(fi, &b[s..e]) {
                put_varint(&mut o, (fi << 3) | 2);
                put_varint(&mut o, n.len() as u64);
                o.extend(n);
                continue;
            }
        }
        o.extend(&b[st..e]);
    }
    o
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let shapes: Vec<Vec<u64>> = a[3]
        .split(';')
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    let m = fs::read(&a[1]).unwrap();
    let mut idx = 0;
    let out = rebuild(&m, &mut |f, g| {
        if f != 7 {
            None
        } else {
            Some(rebuild(g, &mut |f, vi| {
                if f != 11 {
                    None
                } else {
                    let shape = shapes.get(idx).cloned();
                    idx += 1;
                    let shape = shape?;
                    Some(rebuild(vi, &mut |f, ty| {
                        if f != 2 {
                            None
                        } else {
                            Some(rebuild(ty, &mut |f, tt| {
                                if f != 1 {
                                    None
                                } else {
                                    Some(rebuild(tt, &mut |f, _sh| {
                                        if f != 2 {
                                            None
                                        } else {
                                            let mut o = vec![];
                                            for d in &shape {
                                                let mut dim = vec![];
                                                put_varint(&mut dim, 1 << 3);
                                                put_varint(&mut dim, *d);
                                                put_varint(&mut o, (1 << 3) | 2);
                                                put_varint(&mut o, dim.len() as u64);
                                                o.extend(dim);
                                            }
                                            Some(o)
                                        }
                                    }))
                                }
                            }))
                        }
                    }))
                }
            }))
        }
    });
    let out = bake(
        &out,
        std::env::args().nth(4).map(|v| v.parse::<f32>().unwrap()),
    );
    println!("rewrote {} inputs", idx.min(shapes.len()));
    fs::write(&a[2], out).unwrap();
}
fn bake(m: &[u8], ds: Option<f32>) -> Vec<u8> {
    let Some(ds) = ds else { return m.to_vec() };
    rebuild(m, &mut |f, g| {
        if f != 7 {
            None
        } else {
            let mut o = vec![];
            let mut found = false;
            for (fi, w, s, e, st) in fields(g) {
                if fi == 11
                    && w == 2
                    && fields(&g[s..e])
                        .iter()
                        .any(|x| x.0 == 1 && &g[s..e][x.2..x.3] == b"downsample_ratio")
                {
                    found = true;
                    continue;
                }
                o.extend(&g[st..e]);
            }
            assert!(found);
            let mut t = vec![];
            put_varint(&mut t, 1 << 3);
            put_varint(&mut t, 1);
            put_varint(&mut t, 2 << 3);
            put_varint(&mut t, 1);
            put_varint(&mut t, (8 << 3) | 2);
            put_varint(&mut t, 16);
            t.extend(b"downsample_ratio");
            put_varint(&mut t, (9 << 3) | 2);
            put_varint(&mut t, 4);
            t.extend(ds.to_le_bytes());
            put_varint(&mut o, (5 << 3) | 2);
            put_varint(&mut o, t.len() as u64);
            o.extend(t);
            Some(o)
        }
    })
}
