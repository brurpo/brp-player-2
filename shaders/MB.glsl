// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Desfoque de movimento
// @brp pt.param BlurAmount Obturador | Comprimento do desfoque como fração do movimento por quadro (0,5 = obturador de 180°).
// @brp pt.param Samples Amostras | Amostras ao longo do movimento. Mais é mais suave, porém mais lento.
// @brp pt.param ShowVectors Mostrar vetores | 1 mostra os vetores de movimento em vez do desfoque (depuração).
// Motion blur.
//
// 1. Motion: pyramidal Lucas-Kanade optical flow of the current frame against the
//    previous one (5 levels, 1/2 .. 1/32 size, iterative warping), so fast motion
//    is followed and flat or patterned areas don't get random vectors.
// 2. Blur: velocities are reduced to 16x16 tiles (max) and their 3x3
//    neighbourhood, so pixels near a fast object know they may be covered by its
//    blur. Each pixel then gathers samples along that dominant motion and keeps the
//    ones that really smear over it (McGuire et al. 2012 reconstruction filter,
//    with speed standing in for depth: faster = in front).
//
// Runs in mpv/libplacebo too (vo=gpu-next). BRP Player replays the previous frame
// after a seek, so a paused frame looks the same as during playback:
// @brp temporal_warmup 1

//!PARAM BlurAmount
//!DESC Shutter: blur length as a fraction of the motion per frame (0.5 = 180° shutter).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.5

//!PARAM Samples
//!DESC Samples along the motion. Higher is smoother but slower.
//!TYPE int
//!MINIMUM 4
//!MAXIMUM 48
16

//!PARAM ShowVectors
//!DESC 1 shows the motion vectors instead of the blur (debug).
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!TEXTURE MB_PREV
//!SIZE 4096 2304
//!FORMAT r16f
//!STORAGE

// ---- Luma pyramids -----------------------------------------------------------

//!HOOK MAIN
//!BIND HOOKED
//!SAVE MB_Y0
//!DESC Motion blur: luma
vec4 hook() {
    return vec4(dot(HOOKED_texOff(0).rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_Y0
//!SAVE MB_C1
//!WIDTH MB_Y0.w 2 /
//!HEIGHT MB_Y0.h 2 /
//!DESC Motion blur: current 1/2
vec4 hook() { return MB_Y0_tex(MB_Y0_pos); }

//!HOOK MAIN
//!BIND MB_C1
//!SAVE MB_C2
//!WIDTH MB_C1.w 2 /
//!HEIGHT MB_C1.h 2 /
//!DESC Motion blur: current 1/4
vec4 hook() { return MB_C1_tex(MB_C1_pos); }

//!HOOK MAIN
//!BIND MB_C2
//!SAVE MB_C3
//!WIDTH MB_C2.w 2 /
//!HEIGHT MB_C2.h 2 /
//!DESC Motion blur: current 1/8
vec4 hook() { return MB_C2_tex(MB_C2_pos); }

//!HOOK MAIN
//!BIND MB_C3
//!SAVE MB_C4
//!WIDTH MB_C3.w 2 /
//!HEIGHT MB_C3.h 2 /
//!DESC Motion blur: current 1/16
vec4 hook() { return MB_C3_tex(MB_C3_pos); }

//!HOOK MAIN
//!BIND MB_C4
//!SAVE MB_C5
//!WIDTH MB_C4.w 2 /
//!HEIGHT MB_C4.h 2 /
//!DESC Motion blur: current 1/32
vec4 hook() { return MB_C4_tex(MB_C4_pos); }

//!HOOK MAIN
//!BIND MB_Y0
//!BIND MB_PREV
//!SAVE MB_P1
//!WIDTH MB_Y0.w 2 /
//!HEIGHT MB_Y0.h 2 /
//!DESC Motion blur: previous 1/2
vec4 hook() {
    // 2x2 box from the full-size previous luma.
    ivec2 p = ivec2(MB_Y0_pos * MB_Y0_size * 0.5) * 2;
    ivec2 m = ivec2(MB_Y0_size) - 1;
    float s = imageLoad(MB_PREV, min(p, m)).x + imageLoad(MB_PREV, min(p + ivec2(1, 0), m)).x
            + imageLoad(MB_PREV, min(p + ivec2(0, 1), m)).x + imageLoad(MB_PREV, min(p + ivec2(1, 1), m)).x;
    return vec4(s * 0.25, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_P1
//!SAVE MB_P2
//!WIDTH MB_P1.w 2 /
//!HEIGHT MB_P1.h 2 /
//!DESC Motion blur: previous 1/4
vec4 hook() { return MB_P1_tex(MB_P1_pos); }

//!HOOK MAIN
//!BIND MB_P2
//!SAVE MB_P3
//!WIDTH MB_P2.w 2 /
//!HEIGHT MB_P2.h 2 /
//!DESC Motion blur: previous 1/8
vec4 hook() { return MB_P2_tex(MB_P2_pos); }

//!HOOK MAIN
//!BIND MB_P3
//!SAVE MB_P4
//!WIDTH MB_P3.w 2 /
//!HEIGHT MB_P3.h 2 /
//!DESC Motion blur: previous 1/16
vec4 hook() { return MB_P3_tex(MB_P3_pos); }

//!HOOK MAIN
//!BIND MB_P4
//!SAVE MB_P5
//!WIDTH MB_P4.w 2 /
//!HEIGHT MB_P4.h 2 /
//!DESC Motion blur: previous 1/32
vec4 hook() { return MB_P4_tex(MB_P4_pos); }

// ---- Motion: pyramidal Lucas-Kanade optical flow ------------------------------
// Flow textures hold d (in pixels of their level): prev(x + d) = cur(x), so the
// motion is -d.

//!HOOK MAIN
//!BIND MB_C5
//!BIND MB_P5
//!SAVE MB_F5a
//!DESC Motion blur: optical flow 1/32, iteration 1
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return vec2(0.0);
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C5_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C5_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C5_tex(x).x;
            float J = MB_P5_tex(y).x;
            float jx = 0.5 * (MB_P5_tex(y + vec2(pt.x, 0.0)).x - MB_P5_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P5_tex(y + vec2(0.0, pt.y)).x - MB_P5_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C5
//!BIND MB_P5
//!BIND MB_F5a
//!SAVE MB_F5b
//!DESC Motion blur: optical flow 1/32, iteration 2
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F5a_tex(MB_C5_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C5_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C5_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C5_tex(x).x;
            float J = MB_P5_tex(y).x;
            float jx = 0.5 * (MB_P5_tex(y + vec2(pt.x, 0.0)).x - MB_P5_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P5_tex(y + vec2(0.0, pt.y)).x - MB_P5_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C5
//!BIND MB_P5
//!BIND MB_F5b
//!SAVE MB_F5c
//!DESC Motion blur: optical flow 1/32, iteration 3
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F5b_tex(MB_C5_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C5_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C5_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C5_tex(x).x;
            float J = MB_P5_tex(y).x;
            float jx = 0.5 * (MB_P5_tex(y + vec2(pt.x, 0.0)).x - MB_P5_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P5_tex(y + vec2(0.0, pt.y)).x - MB_P5_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C5
//!BIND MB_P5
//!BIND MB_F5c
//!SAVE MB_F5
//!DESC Motion blur: optical flow 1/32, iteration 4
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F5c_tex(MB_C5_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C5_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C5_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C5_tex(x).x;
            float J = MB_P5_tex(y).x;
            float jx = 0.5 * (MB_P5_tex(y + vec2(pt.x, 0.0)).x - MB_P5_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P5_tex(y + vec2(0.0, pt.y)).x - MB_P5_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C4
//!BIND MB_P4
//!BIND MB_F5
//!SAVE MB_F4a
//!DESC Motion blur: optical flow 1/16, iteration 1
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F5_tex(MB_C4_pos).xy * 2.0; // coarser level, upsampled
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C4_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C4_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C4_tex(x).x;
            float J = MB_P4_tex(y).x;
            float jx = 0.5 * (MB_P4_tex(y + vec2(pt.x, 0.0)).x - MB_P4_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P4_tex(y + vec2(0.0, pt.y)).x - MB_P4_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C4
//!BIND MB_P4
//!BIND MB_F4a
//!SAVE MB_F4b
//!DESC Motion blur: optical flow 1/16, iteration 2
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F4a_tex(MB_C4_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C4_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C4_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C4_tex(x).x;
            float J = MB_P4_tex(y).x;
            float jx = 0.5 * (MB_P4_tex(y + vec2(pt.x, 0.0)).x - MB_P4_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P4_tex(y + vec2(0.0, pt.y)).x - MB_P4_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C4
//!BIND MB_P4
//!BIND MB_F4b
//!SAVE MB_F4
//!DESC Motion blur: optical flow 1/16, iteration 3
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F4b_tex(MB_C4_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C4_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C4_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C4_tex(x).x;
            float J = MB_P4_tex(y).x;
            float jx = 0.5 * (MB_P4_tex(y + vec2(pt.x, 0.0)).x - MB_P4_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P4_tex(y + vec2(0.0, pt.y)).x - MB_P4_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C3
//!BIND MB_P3
//!BIND MB_F4
//!SAVE MB_F3a
//!DESC Motion blur: optical flow 1/8, iteration 1
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F4_tex(MB_C3_pos).xy * 2.0; // coarser level, upsampled
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C3_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C3_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C3_tex(x).x;
            float J = MB_P3_tex(y).x;
            float jx = 0.5 * (MB_P3_tex(y + vec2(pt.x, 0.0)).x - MB_P3_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P3_tex(y + vec2(0.0, pt.y)).x - MB_P3_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C3
//!BIND MB_P3
//!BIND MB_F3a
//!SAVE MB_F3b
//!DESC Motion blur: optical flow 1/8, iteration 2
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F3a_tex(MB_C3_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C3_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C3_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C3_tex(x).x;
            float J = MB_P3_tex(y).x;
            float jx = 0.5 * (MB_P3_tex(y + vec2(pt.x, 0.0)).x - MB_P3_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P3_tex(y + vec2(0.0, pt.y)).x - MB_P3_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C3
//!BIND MB_P3
//!BIND MB_F3b
//!SAVE MB_F3
//!DESC Motion blur: optical flow 1/8, iteration 3
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F3b_tex(MB_C3_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C3_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C3_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C3_tex(x).x;
            float J = MB_P3_tex(y).x;
            float jx = 0.5 * (MB_P3_tex(y + vec2(pt.x, 0.0)).x - MB_P3_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P3_tex(y + vec2(0.0, pt.y)).x - MB_P3_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C2
//!BIND MB_P2
//!BIND MB_F3
//!SAVE MB_F2a
//!DESC Motion blur: optical flow 1/4, iteration 1
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F3_tex(MB_C2_pos).xy * 2.0; // coarser level, upsampled
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C2_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C2_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C2_tex(x).x;
            float J = MB_P2_tex(y).x;
            float jx = 0.5 * (MB_P2_tex(y + vec2(pt.x, 0.0)).x - MB_P2_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P2_tex(y + vec2(0.0, pt.y)).x - MB_P2_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C2
//!BIND MB_P2
//!BIND MB_F2a
//!SAVE MB_F2b
//!DESC Motion blur: optical flow 1/4, iteration 2
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F2a_tex(MB_C2_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C2_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C2_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C2_tex(x).x;
            float J = MB_P2_tex(y).x;
            float jx = 0.5 * (MB_P2_tex(y + vec2(pt.x, 0.0)).x - MB_P2_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P2_tex(y + vec2(0.0, pt.y)).x - MB_P2_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C2
//!BIND MB_P2
//!BIND MB_F2b
//!SAVE MB_F2
//!DESC Motion blur: optical flow 1/4, iteration 3
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F2b_tex(MB_C2_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C2_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C2_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C2_tex(x).x;
            float J = MB_P2_tex(y).x;
            float jx = 0.5 * (MB_P2_tex(y + vec2(pt.x, 0.0)).x - MB_P2_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P2_tex(y + vec2(0.0, pt.y)).x - MB_P2_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C1
//!BIND MB_P1
//!BIND MB_F2
//!SAVE MB_F1a
//!DESC Motion blur: optical flow 1/2, iteration 1
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F2_tex(MB_C1_pos).xy * 2.0; // coarser level, upsampled
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C1_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C1_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C1_tex(x).x;
            float J = MB_P1_tex(y).x;
            float jx = 0.5 * (MB_P1_tex(y + vec2(pt.x, 0.0)).x - MB_P1_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P1_tex(y + vec2(0.0, pt.y)).x - MB_P1_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C1
//!BIND MB_P1
//!BIND MB_F1a
//!SAVE MB_F1b
//!DESC Motion blur: optical flow 1/2, iteration 2
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F1a_tex(MB_C1_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C1_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C1_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C1_tex(x).x;
            float J = MB_P1_tex(y).x;
            float jx = 0.5 * (MB_P1_tex(y + vec2(pt.x, 0.0)).x - MB_P1_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P1_tex(y + vec2(0.0, pt.y)).x - MB_P1_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_C1
//!BIND MB_P1
//!BIND MB_F1b
//!SAVE MB_F1
//!DESC Motion blur: optical flow 1/2, iteration 3
// Lucas-Kanade step: find d with prev(x + d) = cur(x) over a Gaussian window,
// linearising prev around the current estimate (warp, then solve 2x2).
vec2 init_d() {
    return MB_F1b_tex(MB_C1_pos).xy;
}
vec4 hook() {
    vec2 d = init_d();
    vec2 pt = MB_C1_pt;
    float g11 = 0.0, g12 = 0.0, g22 = 0.0, b1 = 0.0, b2 = 0.0;
    for (int j = -3; j <= 3; j++)
        for (int i = -3; i <= 3; i++) {
            vec2 o = vec2(i, j);
            float w = exp(-dot(o, o) / 8.0);
            vec2 x = MB_C1_pos + o * pt;
            vec2 y = x + d * pt;
            float I = MB_C1_tex(x).x;
            float J = MB_P1_tex(y).x;
            float jx = 0.5 * (MB_P1_tex(y + vec2(pt.x, 0.0)).x - MB_P1_tex(y - vec2(pt.x, 0.0)).x);
            float jy = 0.5 * (MB_P1_tex(y + vec2(0.0, pt.y)).x - MB_P1_tex(y - vec2(0.0, pt.y)).x);
            float e = I - J;
            g11 += w * jx * jx; g12 += w * jx * jy; g22 += w * jy * jy;
            b1 += w * jx * e; b2 += w * jy * e;
        }
    // Regularised solve: where there is no texture the update fades to 0, so the
    // inherited (coarser) motion is kept.
    float reg = 1e-4;
    g11 += reg; g22 += reg;
    float det = g11 * g22 - g12 * g12;
    vec2 delta = vec2(g22 * b1 - g12 * b2, g11 * b2 - g12 * b1) / det;
    delta = clamp(delta, vec2(-1.0), vec2(1.0));
    return vec4(d + delta, 0.0, 1.0);
}

// ---- Blur ---------------------------------------------------------------------

//!HOOK MAIN
//!BIND MB_F1
//!SAVE MB_VEL
//!DESC Motion blur: smooth vectors (3x3 median)
// Blur vector per half-size pixel, in full-size pixels (already times the shutter).
vec2 at(ivec2 p) {
    return -texelFetch(MB_F1_raw, clamp(p, ivec2(0), ivec2(MB_F1_size) - 1), 0).xy; // motion = -d
}
vec4 hook() {
    ivec2 p = ivec2(MB_F1_pos * MB_F1_size);
    float xs[9], ys[9];
    for (int k = 0; k < 9; k++) {
        vec2 v = at(p + ivec2(k % 3 - 1, k / 3 - 1));
        xs[k] = v.x;
        ys[k] = v.y;
    }
    // Median of 9 per component (partial selection sort).
    for (int i = 0; i <= 4; i++)
        for (int j = i + 1; j < 9; j++) {
            if (xs[j] < xs[i]) { float t = xs[i]; xs[i] = xs[j]; xs[j] = t; }
            if (ys[j] < ys[i]) { float t = ys[i]; ys[i] = ys[j]; ys[j] = t; }
        }
    return vec4(vec2(xs[4], ys[4]) * 2.0 * BlurAmount, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_VEL
//!SAVE MB_TILE
//!WIDTH MB_VEL.w 8 /
//!HEIGHT MB_VEL.h 8 /
//!DESC Motion blur: tile max (16x16)
vec4 hook() {
    ivec2 base = ivec2(MB_VEL_pos * MB_VEL_size / 8.0) * 8;
    ivec2 m = ivec2(MB_VEL_size) - 1;
    vec2 best = vec2(0.0);
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++) {
            vec2 v = texelFetch(MB_VEL_raw, min(base + ivec2(i, j), m), 0).xy;
            if (dot(v, v) > dot(best, best)) best = v;
        }
    return vec4(best, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND MB_TILE
//!SAVE MB_NEIGH
//!DESC Motion blur: neighbour max
vec4 hook() {
    ivec2 p = ivec2(MB_TILE_pos * MB_TILE_size);
    ivec2 m = ivec2(MB_TILE_size) - 1;
    vec2 best = vec2(0.0);
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec2 v = texelFetch(MB_TILE_raw, clamp(p + ivec2(i, j), ivec2(0), m), 0).xy;
            if (dot(v, v) > dot(best, best)) best = v;
        }
    return vec4(best, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND MB_VEL
//!BIND MB_NEIGH
//!SAVE MB_OUT
//!DESC Motion blur: gather
float cone(float d, float v) { return clamp(1.0 - d / max(v, 1e-3), 0.0, 1.0); }
float cylinder(float d, float v) { return 1.0 - smoothstep(0.95 * v, 1.05 * v + 1e-3, d); }
// Soft "in front of": faster motion counts as nearer.
float front(float a, float b) { return clamp(1.0 + (a - b) * 0.5, 0.0, 1.0); }

vec4 hook() {
    vec4 src = HOOKED_texOff(0);
    vec2 vn = MB_NEIGH_tex(HOOKED_pos).xy;
    if (frame == 0 || dot(vn, vn) < 0.25)
        return ShowVectors == 1 ? vec4(0.5, 0.5, 0.5, 1.0) : src;
    vec2 vx = MB_VEL_tex(HOOKED_pos).xy;
    if (ShowVectors == 1)
        return vec4(clamp(vx / 64.0 * 0.5 + 0.5, 0.0, 1.0), 0.5, 1.0);

    float lx = length(vx);
    float ln = length(vn);
    // Per-pixel jitter hides the sampling pattern.
    float jitter = fract(sin(dot(HOOKED_pos * HOOKED_size, vec2(12.9898, 78.233)) + random) * 43758.5453) - 0.5;
    float wsum = 1.0 / max(lx, 1.0);
    vec3 sum = src.rgb * wsum;
    int n = Samples;
    for (int i = 0; i < n; i++) {
        if (i == (n - 1) / 2) continue;
        float t = mix(-0.5, 0.5, (float(i) + jitter * 0.95 + 0.5) / float(n));
        vec2 off = vn * t;
        float d = length(off);
        vec2 pos = HOOKED_pos + off * HOOKED_pt;
        vec2 vy = MB_VEL_tex(pos).xy;
        float ly = length(vy);
        float f = front(ly, lx);
        float b = front(lx, ly);
        float w = f * cone(d, ly * 0.5) + b * cone(d, lx * 0.5) + cylinder(d, ly * 0.5) * cylinder(d, lx * 0.5) * 2.0;
        sum += w * HOOKED_tex(pos).rgb;
        wsum += w;
    }
    return vec4(sum / wsum, src.a);
}

// ---- Keep this frame's luma for the next one ----------------------------------

//!HOOK MAIN
//!BIND MB_OUT
//!BIND MB_Y0
//!BIND MB_PREV
//!DESC Motion blur: remember this frame
vec4 hook() {
    ivec2 p = ivec2(MB_Y0_pos * MB_Y0_size);
    if (all(lessThan(p, imageSize(MB_PREV))))
        imageStore(MB_PREV, p, vec4(MB_Y0_texOff(0).x, 0.0, 0.0, 1.0));
    return MB_OUT_texOff(0);
}
