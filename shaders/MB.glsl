// Motion blur: block-matching motion estimation against the previous frame, then a
// blur along each pixel's motion vector. Rewritten from the original MB.glsl, which
// could not compile (layout qualifier on a function parameter, image format
// mismatch) and read and overwrote its history in the same pass.
//
// BRP Player replays one frame after a seek (TEMPORAL_WARMUP), so the history is
// always the real previous frame.

//!TEMPORAL_WARMUP 1

//!PARAM BlurAmount
//!DESC Length of the blur along the motion (in multiples of the motion per frame).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.5

//!PARAM Samples
//!DESC Samples along the motion path. Higher is smoother but slower.
//!TYPE int
//!MINIMUM 2
//!MAXIMUM 32
12

//!PARAM SearchRadius
//!DESC How far (in pixels) to look for motion. Higher finds faster motion but is slower.
//!TYPE int
//!MINIMUM 2
//!MAXIMUM 24
8

//!PARAM ShowVectors
//!DESC 1 shows the motion vectors instead of the blur (debug).
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!TEXTURE LastFrameTex
//!SIZE 3840 2160
//!FORMAT rgba16f
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!SAVE MB_SRC
//!DESC Motion blur: keep the current frame
vec4 hook() {
    return HOOKED_texOff(vec2(0.0));
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND LastFrameTex
//!DESC Motion blur: estimate motion and blur
float block_sad(ivec2 p, ivec2 d) {
    float s = 0.0;
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            vec3 c = texelFetch(HOOKED_raw, p + ivec2(i, j) * 2, 0).rgb;
            vec3 q = imageLoad(LastFrameTex, p + d + ivec2(i, j) * 2).rgb;
            vec3 a = abs(c - q);
            s += a.r + a.g + a.b;
        }
    return s;
}

vec4 hook() {
    ivec2 size = ivec2(HOOKED_size);
    ivec2 p = ivec2(HOOKED_pos * HOOKED_size);
    // Coarse search (step 2), then refine around the best match.
    float best = block_sad(p, ivec2(0)) - 1e-3; // prefer "no motion" on ties
    ivec2 mv = ivec2(0);
    for (int y = -SearchRadius; y <= SearchRadius; y += 2)
        for (int x = -SearchRadius; x <= SearchRadius; x += 2) {
            float s = block_sad(p, ivec2(x, y));
            if (s < best) { best = s; mv = ivec2(x, y); }
        }
    ivec2 coarse = mv;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++) {
            float s = block_sad(p, coarse + ivec2(x, y));
            if (s < best) { best = s; mv = coarse + ivec2(x, y); }
        }
    // mv points from the current pixel to where it was in the previous frame.
    vec2 motion = -vec2(mv);
    if (ShowVectors == 1)
        return vec4(motion / float(SearchRadius) * 0.5 + 0.5, 0.5, 1.0);

    vec2 step = motion * BlurAmount / float(Samples);
    vec4 sum = vec4(0.0);
    for (int i = 0; i < Samples; i++) {
        float t = float(i) - float(Samples - 1) * 0.5;
        sum += HOOKED_tex(HOOKED_pos + step * t * HOOKED_pt);
    }
    return sum / float(Samples);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND MB_SRC
//!BIND LastFrameTex
//!DESC Motion blur: remember this frame
vec4 hook() {
    ivec2 p = ivec2(HOOKED_pos * HOOKED_size);
    imageStore(LastFrameTex, p, MB_SRC_texOff(vec2(0.0)));
    return HOOKED_texOff(vec2(0.0));
}
