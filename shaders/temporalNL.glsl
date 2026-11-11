// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Redução de ruído NL temporal
// @brp pt.param Strength Intensidade | Força da redução de ruído. Aumente para vídeos mais ruidosos; alto demais borra detalhes finos.
// @brp pt.param Radius Raio | Raio de busca em pixels. Maior encontra mais correspondências e segue movimentos mais rápidos, porém é mais lento.
// @brp pt.param Temporal Temporal | Quanto o quadro anterior contribui (0 = só espacial).
// Temporal non-local means denoiser.
//
// Each pixel is averaged with similar-looking pixels (3x3 patch comparison) from a
// search window in the current frame and in the previous denoised frame. Matching
// against the previous output follows motion within the search radius; areas that
// don't match (fast motion, cuts) fall back to spatial denoising, so there is no
// ghosting. Recursive, so noise keeps dropping over consecutive frames.
//
// Runs in mpv/libplacebo too (vo=gpu-next). BRP Player replays the frames before a
// seek so a paused frame looks the same as during playback:
// @brp temporal_warmup 4

//!PARAM Strength
//!DESC Denoising strength. Raise for noisier footage; too high smears fine detail.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
2.0

//!PARAM Radius
//!DESC Search radius in pixels. Larger finds more matches and follows faster motion, but is slower.
//!TYPE int
//!MINIMUM 1
//!MAXIMUM 6
3

//!PARAM Temporal
//!DESC How much the previous frame contributes (0 = spatial only).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.85

//!HOOK MAIN
//!BIND HOOKED
//!BIND TNL_PREV
//!SAVE TNL_OUT
//!DESC Temporal NL-means: denoise

ivec2 tnl_clampi(ivec2 p) { return clamp(p, ivec2(0), ivec2(HOOKED_size) - 1); }
vec3 cur(ivec2 p) { return texelFetch(HOOKED_raw, tnl_clampi(p), 0).rgb; }
vec3 prev(ivec2 p) { return imageLoad(TNL_PREV, tnl_clampi(p)).rgb; }

vec4 hook() {
    ivec2 c = ivec2(HOOKED_pos * HOOKED_size);
    vec4 centre = texelFetch(HOOKED_raw, c, 0);

    // Patch around the pixel being denoised.
    vec3 ref_patch[9];
    for (int j = 0; j < 9; j++)
        ref_patch[j] = cur(c + ivec2(j % 3 - 1, j / 3 - 1));

    // Patch distance -> weight. h is the expected noise level (0..1 signal units).
    float h = max(Strength * 0.01, 1e-4);
    float inv_h2 = 1.0 / (h * h);
    bool history = frame > 0 && Temporal > 0.0;

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int dy = -Radius; dy <= Radius; dy++) {
        for (int dx = -Radius; dx <= Radius; dx++) {
            ivec2 q = c + ivec2(dx, dy);
            // Current frame.
            float d = 0.0;
            for (int j = 0; j < 9; j++) {
                vec3 e = ref_patch[j] - cur(q + ivec2(j % 3 - 1, j / 3 - 1));
                d += dot(e, e);
            }
            float w = exp(-d / 27.0 * inv_h2);
            sum += w * cur(q);
            wsum += w;
            // Previous denoised frame: the patch noise is lower there, so the
            // distance is mostly the current frame's noise; allow a bit more.
            if (history) {
                float dp = 0.0;
                for (int j = 0; j < 9; j++) {
                    vec3 e = ref_patch[j] - prev(q + ivec2(j % 3 - 1, j / 3 - 1));
                    dp += dot(e, e);
                }
                float wp = Temporal * 2.0 * exp(-dp / 27.0 * inv_h2 * 0.75);
                sum += wp * prev(q);
                wsum += wp;
            }
        }
    }
    return vec4(sum / max(wsum, 1e-6), centre.a);
}

//!HOOK MAIN
//!BIND TNL_OUT
//!BIND TNL_PREV
//!DESC Temporal NL-means: keep for the next frame

vec4 hook() {
    vec4 v = TNL_OUT_texOff(0);
    ivec2 p = ivec2(TNL_OUT_pos * TNL_OUT_size);
    if (all(lessThan(p, imageSize(TNL_PREV))))
        imageStore(TNL_PREV, p, v);
    return v;
}

//!TEXTURE TNL_PREV
//!SIZE 4096 2304
//!FORMAT rgba16f
//!STORAGE
