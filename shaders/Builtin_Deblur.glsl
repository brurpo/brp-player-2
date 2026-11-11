// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Remover desfoque
// @brp pt.param Iterations Iterações | Número de iterações da deconvolução. Mais deixa mais nítido, porém mais lento.
// @brp pt.param BlurRadius Raio do desfoque | Raio estimado do desfoque do vídeo, em pixels. O ajuste mais importante.
// @brp pt.param Regularization Regularização | Proteção contra ruído. Aumente se o grão for amplificado ou aparecerem halos.
// Richardson-Lucy deconvolution on luminance (linear light), Gaussian PSF.
// Colour is kept: the result scales each pixel by its sharpened/original luminance.

//!PARAM Iterations
//!DESC Number of deconvolution iterations. More is sharper but slower.
//!TYPE int
//!MINIMUM 1
//!MAXIMUM 10
5

//!PARAM BlurRadius
//!DESC Estimated blur radius of the footage, in pixels. The most important setting.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 10.0
2.0

//!PARAM Regularization
//!DESC Noise protection. Raise it if you see grain being amplified or ringing.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.1
0.005

//!HOOK MAIN
//!BIND HOOKED
//!SAVE RL_OBS
//!DESC Deblur: luminance

vec4 hook() {
    vec3 c = linearize(HOOKED_texOff(0)).rgb;
    return vec4(max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_OBS
//!SAVE RL_EST
//!DESC Deblur: initial estimate

vec4 hook() { return RL_OBS_texOff(0); }

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 0 >
//!DESC Deblur: iteration 1 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 0 >
//!DESC Deblur: iteration 1 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 0 >
//!DESC Deblur: iteration 1 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 0 >
//!DESC Deblur: iteration 1 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 1 >
//!DESC Deblur: iteration 2 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 1 >
//!DESC Deblur: iteration 2 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 1 >
//!DESC Deblur: iteration 2 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 1 >
//!DESC Deblur: iteration 2 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 2 >
//!DESC Deblur: iteration 3 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 2 >
//!DESC Deblur: iteration 3 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 2 >
//!DESC Deblur: iteration 3 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 2 >
//!DESC Deblur: iteration 3 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 3 >
//!DESC Deblur: iteration 4 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 3 >
//!DESC Deblur: iteration 4 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 3 >
//!DESC Deblur: iteration 4 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 3 >
//!DESC Deblur: iteration 4 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 4 >
//!DESC Deblur: iteration 5 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 4 >
//!DESC Deblur: iteration 5 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 4 >
//!DESC Deblur: iteration 5 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 4 >
//!DESC Deblur: iteration 5 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 5 >
//!DESC Deblur: iteration 6 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 5 >
//!DESC Deblur: iteration 6 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 5 >
//!DESC Deblur: iteration 6 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 5 >
//!DESC Deblur: iteration 6 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 6 >
//!DESC Deblur: iteration 7 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 6 >
//!DESC Deblur: iteration 7 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 6 >
//!DESC Deblur: iteration 7 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 6 >
//!DESC Deblur: iteration 7 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 7 >
//!DESC Deblur: iteration 8 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 7 >
//!DESC Deblur: iteration 8 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 7 >
//!DESC Deblur: iteration 8 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 7 >
//!DESC Deblur: iteration 8 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 8 >
//!DESC Deblur: iteration 9 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 8 >
//!DESC Deblur: iteration 9 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 8 >
//!DESC Deblur: iteration 9 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 8 >
//!DESC Deblur: iteration 9 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_EST
//!SAVE RL_T1
//!WHEN Iterations 9 >
//!DESC Deblur: iteration 10 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_EST_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T1
//!BIND RL_OBS
//!SAVE RL_RATIO
//!WHEN Iterations 9 >
//!DESC Deblur: iteration 10 (y blur, ratio)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T1_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Observed / re-blurred estimate. Regularization damps dark, noisy areas.
    float obs = RL_OBS_texOff(0).x;
    return vec4((obs + Regularization) / (b + Regularization), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_RATIO
//!SAVE RL_T2
//!WHEN Iterations 9 >
//!DESC Deblur: iteration 10 (x blur)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_RATIO_texOff(vec2(float(i), 0.0)).x;
        ws += w;
    }
    float b = acc / ws;
    return vec4(b, 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND RL_T2
//!BIND RL_EST
//!SAVE RL_EST
//!WHEN Iterations 9 >
//!DESC Deblur: iteration 10 (y blur, update)

float g(float x) { float s = max(BlurRadius * 0.5, 0.25); return exp(-0.5 * x * x / (s * s)); }
vec4 hook() {
    int r = int(ceil(BlurRadius * 1.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -r; i <= r; i++) {
        float w = g(float(i));
        acc += w * RL_T2_texOff(vec2(0.0, float(i))).x;
        ws += w;
    }
    float b = acc / ws;
    // Multiplicative update, limited per iteration to avoid ringing.
    return vec4(RL_EST_texOff(0).x * clamp(b, 0.5, 2.0), 0.0, 0.0, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND RL_OBS
//!BIND RL_EST
//!DESC Deblur: apply to colour

vec4 hook() {
    vec4 src = HOOKED_texOff(0);
    vec3 c = linearize(src).rgb;
    float y0 = RL_OBS_texOff(0).x;
    float y1 = RL_EST_texOff(0).x;
    // Gain on all channels keeps hue; the small offset keeps near-black stable.
    float gain = clamp((y1 + 0.002) / (y0 + 0.002), 0.0, 4.0);
    return vec4(delinearize(vec4(max(c * gain, 0.0), 1.0)).rgb, src.a);
}
