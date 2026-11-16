// Grain: shaped noise added to the display picture. The `grain` step of our denoiser's
// training data (tools/denoise_train/gpu_degrade.py), as an effect: the same noise
// shapes and settings, so what is added here is what the denoiser learned to remove.
//
// Works on the display-encoded picture (what an SDR screen shows). New noise on every
// picture, the same on every replay and in exports (seeded by the picture number).
//
// @brp space display
// @brp en.title Grain
// @brp pt.title Granulação
// @brp pt.param Amount Quantidade | Força da granulação, em % do branco (desvio padrão). O treino usa de 0,4 a 6.
// @brp pt.param Size Tamanho | Tamanho do grão em pixels. Até 0,6 é ruído de um pixel; acima disso o ruído é suavizado.
// @brp pt.param Colour Cor | 0 = grão monocromático, 1 = grão independente em cada canal de cor.
// @brp pt.param Shape Forma | Gaussiano: ruído comum. Laplace: mais pontos fortes isolados, com a mesma força média.
// @brp pt.param Sparse Esparso | 0 = desligado. De 1 a 2: só ficam os pontos mais fortes (acima deste limiar), com o dobro da força.

//!PARAM Amount
//!DESC Strength of the grain, in % of white (standard deviation). Training uses 0.4 to 6.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 6.0
2.0

//!PARAM Size
//!DESC Grain size in pixels. Up to 0.6 it is one-pixel noise; above that the noise is softened.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 4.0
1.2

//!PARAM Colour
//!DESC 0 = monochrome grain, 1 = independent grain in each colour channel.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!PARAM Shape
//!DESC Gaussian: ordinary noise. Laplace: more isolated strong specks at the same average strength. (options: Gaussian|Laplace)
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM Sparse
//!DESC 0 = off. From 1 to 2: only the strongest specks stay (above this threshold), twice as strong.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.0

//!HOOK MAIN
//!BIND HOOKED
//!SAVE GRAIN_H
//!HEIGHT HOOKED.h 12 +
//!DESC grain: noise, softened along x

// The noise is a function of (pixel, picture number), so it has no edges: this pass
// makes it for 6 extra rows above and below the picture (the next pass softens along
// y and needs them). Row r of this texture is picture row r - 6.
#define PAD 6

// PCG hash, 4 values from 4 (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020).
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

// Uniform in (0, 1), never 0 or 1.
vec4 u01(uvec4 h) { return (vec4(h >> 8u) + 0.5) * (1.0 / 16777216.0); }

// Four independent unit-variance values for pixel p of this picture: rgb = the colour
// grain of each channel, a = the shared (monochrome) grain.
vec4 noise(ivec2 p) {
    uvec4 h = pcg4d(uvec4(uint(p.x), uint(p.y), uint(frame), 0x9E3779B9u));
    if (Shape == 1) {
        // Laplace: an exponential with a random sign, scaled to variance 1.
        uvec4 s = pcg4d(h ^ uvec4(0x85EBCA6Bu));
        return -log(u01(h)) * (vec4(s >> 31u) * 2.0 - 1.0) * 0.70710678;
    }
    // Gaussian (Box-Muller, two pairs).
    vec4 u = u01(h);
    vec2 r = sqrt(-2.0 * log(u.xz));
    vec2 a = 6.2831853 * u.yw;
    return vec4(r.x * cos(a.x), r.x * sin(a.x), r.y * cos(a.y), r.y * sin(a.y));
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy) - ivec2(0, PAD);
    if (Size <= 0.6)
        return noise(p);
    // Gaussian blur of sigma Size / 2, then back to variance 1 (white noise through
    // weights w has variance sum(w^2)).
    float s = Size * 0.5;
    int rad = min(max(1, int(ceil(3.0 * s))), PAD);
    vec4 acc = vec4(0.0);
    float w2 = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * noise(p + ivec2(i, 0));
        w2 += w * w;
    }
    return acc / sqrt(w2);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND GRAIN_H
//!DESC grain: softened along y, added to the picture

#define PAD 6

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy) + ivec2(0, PAD);
    vec4 n;
    if (Size <= 0.6) {
        n = texelFetch(GRAIN_H_raw, p, 0);
    } else {
        float s = Size * 0.5;
        int rad = min(max(1, int(ceil(3.0 * s))), PAD);
        n = vec4(0.0);
        float w2 = 0.0;
        for (int i = -rad; i <= rad; i++) {
            float w = exp(-float(i * i) / (2.0 * s * s));
            n += w * texelFetch(GRAIN_H_raw, p + ivec2(0, i), 0);
            w2 += w * w;
        }
        n /= sqrt(w2);
    }
    // Monochrome and per-channel grain mixed, keeping variance 1.
    vec3 g = Colour <= 0.0 ? vec3(n.a) : sqrt(1.0 - Colour) * n.a + sqrt(Colour) * n.rgb;
    // Sparse: only where the (red channel's) grain is strong, twice as strong.
    if (Sparse > 0.0)
        g *= abs(g.r) > Sparse ? 2.0 : 0.0;
    vec4 v = HOOKED_tex(HOOKED_pos);
    return vec4(v.rgb + Amount * 0.01 * g, v.a);
}
