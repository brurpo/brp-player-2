// Film grain, the Boolean model (Newson, Delon, Galerne, "A Stochastic Film Grain Model
// for Resolution-Independent Rendering", 2017). The `film_grain` step of our denoiser's
// training data (tools/denoise_train/gpu_degrade.py), as an effect, with the same
// model and settings.
//
// Grains are opaque discs, far smaller than a pixel, dropped at random with a density
// that follows the picture, so that the covered part of a pixel is on average its
// value. Each pixel is rendered as 4 x 4 sub-pixels (covered or not) and averaged (the
// scan), then softened by the scan's blur. What is added is the difference to the
// equally blurred picture, so the picture keeps its sharpness. The grain is strongest
// where film's is and fades at black and white by itself. Colour film: one layer per
// dye (red, green, blue; blue's grains largest) mixed with a shared grain; black and
// white: one layer on luma. Negative film: the grains respond to light, so the model
// runs in linear light and on screen the grain is strongest in the shadows; otherwise
// (prints, reversal) on the display values, strongest in the mid-tones.
//
// Works on the display-encoded picture (what an SDR screen shows). New grains on every
// picture, the same on every replay and in exports (seeded by the picture number).
//
// @brp space display
// @brp en.title Film Grain
// @brp pt.title Grão de filme
// @brp pt.param Strength Intensidade | Quanto do grão do modelo é somado (1 = o grão inteiro do modelo). O treino usa de 0,4 a 1.
// @brp pt.param ScanBlur Desfoque do scan | Desfoque do scan, em pixels: define o tamanho dos grumos visíveis e suaviza o grão. O treino usa de 0,5 a 1,2.
// @brp pt.param Chroma Cor | Filme colorido: quanto do grão é próprio de cada camada de cor (0 = grão igual nas três, sem cor). O treino usa de 0,15 a 0,6.
// @brp pt.param BlackAndWhite Preto e branco | Uma só camada de grão, sobre a luminância (filme preto e branco).
// @brp pt.param Negative Negativo | Filme negativo (a maior parte do cinema): grão mais forte nas sombras. Desligado: cópia ou reversível, grão mais forte nos tons médios.
// @brp pt.param GrainRadius Raio do grão | Raio de cada grão, em pixels. Os grãos são muito menores que um pixel; só perto do máximo passam a cobrir pixels vizinhos e o grão fica mais grosso. O treino usa de 0,05 a 0,25.
// @brp pt.param RadiusSpread Variação do raio | Mistura grãos menores e maiores que o raio (0 = todos iguais). O treino usa de 0 a 0,4.

//!PARAM Strength
//!DESC How much of the model's grain is added (1 = the model's full grain). Training uses 0.4 to 1.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.6

//!PARAM ScanBlur
//!DESC The scan's blur, in pixels: sets the size of the visible clumps and softens the grain. Training uses 0.5 to 1.2.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 1.2
0.8

//!PARAM Chroma
//!DESC Colour film: how much of the grain is each dye layer's own (0 = the same grain in all three, no colour). Training uses 0.15 to 0.6.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.6
0.35

//!PARAM BlackAndWhite
//!DESC One grain layer only, on luma (black and white film).
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM Negative
//!DESC Negative film (most motion-picture film): grain strongest in the shadows. Off: print or reversal film, grain strongest in the mid-tones.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM GrainRadius
//!DESC Radius of each grain, in pixels. Grains are far smaller than a pixel; only near the maximum do they start to cover neighbouring sub-pixels and the grain gets coarser. Training uses 0.05 to 0.25.
//!TYPE float
//!MINIMUM 0.05
//!MAXIMUM 0.25
0.12

//!PARAM RadiusSpread
//!DESC Mixes grains smaller and larger than the radius (0 = all the same). Training uses 0 to 0.4.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.4
0.2

//!HOOK MAIN
//!BIND HOOKED
//!SAVE FILMGRAIN_D
//!DESC film grain: grains rendered at 4x, minus the picture

#define SS 4        // sub-pixels per pixel, each way
#define MARGIN 2    // sub-pixels around the pixel a grain can reach from (radius < 3)
#define WIN (SS + 2 * MARGIN)

// PCG hash, 4 values from 4 (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020).
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

// What the grains follow: the picture's r, g, b and luma (in a), 0..1; linear light
// for negative film.
vec4 layers(ivec2 q) {
    vec3 x = clamp(texelFetch(HOOKED_raw, q, 0).rgb, 0.0, 1.0);
    if (Negative != 0)
        x = mix(x / 12.92, pow((x + 0.055) / 1.055, vec3(2.4)), step(0.04045, x));
    return vec4(x, dot(x, vec3(0.2126, 0.7152, 0.0722)));
}

// Sub-pixels a disc of radius r covers (those whose centre is within max(r, 0.5)).
float disc_area(float r) {
    float t = max(r, 0.5);
    t *= t;
    return 1.0 + 4.0 * step(1.0, t) + 4.0 * step(2.0, t) + 4.0 * step(4.0, t) + 8.0 * step(5.0, t) + 4.0 * step(8.0, t);
}

// Bits of `row` spread to their neighbours within h.
uint widen(uint row, int h) {
    uint o = row;
    for (int i = 1; i <= h; i++)
        o |= (row << uint(i)) | (row >> uint(i));
    return o;
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(HOOKED_size);

    // The pixel and its 8 neighbours (edges repeat).
    vec4 tex[9];
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            tex[j * 3 + i] = layers(clamp(p + ivec2(i - 1, j - 1), ivec2(0), size - 1));

    // Grain radius in sub-pixels, per layer (r, g, b dye layers, luma) and per size
    // (one size, or a smaller and a larger one).
    int sizes = RadiusSpread <= 0.02 ? 1 : 2;
    vec4 scale = vec4(1.0, 0.9, 1.3, 1.0) * (GrainRadius * float(SS));
    vec4 rad[2];
    rad[0] = min(sizes == 1 ? scale : scale * (1.0 - RadiusSpread), 2.99);
    rad[1] = min(scale * (1.0 + RadiusSpread), 2.99);
    vec4 inv_area[2];
    float reach = 0.0;
    for (int k = 0; k < sizes; k++) {
        inv_area[k] = 1.0 / (float(sizes) * vec4(disc_area(rad[k].x), disc_area(rad[k].y), disc_area(rad[k].z), disc_area(rad[k].w)));
        reach = max(reach, max(max(rad[k].x, rad[k].y), max(rad[k].z, rad[k].w)));
    }
    int m = int(floor(max(reach, 0.5)));  // sub-pixels around the pixel that matter

    // Where grains fall: a Poisson process of density -ln(1 - u) / (disc area) per
    // sub-pixel, so one holds at least one grain centre with chance
    // 1 - (1 - u)^(1 / area). One bit per sub-pixel, a row of the window per uint.
    uint centres[2][4][WIN];
    for (int k = 0; k < 2; k++)
        for (int l = 0; l < 4; l++)
            for (int y = 0; y < WIN; y++)
                centres[k][l][y] = 0u;
    for (int y = MARGIN - m; y < MARGIN + SS + m; y++) {
        for (int x = MARGIN - m; x < MARGIN + SS + m; x++) {
            // This sub-pixel in the whole picture's 4x grid; none fall outside it.
            ivec2 sub = p * SS + ivec2(x, y) - MARGIN;
            if (sub.x < 0 || sub.y < 0 || sub.x >= size.x * SS || sub.y >= size.y * SS)
                continue;
            // The picture there (bilinear, 4x up).
            vec2 src = clamp((vec2(sub) + 0.5) / float(SS) - 0.5, vec2(0.0), vec2(size - 1));
            ivec2 i0 = ivec2(floor(src));
            ivec2 i1 = min(i0 + 1, size - 1);
            vec2 f = src - vec2(i0);
            ivec2 a = i0 - p + 1, b = i1 - p + 1;
            vec4 u = mix(mix(tex[a.y * 3 + a.x], tex[a.y * 3 + b.x], f.x), mix(tex[b.y * 3 + a.x], tex[b.y * 3 + b.x], f.x), f.y);
            u = clamp(u, 0.0, 0.995);
            for (int k = 0; k < sizes; k++) {
                vec4 chance = 1.0 - pow(1.0 - u, inv_area[k]);
                uvec4 h = pcg4d(uvec4(uint(sub.x), uint(sub.y), uint(frame), 0x2545F491u + uint(k)));
                vec4 r = (vec4(h >> 8u) + 0.5) * (1.0 / 16777216.0);
                for (int l = 0; l < 4; l++)
                    if (r[l] < chance[l])
                        centres[k][l][y] |= 1u << uint(x);
            }
        }
    }

    // A sub-pixel is covered when a grain centre lies within the disc around it.
    vec4 covered = vec4(0.0);
    for (int l = 0; l < 4; l++) {
        for (int y = MARGIN; y < MARGIN + SS; y++) {
            uint row = 0u;
            for (int k = 0; k < sizes; k++) {
                float t = max(rad[k][l], 0.5);
                int reach_y = int(floor(t));
                for (int dy = -reach_y; dy <= reach_y; dy++)
                    row |= widen(centres[k][l][y + dy], int(floor(sqrt(t * t - float(dy * dy)))));
            }
            covered[l] += float(bitCount((row >> uint(MARGIN)) & 15u));
        }
    }
    // The covered part of the pixel, minus the picture: the grain before the scan blur.
    return covered / float(SS * SS) - tex[4];
}

//!HOOK MAIN
//!BIND FILMGRAIN_D
//!SAVE FILMGRAIN_T
//!DESC film grain: scan blur along x

// Edges mirror (as the training's blur does).
int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(FILMGRAIN_D_size.x);
    float s = ScanBlur;
    int rad = max(1, int(ceil(3.0 * s)));
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * texelFetch(FILMGRAIN_D_raw, ivec2(mirror(p.x + i, n), p.y), 0);
        ws += w;
    }
    return acc / ws;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND FILMGRAIN_T
//!DESC film grain: scan blur along y, added to the picture

int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(FILMGRAIN_T_size.y);
    float s = ScanBlur;
    int rad = max(1, int(ceil(3.0 * s)));
    vec4 g = vec4(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        g += w * texelFetch(FILMGRAIN_T_raw, ivec2(p.x, mirror(p.y + i, n)), 0);
        ws += w;
    }
    g /= ws;

    vec4 v = texelFetch(HOOKED_raw, p, 0);
    vec3 x = clamp(v.rgb, 0.0, 1.0);
    if (Negative != 0)
        x = mix(x / 12.92, pow((x + 0.055) / 1.055, vec3(2.4)), step(0.04045, x));
    // Black and white: the luma layer's grain on every channel. Colour: each dye
    // layer's own grain mixed with the luma layer's.
    vec3 o = x + Strength * (BlackAndWhite != 0 ? vec3(g.a) : mix(vec3(g.a), g.rgb, Chroma));
    if (Negative != 0) {
        o = max(o, 0.0);
        o = mix(12.92 * o, 1.055 * pow(o, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, o));
    }
    // Values outside 0..1 (HDR, overshoots) keep their own offset.
    return vec4(o + (v.rgb - clamp(v.rgb, 0.0, 1.0)), v.a);
}
