// Sensor noise: a camera's noise as it comes out of the camera's processing. The
// `sensor` step of our denoiser's training data (tools/denoise_train/gpu_degrade.py),
// as an effect, with the same model and settings:
//   - per raw channel, shot noise (grows with the light) plus read noise (constant),
//     where raw = the picture divided by the white balance gains;
//   - times the white balance gains (red and blue come out noisier);
//   - mixed by the camera's colour matrix (colour noise along magenta-teal) and
//     the picture style's saturation;
//   - softened by demosaicing; optionally the camera's chroma noise reduction;
//   - optionally partly kept from one picture to the next (temporal noise reduction).
//
// Works in linear light (the LINEAR stage: linear, the file's own primaries, 1.0 = SDR
// white). HDR and linear (EXR) pictures get the noise added to their light as it is.
// SDR video is taken back to its code values and decoded with the sRGB curve first,
// as the training does: a plain 2.4 power is far steeper near black than any camera
// curve and would blow the read noise up in the shadows. Below black is clipped for
// SDR (as a camera file is), kept for HDR and EXR.
//
// New noise on every picture, the same on every replay and in exports (seeded by the
// picture number).
//
// @brp en.title Sensor Noise
// @brp pt.title Ruído de sensor
// @brp pt.param ShotNoise Ruído de fótons | Ruído que cresce com a luz (ganho do sensor × 1000). ISO alto = valor alto. O treino usa de 0,2 a 10.
// @brp pt.param ReadNoise Ruído de leitura | Ruído constante, visível nas sombras (desvio padrão × 1000, em luz linear). O treino usa de 0,1 a 8.
// @brp pt.param Saturation Saturação | Quanto de cor o ruído tem: 0 = só luminância, 1 = como sai da matriz, mais = estilo de imagem saturado. O treino usa de 1 a 2,2.
// @brp pt.param ColourMatrix Matriz de cor | Força da matriz de cor da câmera, que espalha o ruído do vermelho e do azul em ruído de cor (magenta-verde azulado). O treino usa de 0,3 a 1,5.
// @brp pt.param RedGain Ganho do vermelho | Ganho de balanço de branco do vermelho: quanto maior, mais ruído no vermelho. O treino usa de 1 a 3.
// @brp pt.param BlueGain Ganho do azul | Ganho de balanço de branco do azul: quanto maior, mais ruído no azul. O treino usa de 1 a 3.
// @brp pt.param Softness Suavidade | Suavização do ruído pelo demosaico, em pixels (0 = ruído de um pixel). O treino usa de 0,2 a 0,9.
// @brp pt.param ChromaNR Redução de ruído de cor | A redução de ruído de cor da câmera: desfoca a cor do ruído e deixa manchas coloridas largas.
// @brp pt.param ChromaNRSize Tamanho da redução | Tamanho do desfoque da cor do ruído, em pixels. O treino usa de 1,5 a 4.
// @brp pt.param Persistence Persistência | Quanto do ruído da imagem anterior fica (redução de ruído temporal da câmera). O treino usa de 0 a 0,3.

//!PARAM ShotNoise
//!DESC Noise that grows with the light (sensor gain x 1000). High ISO = high value. Training uses 0.2 to 10. (cat: Noise)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
0.5

//!PARAM ReadNoise
//!DESC Constant noise, seen in the shadows (standard deviation x 1000, in linear light). Training uses 0.1 to 8.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 8.0
0.3

//!PARAM Saturation
//!DESC How colourful the noise is: 0 = luma only, 1 = as the matrix leaves it, more = a saturated picture style. Training uses 1 to 2.2. (cat: Colour)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.2
1.5

//!PARAM ColourMatrix
//!DESC Strength of the camera's colour matrix, which spreads red and blue noise into colour noise (magenta-teal). Training uses 0.3 to 1.5.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.5
0.9

//!PARAM RedGain
//!DESC White balance gain of red: the higher, the noisier red is. Training uses 1 to 3.
//!TYPE float
//!MINIMUM 1.0
//!MAXIMUM 3.0
2.0

//!PARAM BlueGain
//!DESC White balance gain of blue: the higher, the noisier blue is. Training uses 1 to 3.
//!TYPE float
//!MINIMUM 1.0
//!MAXIMUM 3.0
2.0

//!PARAM Softness
//!DESC Softening of the noise by demosaicing, in pixels (0 = one-pixel noise). Training uses 0.2 to 0.9. (cat: Camera processing)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.9
0.5

//!PARAM ChromaNR
//!DESC The camera's chroma noise reduction: blurs the noise's colour, leaving wide colour blotches.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM ChromaNRSize
//!DESC Size of the blur of the noise's colour, in pixels. Training uses 1.5 to 4.
//!TYPE float
//!MINIMUM 1.5
//!MAXIMUM 4.0
2.5

//!PARAM Persistence
//!DESC How much of the previous picture's noise stays (the camera's temporal noise reduction). Training uses 0 to 0.3.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.3
0.0

//!HOOK LINEAR
//!BIND HOOKED
//!SAVE SENSOR_N
//!DESC sensor noise: shot + read noise, white balance, colour matrix

// Source transfer, from the player (0 SDR gamma, 1 PQ, 2 HLG, 3 linear).
#ifdef _BRP_TRANSFER
#define SDR_SOURCE (_BRP_TRANSFER == 0)
#else
#define SDR_SOURCE 0
#endif

// PCG hash, 4 values from 4 (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020).
uvec4 pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}

// Three independent unit Gaussians for pixel p of picture f (Box-Muller).
vec3 gauss3(ivec2 p, int f) {
    uvec4 h = pcg4d(uvec4(uint(p.x), uint(p.y), uint(f), 0x51ED270Bu));
    vec4 u = (vec4(h >> 8u) + 0.5) * (1.0 / 16777216.0);
    vec2 r = sqrt(-2.0 * log(u.xz));
    vec2 a = 6.2831853 * u.yw;
    return vec3(r.x * cos(a.x), r.x * sin(a.x), r.y * cos(a.y));
}

// The camera's linear light from the LINEAR stage's.
vec3 camera_linear(vec3 c) {
#if SDR_SOURCE
    // Back to the code values (the player decoded them with a 2.4 power), then sRGB.
    vec3 v = pow(max(c, 0.0), vec3(1.0 / 2.4));
    return mix(v / 12.92, pow((v + 0.055) / 1.055, vec3(2.4)), step(0.04045, v));
#else
    return c;
#endif
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec3 x = camera_linear(texelFetch(HOOKED_raw, p, 0).rgb);
    vec3 wb = vec3(RedGain, 1.0, BlueGain);
    vec3 raw = max(x, 0.0) / wb;
    float read = ReadNoise * 1e-3;
    vec3 sd = sqrt(raw * (ShotNoise * 1e-3) + read * read);
    // Unit noise; with Persistence k, each picture keeps k of the last one's noise
    // (noise[t] = k noise[t-1] + sqrt(1 - k^2) fresh[t]), written as a sum over the
    // last pictures' fresh noise so any picture can be made on its own.
    vec3 g = gauss3(p, frame);
    if (Persistence > 0.0) {
        float w = Persistence, w2 = 1.0;
        for (int j = 1; j <= 4; j++) {
            g += w * gauss3(p, frame - j);
            w2 += w * w;
            w *= Persistence;
        }
        g /= sqrt(w2);
    }
    vec3 n = g * sd * wb;
    // Camera RGB to output RGB: identity + ColourMatrix * a typical matrix's shape
    // (rows sum to 0, so white stays white).
    float k = ColourMatrix;
    n = vec3(dot(vec3(1.0 + 0.6 * k, -0.4 * k, -0.2 * k), n),
             dot(vec3(-0.2 * k, 1.0 + 0.5 * k, -0.3 * k), n),
             dot(vec3(0.0, -0.5 * k, 1.0 + 0.5 * k), n));
    float y = dot(n, vec3(0.2126, 0.7152, 0.0722));
    n = y + Saturation * (n - y);
    return vec4(n, 1.0);
}

//!HOOK LINEAR
//!BIND SENSOR_N
//!SAVE SENSOR_T
//!WHEN Softness 0.1 >
//!DESC sensor noise: demosaic softening along x

// Edges mirror (as the training's blur does).
int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(SENSOR_N_size.x);
    float s = Softness;
    int rad = max(1, int(ceil(3.0 * s)));
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * texelFetch(SENSOR_N_raw, ivec2(mirror(p.x + i, n), p.y), 0);
        ws += w;
    }
    return acc / ws;
}

//!HOOK LINEAR
//!BIND SENSOR_T
//!SAVE SENSOR_N
//!WHEN Softness 0.1 >
//!DESC sensor noise: demosaic softening along y

int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(SENSOR_T_size.y);
    float s = Softness;
    int rad = max(1, int(ceil(3.0 * s)));
    vec4 acc = vec4(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * texelFetch(SENSOR_T_raw, ivec2(p.x, mirror(p.y + i, n)), 0);
        ws += w;
    }
    // The blur lowers the noise; the training gives part of it back.
    return acc / ws * (1.0 + s);
}

//!HOOK LINEAR
//!BIND SENSOR_N
//!SAVE SENSOR_T
//!WHEN ChromaNR
//!DESC sensor noise: chroma noise reduction along x

int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

// The noise's colour part (what the camera's chroma NR blurs).
vec3 chroma(ivec2 q) {
    vec3 c = texelFetch(SENSOR_N_raw, q, 0).rgb;
    return c - dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(SENSOR_N_size.x);
    float s = ChromaNRSize;
    int rad = max(1, int(ceil(3.0 * s)));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * chroma(ivec2(mirror(p.x + i, n), p.y));
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK LINEAR
//!BIND SENSOR_N
//!BIND SENSOR_T
//!SAVE SENSOR_N
//!WHEN ChromaNR
//!DESC sensor noise: chroma noise reduction along y

int mirror(int i, int n) {
    i = i < 0 ? -i : i;
    return clamp(i >= n ? 2 * (n - 1) - i : i, 0, n - 1);
}

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    int n = int(SENSOR_T_size.y);
    float s = ChromaNRSize;
    int rad = max(1, int(ceil(3.0 * s)));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float w = exp(-float(i * i) / (2.0 * s * s));
        acc += w * texelFetch(SENSOR_T_raw, ivec2(p.x, mirror(p.y + i, n)), 0).rgb;
        ws += w;
    }
    // The noise's luma as it was, with the blurred colour.
    float y = dot(texelFetch(SENSOR_N_raw, p, 0).rgb, vec3(0.2126, 0.7152, 0.0722));
    return vec4(y + acc / ws, 1.0);
}

//!HOOK LINEAR
//!BIND HOOKED
//!BIND SENSOR_N
//!DESC sensor noise: added to the picture

#ifdef _BRP_TRANSFER
#define SDR_SOURCE (_BRP_TRANSFER == 0)
#else
#define SDR_SOURCE 0
#endif

vec4 hook() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec4 c = texelFetch(HOOKED_raw, p, 0);
    vec3 n = texelFetch(SENSOR_N_raw, p, 0).rgb;
#if SDR_SOURCE
    // To the camera's linear light (code values, sRGB curve), add, and back.
    vec3 v = pow(max(c.rgb, 0.0), vec3(1.0 / 2.4));
    vec3 x = mix(v / 12.92, pow((v + 0.055) / 1.055, vec3(2.4)), step(0.04045, v)) + n;
    x = max(x, 0.0);
    v = mix(12.92 * x, 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, x));
    return vec4(pow(v, vec3(2.4)), c.a);
#else
    return vec4(c.rgb + n, c.a);
#endif
}
