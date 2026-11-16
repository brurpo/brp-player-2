// Lens flare from the picture itself: what a lens adds around bright lights.
//   - Glare: a glow around the lights, with a long faint tail.
//   - Ghosts: reflections between the lens's elements. They sit on the line from each
//     light through the centre of the frame, from next to the light to beyond the
//     centre, strongest next to the light. Each has its own size, and is the light
//     itself seen through the aperture: a small light gives the aperture's shape, a
//     line of light a soft streak. Tinted as lens coatings tint them, with a faint colour
//     fringe. GhostVariation picks another lens: other places, sizes and kinds.
//   - Halo: a faint ring around each light, split into colours.
//   - Star: thin rays out of each light.
//   - Anamorphic: everything stretched sideways, and a long blue streak through
//     each light.
//   - Dirt: dust and smudges on the front of the lens, lit by all of the above.
// Every part is a blur or a copy of the lights' picture, so it follows the picture
// smoothly from frame to frame. The lights are what is brighter than Threshold and
// compact: a large bright area (a white wall, the sky, a title) is no light, as a lens
// flares on what is far brighter than the rest, and in a video file a lamp and a white
// wall can hold the same value; LargeAreas lets those flare too. Made at a quarter of
// the size (it is all soft), in linear light, and added to the picture.
//
// @brp en.title Lens Flare
// @brp pt.title Reflexo de lente
// @brp pt.param Threshold Limiar | Brilho a partir do qual uma área gera reflexos (1 = branco). Acima de 1: só luzes mais claras que o branco, em imagens HDR e EXR.
// @brp pt.param Intensity Intensidade | Intensidade geral de tudo.
// @brp pt.param LargeAreas Áreas grandes | Quanto as áreas claras grandes (céu, parede branca, títulos) geram reflexos. 0 = só as luzes pequenas, 1 = tudo que é claro.
// @brp pt.param GlareIntensity Clarão | Intensidade do brilho suave em volta das luzes.
// @brp pt.param GlareSize Tamanho do clarão | Tamanho do brilho em volta das luzes.
// @brp pt.param GhostIntensity Fantasmas | Intensidade dos fantasmas: os reflexos em fila na linha que vai de cada luz ao centro do quadro e além dele.
// @brp pt.param GhostCount Número de fantasmas | Quantos fantasmas cada luz gera.
// @brp pt.param GhostVariation Variação | Outra lente: muda a posição, o tamanho, o tipo e a cor de cada fantasma.
// @brp pt.param GhostSize Tamanho dos fantasmas | Tamanho de todos os fantasmas.
// @brp pt.param GhostSoftness Suavidade | Suavidade da borda dos fantasmas: 0 = forma do diafragma nítida, 1 = bem desfocada.
// @brp pt.param GhostBlades Lâminas | Número de lâminas do diafragma, que dá a forma dos fantasmas. Menos de 3 = circular.
// @brp pt.param GhostColor Cor dos fantasmas | Quanto o revestimento da lente tinge os fantasmas (verde, magenta, azul, âmbar). 0 = a cor da própria luz.
// @brp pt.param GhostDispersion Dispersão | Aberração cromática: franja de cor na borda dos fantasmas e cores do halo.
// @brp pt.param Halo Halo | Intensidade do anel em volta de cada luz.
// @brp pt.param HaloSize Tamanho do halo | Raio do anel, em fração da altura do quadro.
// @brp pt.param GlintIntensity Estrela | Intensidade dos raios que saem de cada luz.
// @brp pt.param GlintPoints Pontas da estrela | Número de raios.
// @brp pt.param GlintSize Tamanho da estrela | Comprimento dos raios, em fração da altura do quadro.
// @brp pt.param GlintDispersion Dispersão da estrela | Aberração cromática ao longo dos raios: pontas avermelhadas.
// @brp pt.param Anamorphic Anamórfico | Lente anamórfica: fantasmas, halo e clarão esticados na horizontal, e um risco comprido em cada luz.
// @brp pt.param StreakIntensity Risco | Intensidade do risco horizontal azul da lente anamórfica.
// @brp pt.param DirtIntensity Sujeira da lente | Poeira e manchas na lente, iluminadas pelos reflexos (0 = lente limpa).

//!PARAM Threshold
//!DESC Brightness above which an area makes flares (1 = white). Above 1: only lights brighter than white, in HDR and EXR pictures. (cat: Light)
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 4.0
0.9

//!PARAM Intensity
//!DESC Overall intensity of everything.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.5

//!PARAM LargeAreas
//!DESC How much large bright areas (sky, a white wall, titles) flare. 0 = compact lights only, 1 = everything bright.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM GlareIntensity
//!DESC Strength of the soft glow around the lights. (cat: Glare)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM GlareSize
//!DESC Size of the glow around the lights.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM GhostIntensity
//!DESC Strength of the ghosts: the reflections in a row on the line from each light to the centre of the frame and beyond it. (cat: Ghosts)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM GhostCount
//!DESC How many ghosts each light gives.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 12
7

//!PARAM GhostVariation
//!DESC Another lens: changes where each ghost sits, its size, kind and tint.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 30
0

//!PARAM GhostSize
//!DESC Size of all the ghosts.
//!TYPE float
//!MINIMUM 0.3
//!MAXIMUM 3.0
1.0

//!PARAM GhostSoftness
//!DESC Softness of the ghosts' edge: 0 = the aperture's shape, crisp; 1 = well out of focus.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM GhostBlades
//!DESC Number of aperture blades, which shapes the ghosts. Fewer than 3 = round.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 12
6

//!PARAM GhostColor
//!DESC How much the lens coating tints the ghosts (green, magenta, blue, amber). 0 = the light's own colour.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM GhostDispersion
//!DESC Chromatic aberration: a colour fringe on the ghosts' edges, and the halo's colours.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.2
0.04

//!PARAM Halo
//!DESC Strength of the ring around each light. (cat: Halo)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.4

//!PARAM HaloSize
//!DESC Radius of the ring, as a fraction of the frame's height.
//!TYPE float
//!MINIMUM 0.1
//!MAXIMUM 0.6
0.3

//!PARAM GlintIntensity
//!DESC Strength of the rays out of each light. (cat: Star)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM GlintPoints
//!DESC Number of rays.
//!TYPE int
//!MINIMUM 3
//!MAXIMUM 20
8

//!PARAM GlintSize
//!DESC Length of the rays, as a fraction of the frame's height.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.5
0.2

//!PARAM GlintDispersion
//!DESC Chromatic aberration along the rays: reddish tips.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.1
0.04

//!PARAM Anamorphic
//!DESC An anamorphic lens: ghosts, halo and glare stretched sideways, and a long streak through each light. (cat: Lens)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM StreakIntensity
//!DESC Strength of the anamorphic lens's blue horizontal streak.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0

//!PARAM DirtIntensity
//!DESC Dust and smudges on the lens, lit by the flare (0 = a clean lens).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1

//!HOOK MAIN
//!BIND HOOKED
//!SAVE FLARE_RAW
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: the bright parts, at a quarter of the size

vec4 hook() {
    // 4 x 4 pixels of the picture per pixel here: four taps between pixels.
    // rgb: the bright light; a: how much of the pixel is bright.
    vec4 acc = vec4(0.0);
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            vec3 lin = linearize(HOOKED_tex(HOOKED_pos + (vec2(i, j) * 2.0 - 1.0) * HOOKED_pt).rgb);
            // Brightness as a screen shows it (1 = SDR white; HDR goes above). A wide
            // step, so that noise around the threshold does not switch a light on and off.
            float seen = pow(max(dot(lin, vec3(0.2126, 0.7152, 0.0722)), 0.0), 1.0 / 2.4);
            float bright = smoothstep(Threshold - 0.15, Threshold + 0.03, seen);
            acc += vec4(clamp(lin, 0.0, 64.0) * bright, bright);
        }
    }
    return acc * 0.25;
}

//!HOOK MAIN
//!BIND FLARE_RAW
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: how much of the surroundings is bright, along x

// A wide blur (sigma: a twentieth of the height) of the bright share.
vec4 hook() {
    float s = 0.05 * FLARE_RAW_size.y;
    int rad = int(ceil(2.5 * s * 0.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float x = float(i) * 2.0;
        float w = exp(-x * x / (2.0 * s * s));
        acc += w * FLARE_RAW_tex(FLARE_RAW_pos + vec2(x + 0.5, 0.0) * FLARE_RAW_pt).a;
        ws += w;
    }
    return vec4(acc / ws);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_AREA
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: how much of the surroundings is bright, along y

vec4 hook() {
    float s = 0.05 * FLARE_T_size.y;
    int rad = int(ceil(2.5 * s * 0.5));
    float acc = 0.0, ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float y = float(i) * 2.0;
        float w = exp(-y * y / (2.0 * s * s));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, y + 0.5) * FLARE_T_pt).a;
        ws += w;
    }
    return vec4(acc / ws);
}

//!HOOK MAIN
//!BIND FLARE_RAW
//!BIND FLARE_AREA
//!SAVE FLARE_BRIGHT
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: the lights (bright and compact)

vec4 hook() {
    // A lamp, the sun or a glint has little brightness around it; inside a white
    // wall or a title nearly all the surroundings are bright. The more of the
    // surroundings is bright, the less of a light this is: gradually, with no step
    // that a moving or growing shape could cross from one frame to the next (half
    // when a tenth of the surroundings is bright, a hundredth inside a white wall).
    // So a bright shape flares about as much whatever its width: mostly from its rim.
    float around = FLARE_AREA_tex(FLARE_AREA_pos).a / 0.1;
    float light = mix(1.0 / (1.0 + around * around), 1.0, LargeAreas);
    return vec4(FLARE_RAW_tex(FLARE_RAW_pos).rgb * light, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_BRIGHT
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened a little, along x

vec4 hook() {
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -6; i <= 6; i++) {
        float w = exp(-float(i * i) / (2.0 * 2.2 * 2.2));
        acc += w * FLARE_BRIGHT_tex(FLARE_BRIGHT_pos + vec2(float(i), 0.0) * FLARE_BRIGHT_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_SOFT
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened a little, along y

vec4 hook() {
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -6; i <= 6; i++) {
        float w = exp(-float(i * i) / (2.0 * 2.2 * 2.2));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, float(i)) * FLARE_T_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_SOFT
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: glare, along x

vec4 hook() {
    // Wider sideways with an anamorphic lens.
    float s = max(GlareSize, 0.02) * 0.035 * FLARE_SOFT_size.y * (1.0 + 2.0 * Anamorphic);
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    // Taps two pixels apart, each between two pixels.
    for (int i = -rad; i <= rad; i++) {
        float x = float(i) * 2.0;
        float w = exp(-x * x / (2.0 * s * s));
        acc += w * FLARE_SOFT_tex(FLARE_SOFT_pos + vec2(x + 0.5, 0.0) * FLARE_SOFT_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_GLARE
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: glare, along y

vec4 hook() {
    float s = max(GlareSize, 0.02) * 0.035 * FLARE_T_size.y;
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float y = float(i) * 2.0;
        float w = exp(-y * y / (2.0 * s * s));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, y + 0.5) * FLARE_T_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_SOFT
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: the wide glow (glare's tail, light on the dirt), along x

vec4 hook() {
    float s = 0.10 * FLARE_SOFT_size.y * (1.0 + Anamorphic);
    int rad = int(ceil(2.5 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float x = float(i) * 2.0;
        float w = exp(-x * x / (2.0 * s * s));
        acc += w * FLARE_SOFT_tex(FLARE_SOFT_pos + vec2(x + 0.5, 0.0) * FLARE_SOFT_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_WIDE
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: the wide glow, along y

vec4 hook() {
    float s = 0.10 * FLARE_T_size.y;
    int rad = int(ceil(2.5 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float y = float(i) * 2.0;
        float w = exp(-y * y / (2.0 * s * s));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, y + 0.5) * FLARE_T_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_SOFT
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened more, along x

vec4 hook() {
    float s = 5.6 * FLARE_SOFT_size.y / 270.0;
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float x = float(i) * 2.0;
        float w = exp(-x * x / (2.0 * s * s));
        acc += w * FLARE_SOFT_tex(FLARE_SOFT_pos + vec2(x + 0.5, 0.0) * FLARE_SOFT_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_MID
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened more, along y

vec4 hook() {
    float s = 5.6 * FLARE_T_size.y / 270.0;
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float y = float(i) * 2.0;
        float w = exp(-y * y / (2.0 * s * s));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, y + 0.5) * FLARE_T_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_MID
//!SAVE FLARE_T
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened most, along x

vec4 hook() {
    float s = 12.6 * FLARE_MID_size.y / 270.0;
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float x = float(i) * 2.0;
        float w = exp(-x * x / (2.0 * s * s));
        acc += w * FLARE_MID_tex(FLARE_MID_pos + vec2(x + 0.5, 0.0) * FLARE_MID_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_T
//!SAVE FLARE_BIG
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: lights softened most, along y

vec4 hook() {
    float s = 12.6 * FLARE_T_size.y / 270.0;
    int rad = int(ceil(3.0 * s * 0.5));
    vec3 acc = vec3(0.0);
    float ws = 0.0;
    for (int i = -rad; i <= rad; i++) {
        float y = float(i) * 2.0;
        float w = exp(-y * y / (2.0 * s * s));
        acc += w * FLARE_T_tex(FLARE_T_pos + vec2(0.0, y + 0.5) * FLARE_T_pt).rgb;
        ws += w;
    }
    return vec4(acc / ws, 1.0);
}

//!HOOK MAIN
//!BIND FLARE_BRIGHT
//!SAVE FLARE_STREAK
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: the anamorphic streak

vec4 hook() {
    if (Anamorphic <= 0.0 || StreakIntensity <= 0.0)
        return vec4(0.0, 0.0, 0.0, 1.0);
    // Half length of the streak, in pixels here.
    float len = Anamorphic * 0.45 * FLARE_BRIGHT_size.x;
    int n = int(ceil(len * 0.5));
    vec3 acc = FLARE_BRIGHT_tex(FLARE_BRIGHT_pos).rgb;
    for (int i = 1; i <= n; i++) {
        float x = float(i) * 2.0 - 0.5;
        float t = x / len;
        // Bright at the light, with a long thin tail.
        float w = (1.0 - t) * (1.0 - t) / (1.0 + 30.0 * t);
        acc += w * (FLARE_BRIGHT_tex(FLARE_BRIGHT_pos + vec2(x, 0.0) * FLARE_BRIGHT_pt).rgb + FLARE_BRIGHT_tex(FLARE_BRIGHT_pos - vec2(x, 0.0) * FLARE_BRIGHT_pt).rgb);
    }
    // Not divided by the weights: a streak, not a blur. A wider light gives a
    // brighter one, up to a limit.
    acc *= 0.5;
    return vec4(acc / (1.0 + max(max(acc.r, acc.g), acc.b) / 1.5), 1.0);
}

//!HOOK MAIN
//!BIND FLARE_SOFT
//!SAVE FLARE_STAR
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: star

// 1 inside the picture, fading to 0 at its edges (no light comes from outside it).
float inside(vec2 q) {
    vec2 e = smoothstep(vec2(0.0), vec2(0.04), q) * smoothstep(vec2(0.0), vec2(0.04), 1.0 - q);
    return e.x * e.y;
}

// How bright a ray is at t (0 at the light, 1 at its tip): strong at the light, a
// long faint tail.
float ray(float t) {
    float u = clamp(1.0 - t, 0.0, 1.0);
    return u * u / (1.0 + 16.0 * max(t, 0.0));
}

// Rays out of each light. Gathered here by looking back along each ray; red reaches
// farther than blue (GlintDispersion). No two rays are quite alike.
vec4 hook() {
    if (GlintSize <= 0.0 || GlintIntensity <= 0.0)
        return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 uv = FLARE_SOFT_pos;
    vec2 size = FLARE_SOFT_size;
    const int STEPS = 40;
    float d = GlintDispersion * 4.0;
    vec3 star = vec3(0.0);
    for (int k = 0; k < GlintPoints; k++) {
        float a = 6.2831853 * float(k) / float(GlintPoints);
        // Each ray's own length and strength (fixed: from its number).
        float own = fract(sin(float(k) * 12.9898 + 4.1) * 43758.5453);
        float len = GlintSize * size.y * (0.7 + 0.5 * own);
        vec2 dir = vec2(sin(a), -cos(a)) / size;
        vec3 one = vec3(0.0);
        for (int j = 0; j < STEPS; j++) {
            float t = (float(j) + 0.5) / float(STEPS);
            vec2 q = uv - dir * (t * len);
            one += FLARE_SOFT_tex(q).rgb * inside(q) * vec3(ray(t / (1.0 + d)), ray(t), ray(t / (1.0 - d)));
        }
        // Per pixel of ray, so a ray's brightness does not depend on its length.
        star += one * (len / float(STEPS)) * (0.75 + 0.5 * fract(own * 7.3));
    }
    // A wide light gives brighter rays, up to a limit.
    star *= 0.5;
    return vec4(star / (1.0 + max(max(star.r, star.g), star.b) / 1.5), 1.0);
}

//!HOOK MAIN
//!BIND FLARE_SOFT
//!BIND FLARE_MID
//!BIND FLARE_BIG
//!SAVE FLARE_GHOSTS
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!DESC lens flare: ghosts and halo

// PCG hash, 3 values from 3 (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020), as numbers in 0..1.
vec3 hash3(ivec3 v0) {
    uvec3 v = uvec3(v0) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return vec3(v >> 8u) * (1.0 / 16777216.0);
}

// 1 inside the picture, fading to 0 at its edges (no light comes from outside it).
float inside(vec2 q) {
    vec2 e = smoothstep(vec2(0.0), vec2(0.04), q) * smoothstep(vec2(0.0), vec2(0.04), 1.0 - q);
    return e.x * e.y;
}

// How far the aperture reaches at angle a, relative to its corners: a polygon with
// `blades` sides, or a circle.
float aperture(float a, int blades) {
    if (blades < 3)
        return 1.0;
    float seg = 6.2831853 / float(blades);
    return cos(0.5 * seg) / cos(mod(a, seg) - 0.5 * seg);
}

// How bright a ghost is at r (0 its centre, 1 its edge), by kind: 0 evenly lit, 1
// brighter at its centre, 2 a ring (lit at its edge), 3 a soft blob.
float lit(int kind, float r, float soft) {
    float edge = 1.0 - smoothstep(1.0 - soft, 1.0, r);
    if (kind == 0)
        return edge * (0.9 + 0.1 * r * r);
    if (kind == 1)
        return edge * (1.0 - 0.5 * r);
    if (kind == 2)
        return edge * (0.10 + 0.90 * smoothstep(0.45, 0.85, r));
    return exp(-3.5 * r * r);
}

// The lights at q, softened a little (0), more (1) or most (2), inside the picture only.
vec3 lights(int level, vec2 q) {
    vec3 c = level == 0 ? FLARE_SOFT_tex(q).rgb : level == 1 ? FLARE_MID_tex(q).rgb : FLARE_BIG_tex(q).rgb;
    return c * inside(q);
}

vec4 hook() {
    vec2 uv = FLARE_SOFT_pos;
    vec2 size = FLARE_SOFT_size;
    vec2 c = uv - 0.5;
    vec3 sum = vec3(0.0);

    // Ghosts. Ghost g of a light sits on the line from the light through the centre of
    // the frame, at `where`: 1 = at the light, 0 = at the centre, -1 = as far beyond
    // the centre as the light is from it. Each has its own size, whatever its place
    // (sizes that grew with the distance from the centre made the ghosts look thrown
    // out from the centre), and the chain is strongest next to the light: it reads as
    // coming out of the light. Where, how large, which kind, how strong and which
    // coating tint: from the ghost's number and GhostVariation, the places spread
    // along the line so that a few ghosts already span it.
    //
    // A ghost at a pixel is the lights' picture, reduced `where` times about the
    // centre, seen through the aperture: the sum, over the aperture's shape around
    // the pixel, of the lights there. The more the lights are reduced, the more
    // softened a copy of them is read, so that no light falls between two taps.
    if (GhostCount > 0 && GhostIntensity > 0.0) {
        // What anti-reflection coatings reflect: green, magenta, violet-blue, amber, cyan.
        const vec3 COATING[5] = vec3[5](vec3(0.50, 1.15, 0.60), vec3(1.25, 0.72, 1.10), vec3(0.75, 0.85, 1.60), vec3(1.30, 0.95, 0.50), vec3(0.55, 1.10, 1.20));
        float unit = size.y / 270.0;
        float lamp = 16.0 * unit * unit;  // a small lamp: 16 pixels of white here at 1080p
        vec2 stretch = vec2(1.0 + Anamorphic, 1.0);
        float soft = mix(0.10, 0.75, GhostSoftness);
        float start = hash3(ivec3(GhostVariation, 3, 11)).x;
        for (int g = 0; g < GhostCount; g++) {
            vec3 h = hash3(ivec3(g, GhostVariation, 29));
            vec3 k = hash3(ivec3(g, GhostVariation, 53));
            float where = mix(0.92, -1.7, fract(start + 0.61803399 * float(g + 1) + 0.05 * (h.x - 0.5)));
            // Not on the centre itself (its lights would shrink to a dot there).
            where = abs(where) < 0.12 ? (where < 0.0 ? -0.12 : 0.12) : where;
            int kind = h.y < 0.30 ? 0 : h.y < 0.50 ? 1 : h.y < 0.80 ? 2 : 3;
            vec3 tint = mix(vec3(1.0), COATING[int(h.z * 4.999)], GhostColor);
            // Strongest next to the light, fading along the line away from it.
            float strength = mix(0.35, 0.85, k.y) * mix(1.0, 0.45, clamp((0.92 - where) / 2.62, 0.0, 1.0)) * (kind == 2 ? 1.3 : 1.0);
            // Its radius, in pixels here: most ghosts small, a few large.
            float radius = GhostSize * mix(0.010, 0.065, k.x * k.x) * (kind == 3 ? 1.6 : 1.0) * size.y;
            // Rings of taps over the shape; the copy of the lights soft enough for their spacing.
            int rings = clamp(int(ceil(radius * stretch.x / 2.0)), 3, 7);
            float need = radius * stretch.x / float(rings) / (2.0 * abs(where) * unit);
            int level = need <= 2.6 ? 0 : need <= 7.0 ? 1 : 2;
            // Red and blue reduced slightly differently: the lens bends each colour
            // differently, which shows as a fringe at the ghost's edge.
            float d = GhostDispersion * 0.25;
            float to_r = 1.0 / (where * (1.0 + d)), to_g = 1.0 / where, to_b = 1.0 / (where * (1.0 - d));
            vec3 acc = lights(level, vec2(0.5) + c * to_g) * lit(kind, 0.0, soft);
            float taps = 1.0;
            for (int ring = 1; ring <= rings; ring++) {
                float r = (float(ring) - 0.25) / float(rings);
                float w = lit(kind, r, soft);
                int n = 6 * ring;
                for (int j = 0; j < n; j++) {
                    float a = 6.2831853 * (float(j) + 0.5 * float(ring & 1)) / float(n);
                    vec2 at = c + vec2(sin(a), -cos(a)) * (r * radius * aperture(a, GhostBlades)) * stretch / size;
                    if (d > 0.0) {
                        vec3 red = lights(level, vec2(0.5) + at * to_r), blue = lights(level, vec2(0.5) + at * to_b);
                        acc += w * vec3(red.r, 0.5 * (red.g + blue.g), blue.b);
                    } else {
                        acc += w * lights(level, vec2(0.5) + at * to_g);
                    }
                }
                taps += float(n);
            }
            // As light, not as an average: each tap stands for its share of the shape,
            // which is `where` squared times as much of the lights' picture. In units of
            // the small lamp, which so gives 1 whatever the ghost's size and place.
            vec3 v = acc * (3.14159 * radius * radius * stretch.x / taps) / (lamp * where * where);
            // A larger light gives a brighter ghost, up to a limit.
            v *= tint * (0.40 * strength);
            sum += v / (1.0 + max(max(v.r, v.g), v.b) / 0.6);
        }
        sum *= GhostIntensity;
    }

    // Halo: every light spread into a ring around itself (an ellipse with an
    // anamorphic lens), each colour at its own radius. Taps along the ring, about 3
    // pixels apart.
    if (Halo > 0.0) {
        float radius = HaloSize * size.y;
        int n = clamp(int(6.2831853 * radius * (1.0 + Anamorphic) / 3.0), 48, 220);
        vec3 ring = vec3(0.0);
        for (int j = 0; j < n; j++) {
            float a = 6.2831853 * (float(j) + 0.5) / float(n);
            vec2 dir = vec2(cos(a) * (1.0 + Anamorphic), sin(a)) / size;
            for (int k = 0; k < 3; k++) {
                vec2 q = uv + dir * radius * (1.0 - (float(k) - 1.0) * GhostDispersion * 1.2);
                ring[k] += FLARE_SOFT_tex(q)[k] * inside(q);
            }
        }
        // As light: a tap stands for its part of the ring (about 5 pixels wide).
        float lamp = 16.0 * (size.y / 270.0) * (size.y / 270.0);
        ring *= (6.2831853 * radius * 5.0 / float(n)) / lamp * 0.05;
        sum += ring / (1.0 + max(max(ring.r, ring.g), ring.b) / 0.3) * Halo;
    }
    return vec4(sum, 1.0);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND FLARE_STAR
//!BIND FLARE_GHOSTS
//!BIND FLARE_GLARE
//!BIND FLARE_STREAK
//!BIND FLARE_WIDE
//!DESC lens flare: with the dirt on the lens, added to the picture

// PCG hash, 3 values from 3 (Jarzynski and Olano, "Hash Functions for GPU Rendering",
// JCGT 2020), as numbers in 0..1.
vec3 hash3(ivec3 v0) {
    uvec3 v = uvec3(v0) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return vec3(v >> 8u) * (1.0 / 16777216.0);
}

// Smooth noise, 0..1.
float noise(vec2 p, int seed) {
    ivec2 i = ivec2(floor(p));
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash3(ivec3(i, seed)).x, b = hash3(ivec3(i + ivec2(1, 0), seed)).x;
    float c = hash3(ivec3(i + ivec2(0, 1), seed)).x, d = hash3(ivec3(i + ivec2(1, 1), seed)).x;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// The dirt on the lens at p (in frame heights from the centre): out-of-focus dust
// (soft discs, a little brighter at the rim, in three sizes) and smudges. It never moves.
float dirt(vec2 p) {
    float sum = 0.0;
    const float CELL[3] = float[3](0.17, 0.08, 0.036);
    const float SHARE[3] = float[3](0.55, 0.5, 0.4);
    const float LIGHT[3] = float[3](0.9, 0.75, 0.6);
    for (int l = 0; l < 3; l++) {
        vec2 g = p / CELL[l];
        ivec2 id = ivec2(floor(g));
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                ivec2 cell = id + ivec2(x, y);
                vec3 h = hash3(ivec3(cell, l));
                vec3 k = hash3(ivec3(cell, l + 7));
                if (k.x > SHARE[l])
                    continue;
                float r = mix(0.16, 0.5, h.z * h.z);
                float d = length(g - (vec2(cell) + h.xy)) / r;
                sum += smoothstep(1.0, 0.65, d) * (0.55 + 0.45 * smoothstep(0.3, 0.9, d)) * LIGHT[l] * mix(0.35, 1.0, k.y);
            }
        }
    }
    float smudge = 0.6 * noise(p * 2.3, 11) + 0.3 * noise(p * 5.1, 12) + 0.1 * noise(p * 23.0, 13);
    return min(sum, 1.2) + 0.5 * smoothstep(0.5, 0.8, smudge);
}

vec4 hook() {
    vec4 o = HOOKED_tex(HOOKED_pos);
    vec3 wide = FLARE_WIDE_tex(FLARE_WIDE_pos).rgb;
    vec3 shapes = FLARE_STAR_tex(FLARE_STAR_pos).rgb * GlintIntensity + FLARE_GHOSTS_tex(FLARE_GHOSTS_pos).rgb;
    // The anamorphic streak is blue, as the coating of those lenses makes it.
    shapes += FLARE_STREAK_tex(FLARE_STREAK_pos).rgb * vec3(0.35, 0.6, 1.0) * (1.2 * StreakIntensity);
    // The glare: a glow of the chosen size, and a faint tail far around the lights.
    vec3 glare = (FLARE_GLARE_tex(FLARE_GLARE_pos).rgb * 0.8 + wide * 0.5) * GlareIntensity;
    vec2 size = HOOKED_size;
    float d = DirtIntensity > 0.0 ? dirt((HOOKED_pos - 0.5) * vec2(size.x / size.y, 1.0)) : 0.0;
    // The dirt scatters the light that falls on it: the flare, and the lights' glow
    // far around them (held below 1, however many lights there are).
    vec3 around = wide * 25.0;
    around /= 1.0 + max(max(around.r, around.g), around.b);
    vec3 flare = shapes + glare + DirtIntensity * d * (shapes * 0.6 + glare + around * 0.7);
    // Very strong flares (HDR lights) are held back, colours kept.
    flare /= 1.0 + 0.12 * max(max(flare.r, flare.g), flare.b);
    return delinearize(vec4(linearize(o.rgb) + Intensity * flare, o.a));
}
