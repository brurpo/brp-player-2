// Gaussian blur, in linear light.
//
// The slider is not the blur's size itself: the size grows with its square, so the low
// end of the slider moves in small steps (0.1 is about half a pixel at 1080p, 0.3 about
// 5 pixels, 1 about 58: the blur's sigma is Size^2 x 3 % of the picture's longer side).
//
// @brp en.title Gaussian Blur
// @brp pt.title Desfoque gaussiano
// @brp pt.param Size Tamanho | Tamanho do desfoque (0 = nenhum). Cresce devagar no começo do controle: 0,3 dá uns 5 pixels em 1080p, 1 dá uns 58.

//!PARAM Size
//!DESC Blur size (0 = none). It grows slowly at the low end of the slider: 0.3 is about 5 pixels at 1080p, 1 about 58.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!HOOK MAIN
//!BIND HOOKED
//!SAVE BLUR_LINEAR
//!DESC gaussian blur: to linear light

vec4 hook() {
    return linearize(HOOKED_tex(HOOKED_pos));
}

//!HOOK MAIN
//!BIND BLUR_LINEAR
//!SAVE BLUR_Y
//!DESC gaussian blur: along y

// The blur's sigma in pixels.
float sigma(vec2 size) {
    return Size * Size * 0.03 * max(size.x, size.y);
}

// Blur along `axis` (one pixel that way, in texture coordinates). Two pixels per tap:
// each tap sits between two pixels, where the sampler mixes them in the ratio of
// their weights.
vec4 blur(vec2 pos, vec2 axis, float s) {
    vec4 sum = BLUR_LINEAR_tex(pos);
    if (s < 0.05)
        return sum;
    float ws = 1.0;
    int taps = int(ceil(3.0 * s * 0.5));
    for (int i = 1; i <= taps; i++) {
        float a = float(2 * i - 1), b = float(2 * i);
        float wa = exp(-a * a / (2.0 * s * s)), wb = exp(-b * b / (2.0 * s * s));
        float w = wa + wb;
        vec2 off = axis * (a + wb / w);
        sum += w * (BLUR_LINEAR_tex(pos + off) + BLUR_LINEAR_tex(pos - off));
        ws += 2.0 * w;
    }
    return sum / ws;
}

vec4 hook() {
    return blur(BLUR_LINEAR_pos, vec2(0.0, BLUR_LINEAR_pt.y), sigma(BLUR_LINEAR_size));
}

//!HOOK MAIN
//!BIND BLUR_Y
//!DESC gaussian blur: along x, back from linear light

float sigma(vec2 size) {
    return Size * Size * 0.03 * max(size.x, size.y);
}

vec4 blur(vec2 pos, vec2 axis, float s) {
    vec4 sum = BLUR_Y_tex(pos);
    if (s < 0.05)
        return sum;
    float ws = 1.0;
    int taps = int(ceil(3.0 * s * 0.5));
    for (int i = 1; i <= taps; i++) {
        float a = float(2 * i - 1), b = float(2 * i);
        float wa = exp(-a * a / (2.0 * s * s)), wb = exp(-b * b / (2.0 * s * s));
        float w = wa + wb;
        vec2 off = axis * (a + wb / w);
        sum += w * (BLUR_Y_tex(pos + off) + BLUR_Y_tex(pos - off));
        ws += 2.0 * w;
    }
    return sum / ws;
}

vec4 hook() {
    return delinearize(blur(BLUR_Y_pos, vec2(BLUR_Y_pt.x, 0.0), sigma(BLUR_Y_size)));
}
