#version 450

// Converts a decoded frame to display RGB.
//  mode.x: 0 = planar YUV (t0 Y, t1 U, t2 V, t3 A), 1 = packed RGBA in t0
//  mode.y: transfer 0 = SDR (pass through), 1 = PQ, 2 = HLG, 3 = linear
//  mode.z: 1 = has alpha (composite over checkerboard)
//  mode.w: bit 0 = BT.2020 primaries (convert to BT.709 for HDR paths)
//          bit 1 = HDR output: write linear light, Rec.2020, 1.0 = SDR white, for
//          the HDR composite (DESIGN.md §7b); m0.w = SDR white (nits), m1.w = the
//          screen's peak (nits)

layout(set = 0, binding = 0) uniform sampler2D t0;
layout(set = 0, binding = 1) uniform sampler2D t1;
layout(set = 0, binding = 2) uniform sampler2D t2;
layout(set = 0, binding = 3) uniform sampler2D t3;

layout(push_constant) uniform PC {
    vec4 rect;
    vec4 m0;   // YUV->RGB matrix rows
    vec4 m1;
    vec4 m2;
    vec4 off;  // y offset, chroma offset, y scale, chroma scale
    vec4 misc; // x: sample norm, y: HDR peak (relative to SDR white), z: checker size (px)
    ivec4 mode;
    ivec4 swz; // packed RGBA: channel index of R, G, B, A (A < 0 = opaque)
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float SDR_WHITE_NITS = 203.0;

vec3 pq_to_nits(vec3 e) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 p = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), vec3(1.0 / m1)) * 10000.0;
}

vec3 hlg_to_nits(vec3 e) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    vec3 lo = e * e / 3.0;
    vec3 hi = (exp((e - c) / a) + b) / 12.0;
    vec3 scene = mix(lo, hi, step(0.5, e));
    // OOTF for a 1000 nit display (system gamma 1.2).
    float ys = dot(scene, vec3(0.2627, 0.6780, 0.0593));
    return scene * pow(max(ys, 1e-6), 0.2) * 1000.0;
}

vec3 bt2020_to_709(vec3 c) {
    return mat3(
        1.6605, -0.1246, -0.0182,
       -0.5876,  1.1329, -0.1006,
       -0.0728, -0.0083,  1.1187) * c;
}

// Extended Reinhard on the max channel: keeps hue, rolls off highlights to SDR white.
vec3 tonemap(vec3 c, float peak) {
    float m = max(max(c.r, c.g), c.b);
    if (m <= 0.0) return vec3(0.0);
    float t = m * (1.0 + m / (peak * peak)) / (1.0 + m);
    return c * (t / m);
}

vec3 bt709_to_2020(vec3 c) {
    return mat3(
        0.6274, 0.0691, 0.0164,
        0.3293, 0.9195, 0.0880,
        0.0433, 0.0114, 0.8956) * c;
}

// Highlights above the screen's peak `dp` rolled off (hue kept, on the largest
// channel): unchanged up to 3/4 of the peak, then a curve that reaches the peak at the
// content's assumed peak `cp` (all relative to SDR white).
vec3 rolloff(vec3 c, float dp, float cp) {
    float m = max(max(c.r, c.g), c.b);
    float k = 0.75 * dp;
    if (cp <= dp || m <= k) return c;
    float b = (cp - k) * (dp - k) / (cp - dp);
    float x = m - k;
    return c * ((k + x / (1.0 + x / b)) / m);
}

// HDR screen: the picture as linear light, Rec.2020, 1.0 = SDR white (the screen's
// SDR white level); HDR in absolute brightness, rolled off to the screen's peak.
vec3 hdr_out(vec3 rgb, float alpha) {
    float sdr = pc.m0.w, peak = pc.m1.w;
    bool b2020 = (pc.mode.w & 1) == 1;
    vec3 lin;
    float cp; // content peak, nits
    if (pc.mode.y == 1 || pc.mode.y == 2) {
        lin = (pc.mode.y == 1 ? pq_to_nits(rgb) : hlg_to_nits(clamp(rgb, 0.0, 1.0))) / sdr;
        if (!b2020) lin = bt709_to_2020(lin);
        cp = 1000.0;
    } else if (pc.mode.y == 3) {
        // Linear (EXR): 1.0 is SDR white, brighter is HDR.
        lin = b2020 ? rgb : bt709_to_2020(rgb);
        cp = 10000.0;
    } else {
        vec3 l = pow(clamp(rgb, 0.0, 1.0), vec3(2.4));
        lin = b2020 ? l : bt709_to_2020(l);
        cp = sdr;
    }
    lin = rolloff(max(lin, 0.0), peak / sdr, cp / sdr);
    if (pc.mode.z == 1) {
        ivec2 cell = ivec2(gl_FragCoord.xy / pc.misc.z);
        float checker = pow(((cell.x + cell.y) & 1) == 0 ? 0.20 : 0.30, 2.2);
        lin = mix(vec3(checker), lin, clamp(alpha, 0.0, 1.0));
    }
    return lin;
}

void main() {
    vec3 rgb;
    float alpha = 1.0;
    if (pc.mode.x == 0) {
        float n = pc.misc.x;
        float y = (texture(t0, v_uv).r * n - pc.off.x) * pc.off.z;
        float u = (texture(t1, v_uv).r * n - pc.off.y) * pc.off.w;
        float v = (texture(t2, v_uv).r * n - pc.off.y) * pc.off.w;
        vec3 yuv = vec3(y, u, v);
        rgb = vec3(dot(pc.m0.xyz, yuv), dot(pc.m1.xyz, yuv), dot(pc.m2.xyz, yuv));
        if (pc.mode.z == 1) alpha = texture(t3, v_uv).r * n;
    } else {
        vec4 s = texture(t0, v_uv);
        rgb = vec3(s[pc.swz.x], s[pc.swz.y], s[pc.swz.z]);
        if (pc.swz.w >= 0) alpha = s[pc.swz.w];
    }

    if ((pc.mode.w & 2) != 0) {
        o_color = vec4(hdr_out(rgb, alpha), 1.0);
        return;
    }

    if (pc.mode.y == 1 || pc.mode.y == 2) {
        vec3 nits = pc.mode.y == 1 ? pq_to_nits(rgb) : hlg_to_nits(clamp(rgb, 0.0, 1.0));
        vec3 lin = nits / SDR_WHITE_NITS;
        if ((pc.mode.w & 1) == 1) lin = bt2020_to_709(lin);
        rgb = pow(clamp(tonemap(max(lin, 0.0), pc.misc.y), 0.0, 1.0), vec3(1.0 / 2.4));
    } else if (pc.mode.y == 3) {
        rgb = pow(clamp(rgb, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    rgb = clamp(rgb, 0.0, 1.0);

    if (pc.mode.z == 1) {
        ivec2 cell = ivec2(gl_FragCoord.xy / pc.misc.z);
        float checker = ((cell.x + cell.y) & 1) == 0 ? 0.20 : 0.30;
        rgb = mix(vec3(checker), rgb, clamp(alpha, 0.0, 1.0));
    }
    o_color = vec4(rgb, 1.0);
}
