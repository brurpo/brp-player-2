// Colour spaces of the effect working space (DESIGN.md §7a). Shared by the
// renderer's chain (space.comp) and the AI worker's tensor kernel.
//
// Every conversion goes through linear light, Rec.2020 primaries, 1.0 = SDR white
// (203 nits). Spaces:
//   0 source      the file's own encoding and primaries (mpv MAIN)
//   1 linear      linear Rec.2020
//   2 display     the SDR viewer's picture, extended invertibly beyond 0..1
//   3 pq          PQ code values, Rec.2020
//   4 mpv-linear  linear, source primaries (libplacebo LINEAR)
//   5 sigmoid     libplacebo's sigmoidised mpv-linear (of light up to the peak)
// Source description: transfer 0 SDR, 1 PQ, 2 HLG, 3 linear; src2020 = 1 for Rec.2020
// primaries; peak = HDR peak relative to SDR white (1 for SDR and linear sources).

const float BRP_SDR_NITS = 203.0;

// Rec.709 <-> Rec.2020 (ITU-R BT.2087), column-major.
const mat3 BRP_709_TO_2020 = mat3(
    0.6274038959, 0.0690972894, 0.0163914389,
    0.3292830384, 0.9195403951, 0.0880133079,
    0.0433130657, 0.0113623156, 0.8955952532);
const mat3 BRP_2020_TO_709 = mat3(
    1.6604910021, -0.1245504745, -0.0181507634,
   -0.5876411388,  1.1328998971, -0.1005788980,
   -0.0728498633, -0.0083494226,  1.1187296614);

vec3 brp_spow(vec3 x, float g) { return sign(x) * pow(abs(x), vec3(g)); }

// PQ code value <-> linear (relative to SDR white), sign-preserving.
vec3 brp_pq_to_lin(vec3 e) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 p = pow(min(abs(e), 1.0), vec3(1.0 / m2));
    return sign(e) * pow(max(p - c1, 0.0) / (c2 - c3 * p), vec3(1.0 / m1)) * (10000.0 / BRP_SDR_NITS);
}
vec3 brp_lin_to_pq(vec3 l) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 y = pow(min(abs(l) * (BRP_SDR_NITS / 10000.0), 1.0), vec3(m1));
    return sign(l) * pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

// HLG signal <-> display light (relative to SDR white) for a 1000 nit display (system
// gamma 1.2), as the viewer shows HLG. Rec.2020 luma weights for the OOTF.
const vec3 BRP_Y2020 = vec3(0.2627, 0.6780, 0.0593);
vec3 brp_hlg_to_lin(vec3 e) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    vec3 x = abs(e);
    vec3 scene = sign(e) * mix(x * x / 3.0, (exp((x - c) / a) + b) / 12.0, step(0.5, x));
    float ys = dot(scene, BRP_Y2020);
    return scene * pow(max(ys, 1e-6), 0.2) * (1000.0 / BRP_SDR_NITS);
}
vec3 brp_lin_to_hlg(vec3 l) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    vec3 d = l * (BRP_SDR_NITS / 1000.0);
    // Display luma = scene luma ^ 1.2: undo the OOTF (with the forward's floor on the
    // scene luma, 1e-6, which gives display luma 1e-6^1.2).
    float yd = dot(d, BRP_Y2020);
    float ys = yd >= 6.3095734e-8 ? pow(yd, 1.0 / 1.2) : 1e-6;
    vec3 scene = d / pow(ys, 0.2);
    vec3 x = abs(scene);
    return sign(scene) * mix(sqrt(3.0 * x), a * log(max(12.0 * x - b, 1e-6)) + c, step(1.0 / 12.0, x));
}

// The viewer's tone curve (extended Reinhard on the largest channel) and its inverse.
float brp_tone(float m, float peak) { return m * (1.0 + m / (peak * peak)) / (1.0 + m); }
float brp_untone(float t, float peak) {
    float u = 1.0 - t;
    return 2.0 * t / (u + sqrt(u * u + 4.0 * t / (peak * peak)));
}

// libplacebo's sigmoid (centre 0.75, slope 6.5).
const float BRP_SIG_C = 0.75, BRP_SIG_S = 6.5;
vec3 brp_sigmoid(vec3 x) {
    float off = 1.0 / (1.0 + exp(BRP_SIG_S * BRP_SIG_C));
    float scale = 1.0 / (1.0 + exp(BRP_SIG_S * (BRP_SIG_C - 1.0))) - off;
    x = clamp(x, 0.0, 1.0);
    return BRP_SIG_C - log(1.0 / (x * scale + off) - 1.0) / BRP_SIG_S;
}
vec3 brp_unsigmoid(vec3 y) {
    float off = 1.0 / (1.0 + exp(BRP_SIG_S * BRP_SIG_C));
    float scale = 1.0 / (1.0 + exp(BRP_SIG_S * (BRP_SIG_C - 1.0))) - off;
    return (1.0 / (1.0 + exp(BRP_SIG_S * (BRP_SIG_C - y))) - off) / scale;
}

// Light the sigmoid covers (its 0..1): all a PQ file can hold, HLG and SDR up to
// their nominal peak.
float brp_sig_range(int transfer, float peak) { return transfer == 1 ? 10000.0 / BRP_SDR_NITS : max(peak, 1.0); }

// Source primaries <-> Rec.2020.
vec3 brp_src_to_2020(vec3 c, int src2020) { return src2020 == 1 ? c : BRP_709_TO_2020 * c; }
vec3 brp_2020_to_src(vec3 c, int src2020) { return src2020 == 1 ? c : BRP_2020_TO_709 * c; }

vec3 brp_to_linear(int space, vec3 c, int transfer, int src2020, float peak) {
    if (space == 1) return c;
    if (space == 3) return brp_pq_to_lin(c);
    if (space == 0) {
        vec3 l;
        if (transfer == 1) l = brp_pq_to_lin(c);
        else if (transfer == 2) l = brp_hlg_to_lin(c);
        else if (transfer == 3) l = c;
        else l = brp_spow(c, 2.4);
        return brp_src_to_2020(l, src2020);
    }
    if (space == 2) {
        bool hdr = transfer == 1 || transfer == 2;
        vec3 d = brp_spow(c, transfer == 3 ? 2.2 : 2.4);
        if (hdr && peak > 1.0) {
            float t = max(max(d.r, d.g), d.b);
            if (t > 0.0) d *= brp_untone(t, peak) / t;
        }
        return hdr ? BRP_709_TO_2020 * d : brp_src_to_2020(d, src2020);
    }
    // 4 mpv-linear, 5 sigmoid
    vec3 l = space == 5 ? brp_unsigmoid(c) * brp_sig_range(transfer, peak) : c;
    return brp_src_to_2020(l, src2020);
}

vec3 brp_from_linear(int space, vec3 l, int transfer, int src2020, float peak) {
    if (space == 1) return l;
    if (space == 3) return brp_lin_to_pq(l);
    if (space == 0) {
        vec3 c = brp_2020_to_src(l, src2020);
        if (transfer == 1) return brp_lin_to_pq(c);
        if (transfer == 2) return brp_lin_to_hlg(c);
        if (transfer == 3) return c;
        return brp_spow(c, 1.0 / 2.4);
    }
    if (space == 2) {
        bool hdr = transfer == 1 || transfer == 2;
        vec3 d = hdr ? BRP_2020_TO_709 * l : brp_2020_to_src(l, src2020);
        if (hdr && peak > 1.0) {
            float m = max(max(d.r, d.g), d.b);
            if (m > 0.0) d *= brp_tone(m, peak) / m;
        }
        return brp_spow(d, transfer == 3 ? 1.0 / 2.2 : 1.0 / 2.4);
    }
    vec3 c = brp_2020_to_src(l, src2020);
    return space == 5 ? brp_sigmoid(c / brp_sig_range(transfer, peak)) : c;
}

vec3 brp_convert(int from, int to, vec3 c, int transfer, int src2020, float peak) {
    if (from == to) return c;
    return brp_from_linear(to, brp_to_linear(from, c, transfer, src2020, peak), transfer, src2020, peak);
}
