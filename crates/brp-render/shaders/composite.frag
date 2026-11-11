#version 450
// HDR screens (DESIGN.md §7b): the video (scene: linear light, Rec.2020, 1.0 = SDR
// white) under the UI (egui as for an SDR window: sRGB, premultiplied alpha, on a
// transparent layer).
//  mode.x: 0 = HDR10 out (PQ, Rec.2020), 1 = SDR out (sRGB; the popups' backdrop)
//  f.x: SDR white (nits)
// The UI blends as on an SDR window: in sRGB, over the video's SDR part. Light above
// SDR white is added back, dimmed by the UI's coverage, so HDR highlights stay HDR
// under translucent panels and the look of the UI does not change.

layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D ui;
layout(push_constant) uniform PC {
    ivec4 mode;
    vec4 f;
} pc;
layout(location = 0) out vec4 o_color;

vec3 srgb_encode(vec3 l) {
    l = clamp(l, 0.0, 1.0);
    return mix(l * 12.92, 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, l));
}
vec3 srgb_decode(vec3 e) {
    return mix(e / 12.92, pow((e + 0.055) / 1.055, vec3(2.4)), step(0.04045, e));
}
vec3 pq_encode(vec3 nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375;
    const float c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}
const mat3 M2020_TO_709 = mat3(
    1.6604910021, -0.1245504745, -0.0181507634,
   -0.5876411388,  1.1328998971, -0.1005788980,
   -0.0728498633, -0.0083494226,  1.1187296614);
const mat3 M709_TO_2020 = mat3(
    0.6274038959, 0.0690972894, 0.0163914389,
    0.3292830384, 0.9195403951, 0.0880133079,
    0.0433130657, 0.0113623156, 0.8955952532);

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 s = M2020_TO_709 * texelFetch(scene, p, 0).rgb;
    vec4 u = texelFetch(ui, p, 0);
    vec3 sdr = clamp(s, 0.0, 1.0);
    vec3 g = u.rgb + (1.0 - u.a) * srgb_encode(sdr);
    if (pc.mode.x == 1) {
        o_color = vec4(clamp(g, 0.0, 1.0), 1.0);
        return;
    }
    vec3 lin = srgb_decode(clamp(g, 0.0, 1.0)) + (1.0 - u.a) * (s - sdr);
    o_color = vec4(pq_encode(max(M709_TO_2020 * lin, 0.0) * pc.f.x), 1.0);
}
