#version 450

// Glass background of a popup window. Where the popup lies over the player window,
// the player's own picture (a blurred eighth-size copy the main renderer keeps) is
// sampled: lightly blurred, bent near the rounded border like the rim of a lens, with
// a little colour fringing and a specular rim. Elsewhere the pixel stays translucent
// (premultiplied alpha) so the system backdrop (Windows acrylic) blurs the desktop.

layout(location = 0) out vec4 o;
layout(binding = 0) uniform sampler2D backdrop;

layout(push_constant) uniform PC {
    vec4 place;   // popup client origin in player-window pixels (xy), player size (zw)
    vec4 shape;   // popup size in pixels (xy), corner radius px (z), pixels per point (w)
    vec4 tint;    // straight colour (rgb) and amount (a)
    vec4 misc;    // backdrop on (x), backdrop texel size in uv (yz), refraction px (w)
} pc;

float sd_round_rect(vec2 p, vec2 half_size, float r) {
    vec2 q = abs(p) - half_size + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Outward normal of the rounded rectangle at p (centre-relative).
vec2 rim_normal(vec2 p, vec2 half_size, float r) {
    vec2 q = abs(p) - half_size + r;
    vec2 s = vec2(p.x < 0.0 ? -1.0 : 1.0, p.y < 0.0 ? -1.0 : 1.0);
    if (q.x > 0.0 && q.y > 0.0) return normalize(q) * s;
    return q.x > q.y ? vec2(s.x, 0.0) : vec2(0.0, s.y);
}

// Cubic B-spline read of the 1/8-size copy from 4 bilinear reads (Sigg & Hadwiger).
// Plain bilinear magnified 8x shows the texel grid as blocks and diamonds; the
// B-spline is smooth across texels, so no grid shows.
vec3 bspline(vec2 uv) {
    vec2 texel = pc.misc.yz;
    vec2 p = uv / texel - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1;
    vec2 g1 = w2 + w3;
    vec2 h0 = (i - 0.5 + w1 / g0) * texel;
    vec2 h1 = (i + 1.5 + w3 / g1) * texel;
    return g0.y * (g0.x * texture(backdrop, h0).rgb + g1.x * texture(backdrop, vec2(h1.x, h0.y)).rgb)
         + g1.y * (g0.x * texture(backdrop, vec2(h0.x, h1.y)).rgb + g1.x * texture(backdrop, h1).rgb);
}

// The blur: five B-spline reads in a small tent (centre and four diagonals). With the
// box-filtered 1/8 copy that is a smooth, roughly Gaussian blur of ~8 px at full
// resolution, for 20 texture reads.
vec3 blurred(vec2 px) {
    vec2 uv = px / pc.place.zw;
    vec2 d = pc.misc.yz * 0.9;
    vec3 s = bspline(uv) * 0.4;
    s += (bspline(uv + d) + bspline(uv - d) + bspline(uv + vec2(d.x, -d.y)) + bspline(uv + vec2(-d.x, d.y))) * 0.15;
    return s;
}

void main() {
    vec2 p = gl_FragCoord.xy;
    vec2 size = pc.shape.xy;
    float r = pc.shape.z;
    float ppp = pc.shape.w;
    vec2 c = p - size * 0.5;
    float d = sd_round_rect(c, size * 0.5, r);
    // Anti-aliased shape (the system also rounds the window corners).
    float inside = clamp(0.5 - d, 0.0, 1.0);
    if (inside <= 0.0) {
        o = vec4(0.0);
        return;
    }
    vec2 n = rim_normal(c, size * 0.5, r);
    // 0 in the middle, 1 at the border, over a bevel of ~16 points.
    float bevel = 16.0 * ppp;
    float t = clamp(1.0 + d / bevel, 0.0, 1.0);
    float bend = t * t * t;

    // Behind the glass: the player's picture where the popup overlaps it.
    vec2 at = pc.place.xy + p;
    vec2 rel = at / pc.place.zw;
    vec2 edge = min(rel, 1.0 - rel) * pc.place.zw;
    float cover = pc.misc.x * clamp(min(edge.x, edge.y) + 0.5, 0.0, 1.0);

    vec3 glass = vec3(0.0);
    if (cover > 0.0) {
        // Rim: sample from further in, so the picture swells toward the border.
        vec2 off = -n * bend * pc.misc.w;
        if (bend > 0.002) {
            // A little dispersion: each channel bends slightly differently.
            glass = vec3(blurred(at + off * 1.12).r, blurred(at + off).g, blurred(at + off * 0.88).b);
        } else {
            glass = blurred(at);
        }
        // Frosted: lift the darks, keep the colour.
        glass = mix(glass, pc.tint.rgb, pc.tint.a);
    }
    // Premultiplied: opaque glass over the player, translucent tint elsewhere.
    // Outside the player the system backdrop may not blur (inactive window): a
    // denser tint keeps the text readable there.
    float outside = max(pc.tint.a, 0.74);
    float a_out = mix(outside, 1.0, cover);
    vec3 rgb = glass * cover + pc.tint.rgb * outside * (1.0 - cover);

    // Light: a specular rim brighter toward the top-left, and a soft top sheen.
    vec2 light = normalize(vec2(-0.55, -0.85));
    float spec = pow(t, 6.0) * (0.25 + 0.75 * max(dot(n, light), 0.0)) * 0.38;
    float sheen = (1.0 - clamp(p.y / (size.y * 0.45), 0.0, 1.0)) * 0.035;
    float h = spec + sheen;
    rgb += vec3(h);
    a_out = a_out + h * (1.0 - a_out);

    o = vec4(rgb, a_out) * inside;
}
