#version 450

// Draws the video quad. The rectangle (in NDC) comes from push constants.
layout(push_constant) uniform PC {
    vec4 rect;      // x0, y0, x1, y1 in NDC
    vec4 m0;
    vec4 m1;
    vec4 m2;
    vec4 off;
    vec4 misc;
    ivec4 mode;
    ivec4 swz;
} pc;

layout(location = 0) out vec2 v_uv;

void main() {
    vec2 c = vec2(float(gl_VertexIndex & 1), float((gl_VertexIndex >> 1) & 1));
    // misc.w = clockwise quarter turns for display (phone footage).
    int q = int(pc.misc.w + 0.5) & 3;
    if (q == 1) v_uv = vec2(c.y, 1.0 - c.x);
    else if (q == 2) v_uv = vec2(1.0 - c.x, 1.0 - c.y);
    else if (q == 3) v_uv = vec2(1.0 - c.y, c.x);
    else v_uv = c;
    gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, c), 0.0, 1.0);
}
