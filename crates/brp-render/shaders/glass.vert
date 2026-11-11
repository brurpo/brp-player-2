#version 450

// One triangle covering the whole popup window.
void main() {
    vec2 c = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0);
}
