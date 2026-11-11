// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Inverter
// @brp pt.param intensity Intensidade | Quanto inverter (0 = original, 1 = totalmente invertido).
//!PARAM intensity
//!DESC Amount of inversion (0 = original, 1 = fully inverted)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!HOOK MAIN
//!BIND HOOKED
//!DESC Invert

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);
    return vec4(mix(c.rgb, vec3(1.0) - c.rgb, intensity), c.a);
}
