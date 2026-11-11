// Test effect: exposure in linear light (GAIN times the light), Rec.2020.
// HDR highlights must scale, not clip at SDR white.
// @brp space linear
//!PARAM GAIN
//!DESC Light multiplied by this
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 16.0
0.25
//!HOOK MAIN
//!BIND HOOKED
//!DESC linear gain
vec4 hook() { vec4 c = HOOKED_tex(HOOKED_pos); return vec4(c.rgb * GAIN, c.a); }
