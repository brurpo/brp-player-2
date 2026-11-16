// Test effect: a 5x5 binomial blur in linear light (Rec.2020): light adds up, so the
// result depends on which light the chain starts from (OCIO's or the player's).
// @brp space linear
//!HOOK MAIN
//!BIND HOOKED
//!DESC linear blur
vec4 hook() {
    const float k[5] = float[5](1.0, 4.0, 6.0, 4.0, 1.0);
    vec4 s = vec4(0.0);
    for (int y = -2; y <= 2; y++)
        for (int x = -2; x <= 2; x++)
            s += k[x + 2] * k[y + 2] * HOOKED_texOff(vec2(x, y));
    return s / 256.0;
}
