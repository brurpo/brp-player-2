// Test effect: changes nothing, in the source's own encoding (mpv MAIN): no conversion.
//!HOOK MAIN
//!BIND HOOKED
//!DESC pass-through (source)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
