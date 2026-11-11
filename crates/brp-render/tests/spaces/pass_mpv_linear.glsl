// Test effect: changes nothing, hooking libplacebo's LINEAR stage.
//!HOOK LINEAR
//!BIND HOOKED
//!DESC pass-through (LINEAR)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
