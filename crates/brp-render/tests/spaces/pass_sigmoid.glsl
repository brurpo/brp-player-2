// Test effect: changes nothing, hooking libplacebo's SIGMOID stage.
//!HOOK SIGMOID
//!BIND HOOKED
//!DESC pass-through (SIGMOID)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
