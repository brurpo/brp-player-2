// Test effect: changes nothing, working in the "linear" colour space (DESIGN.md §7a).
// The chain converts to linear before it and back after: the picture must not move.
// @brp space linear
//!HOOK MAIN
//!BIND HOOKED
//!DESC pass-through (linear)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
