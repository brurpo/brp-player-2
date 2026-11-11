// Test effect: changes nothing, working in the "display" colour space (DESIGN.md §7a).
// The chain converts to display before it and back after: the picture must not move.
// @brp space display
//!HOOK MAIN
//!BIND HOOKED
//!DESC pass-through (display)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
