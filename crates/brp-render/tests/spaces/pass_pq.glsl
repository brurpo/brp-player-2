// Test effect: changes nothing, working in the "pq" colour space (DESIGN.md §7a).
// The chain converts to pq before it and back after: the picture must not move.
// @brp space pq
//!HOOK MAIN
//!BIND HOOKED
//!DESC pass-through (pq)
vec4 hook() { return HOOKED_tex(HOOKED_pos); }
