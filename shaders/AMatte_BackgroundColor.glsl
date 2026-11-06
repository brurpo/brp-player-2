//!PARAM ColorR
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM ColorG
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM ColorB
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0


//!HOOK MAIN
//!BIND HOOKED
//!DESC High-Quality Chroma Key Shader with Spill Suppressor
vec4 hook() { 
    vec4 col = textureLod(HOOKED_raw, HOOKED_pos, 0.0);
    col.rgb = mix(col.rgb, vec3(ColorR,ColorG,ColorB),1.0-col.a);

    
    return vec4(col.rgb,1.0); }

