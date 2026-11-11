// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Reflexo de lente
// @brp pt.param Threshold Limiar | Brilho a partir do qual surge o reflexo.
// @brp pt.param Intensity Intensidade | Intensidade geral de todos os reflexos.
// @brp pt.param Anamorphic Anamórfico | Estiramento horizontal, para um visual anamórfico.
// @brp pt.param GlintPoints Pontas da estrela | Número de pontas do efeito estrela.
// @brp pt.param GlintSize Tamanho da estrela | Comprimento dos raios da estrela.
// @brp pt.param GlintDispersion Dispersão da estrela | Aberração cromática ao longo dos raios.
// @brp pt.param GhostCount Fantasmas | Número de fantasmas da íris.
// @brp pt.param GhostBlades Lâminas | Número de lâminas do diafragma na forma dos fantasmas. 0 = circular.
// @brp pt.param GhostDispersion Dispersão dos fantasmas | Aberração cromática dos fantasmas.
// @brp pt.param GlareSize Tamanho do clarão | Tamanho do clarão suave (bloom).
// @brp pt.param DirtIntensity Sujeira da lente | Intensidade da sujeira na lente. Precisa de uma textura de sujeira.
//!PARAM Threshold
//!DESC Brightness threshold to trigger the lens flare.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 1.0
0.95

//!PARAM Intensity
//!DESC Overall intensity of all lens flare effects.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.5

//!PARAM Anamorphic
//!DESC Amount of horizontal stretch for an anamorphic look.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM GlintPoints
//!DESC Number of points for the starburst effect.
//!TYPE int
//!MINIMUM 3
//!MAXIMUM 20
8

//!PARAM GlintSize
//!DESC The length of the starburst streaks.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.5
0.2

//!PARAM GlintDispersion
//!DESC Amount of chromatic aberration along the streaks.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.1
0.03


//!PARAM GhostCount
//!DESC Number of iris ghosts to generate.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 10
4

//!PARAM GhostBlades
//!DESC Number of aperture blades for the ghost shape. 0 is circular.
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 12
6

//!PARAM GhostDispersion
//!DESC Amount of chromatic aberration for the ghosts.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.2
0.05


//!PARAM GlareSize
//!DESC Size of the soft glare/bloom effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM DirtIntensity
//!DESC Intensity of the lens dirt effect. Requires a dirt texture.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
1.0



//!HOOK MAIN
//!BIND HOOKED
//!SAVE PASS_BRIGHT
//!DESC Lens Flare - Pass 1: Isolate Bright Areas

vec4 hook() {
    vec3 color = texture(HOOKED_raw, HOOKED_pos).rgb;
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float bright = smoothstep(Threshold - 0.1, Threshold, luma);
    return vec4(color * bright, 1.0);
}
//======================================================================================
// PASS 1: Threshold - Isolate bright spots
// We downsample the output of this pass for much better performance on subsequent steps.
//======================================================================================

//======================================================================================
// PASS 2: Flare Generation (Glint, Ghosts, Glare)
// This pass works on the small downsampled texture to calculate all flare elements.
//======================================================================================
//!HOOK MAIN
//!BIND HOOKED
//!BIND PASS_BRIGHT
//!SAVE PASS_FLARE
//!DESC Lens Flare - Pass 2: Generate Flare Elements

// Helper function for signed distance to a regular polygon
float sdf_polygon(vec2 p, int N) {
    float a = atan(p.x, p.y) + 0.5;
    float r = length(p);
    float pi = 3.1415926535;
    float v = 2.0 * pi / float(N);
    return cos(floor(0.5 + a / v) * v - a) * r;
}

vec4 hook() {
    vec2 uv = HOOKED_pos;
    vec2 center = vec2(0.5);
    vec2 pos_from_center = uv - center;
    
    vec3 flare_color = vec3(0.0);
    
    // --- 1. Main Glare/Bloom Calculation ---
    // A simple (but slow) wide blur on the bright pass
    int glare_samples = 16;
    for (int i = 0; i < glare_samples; i++) {
        float angle = float(i) / float(glare_samples) * 6.28318;
        vec2 offset = vec2(cos(angle), sin(angle)) * GlareSize * 0.05;
        flare_color += texture(PASS_BRIGHT_raw, uv + offset).rgb;
    }
    flare_color /= float(glare_samples);

    // --- 2. Iris Ghost Calculation ---
    for (int i = 0; i < GhostCount; i++) {
        float dist_factor = (fract(float(i) * 1.61803398875) * 2.0 - 1.0) * 0.8;
        vec2 ghost_uv = center + pos_from_center * dist_factor;
        
        vec3 ghost_color = texture(PASS_BRIGHT_raw, ghost_uv).rgb;

        // Shape the ghost using an SDF
        vec2 shape_uv = (uv - center) / (0.1 + fract(float(i) * 2.71828)) * 5.0;
        float shape = 0.0;
        if (GhostBlades < 3) { // Circular
            shape = 1.0 - smoothstep(0.9, 1.0, length(shape_uv));
        } else { // Polygonal
            shape = 1.0 - smoothstep(0.9, 1.0, sdf_polygon(shape_uv, GhostBlades));
        }
        
        // Add chromatic aberration and accumulate
        float t = fract(float(i) * 0.31415); // Interpolation factor for color
        float strength = mix(-GhostDispersion, GhostDispersion, t);
        flare_color.r += ghost_color.r * shape * (1.0 + strength);
        flare_color.g += ghost_color.g * shape;
        flare_color.b += ghost_color.b * shape * (1.0 - strength);
    }

    // --- 3. Glint / Starburst Calculation ---
    float star_angle_step = 3.14159 / float(GlintPoints);
    for (int i = 0; i < GlintPoints; i++) {
        float angle = float(i) * star_angle_step;
        vec2 dir = vec2(cos(angle), sin(angle));
        
        for (int j = 1; j < 12; j++) {
            float dist = float(j*j) * 0.001 * GlintSize;
            vec3 sample_color = texture(PASS_BRIGHT_raw, uv - dir * dist).rgb;
            
            // Add aberration to the streak
            float strength = mix(-GlintDispersion, GlintDispersion, fract(angle / 3.14159));
            flare_color.r += sample_color.r * (1.0 - float(j)/12.0) * 0.2 * (1.0 + strength);
            flare_color.g += sample_color.g * (1.0 - float(j)/12.0) * 0.2;
            flare_color.b += sample_color.b * (1.0 - float(j)/12.0) * 0.2 * (1.0 - strength);
        }
    }
    
    return vec4(flare_color, 1.0);
}


//======================================================================================
// PASS 3: Final Composite
// Upsamples the flare, adds dirt, and composites over the original image.
//======================================================================================
//!HOOK MAIN
//!BIND HOOKED
//!BIND PASS_FLARE
//!DESC Lens Flare - Final Pass: Composite

vec4 hook() {
    vec4 original_color = texture(HOOKED_raw, HOOKED_pos);
    vec2 uv = HOOKED_pos;
    vec2 center = vec2(0.5);

    // Apply anamorphic stretch by modifying UVs for flare textures
    vec2 flare_uv = uv;
    flare_uv.x = (uv.x - center.x) / (1.0 + Anamorphic) + center.x;
    
    // Sample the flare texture (which is small) with the stretched UVs
    vec3 flare_color = texture(PASS_FLARE_raw, flare_uv).rgb;

    // Add lens dirt, illuminated by the flare
    //vec3 dirt_color = texture(DIRT_TEX_raw, uv).rgb * flare_color * DirtIntensity;

    // Additive blend of all components
    vec3 final_color = original_color.rgb + (flare_color ) * Intensity;
    
    return vec4(clamp(final_color, 0.0, 10.0), original_color.a);
}