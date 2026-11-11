// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Distorção de lente
// @brp pt.param Distortion Distorção | Força principal da distorção da lente. Negativo: almofada; positivo: barril.
// @brp pt.param Aberration Aberração | Aberração espectral. Valores positivos levam o azul para fora, negativos o vermelho.
// @brp pt.param CenterX Centro X | Centro horizontal (eixo X) da distorção.
// @brp pt.param CenterY Centro Y | Centro vertical (eixo Y) da distorção.
// @brp pt.param SpectralSamples Amostras espectrais | Número de amostras da aberração espectral. Mais é mais suave, porém mais pesado.
// @brp pt.param AutoZoom Zoom automático | Se 1, aproxima a imagem para esconder as bordas causadas pela distorção.
//!PARAM Distortion
//!DESC Controls the main lens distortion strength. Negative for pincushion, positive for barrel.
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM Aberration
//!DESC Controls spectral aberration. Positive values shift blue outward, negative values shift red outward.
//!TYPE float
//!MINIMUM -0.5
//!MAXIMUM 0.5
0.02

//!PARAM CenterX
//!DESC Controls the horizontal center (X-axis) of the distortion effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM CenterY
//!DESC Controls the vertical center (Y-axis) of the distortion effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM SpectralSamples
//!DESC Number of samples for the spectral aberration. Higher values are smoother but more expensive.
//!TYPE int
//!MINIMUM 8
//!MAXIMUM 128
32

//!PARAM AutoZoom
//!DESC If 1, automatically zooms the image to hide borders caused by distortion.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1


//!HOOK MAIN
//!BIND HOOKED
//!DESC High-quality lens distortion with smooth spectral aberration and auto-zoom.

// --- Helper Functions ---

// Calculates a spectral weight vector (RGB) for a given point 't' on a simulated spectrum [0,1].
vec3 get_spectral_weight(float t) {
    float w = 0.25; 
    float w2 = 2.0 * w * w;
    float r = exp(-pow(t - 0.0, 2.0) / w2);
    float g = exp(-pow(t - 0.5, 2.0) / w2);
    float b = exp(-pow(t - 1.0, 2.0) / w2);
    return vec3(r, g, b);
}

// Applies lens distortion to a given UV coordinate, including a zoom factor and a center point.
vec2 distort_uv(vec2 uv, float strength, float zoom, vec2 center) {
    vec2 uv_from_center = uv - center;
    
    // Apply zoom by scaling coordinates towards the center. A zoom > 1.0 zooms in.
    uv_from_center /= zoom; 
    
    float r2 = dot(uv_from_center, uv_from_center);
    vec2 distorted_uv_from_center = uv_from_center * (1.0 + strength * r2);
    
    return distorted_uv_from_center + center;
}


vec4 hook() {
    // --- Initial Setup ---
    vec2 uv = HOOKED_pos;
    vec4 original_color = texture(HOOKED_raw, uv);
    vec2 distortion_center = vec2(CenterX, CenterY);
    
    // --- Auto Zoom Calculation ---
    float zoom_factor = 1.0;
    if (AutoZoom == 1u) {
        // Find the distortion strength with the largest effect at the corner.
        float max_dist = Distortion;
        float min_dist = Distortion;
        float corner_strength = abs(max_dist) > abs(min_dist) ? max_dist : min_dist;
        
        // **FIXED ZOOM LOGIC**
        // The logic is now split for positive (barrel) and negative (pincushion) distortion,
        // as they require different compensation strengths.
        if (corner_strength > 0.0) { // Barrel distortion
            // This formula was working well and remains unchanged.
            zoom_factor = 1.0 + pow(corner_strength,.8) * 0.3;
        } else { // Pincushion distortion
            // For negative distortion, the previous zoom was too strong.
            // This uses a gentler multiplier for a more subtle and accurate zoom.
            zoom_factor = 1.0 + pow(abs(corner_strength),2.0) * -0.3;
        }
    }

    // --- Spectral Multi-Sampling Loop ---
    vec3 final_color = vec3(0.0);
    vec3 total_weight = vec3(0.0);
    
    if (SpectralSamples <= 1) {
        vec2 distorted_uv = distort_uv(uv, Distortion, zoom_factor, distortion_center);
        if (distorted_uv.x < 0.0 || distorted_uv.x > 1.0 || distorted_uv.y < 0.0 || distorted_uv.y > 1.0) {
            return vec4(0.0);
        }
        final_color = texture(HOOKED_raw, distorted_uv).rgb;
    } else {
        for (int i = 0; i < SpectralSamples; i++) {
            float t = float(i) / float(SpectralSamples - 1);
            float strength = mix(Distortion - Aberration, Distortion + Aberration, t);
            
            vec2 distorted_uv = distort_uv(uv, strength, zoom_factor, distortion_center);
            
            if (distorted_uv.x < 0.0 || distorted_uv.x > 1.0 || distorted_uv.y < 0.0 || distorted_uv.y > 1.0) {
                continue;
            }

            vec3 sample_color = texture(HOOKED_raw, distorted_uv).rgb;
            vec3 spectral_weight = get_spectral_weight(t);
            
            final_color += sample_color * spectral_weight;
            total_weight += spectral_weight;
        }
        
        final_color /= (total_weight + vec3(1e-6));
    }

    return vec4(clamp(final_color, 0.0, 1.0), original_color.a);
}