// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Pintura
// @brp pt.param SmoothingRadius Raio de suavização | Tamanho da suavização. Valores maiores dão um visual mais abstrato, de pintura.
// @brp pt.param EdgeSharpness Nitidez das bordas | Anisotropia (alongamento) do filtro ao longo das bordas. Valores maiores respeitam mais as bordas.
// @brp pt.param EffectMix Mistura | Mistura a imagem filtrada com a original. 0 = original, 1 = efeito completo.
//!PARAM SmoothingRadius
//!DESC Controls the size of the smoothing effect. Higher values create a more abstract, painterly look.
//!TYPE int
//!MINIMUM 2
//!MAXIMUM 15
5

//!PARAM EdgeSharpness
//!DESC Controls the anisotropy (elongation) of the filter along edges. Higher values respect edges more.
//!TYPE float
//!MINIMUM 1.0
//!MAXIMUM 12.0
8.0

//!PARAM EffectMix
//!DESC Blends the filtered image with the original. 0.0 is original image, 1.0 is full effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!HOOK MAIN
//!BIND HOOKED
//!DESC Anisotropic Kuwahara, an edge-preserving painterly filter.

// This is a computationally intensive filter. Lowering the SmoothingRadius is the best way to improve performance.
// Based on the work by Kyprianidis, J. E., Kang, H., & Döllner, J. (2009).

const float PI = 3.14159265359;
const int SECTORS = 8; // Number of angular sectors to sample

vec4 hook() {
    vec2 uv = HOOKED_pos;
    vec2 pt = HOOKED_pt;
    vec3 original_color = texture(HOOKED_raw, uv).rgb;

    // If mix is zero, skip all calculations
    if (EffectMix == 0.0) {
        return vec4(original_color, 1.0);
    }

    // --- 1. Estimate Local Orientation (Structure Tensor) ---
    // We use image gradients (Sobel operator) to find the dominant local orientation.
    mat3x3 sobel_x = mat3x3(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
    mat3x3 sobel_y = mat3x3(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    vec3 Gx = vec3(0.0), Gy = vec3(0.0);

    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec3 Sample = texture(HOOKED_raw, uv + vec2(i, j) * pt).rgb;
            Gx += Sample * sobel_x[i+1][j+1];
            Gy += Sample * sobel_y[i+1][j+1];
        }
    }

    // Convert gradients to a single luminance value
    float lum_gx = dot(Gx, vec3(0.299, 0.587, 0.114));
    float lum_gy = dot(Gy, vec3(0.299, 0.587, 0.114));

    // The tangent direction is perpendicular to the gradient direction
    vec2 tangent = normalize(vec2(-lum_gy, lum_gx));
    
    // --- 2. Anisotropic Kuwahara Filtering ---
    vec3 means[SECTORS];
    float variances[SECTORS];
    
    float radius = float(SmoothingRadius);
    float alpha = EdgeSharpness; // Renaming for clarity from the paper's 'alpha' parameter

    for (int i = 0; i < SECTORS; i++) {
        vec3 sum_colors = vec3(0.0);
        vec3 sum_sq_colors = vec3(0.0);
        float count = 0.0;
        
        float sector_angle = float(i) * 2.0 * PI / float(SECTORS);
        vec2 sector_dir = vec2(cos(sector_angle), sin(sector_angle));
        
        for (int x = -SmoothingRadius; x <= SmoothingRadius; x++) {
            for (int y = -SmoothingRadius; y <= SmoothingRadius; y++) {
                vec2 offset = vec2(x, y);
                
                // Rotate the sampling offset to align with the local edge orientation
                mat2 rot_matrix = mat2(tangent.x, -tangent.y, tangent.y, tangent.x);
                offset = rot_matrix * offset;
                
                // Weight the sample based on its position within the anisotropic kernel
                float d = dot(offset, sector_dir);
                if (d > 0.0) {
                    float len_sq = dot(offset, offset);
                    float gaussian_weight = exp(-len_sq / (2.0 * (radius / 2.0) * (radius / 2.0)));
                    float anisotropy_weight = exp(-pow(dot(offset, tangent), 2.0) / alpha);
                    float weight = gaussian_weight * anisotropy_weight;
                    
                    vec3 sample_color = texture(HOOKED_raw, uv + offset * pt).rgb;
                    
                    sum_colors += sample_color * weight;
                    sum_sq_colors += sample_color * sample_color * weight;
                    count += weight;
                }
            }
        }
        
        if (count == 0.0) {
            means[i] = original_color;
            variances[i] = 1e6; // A large number to avoid selection
            continue;
        }

        vec3 mean = sum_colors / count;
        means[i] = mean;
        vec3 variance_vec = (sum_sq_colors / count) - (mean * mean);
        variances[i] = dot(variance_vec, vec3(1.0)); // Total variance
    }
    
    // --- 3. Find Sector with Minimum Variance ---
    float min_variance = variances[0];
    vec3 final_color = means[0];
    for (int i = 1; i < SECTORS; i++) {
        if (variances[i] < min_variance) {
            min_variance = variances[i];
            final_color = means[i];
        }
    }
    
    // --- 4. Mix and Output ---
    final_color = mix(original_color, final_color, EffectMix);
    
    return vec4(final_color, 1.0);
}