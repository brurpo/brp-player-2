//!PARAM SharpenStrength
//!DESC Amount of sharpening to apply. 0.0 means no effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 3.0
0.0

//!PARAM SharpenRadius
//!DESC Radius of the blur used for the unsharp mask. Larger values affect broader edges.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 5.0
1.0

//!PARAM Exposure
//!DESC Multiplies the brightness of the image. 1.0 is no change.
//!TYPE float
//!MINIMUM -10.0
//!MAXIMUM 10.0
0.0

//!PARAM Temperature
//!DESC Adjusts the color temperature. < 0 is cooler (blue), > 0 is warmer (orange).
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM Saturation
//!DESC Adjusts the overall color intensity. 0.0 is grayscale, 1.0 is original.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.5
1.0

//!PARAM Vibrance
//!DESC Intelligently adjusts saturation, protecting skin tones.
//!TYPE float
//!MINIMUM -1.5
//!MAXIMUM 1.5
0.0

//!PARAM Gamma
//!DESC Adjusts the mid-tones for the overall image.
//!TYPE float
//!MINIMUM 0.2
//!MAXIMUM 2.5
1.0

//!PARAM GammaRed
//!DESC Adjusts the mid-tones for the Red channel.
//!TYPE float
//!MINIMUM 0.2
//!MAXIMUM 2.5
1.0

//!PARAM GammaGreen
//!DESC Adjusts the mid-tones for the Green channel.
//!TYPE float
//!MINIMUM 0.2
//!MAXIMUM 2.5
1.0

//!PARAM GammaBlue
//!DESC Adjusts the mid-tones for the Blue channel.
//!TYPE float
//!MINIMUM 0.2
//!MAXIMUM 2.5
1.0

//!PARAM VignetteStrength
//!DESC Adjusts the vignette. < 0 darkens edges, > 0 lightens edges.
//!TYPE float
//!MINIMUM -2.0
//!MAXIMUM 2.0
0.0

//!PARAM VignetteRadius
//!DESC The inner radius of the vignette, before it starts to fade.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM VignetteSoftness
//!DESC How soft the transition of the vignette is.
//!TYPE float
//!MINIMUM 0.01
//!MAXIMUM 1.5
0.4

//!PARAM LinearizeInput
//!DESC How soft the transition of the vignette is.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM DeLinearizeOutput
//!DESC How soft the transition of the vignette is.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1


//!HOOK MAIN
//!BIND HOOKED
//!SAVE PASS_LIN
//!DESC Unsharp Mask - Horizontal Blur Pass

vec3 inverse_tonemap_filmic(vec3 ldr_linear, float exposure) {
    // **CORREÇÃO:** Adicionamos um clamp para garantir que a entrada para o log() seja sempre > 0.
    // Isso previne a geração de valores NaN ("buracos negros") se a cor de entrada for >= 1.0.
    vec3 clamped_ldr = clamp(ldr_linear, 0.0, 0.9999);
    
    // Usa a função logarítmica, que é a inversa da exponencial.
    
    return -log(1.0 - clamped_ldr) / exposure;
}
vec3 srgb_to_acescg(vec3 srgb) {
    // 1. Linearização do sRGB
    vec3 linear_ldr;
    linear_ldr.r = (srgb.r <= 0.04045) ? srgb.r / 12.92 : pow((srgb.r + 0.055) / 1.055, 2.4);
    linear_ldr.g = (srgb.g <= 0.04045) ? srgb.g / 12.92 : pow((srgb.g + 0.055) / 1.055, 2.4);
    linear_ldr.b = (srgb.b <= 0.04045) ? srgb.b / 12.92 : pow((srgb.b + 0.055) / 1.055, 2.4);

    // 2. Expansão de Alcance com Mapeamento de Tom Inverso
    float exposure = 1.0; // Este valor pode ser exposto como um slider
    vec3 linear_hdr = inverse_tonemap_filmic(linear_ldr, exposure);

    // 3. Conversão de Gamut (Rec.709 para ACEScg AP1)
    const mat3 srgb_to_acescg_matrix = mat3(
        0.6165560, 0.3400713, 0.0433727,
        0.0772421, 0.9161245, 0.0066334,
        0.0157941, 0.1094681, 0.8747378
    );
    
    return linear_hdr * srgb_to_acescg_matrix;
}

vec4 hook() {
    vec4 ocol = texture(HOOKED_raw, HOOKED_pos);
    if (LinearizeInput==1u){
    ocol.rgb = clamp(srgb_to_acescg(ocol.rgb),vec3(0.0),vec3(50.0));
    }
    
    return ocol;


}


//!HOOK MAIN
//!BIND HOOKED
//!SAVE PASS_H
//!DESC Unsharp Mask - Horizontal Blur Pass

// Universal Color Correction and Grading Shader
// Combines Sharpen, Color Temperature, Exposure, Gamma, Saturation, Vibrance, and Vignette.

vec4 hook() {
    float radius = SharpenRadius;
    vec2 pt = HOOKED_pt;
    vec4 sum = vec4(0.0);

    // 9-tap Gaussian blur kernel
    sum += texture(HOOKED_raw, HOOKED_pos - 4.0 * pt * vec2(radius, 0.0)) * 0.0162162162;
    sum += texture(HOOKED_raw, HOOKED_pos - 3.0 * pt * vec2(radius, 0.0)) * 0.0540540541;
    sum += texture(HOOKED_raw, HOOKED_pos - 2.0 * pt * vec2(radius, 0.0)) * 0.1216216216;
    sum += texture(HOOKED_raw, HOOKED_pos - 1.0 * pt * vec2(radius, 0.0)) * 0.1945945946;
    sum += texture(HOOKED_raw, HOOKED_pos)                            * 0.2270270270;
    sum += texture(HOOKED_raw, HOOKED_pos + 1.0 * pt * vec2(radius, 0.0)) * 0.1945945946;
    sum += texture(HOOKED_raw, HOOKED_pos + 2.0 * pt * vec2(radius, 0.0)) * 0.1216216216;
    sum += texture(HOOKED_raw, HOOKED_pos + 3.0 * pt * vec2(radius, 0.0)) * 0.0540540541;
    sum += texture(HOOKED_raw, HOOKED_pos + 4.0 * pt * vec2(radius, 0.0)) * 0.0162162162;

    return sum;
}

//!HOOK MAIN
//!BIND PASS_H
//!SAVE PASS_BLUR
//!DESC Unsharp Mask - Vertical Blur Pass

vec4 hook() {
    float radius = SharpenRadius;
    vec2 pt = PASS_H_pt;
    vec4 sum = vec4(0.0);
    
    // 9-tap Gaussian blur kernel
    sum += texture(PASS_H_raw, PASS_H_pos - 4.0 * pt * vec2(0.0, radius)) * 0.0162162162;
    sum += texture(PASS_H_raw, PASS_H_pos - 3.0 * pt * vec2(0.0, radius)) * 0.0540540541;
    sum += texture(PASS_H_raw, PASS_H_pos - 2.0 * pt * vec2(0.0, radius)) * 0.1216216216;
    sum += texture(PASS_H_raw, PASS_H_pos - 1.0 * pt * vec2(0.0, radius)) * 0.1945945946;
    sum += texture(PASS_H_raw, PASS_H_pos)                             * 0.2270270270;
    sum += texture(PASS_H_raw, PASS_H_pos + 1.0 * pt * vec2(0.0, radius)) * 0.1945945946;
    sum += texture(PASS_H_raw, PASS_H_pos + 2.0 * pt * vec2(0.0, radius)) * 0.1216216216;
    sum += texture(PASS_H_raw, PASS_H_pos + 3.0 * pt * vec2(0.0, radius)) * 0.0540540541;
    sum += texture(PASS_H_raw, PASS_H_pos + 4.0 * pt * vec2(0.0, radius)) * 0.0162162162;

    return sum;
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND PASS_LIN
//!BIND PASS_BLUR
//!DESC Applies all color adjustments and composites the final image.

// Helper function to calculate luminance
float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}
vec3 tonemap_filmic(vec3 hdr_linear, float exposure) {
    // Aplica a exposição e a curva exponencial
    return vec3(1.0) - exp(-hdr_linear * exposure);
}

vec3 acescg_to_srgb(vec3 acescg) {
    // 1. Conversão de Gamut (ACEScg AP1 para Rec.709)
    const mat3 acescg_to_srgb_matrix = mat3(
        1.6412204, -0.6015132, -0.0397072,
       -0.1242013,  1.1329241, -0.0087228,
       -0.0178373, -0.1293393,  1.1471766
    );

    vec3 linear_hdr = acescg * acescg_to_srgb_matrix;

    // 2. Mapeamento de Tom (Compressão de HDR para LDR)
    float exposure = 1.0; // Este valor deve ser o mesmo usado na função inversa
    vec3 linear_ldr = tonemap_filmic(linear_hdr, exposure);

    // 3. Aplicação da Curva Gama sRGB
    vec3 srgb;
    srgb.r = (linear_ldr.r <= 0.0031308) ? linear_ldr.r * 12.92 : 1.055 * pow(linear_ldr.r, 1.0/2.4) - 0.055;
    srgb.g = (linear_ldr.g <= 0.0031308) ? linear_ldr.g * 12.92 : 1.055 * pow(linear_ldr.g, 1.0/2.4) - 0.055;
    srgb.b = (linear_ldr.b <= 0.0031308) ? linear_ldr.b * 12.92 : 1.055 * pow(linear_ldr.b, 1.0/2.4) - 0.055;
    
    return clamp(srgb, 0.0, 1.0);
}

vec3 inverse_tonemap_filmic(vec3 ldr_linear, float exposure) {
    // **CORREÇÃO:** Adicionamos um clamp para garantir que a entrada para o log() seja sempre > 0.
    // Isso previne a geração de valores NaN ("buracos negros") se a cor de entrada for >= 1.0.
    vec3 clamped_ldr = clamp(ldr_linear, 0.0, 0.9999);
    
    // Usa a função logarítmica, que é a inversa da exponencial.
    
    return -log(1.0 - clamped_ldr) / exposure;
}
vec3 srgb_to_acescg(vec3 srgb) {
    // 1. Linearização do sRGB
    vec3 linear_ldr;
    linear_ldr.r = (srgb.r <= 0.04045) ? srgb.r / 12.92 : pow((srgb.r + 0.055) / 1.055, 2.4);
    linear_ldr.g = (srgb.g <= 0.04045) ? srgb.g / 12.92 : pow((srgb.g + 0.055) / 1.055, 2.4);
    linear_ldr.b = (srgb.b <= 0.04045) ? srgb.b / 12.92 : pow((srgb.b + 0.055) / 1.055, 2.4);

    // 2. Expansão de Alcance com Mapeamento de Tom Inverso
    float exposure = 1.0; // Este valor pode ser exposto como um slider
    vec3 linear_hdr = inverse_tonemap_filmic(linear_ldr, exposure);

    // 3. Conversão de Gamut (Rec.709 para ACEScg AP1)
    const mat3 srgb_to_acescg_matrix = mat3(
        0.6165560, 0.3400713, 0.0433727,
        0.0772421, 0.9161245, 0.0066334,
        0.0157941, 0.1094681, 0.8747378
    );
    
    return linear_hdr * srgb_to_acescg_matrix;
}

vec4 hook() {
    // --- 1. Fetch Textures ---
    vec4 original = texture(PASS_LIN_raw, PASS_LIN_pos);
    vec3 color = original.rgb;

    // --- 2. Sharpen (Unsharp Mask) ---
    
    
    if (SharpenStrength > 0.0) {
        vec3 blurred = texture(PASS_BLUR_raw, PASS_BLUR_pos).rgb;
        vec4 Poriginal = texture(HOOKED_raw, HOOKED_pos);
        color = Poriginal.rgb;
        color += (color - blurred) * SharpenStrength;
        color = srgb_to_acescg(color);
    }

    
    
    // --- 3. Exposure ---
    color *= pow(2.0, Exposure);

    // --- 4. Color Temperature ---
    if (Temperature != 0.0) {
        vec3 warm_filter = vec3(1.0, 0.93, 0.85);
        vec3 cool_filter = vec3(0.85, 0.93, 1.0);
        color *= mix(vec3(1.0), (Temperature > 0.0 ? warm_filter : cool_filter), abs(Temperature));
    }

    // --- 5. Saturation & Vibrance ---
    float color_luma = luma(color);
    float color_sat = max(max(color.r, color.g), color.b) - min(min(color.r, color.g), color.b);
    
    float vibrance_adjust = Vibrance * (1.0 - smoothstep(0.2, 0.5, color_sat));
    color = mix(vec3(color_luma), color, Saturation + vibrance_adjust);

    // --- 6. Gamma Correction (Combined) ---
    vec3 gamma_vec = vec3(1.0 / (Gamma * GammaRed), 1.0 / (Gamma * GammaGreen), 1.0 / (Gamma * GammaBlue));
    color = pow(color, gamma_vec);

    // --- 7. Vignette ---
    if (VignetteStrength != 0.0) {
        vec2 uv = PASS_LIN_pos - 0.5;
        float dist = length(uv * 1.25);
        
        // vignette_effect is 0.0 in the center and smoothly goes to 1.0 at the edges
        float vignette_effect = smoothstep(VignetteRadius, VignetteRadius + VignetteSoftness, dist);

        if (VignetteStrength < 0.0) { // Darken (multiply)
            color *= (1.0 - abs(VignetteStrength) * vignette_effect);
        } else { // Lighten (add)
            color += VignetteStrength * vignette_effect;
        }
    }

    // --- 8. Final Output ---
    // Final clamp to ensure colors are in the valid [0.0, 1.0] range
    if (DeLinearizeOutput==1u){
        color.rgb = acescg_to_srgb(color.rgb);
    }
    
    return (vec4(clamp(color, 0.0, 1.0), original.a));
}