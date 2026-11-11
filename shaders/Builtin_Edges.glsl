// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Bordas
// @brp pt.param LowThreshold Limiar baixo | Limiar inferior da detecção de bordas. Ajuda a ligar as linhas.
// @brp pt.param HighThreshold Limiar alto | Limiar superior da detecção de bordas. Define os pixels de borda forte.
// @brp pt.param Thickness Espessura | Espessura das linhas de borda.
// @brp pt.param EdgeColorR Cor da borda | Cor das linhas de borda.
// @brp pt.param EdgeOpacity Opacidade | Opacidade das linhas de borda.
// @brp pt.param TransparentBg Fundo transparente | Se 1, as bordas ficam sobre fundo transparente. Se 0, sobre a imagem original.
//!PARAM LowThreshold
//!DESC Lower threshold for the edge detection. Helps connect edge lines.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1

//!PARAM HighThreshold
//!DESC Higher threshold for the edge detection. Defines strong edge pixels.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!PARAM Thickness
//!DESC Controls the thickness of the edge lines.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
1.0

//!PARAM EdgeColorR
//!DESC Red component of the edge line color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM EdgeColorG
//!DESC Green component of the edge line color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM EdgeColorB
//!DESC Blue component of the edge line color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM EdgeOpacity
//!DESC Opacity of the edge lines.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM TransparentBg
//!DESC If 1, composes edges on a transparent background. If 0, on the original image.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0



//!HOOK MAIN
//!BIND HOOKED
//!SAVE PASS_H
//!DESC Canny - Pass 1: Horizontal Blur

//=========================================================================================
// PASS 1: Grayscale & Horizontal Blur
//=========================================================================================

float luma(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }
vec4 hook() {
    vec2 pt = HOOKED_pt; float sum = 0.0;
    float weights[] = float[](0.153388, 0.221461, 0.250301, 0.221461, 0.153388);
    for (int i = -2; i <= 2; i++) {
        sum += luma(texture(HOOKED_raw, HOOKED_pos + vec2(float(i), 0.0) * pt).rgb) * weights[i + 2];
    }
    return vec4(sum, sum, sum, 1.0);
}


//=========================================================================================
// PASS 2: Vertical Blur
//=========================================================================================
//!HOOK MAIN
//!BIND PASS_H
//!SAVE PASS_BLUR
//!DESC Canny - Pass 2: Vertical Blur

vec4 hook() {
    vec2 pt = PASS_H_pt; float sum = 0.0;
    float weights[] = float[](0.153388, 0.221461, 0.250301, 0.221461, 0.153388);
    for (int i = -2; i <= 2; i++) {
        sum += texture(PASS_H_raw, PASS_H_pos + vec2(0.0, float(i)) * pt).r * weights[i + 2];
    }
    return vec4(sum, sum, sum, 1.0);
}


//=========================================================================================
// PASS 3: Gradient Calculation
//=========================================================================================
//!HOOK MAIN
//!BIND PASS_BLUR
//!SAVE PASS_GRADIENT
//!DESC Canny - Pass 3: Gradient Calculation

vec4 hook() {
    vec2 pt = PASS_BLUR_pt;
    float gx = -texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(-1,-1)).r -2.0*texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(-1,0)).r -texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(-1,1)).r +texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(1,-1)).r +2.0*texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(1,0)).r +texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(1,1)).r;
    float gy = -texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(-1,-1)).r -2.0*texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(0,-1)).r -texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(1,-1)).r +texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(-1,1)).r +2.0*texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(0,1)).r +texture(PASS_BLUR_raw, PASS_BLUR_pos + pt * vec2(1,1)).r;
    return vec4(sqrt(gx*gx + gy*gy), atan(gy, gx), 0.0, 1.0);
}


//=========================================================================================
// PASS 4: Non-Maximum Suppression and Double Threshold (Generates 1px wide edges)
//=========================================================================================
//!HOOK MAIN
//!BIND PASS_GRADIENT
//!SAVE PASS_EDGES_RAW
//!DESC Canny - Pass 4: Edge Finalization

vec4 hook() {
    vec2 pt = PASS_GRADIENT_pt;
    vec2 center_grad = texture(PASS_GRADIENT_raw, PASS_GRADIENT_pos).rg;
    float mag = center_grad.r; float dir = center_grad.g;
    vec2 p1, p2;
         if (abs(dir) > 2.74889) { p1 = vec2(-1, 0); p2 = vec2(1, 0); }
    else if (abs(dir) > 1.96349) { p1 = vec2(-1, 1); p2 = vec2(1, -1); }
    else if (abs(dir) > 1.17809) { p1 = vec2(0, 1); p2 = vec2(0, -1); }
    else if (abs(dir) > 0.39269) { p1 = vec2(1, 1); p2 = vec2(-1, -1); }
    else { p1 = vec2(1, 0); p2 = vec2(-1, 0); }
    float mag1 = texture(PASS_GRADIENT_raw, PASS_GRADIENT_pos + p1 * pt).r;
    float mag2 = texture(PASS_GRADIENT_raw, PASS_GRADIENT_pos + p2 * pt).r;
    if (mag < mag1 || mag < mag2) mag = 0.0;
    
    float edge = 0.0;
    if (mag >= HighThreshold) {
        edge = 1.0;
    } else if (mag >= LowThreshold) {
        for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++) {
            if (x == 0 && y == 0) continue;
            if (texture(PASS_GRADIENT_raw, PASS_GRADIENT_pos + vec2(x, y) * pt).r >= HighThreshold) {
                edge = 1.0; break;
            }
        }
    }
    return vec4(edge);
}


//=========================================================================================
// PASS 5: Dilate (Thickness)
//=========================================================================================
//!HOOK MAIN
//!BIND PASS_EDGES_RAW
//!SAVE PASS_DILATED
//!DESC Canny - Pass 5: Apply Thickness

#define CIRCULAR_KERNEL

vec4 hook() {
    // If thickness is minimal, just return the raw edge map.
    if (Thickness <= 0.5) {
        return texture(PASS_EDGES_RAW_raw, PASS_EDGES_RAW_pos);
    }

    float max_val = 0.0;
    int radius = int(Thickness);
    vec2 pt = PASS_EDGES_RAW_pt;

    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            #if defined(CIRCULAR_KERNEL)
                if (distance(vec2(x, y), vec2(0, 0)) > float(radius)) continue;
            #endif
            max_val = max(max_val, texture(PASS_EDGES_RAW_raw, PASS_EDGES_RAW_pos + vec2(x, y) * pt).r);
        }
    }
    return vec4(max_val);
}


//=========================================================================================
// PASS 6: Final Composition
//=========================================================================================
//!HOOK MAIN
//!BIND HOOKED
//!BIND PASS_DILATED
//!DESC Canny - Final Pass: Composition

vec4 hook() {
    vec4 base_color = (TransparentBg == 1u) ? vec4(0.0) : texture(HOOKED_raw, HOOKED_pos);
    float edge_value = texture(PASS_DILATED_raw, PASS_DILATED_pos).r;
    vec4 edge_color = vec4(EdgeColorR, EdgeColorG, EdgeColorB, EdgeOpacity);
    return mix(base_color, edge_color, edge_value * edge_color.a);
}