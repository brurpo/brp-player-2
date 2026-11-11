// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Chroma key
// @brp pt.param KeyColorR Cor da chave | Primeira cor da chave.
// @brp pt.param KeyColor2R Segunda cor da chave | Segunda cor da chave.
// @brp pt.param Tolerance Tolerância | Tolerância da comparação com a cor da chave. Valores menores são mais rígidos.
// @brp pt.param Softness Suavidade | Suavização da borda da máscara.
// @brp pt.param ErodeDilate Erodir/dilatar | Erosão (negativo) ou expansão (positivo) da máscara.
// @brp pt.param BlackRemap Remapear pretos | Remapeia os pretos da máscara. Valores maiores deixam a máscara mais transparente nas bordas.
// @brp pt.param WhiteRemap Remapear brancos | Remapeia os brancos da máscara. Valores menores deixam a máscara mais opaca.
// @brp pt.param SpillSuppressor Supressão de vazamento | Força da supressão do vazamento de cor.
// @brp pt.param SpillColorR Cor do vazamento | Cor do vazamento a suprimir.
//!PARAM KeyColorR
//!DESC First key color red component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColorG
//!DESC First key color green component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM KeyColorB
//!DESC First key color blue component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM KeyColor2R
//!DESC Second key color red component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColor2G
//!DESC Second key color green component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM KeyColor2B
//!DESC Second key color blue component (0-1).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM Tolerance
//!DESC Tolerance for the key color match. Lower values are stricter.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.2
//!PARAM Softness
//!DESC Feathering amount for the key mask edge.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1

//!PARAM ErodeDilate
//!DESC Mask erosion (negative) or expansion (positive).
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM BlackRemap
//!DESC Remap of mask blacks. Higher values make the mask more transparent at the edges.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM WhiteRemap
//!DESC Remap of mask whites. Lower values make the mask more opaque.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM SpillSuppressor
//!DESC Strength of the spill suppression.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5
//!PARAM SpillColorR
//!DESC Spill color to suppress (red component).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM SpillColorG
//!DESC Spill color to suppress (green component).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM SpillColorB
//!DESC Spill color to suppress (blue component).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0


//!HOOK MAIN
//!BIND HOOKED
//!DESC High-Quality Chroma Key Shader with Spill Suppressor


// Calcula a luminância de uma cor
float luma(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// Converte RGB para HSL
vec3 rgb_to_hsl(vec3 c) {
    float max_c = max(c.r, max(c.g, c.b));
    float min_c = min(c.r, min(c.g, c.b));
    float delta = max_c - min_c;
    
    float h = 0.0;
    if (delta > 0.0) {
        if (max_c == c.r) {
            h = mod((c.g - c.b) / delta, 6.0);
        } else if (max_c == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
    }
    
    float l = (max_c + min_c) / 2.0;
    float s = (l > 0.0 && l < 1.0) ? delta / (1.0 - abs(2.0 * l - 1.0)) : 0.0;
    
    return vec3(h, s, l);
}

// Chroma key com base na distância de cor
float key_mask(vec3 color, vec3 key_color) {
    float dist = distance(color.rgb, key_color);
    return clamp(dist / Tolerance, 0.0, 1.0);
}

// Aplica remapeamento de preto e branco
float remap(float mask_value, float black, float white) {
    float remapped = (mask_value - black) / (white - black);
    return clamp(remapped, 0.0, 1.0);
}

// Spill suppressor (desaturar cores próximas ao spill)
vec3 suppress_spill(vec3 color, vec3 spill_color, float strength) {
    vec3 hsl_color = rgb_to_hsl(color);
    vec3 hsl_spill = rgb_to_hsl(spill_color);

    // Usa a saturação e a distância de matiz para calcular o fator de supressão
    float hue_dist = abs(hsl_color.r - hsl_spill.r);
    hue_dist = min(hue_dist, 1.0 - hue_dist);

    float suppression_factor = clamp(1.0 - hue_dist * 5.0, 0.0, 1.0) * strength;
    
    // Desatura a cor em direção à luminância
    return mix(color, vec3(luma(color)), suppression_factor);
}

vec4 hook() {
    vec4 original_pixel = textureLod(HOOKED_raw, HOOKED_pos, 0.0);
    vec3 color = original_pixel.rgb;
    vec3 KeyColor = vec3(KeyColorR,KeyColorG,KeyColorB);
    vec3 KeyColor2 = vec3(KeyColor2R,KeyColor2G,KeyColor2B);
    vec3 SpillColor = vec3(SpillColorR,SpillColorG,SpillColorB);
    //float Tolerance;
    //float Softness;
    //float ErodeDilate;
    //float BlackRemap;
    //float WhiteRemap;
    //float SpillSuppressor;
  

    // Calcular a máscara para as duas cores de key
    float mask1 = key_mask(color, KeyColor);
    float mask2 = key_mask(color, KeyColor2);
    
    float final_mask = min(mask1, mask2);

    // Ajuste de "softness" e "expand/erode"
    float final_mask_adjusted = smoothstep(Softness + ErodeDilate, 1.0, final_mask);

    // Reampear os brancos e pretos da máscara
    float remapped_mask = remap(final_mask_adjusted, BlackRemap, WhiteRemap);
    
    // Aplica spill suppressor com base na máscara
    vec3 final_color = suppress_spill(color, SpillColor, SpillSuppressor );
    final_color *= remapped_mask;
    return vec4(final_color, remapped_mask);
}