// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Brilho (rápido)
// @brp pt.param Strength Intensidade | Tamanho do desfoque.
// @brp pt.param Radius Raio | Raio do desfoque.
// @brp pt.param Curve Decaimento | Decaimento do desfoque.
// @brp pt.param Threshold Limiar | A partir de que luminância a imagem brilha.
// @brp pt.param ThresholdSmooth Suavidade do limiar | Suavidade do limiar.
// @brp pt.param Proportion Proporção | Proporções do brilho.
// @brp pt.param Tonemap Mapeamento de tons | Aplica mapeamento de tons.
// @brp pt.param Screen Modo tela | Compõe com o modo de mesclagem tela.
//////////////////////////////////////////////////////////////////
//  BRP_GLOW Free for any use, I have no liability though, lol  //
//////////////////////////////////////////////////////////////////

//!PARAM Strength
//!DESC Blur size
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
1.0

//!PARAM Radius
//!DESC Blur Radius
//!TYPE float
//!MINIMUM 0.01
//!MAXIMUM 1.0
1.0

//!PARAM Curve
//!DESC Blur decay
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 4.0
1.0

//!PARAM Threshold
//!DESC What luminance to start glowing
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
0.8

//!PARAM ThresholdSmooth
//!DESC Threshold softness
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.2

//!PARAM Proportion
//!DESC Glow proportions
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM Tonemap
//!DESC Aplly Tone Mapping
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1
//!PARAM Screen
//!DESC Composite with screen blend mode
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1

//!HOOK MAIN
//!BIND HOOKED
//!SAVE PASSX
//!DESC gaussian blur pass1
vec4 hook() {    
    return linearize(textureLod(HOOKED_raw, HOOKED_pos, 0.0) * HOOKED_mul);
}
//!HOOK MAIN
//!BIND PASSX
//!SAVE PASS0
//!DESC gaussian blur pass1

float rgbToLuma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
vec4 hook() {    
    vec4 ocol = textureLod(PASSX_raw, PASSX_pos, 0.0);
    float luma =  rgbToLuma(ocol.rgb);
    float range = max(0.0,Threshold - Threshold*ThresholdSmooth);
    float mask = smoothstep(range, Threshold, luma);
    vec4 col = ocol*vec4(mask);
    return col;
}
///////////////////////////////////////////////////////PASS1////////////////////////////////
//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS1
//!DESC gaussian blur pass2


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.01;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; ++i) {
        weight = get_weight(i);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) * PASS0_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
}

//!HOOK MAIN
//!BIND PASS1
//!SAVE PASS2
//!DESC gaussian blur pass3


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (Proportion - 0.5) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.01;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; ++i) {
        weight = get_weight(i);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) * PASS1_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
    //return delinearize(vec4(col.r,col.g,col.b,0.0)+texture(PASSX_raw, PASSX_pos));
}

///////////////////////////////////////////////////////PASS2////////////////////////////////
//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS3
//!DESC gaussian blur pass2


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.05;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=2.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) * PASS0_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
}

//!HOOK MAIN
//!BIND PASS3
//!SAVE PASS4
//!DESC gaussian blur pass3


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (Proportion - 0.5) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.05;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS3_raw, PASS3_pos, 0.0) * PASS3_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=2.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS3_raw, PASS3_pos + PASS3_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS3_raw, PASS3_pos + PASS3_pt * vec2(i, 0.0), 0.0)) * PASS3_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
    //return delinearize(vec4(col.r,col.g,col.b,0.0)+texture(PASSX_raw, PASSX_pos));
}

///////////////////////////////////////////////////////PASS3////////////////////////////////
//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS5
//!DESC gaussian blur pass2


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.15;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=3.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) * PASS0_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
}

//!HOOK MAIN
//!BIND PASS5
//!SAVE PASS6
//!DESC gaussian blur pass3


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (Proportion - 0.5) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.15;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS5_raw, PASS5_pos, 0.0) * PASS5_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=3.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS5_raw, PASS5_pos + PASS5_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS5_raw, PASS5_pos + PASS5_pt * vec2(i, 0.0), 0.0)) * PASS5_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
    //return delinearize(vec4(col.r,col.g,col.b,0.0)+texture(PASSX_raw, PASSX_pos));
}

///////////////////////////////////////////////////////PASS4////////////////////////////////
//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS7
//!DESC gaussian blur pass2


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.35;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=4.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) * PASS0_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
}

//!HOOK MAIN
//!BIND PASS7
//!SAVE PASS8
//!DESC gaussian blur pass3


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (Proportion - 0.5) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 0.35;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS7_raw, PASS7_pos, 0.0) * PASS7_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=4.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS7_raw, PASS7_pos + PASS7_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS7_raw, PASS7_pos + PASS7_pt * vec2(i, 0.0), 0.0)) * PASS7_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
    //return delinearize(vec4(col.r,col.g,col.b,0.0)+texture(PASSX_raw, PASSX_pos));
}

///////////////////////////////////////////////////////PASS5////////////////////////////////
//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS9
//!DESC gaussian blur pass2


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 1.0;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=5.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) * PASS0_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    return col;
}

//!HOOK MAIN
//!BIND PASS9
//!BIND PASS8
//!BIND PASS6
//!BIND PASS4
//!BIND PASS2
//!BIND PASSX
//!DESC gaussian blur pass3


#define get_weight(x) (exp(-(x) * (x) / (2.0 * SIGMA * SIGMA)))
float blendScreen(float base, float blend) {
	return 1.0 - ((1.0 - base) * (1.0 - blend));
}
vec3 blendScreen(vec3 base, vec3 blend) {
	return vec3(blendScreen(base.r, blend.r), blendScreen(base.g, blend.g), blendScreen(base.b, blend.b));
}

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (Proportion - 0.5) * 2.0, 0.0, 1.0);
    float RADIUS = input_size.x > input_size.y ? 2.0*Radius*input_size.x*prop : 2.0*Radius*input_size.y*prop;
    RADIUS *= 1.0;
    float SIGMA = (RADIUS - 1.0) / 6.0;
    vec4 csum = textureLod(PASS9_raw, PASS9_pos, 0.0) * PASS9_mul;
    float wsum = 1.0;
    vec4 col = vec4(0.0);
    for(float i = 1.0; i <= RADIUS; i+=5.0) {
        weight = get_weight(i);
        csum += (textureLod(PASS9_raw, PASS9_pos + PASS9_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS9_raw, PASS9_pos + PASS9_pt * vec2(i, 0.0), 0.0)) * PASS9_mul * weight;
        wsum += 2.0 * weight;
    }
    col += csum / wsum;
    col += texture(PASS8_raw, PASS8_pos)+texture(PASS6_raw, PASS6_pos)+texture(PASS4_raw, PASS4_pos)+texture(PASS2_raw, PASS2_pos);
    col = pow(col,vec4(Curve));
    col *= Strength;
    if(Tonemap == 1u){
        col = 1.0 - exp(-col);
    }
    if(Screen == 1u){
        vec4 orig = texture(PASSX_raw, PASSX_pos);
        vec3 pre = blendScreen(col.rgb, orig.rgb);
        return delinearize(vec4(pre.r, pre.g, pre.b, orig.a));
    } else {
        // CORREÇÃO: Usa '0.0' em vec4
        return delinearize(vec4(col.r, col.g, col.b, 0.0) + texture(PASSX_raw, PASSX_pos));
    }
}
