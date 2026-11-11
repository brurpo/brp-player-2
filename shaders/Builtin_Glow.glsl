// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Brilho
// @brp pt.param Strength Intensidade | Tamanho do desfoque.
// @brp pt.param Radius Raio | Raio do desfoque.
// @brp pt.param Power Decaimento | Decaimento do desfoque.
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



//!PARAM Power
//!DESC Blur decay
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 4.0
2.0

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
//!DESC gaussian blur passX


vec4 hook() {
    
    return linearize(textureLod(HOOKED_raw, HOOKED_pos, 0.0) * HOOKED_mul);
}

//!HOOK MAIN
//!BIND PASSX
//!SAVE PASS0
//!DESC gaussian blur pass1


vec4 hook() {
    vec4 ocol = textureLod(PASSX_raw, PASSX_pos, 0.0);
    float range = Threshold - Threshold*ThresholdSmooth;
    //vec4 lin = linearize(ocol);
    vec4 col = smoothstep(range, Threshold, ocol);
    return col;
}

//!HOOK MAIN
//!BIND PASS0
//!SAVE PASS1
//!DESC gaussian blur pass2



float RADIUS =  input_size.x > input_size.y ? 1.0*Radius*input_size.x : 1.0*Radius*input_size.y; //kernel radius (integer as float, e.g. 3.0), (0.0, 10+]


#define get_weight(x, sig) (exp(-(x) * (x) / (2.0 * sig * sig)))

vec4 hook() {
    float weight;
    float prop = clamp(1.0 + (0.5 - Proportion)*2 ,0.0 ,1.0);
    float rad = RADIUS*prop*.05;
    float SIGMA = (rad-1)/6;
    vec4 csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    float wsum = 1.0;
    float dec = 1.0/rad;
    for(float i = 1.0; i <= rad; ++i) {
        
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) *weight;
        wsum += 2.0 * weight ;
    }
    vec4 col = (csum/RADIUS)*Strength*pow(1,Power)*10;
    
    rad = RADIUS*prop*.1;
    csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) *weight;
        wsum += 2.0 * weight ;
    }
    col += (csum/RADIUS)*Strength*pow(.8,Power)*10;
    
    
    rad = RADIUS*prop*.25;
    csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) *weight;
        wsum += 2.0 * weight ;
    }
    col += (csum/RADIUS)*Strength*pow(.6,Power)*10;
    

    rad = RADIUS*prop*.5;
    csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) *weight;
        wsum += 2.0 * weight ;
    }
    col += (csum/RADIUS)*Strength*pow(.4,Power)*10;
    

    rad = RADIUS*prop;
    csum = textureLod(PASS0_raw, PASS0_pos, 0.0) * PASS0_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, -i), 0.0) + textureLod(PASS0_raw, PASS0_pos + PASS0_pt * vec2(0.0, i), 0.0)) *weight;
        wsum += 2.0 * weight ;
    }
    col += (csum/RADIUS)*Strength*pow(.2,Power)*10;
    return vec4(col.r, col.g, col.b, texture(PASS0_raw, PASS0_pos).a);
}

//!HOOK MAIN
//!BIND PASS1
//!BIND PASS0
//!BIND PASSX
//!DESC gaussian blur pass3


float RADIUS =  input_size.x > input_size.y ? 1.0*Radius*input_size.x : 1.0*Radius*input_size.y; //kernel radius (integer as float, e.g. 3.0), (0.0, 10+]

#define get_weight(x,sig) (exp(-(x) * (x) / (2.0 * sig * sig)))
float blendScreen(float base, float blend) {
	return 1.0-((1.0-base)*(1.0-blend));
}
vec3 blendScreen(vec3 base, vec3 blend) {
	return vec3(blendScreen(base.r,blend.r),blendScreen(base.g,blend.g),blendScreen(base.b,blend.b));
}

vec4 hook() {
    float weight;
    float prop = clamp( 1.0 + (Proportion - 0.5)*2 ,0.0 ,1.0);
    float rad = RADIUS*prop*.05;
    vec4 csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    float wsum = 1.0;
    float dec = 1.0/rad;
    float SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) *weight;
        wsum += 2.0 * weight;
    }
    vec4 col = (csum/(RADIUS))*Strength*pow(1,Power)*10;

    rad = RADIUS*prop*.1;
    csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) *weight;
        wsum += 2.0 * weight;
    }
    col += (csum/(RADIUS))*Strength*pow(.8,Power)*10;

    rad = RADIUS*prop*.25;
    csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) *weight;
        wsum += 2.0 * weight;
    }
    col += (csum/(RADIUS))*Strength*pow(.6,Power)*10;

    rad = RADIUS*prop*.5;
    csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) *weight;
        wsum += 2.0 * weight;
    }
    col += (csum/(RADIUS))*Strength*pow(.4,Power)*10;

    rad = RADIUS*prop;
    csum = textureLod(PASS1_raw, PASS1_pos, 0.0) * PASS1_mul;
    wsum = 1.0;
    dec = 1.0/rad;
    SIGMA = (rad-1)/6;
    for(float i = 1.0; i <= rad; ++i) {
        weight = get_weight(i,SIGMA);
        csum += (textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS1_raw, PASS1_pos + PASS1_pt * vec2(i, 0.0), 0.0)) *weight;
        wsum += 2.0 * weight;
    }
    col += (csum/(RADIUS))*Strength*pow(.2,Power)*10;
    if(Tonemap==1){
    col = 1.0 - exp( -col );
    }
    if(Screen==1){
        vec4 orig = texture(PASSX_raw, PASSX_pos);
        vec3 pre = blendScreen(col.rgb, orig.rgb);
        return delinearize(vec4(pre.r, pre.g, pre.b, orig.a));

    }else{
        return delinearize(vec4(col.r, col.g, col.b, 0)+texture(PASSX_raw, PASSX_pos));

    }
    //col = 1.0 - exp( -col );
    
}
