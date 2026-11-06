//!PARAM SplitView
//!DESC Split View for color picking.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
0.5
//!PARAM DistanceOrDifference
//!DESC 0=Delta Keyer (Rápido), 1=IBK Keyer (Usa Clean Plate gerada, mais pesado).
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM KeyColorR
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColorG
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM KeyColorB
//!DESC Primary key color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColor2R
//!DESC Second key color for non-uniform backgrounds.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColor2G
//!DESC Second key color for non-uniform backgrounds.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM KeyColor2B
//!DESC Second key color for non-uniform backgrounds.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyTolerance
//!DESC Sets the center point of the key. Higher values make the key stricter.
//!TYPE float
//!MINIMUM -0.5
//!MAXIMUM 1.0
0.2
//!PARAM KeySoftness
//!DESC Controls the softness/feather of the key's edge transition.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.2
//!PARAM ScreenBalance
//!DESC (Modo Delta Keyer) Balances the key against red/blue.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM UseProtectionMatte
//!DESC If 1, enables the Foreground Protection matte.(cat: ProtectionMask)
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0
//!PARAM CoreMatteColorR
//!DESC Color to protect and force as opaque foreground.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM CoreMatteColorG
//!DESC Color to protect and force as opaque foreground.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM CoreMatteColorB
//!DESC Color to protect and force as opaque foreground.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM CoreMatteTolerance
//!DESC Tolerance for the foreground protection color match.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1
//!PARAM CoreMatteSoftness
//!DESC Softness for the foreground protection matte transition.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1
//!PARAM UseHairDetection
//!DESC If 1, enables the hair strand detection algorithm.Performance intensive.(cat: HairRecovery)
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0
//!PARAM HairWidth
//!DESC The expected width of a hair strand in pixels.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 5.0
1.5
//!PARAM HairIntensity
//!DESC The intensity of the detected hair matte.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
1.0
//!PARAM GarbageMatteClean
//!DESC Strength of the noise removal (median filter) on the matte. 0=off.(cat: Masking Adjustments)
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 2
0
//!PARAM ErodeDilate1
//!DESC Mask erosion (negative) or expansion (positive).
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0
//!PARAM ErodeDilate2
//!DESC Mask erosion (negative) or expansion (positive).
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0
//!PARAM MatteSoftenBlur
//!DESC Applies a blur to the matte before the border fixing stage.
//!TYPE float
//!MINIMUM 0.0
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
//!PARAM MatteShrink
//!DESC Pushes semi-transparent areas towards black, thinning the matte edge.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
0.0
//!PARAM MatteExpand
//!DESC Pushes semi-transparent areas towards white, solidifying the matte edge.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
0.0
//!PARAM FinalMatteBlur
//!DESC Applies a final blur to the processed matte for feathering.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM ColorUnmix
//!DESC Use color unmixing.(cat: SpillSupressor)
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1
//!PARAM Unmix
//!DESC Strength of the unmix.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM SpillSuppressor
//!DESC Strength of the spill suppression.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM SpillTolerance
//!DESC (Simple Mode) Controls the range of hues affected by the spill suppressor.Lower values affect a wider range.
//!TYPE float
//!MINIMUM 1.0
//!MAXIMUM 10.0
3.0

//!PARAM DespillMode
//!DESC Spill suppression mode. 0=Simple Desaturate, 1=Advanced Reconstruct Color.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1
//!PARAM FixBorders
//!DESC Extends edge colors to fix borders. Defines the number of iterations.(cat: BorderRecovery)
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 10
0
//!PARAM ViewMatte
//!DESC View Matte only. (cat: MaskTesting)
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0
//!PARAM ViewCleanPlate
//!DESC View clean plate only.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM TestBackground
//!DESC If 1, composite over a solid test color instead of transparent black.
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
0
//!PARAM TestBgR
//!DESC Red component of the test background color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM TestBgG
//!DESC Green component of the test background color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM TestBgB
//!DESC Blue component of the test background color.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!HOOK MAIN
//!BIND HOOKED
//!SAVE OCOL
//!DESC High-Quality Chroma Key Shader with Spill Suppressor
vec4 hook() { 
    vec4 col = textureLod(HOOKED_raw, HOOKED_pos, 0.0);
    if(col.a > 0.0){
        col.rgb = col.rgb/col.a;
    }else{
        col.rgb = vec3(0.0);
    }
    return col;
}

// =========================================================================================
// INÍCIO DA CADEIA DE GERAÇÃO DE CLEAN PLATE E KEYER IBK (SÓ EXECUTA SE DistanceOrDifference = 1)
// =========================================================================================

//!HOOK MAIN
//!BIND OCOL
//!SAVE PASS0_DELTA_KEYER
//!DESC Etapa 1: Gera uma máscara inicial usando o Keyer Delta (base do seu shader original).
float calculate_screen_color(vec3 color, vec3 key_color, float balance) { float p; float r; float b; if (key_color.g > key_color.r && key_color.g > key_color.b) { p = color.g; r = color.r; b = color.b; } else if (key_color.b > key_color.r && key_color.b > key_color.g) { p = color.b; r = color.r; b = color.g; } else { p = color.r; r = color.g; b = color.b; } return p - mix(b, r, balance); }
vec4 hook() {
    if (DistanceOrDifference == 0u) { return vec4(0.0); } // Otimização
    vec3 color = textureLod(OCOL_raw, OCOL_pos, 0.0).rgb;
    vec3 key_color1 = vec3(KeyColorR, KeyColorG, KeyColorB);
    vec3 key_color2 = vec3(KeyColor2R, KeyColor2G, KeyColor2B);
    float screen_color1 = calculate_screen_color(color, key_color1, ScreenBalance);
    float screen_color2 = calculate_screen_color(color, key_color2, ScreenBalance);
    float final_screen_color = max(screen_color1, screen_color2);
    float initial_tolerance = 0.28;
    float initial_softness = 0.2;
    float clip_black = initial_tolerance - initial_softness / 2.0;
    float clip_white = initial_tolerance + initial_softness / 2.0;
    float keyer_mask = 1.0 - smoothstep(clip_black, clip_white, final_screen_color);
    return vec4(keyer_mask);
}

//!HOOK MAIN
//!BIND PASS0_DELTA_KEYER
//!SAVE ROUGH_MATTE
//!DESC Etapa 2: Isola apenas os pixels de fundo garantidos.
vec4 hook() {
    if (DistanceOrDifference == 0u) { return vec4(0.0); } // Otimização
    float m = textureLod(PASS0_DELTA_KEYER_raw, PASS0_DELTA_KEYER_pos, 0.0).r;
    float bg_mask = 1.0 - m;
    bg_mask = smoothstep(0.99, 1.0, bg_mask);
    return vec4(bg_mask);
}

//!HOOK MAIN
//!BIND ROUGH_MATTE
//!SAVE ROUGH_MATTE_ERODED
//!DESC Etapa 3: Aplica uma forte erosão para criar a "folga" de segurança.
vec4 hook() {
    if (DistanceOrDifference == 0u) { return vec4(0.0); } // Otimização
    float min_m = 1.0;
    int radius = 35;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            min_m = min(min_m, textureLod(ROUGH_MATTE_raw, ROUGH_MATTE_pos + vec2(x, y) * ROUGH_MATTE_pt, 0.0).r);
        }
    }
    return vec4(min_m);
}

//!HOOK MAIN
//!BIND OCOL
//!BIND ROUGH_MATTE_ERODED
//!SAVE SEED_PLATE
//!DESC Etapa 4: Cria a placa de semente com um buraco no primeiro plano.
vec4 hook() {
    if (DistanceOrDifference == 0u) { return vec4(0.0); } // Otimização
    float is_bg = textureLod(ROUGH_MATTE_ERODED_raw, ROUGH_MATTE_ERODED_pos, 0.0).r;
    if (is_bg > 0.5) {
        return vec4(textureLod(OCOL_raw, OCOL_pos, 0.0).rgb, 1.0);
    } else {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }
}

// --- Etapa 5: Cadeia de Expansão Iterativa (32 passes) ---
//!HOOK MAIN
//!BIND SEED_PLATE
//!SAVE EXPAND_1
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(SEED_PLATE_raw,SEED_PLATE_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(SEED_PLATE_raw,SEED_PLATE_pos+vec2(x,y)*SEED_PLATE_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_1
//!SAVE EXPAND_2
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_1_raw,EXPAND_1_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_1_raw,EXPAND_1_pos+vec2(x,y)*EXPAND_1_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_2
//!SAVE EXPAND_3
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_2_raw,EXPAND_2_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_2_raw,EXPAND_2_pos+vec2(x,y)*EXPAND_2_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_3
//!SAVE EXPAND_4
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_3_raw,EXPAND_3_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_3_raw,EXPAND_3_pos+vec2(x,y)*EXPAND_3_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_4
//!SAVE EXPAND_5
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_4_raw,EXPAND_4_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_4_raw,EXPAND_4_pos+vec2(x,y)*EXPAND_4_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_5
//!SAVE EXPAND_6
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_5_raw,EXPAND_5_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_5_raw,EXPAND_5_pos+vec2(x,y)*EXPAND_5_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_6
//!SAVE EXPAND_7
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_6_raw,EXPAND_6_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_6_raw,EXPAND_6_pos+vec2(x,y)*EXPAND_6_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_7
//!SAVE EXPAND_8
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_7_raw,EXPAND_7_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_7_raw,EXPAND_7_pos+vec2(x,y)*EXPAND_7_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_8
//!SAVE EXPAND_9
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_8_raw,EXPAND_8_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_8_raw,EXPAND_8_pos+vec2(x,y)*EXPAND_8_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_9
//!SAVE EXPAND_10
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_9_raw,EXPAND_9_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_9_raw,EXPAND_9_pos+vec2(x,y)*EXPAND_9_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_10
//!SAVE EXPAND_11
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_10_raw,EXPAND_10_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_10_raw,EXPAND_10_pos+vec2(x,y)*EXPAND_10_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_11
//!SAVE EXPAND_12
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_11_raw,EXPAND_11_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_11_raw,EXPAND_11_pos+vec2(x,y)*EXPAND_11_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_12
//!SAVE EXPAND_13
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_12_raw,EXPAND_12_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_12_raw,EXPAND_12_pos+vec2(x,y)*EXPAND_12_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_13
//!SAVE EXPAND_14
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_13_raw,EXPAND_13_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_13_raw,EXPAND_13_pos+vec2(x,y)*EXPAND_13_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_14
//!SAVE EXPAND_15
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_14_raw,EXPAND_14_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_14_raw,EXPAND_14_pos+vec2(x,y)*EXPAND_14_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_15
//!SAVE EXPAND_16
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_15_raw,EXPAND_15_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_15_raw,EXPAND_15_pos+vec2(x,y)*EXPAND_15_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_16
//!SAVE EXPAND_17
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_16_raw,EXPAND_16_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_16_raw,EXPAND_16_pos+vec2(x,y)*EXPAND_16_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_17
//!SAVE EXPAND_18
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_17_raw,EXPAND_17_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_17_raw,EXPAND_17_pos+vec2(x,y)*EXPAND_17_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_18
//!SAVE EXPAND_19
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_18_raw,EXPAND_18_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_18_raw,EXPAND_18_pos+vec2(x,y)*EXPAND_18_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_19
//!SAVE EXPAND_20
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_19_raw,EXPAND_19_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_19_raw,EXPAND_19_pos+vec2(x,y)*EXPAND_19_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_20
//!SAVE EXPAND_21
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_20_raw,EXPAND_20_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_20_raw,EXPAND_20_pos+vec2(x,y)*EXPAND_20_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_21
//!SAVE EXPAND_22
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_21_raw,EXPAND_21_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_21_raw,EXPAND_21_pos+vec2(x,y)*EXPAND_21_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_22
//!SAVE EXPAND_23
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_22_raw,EXPAND_22_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_22_raw,EXPAND_22_pos+vec2(x,y)*EXPAND_22_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_23
//!SAVE EXPAND_24
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_23_raw,EXPAND_23_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_23_raw,EXPAND_23_pos+vec2(x,y)*EXPAND_23_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_24
//!SAVE EXPAND_25
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_24_raw,EXPAND_24_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_24_raw,EXPAND_24_pos+vec2(x,y)*EXPAND_24_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_25
//!SAVE EXPAND_26
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_25_raw,EXPAND_25_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_25_raw,EXPAND_25_pos+vec2(x,y)*EXPAND_25_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_26
//!SAVE EXPAND_27
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_26_raw,EXPAND_26_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_26_raw,EXPAND_26_pos+vec2(x,y)*EXPAND_26_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_27
//!SAVE EXPAND_28
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_27_raw,EXPAND_27_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_27_raw,EXPAND_27_pos+vec2(x,y)*EXPAND_27_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_28
//!SAVE EXPAND_29
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_28_raw,EXPAND_28_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_28_raw,EXPAND_28_pos+vec2(x,y)*EXPAND_28_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_29
//!SAVE EXPAND_30
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_29_raw,EXPAND_29_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_29_raw,EXPAND_29_pos+vec2(x,y)*EXPAND_29_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}
//!HOOK MAIN
//!BIND EXPAND_30
//!SAVE EXPAND_31
vec4 hook() { if(DistanceOrDifference==0u){return vec4(0.0);}vec4 c=textureLod(EXPAND_30_raw,EXPAND_30_pos,0.0);if(c.a>0.5){return c;}vec4 s=vec4(0.0);float w=0.0;for(int y=-1;y<=1;y++){for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;vec4 n=textureLod(EXPAND_30_raw,EXPAND_30_pos+vec2(x,y)*EXPAND_30_pt,0.0);if(n.a>0.5){s+=n;w+=1.0;}}}if(w>0.0){return vec4(s.rgb/w,1.0);}return c;}

//!HOOK MAIN
//!BIND EXPAND_31
//!SAVE CLEAN_PLATE_FINAL
//!DESC Pass 5.32: Expansão Final com Fallback de Cor Chave.
vec4 hook() {
    if (DistanceOrDifference == 0u) { return vec4(0.0); } // Otimização
    vec4 center_pixel = textureLod(EXPAND_31_raw, EXPAND_31_pos, 0.0);
    if (center_pixel.a > 0.5) {
        return center_pixel;
    }
    vec4 csum = vec4(0.0);
    float wsum = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            vec4 n = textureLod(EXPAND_31_raw, EXPAND_31_pos + vec2(x, y) * EXPAND_31_pt, 0.0);
            if (n.a > 0.5) {
                csum += n;
                wsum += 1.0;
            }
        }
    }
    if (wsum > 0.0) {
        return vec4(csum.rgb / wsum, 1.0);
    } else {
        return vec4(KeyColorR, KeyColorG, KeyColorB, 1.0);
    }
}

//!HOOK MAIN
//!BIND OCOL
//!BIND CLEAN_PLATE_FINAL
//!SAVE PASS0
//!DESC Etapa Final: Gera a máscara usando a lógica IBK ou Delta Keyer.
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float calculate_screen_color_final(vec3 color, vec3 key_color, float balance) { float p; float r; float b; if (key_color.g > key_color.r && key_color.g > key_color.b) { p = color.g; r = color.r; b = color.b; } else if (key_color.b > key_color.r && key_color.b > key_color.g) { p = color.b; r = color.r; b = color.g; } else { p = color.r; r = color.g; b = color.b; } return p - mix(b, r, balance); }
vec3 linearize(vec3 c) { return pow(c, vec3(2.2)); }
vec3 delinearize(vec3 c) { return pow(c, vec3(1.0/2.2)); }
vec4 hook() {
    if (DistanceOrDifference == 1u){
        vec3 C = linearize(textureLod(OCOL_raw, OCOL_pos, 0.0).rgb);
        vec3 B = linearize(textureLod(CLEAN_PLATE_FINAL_raw, CLEAN_PLATE_FINAL_pos, 0.0).rgb);

        if (dot(B,B) < 1e-8) return vec4(1.0);
        
        const float EPS = 1e-6;
        vec3 r = C / (B + EPS);
        float k = clamp(min(min(r.r, r.g), r.b), 0.0, 1.0);
        vec3 resid = max(C - k * B, 0.0);
        
        float diff = max(resid.r, max(resid.g, resid.b));
        
        float YC = dot(C, vec3(0.2126, 0.7152, 0.0722));
        float DarkBias   = 0.015;
        float DarkLift   = 0.1;
        float NearBlack  = 0.010;
        float DarkWidth  = 0.030;

        float dark_w = 1.0 - smoothstep(NearBlack, NearBlack + DarkWidth, YC);
        float diff_dark = max(diff - DarkBias * dark_w, 0.0);

        float tol = KeyTolerance + DarkLift * dark_w;
        float clip_black = tol - 0.5 * KeySoftness;
        float clip_white = tol + 0.5 * KeySoftness;
        float ibk_mask = smoothstep(clip_black, clip_white, diff_dark);

        float near_black_keep = 1.0 - smoothstep(0.0, NearBlack, YC);
        float matte = max(ibk_mask, near_black_keep);

        if (UseProtectionMatte == 1u) {
            vec3 core_color = linearize(vec3(CoreMatteColorR, CoreMatteColorG, CoreMatteColorB));
            float core_dist = distance(C, core_color);
            float p_blk = CoreMatteTolerance - 0.5 * CoreMatteSoftness;
            float p_wht = CoreMatteTolerance + 0.5 * CoreMatteSoftness;
            float protection_mask = 1.0 - smoothstep(p_blk, p_wht, core_dist);
            matte = max(matte, protection_mask);
        }
        float blackmask = clamp(luma(textureLod(OCOL_raw, OCOL_pos, 0.0).rgb), 0.0, 1.0);
        blackmask = smoothstep(0.1, 0.0, blackmask);
        matte = clamp(matte + blackmask,0.0,1.0);
 
        return vec4(matte);
    }else{
        vec3 color = textureLod(OCOL_raw, OCOL_pos, 0.0).rgb;
        vec3 key_color1 = vec3(KeyColorR, KeyColorG, KeyColorB);
        vec3 key_color2 = vec3(KeyColor2R, KeyColor2G, KeyColor2B);
        float screen_color1 = calculate_screen_color_final(color, key_color1, ScreenBalance);
        float screen_color2 = calculate_screen_color_final(color, key_color2, ScreenBalance);
        float final_screen_color = max(screen_color1, screen_color2);
        float clip_black = KeyTolerance - KeySoftness / 2.0;
        float clip_white = KeyTolerance + KeySoftness / 2.0;
        float keyer_mask = 1.0 - smoothstep(clip_black, clip_white, final_screen_color);
        if (UseProtectionMatte == 1u) {
            vec3 core_color = vec3(CoreMatteColorR, CoreMatteColorG, CoreMatteColorB);
            float core_dist = distance(color, core_color);
            float protect_clip_black = CoreMatteTolerance - CoreMatteSoftness / 2.0;
            float protect_clip_white = CoreMatteTolerance + CoreMatteSoftness / 2.0;
            float protection_mask = 1.0 - smoothstep(protect_clip_black, protect_clip_white, core_dist);
            keyer_mask = max(keyer_mask, protection_mask);
        }
        return vec4(keyer_mask);
    }
}

// =========================================================================================
// FIM DA CADEIA DE GERAÇÃO DE MATTE
// O restante do shader continua a partir daqui, lendo de PASS0
// =========================================================================================

//!HOOK MAIN
//!BIND OCOL
//!BIND PASS0
//!SAVE HAIR_MATTE
//!DESC Hair Detection (Orientation Filter + BG affinity)
vec4 hook() {
    if (UseHairDetection == 0u) {
        return vec4(0.0);
    }
    vec3 key_color = vec3(KeyColorR, KeyColorG, KeyColorB);
    float min_neg = 0.0;
    int num_orientations = 18;
    vec2 pt = OCOL_pt * max(0.5, HairWidth);
    const float k0 = -1.0, k1 = -2.0, k2 = 6.0, k3 = -2.0, k4 = -1.0;
    for (int i = 0; i < num_orientations; i++) {
        float angle = float(i) * 3.14159265 / float(num_orientations);
        vec2 dir = vec2(cos(angle), sin(angle));
        vec3 c0 = textureLod(OCOL_raw, OCOL_pos + dir * pt * -2.0, 0.0).rgb;
        vec3 c1 = textureLod(OCOL_raw, OCOL_pos + dir * pt * -1.0, 0.0).rgb;
        vec3 c2 = textureLod(OCOL_raw, OCOL_pos, 0.0).rgb;
        vec3 c3 = textureLod(OCOL_raw, OCOL_pos + dir * pt * 1.0, 0.0).rgb;
        vec3 c4 = textureLod(OCOL_raw, OCOL_pos + dir * pt * 2.0, 0.0).rgb;
        float l0 = dot(c0, vec3(0.2126, 0.7152, 0.0722));
        float l1 = dot(c1, vec3(0.2126, 0.7152, 0.0722));
        float l2 = dot(c2, vec3(0.2126, 0.7152, 0.0722));
        float l3 = dot(c3, vec3(0.2126, 0.7152, 0.0722));
        float l4 = dot(c4, vec3(0.2126, 0.7152, 0.0722));
        float resp = l0*k0 + l1*k1 + l2*k2 + l3*k3 + l4*k4;
        min_neg = min(min_neg, resp);
    }
    float raw_line = max(0.0, -min_neg);
    vec3 col = textureLod(OCOL_raw, OCOL_pos, 0.0).rgb;
    float d_key = distance(col, key_color);
    float bg_aff = 1.0 - smoothstep(0.0, 1.0, d_key);
    bg_aff *= smoothstep(0.5, 0.9, d_key);
    float mC = textureLod(PASS0_raw, PASS0_pos, 0.0).r;
    float core_suppress = smoothstep(0.88, 1.00, mC);
    float hair = raw_line * (1.0 - core_suppress)*HairIntensity;
    hair = pow(hair, 1.25) * HairIntensity * 18.0;
    return vec4(clamp(hair, 0.0, 1.0));
}

//!HOOK MAIN
//!BIND PASS0
//!BIND HAIR_MATTE
//!BIND OCOL
//!SAVE PASS_COMBINED
//!DESC Combine Keyer and Hair Mattes
vec4 hook() {
    float keyer_matte = textureLod(PASS0_raw, PASS0_pos, 0.0).r;
    if (UseHairDetection == 1u) {
        float hair_matte = textureLod(HAIR_MATTE_raw, HAIR_MATTE_pos, 0.0).r;
        vec3 original_color = textureLod(OCOL_raw, OCOL_pos, 0.0).rgb;
        vec3 key_color = vec3(KeyColorR, KeyColorG, KeyColorB);
        float core_suppression = 1.0 - smoothstep(0.95, 1.0, keyer_matte);
        float key_dist = distance(original_color, key_color);
        float key_color_suppression = smoothstep(KeyTolerance, KeyTolerance + 0.2, key_dist);
        float combined_suppression = core_suppression * key_color_suppression;
        float constrained_hair_matte = hair_matte * combined_suppression;
        return vec4(min(1.0, keyer_matte + constrained_hair_matte));
    }
    return vec4(keyer_matte);
}

//!HOOK MAIN
//!BIND PASS_COMBINED
//!SAVE PASS1
//!DESC Garbage Matte (Median Filter)
vec4 hook() {
    if (GarbageMatteClean == 0) {
        return textureLod(PASS_COMBINED_raw, PASS_COMBINED_pos, 0.0);
    }
    float m[9]; int k=0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) m[k++] = textureLod(PASS_COMBINED_raw, PASS_COMBINED_pos + vec2(i, j) * PASS_COMBINED_pt, 0.0).r;
    for (int i = 0; i < 5; ++i) for (int j = i + 1; j < 9; ++j) { if (m[i] > m[j]) { float t = m[i];
    m[i] = m[j]; m[j] = t; } }
    return vec4(m[4]);
}

//!HOOK MAIN
//!BIND PASS1
//!SAVE erode1
//!DESC Mask Erode/Dilate 1
vec4 hook() { float R = max(input_size.x,input_size.y); vec4 m = texture(PASS1_raw, PASS1_pos);
if(ErodeDilate1 > 0.0){ float max_m=0.; for(int x=-int(abs(ErodeDilate1*R*.03)); x<=int(abs(ErodeDilate1*R*.03)); ++x) for(int y=-int(abs(ErodeDilate1*R*.03)); y<=int(abs(ErodeDilate1*R*.03)); ++y) { if(distance(vec2(x,y),vec2(0))>abs(ErodeDilate1*R*.03)) continue; max_m=max(max_m,texture(PASS1_raw,PASS1_pos+vec2(x,y)/input_size).r); } m.r=max_m;
} else { float min_m=1.; for(int x=int(ErodeDilate1*R*.03); x<=-int(ErodeDilate1*R*.03); ++x) for(int y=int(ErodeDilate1*R*.03); y<=-int(ErodeDilate1*R*.03); ++y) { if(distance(vec2(x,y),vec2(0))>abs(ErodeDilate1*R*.03)) continue; min_m=min(min_m,texture(PASS1_raw,PASS1_pos+vec2(x,y)/input_size).r); } m.r=min_m;
} return m; }

//!HOOK MAIN
//!BIND erode1
//!SAVE PASS2
//!DESC Mask Erode/Dilate 1
vec4 hook() { float R = max(input_size.x,input_size.y);
vec4 m = texture(erode1_raw, erode1_pos); if(ErodeDilate2 > 0.0){ float max_m=0.;
for(int x=-int(abs(ErodeDilate2*R*.03)); x<=int(abs(ErodeDilate2*R*.03)); ++x) for(int y=-int(abs(ErodeDilate2*R*.03)); y<=int(abs(ErodeDilate2*R*.03)); ++y) { if(distance(vec2(x,y),vec2(0))>abs(ErodeDilate2*R*.03)) continue; max_m=max(max_m,texture(erode1_raw,erode1_pos+vec2(x,y)/input_size).r); } m.r=max_m; } else { float min_m=1.;
for(int x=int(ErodeDilate2*R*.03); x<=-int(ErodeDilate2*R*.03); ++x) for(int y=int(ErodeDilate2*R*.03); y<=-int(ErodeDilate2*R*.03); ++y) { if(distance(vec2(x,y),vec2(0))>abs(ErodeDilate2*R*.03)) continue; min_m=min(min_m,texture(erode1_raw,erode1_pos+vec2(x,y)/input_size).r); } m.r=min_m; } return m;
}

//!HOOK MAIN
//!BIND PASS2
//!SAVE PASS3
//!DESC Initial Softness Blur H (Controlled by MatteSoftenBlur)
#define get_weight(x, sigma) (exp(-(x) * (x) / (2.0 * sigma * sigma)))
vec4 hook() { float RADIUS = floor(max(input_size.x, input_size.y) * 0.05 * MatteSoftenBlur);
if (RADIUS < 1.0) return textureLod(PASS2_raw, PASS2_pos, 0.0); float SIGMA = RADIUS / 3.0;
vec4 csum = textureLod(PASS2_raw, PASS2_pos, 0.0) * get_weight(0.0, SIGMA); float wsum = get_weight(0.0, SIGMA);
for(float i = 1.0; i <= RADIUS; ++i) { float weight = get_weight(i, SIGMA);
csum += (textureLod(PASS2_raw, PASS2_pos + PASS2_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS2_raw, PASS2_pos + PASS2_pt * vec2(i, 0.0), 0.0)) * weight;
wsum += 2.0 * weight; } return csum / wsum;
}

//!HOOK MAIN
//!BIND PASS3
//!SAVE PASS4
//!DESC Initial Softness Blur V (Controlled by MatteSoftenBlur)
#define get_weight(x, sigma) (exp(-(x) * (x) / (2.0 * sigma * sigma)))
vec4 hook() { float RADIUS = floor(max(input_size.x, input_size.y) * 0.05 * MatteSoftenBlur);
if (RADIUS < 1.0) return textureLod(PASS3_raw, PASS3_pos, 0.0); float SIGMA = RADIUS / 3.0;
vec4 csum = textureLod(PASS3_raw, PASS3_pos, 0.0) * get_weight(0.0, SIGMA); float wsum = get_weight(0.0, SIGMA);
for(float i = 1.0; i <= RADIUS; ++i) { float weight = get_weight(i, SIGMA);
csum += (textureLod(PASS3_raw, PASS3_pos + PASS3_pt * vec2(0.0, -i), 0.0) + textureLod(PASS3_raw, PASS3_pos + PASS3_pt * vec2(0.0, i), 0.0)) * weight;
wsum += 2.0 * weight; } return csum / wsum;
}

//!HOOK MAIN
//!BIND PASS4
//!SAVE PASS5
//!DESC Matte Post-Remap (Black/White Remap)
float remap(float mask_value, float black, float white) { if (white == black) return (mask_value >= black) ? 1.0 : 0.0; float remapped = (mask_value - black) / (white - black); return clamp(remapped, 0.0, 1.0); }
vec4 hook() { float remapped_mask = remap(textureLod(PASS4_raw, PASS4_pos, 0.0).r, BlackRemap, WhiteRemap); return vec4(remapped_mask); }

//!HOOK MAIN
//!BIND PASS5
//!SAVE PASS_REFINED
//!DESC Matte Edge Refinement
vec4 hook() { float matte = textureLod(PASS5_raw, PASS5_pos, 0.0).r;
if(MatteShrink > 0.0) matte = pow(matte, 1.0 + MatteShrink); if(MatteExpand > 0.0) matte = 1.0 - pow(1.0 - matte, 1.0 + MatteExpand);
return vec4(matte); }

//!HOOK MAIN
//!BIND OCOL
//!BIND PASS_REFINED
//!SAVE PASS6
//!DESC FixBorders: PRE Pass
vec4 hook() { vec4 ocol = textureLod(OCOL_raw, OCOL_pos, 0.0);
vec4 final_mask = textureLod(PASS_REFINED_raw, PASS_REFINED_pos, 0.0); return vec4(ocol.rgb, final_mask.r); }

//!HOOK MAIN
//!BIND PASS6
//!SAVE PASS7
//!DESC FixBorders: Choke Pass
vec4 hook() { if (FixBorders == 0) return textureLod(PASS6_raw, PASS6_pos, 0.0);
int size = FixBorders; vec2 tex_pt = PASS6_pt; vec4 min_alpha_color = textureLod(PASS6_raw, PASS6_pos, 0.0);
for (int x = -size; x <= size; ++x) { for (int y = -size; y <= size; ++y) { if (distance(vec2(x, y), vec2(0.0, 0.0)) > float(size)) continue;
vec4 neighbor_color = textureLod(PASS6_raw, PASS6_pos + vec2(x, y) * tex_pt, 0.0); if (neighbor_color.a < min_alpha_color.a) { min_alpha_color = neighbor_color;
} } } return min_alpha_color; }

//!HOOK MAIN
//!BIND PASS7
//!SAVE DILATE_1
//!DESC FixBorders: Dilate Iteration 1
vec4 hook() { if ((FixBorders * 2) < 1) return textureLod(PASS7_raw, PASS7_pos, 0.0);
vec2 tex_pt = PASS7_pt; vec4 max_alpha_color = textureLod(PASS7_raw, PASS7_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(PASS7_raw, PASS7_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_1
//!SAVE DILATE_2
//!DESC FixBorders: Dilate Iteration 2
vec4 hook() { if ((FixBorders * 2) < 2) return textureLod(DILATE_1_raw, DILATE_1_pos, 0.0);
vec2 tex_pt = DILATE_1_pt; vec4 max_alpha_color = textureLod(DILATE_1_raw, DILATE_1_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_1_raw, DILATE_1_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_2
//!SAVE DILATE_3
//!DESC FixBorders: Dilate Iteration 3
vec4 hook() { if ((FixBorders * 2) < 3) return textureLod(DILATE_2_raw, DILATE_2_pos, 0.0);
vec2 tex_pt = DILATE_2_pt; vec4 max_alpha_color = textureLod(DILATE_2_raw, DILATE_2_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_2_raw, DILATE_2_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_3
//!SAVE DILATE_4
//!DESC FixBorders: Dilate Iteration 4
vec4 hook() { if ((FixBorders * 2) < 4) return textureLod(DILATE_3_raw, DILATE_3_pos, 0.0);
vec2 tex_pt = DILATE_3_pt; vec4 max_alpha_color = textureLod(DILATE_3_raw, DILATE_3_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_3_raw, DILATE_3_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_4
//!SAVE DILATE_5
//!DESC FixBorders: Dilate Iteration 5
vec4 hook() { if ((FixBorders * 2) < 5) return textureLod(DILATE_4_raw, DILATE_4_pos, 0.0);
vec2 tex_pt = DILATE_4_pt; vec4 max_alpha_color = textureLod(DILATE_4_raw, DILATE_4_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_4_raw, DILATE_4_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_5
//!SAVE DILATE_6
//!DESC FixBorders: Dilate Iteration 6
vec4 hook() { if ((FixBorders * 2) < 6) return textureLod(DILATE_5_raw, DILATE_5_pos, 0.0);
vec2 tex_pt = DILATE_5_pt; vec4 max_alpha_color = textureLod(DILATE_5_raw, DILATE_5_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_5_raw, DILATE_5_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_6
//!SAVE DILATE_7
//!DESC FixBorders: Dilate Iteration 7
vec4 hook() { if ((FixBorders * 2) < 7) return textureLod(DILATE_6_raw, DILATE_6_pos, 0.0);
vec2 tex_pt = DILATE_6_pt; vec4 max_alpha_color = textureLod(DILATE_6_raw, DILATE_6_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_6_raw, DILATE_6_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_7
//!SAVE DILATE_8
//!DESC FixBorders: Dilate Iteration 8
vec4 hook() { if ((FixBorders * 2) < 8) return textureLod(DILATE_7_raw, DILATE_7_pos, 0.0);
vec2 tex_pt = DILATE_7_pt; vec4 max_alpha_color = textureLod(DILATE_7_raw, DILATE_7_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_7_raw, DILATE_7_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_8
//!SAVE DILATE_9
//!DESC FixBorders: Dilate Iteration 9
vec4 hook() { if ((FixBorders * 2) < 9) return textureLod(DILATE_8_raw, DILATE_8_pos, 0.0);
vec2 tex_pt = DILATE_8_pt; vec4 max_alpha_color = textureLod(DILATE_8_raw, DILATE_8_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_8_raw, DILATE_8_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_9
//!SAVE DILATE_10
//!DESC FixBorders: Dilate Iteration 10
vec4 hook() { if ((FixBorders * 2) < 10) return textureLod(DILATE_9_raw, DILATE_9_pos, 0.0);
vec2 tex_pt = DILATE_9_pt; vec4 max_alpha_color = textureLod(DILATE_9_raw, DILATE_9_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_9_raw, DILATE_9_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_10
//!SAVE DILATE_11
//!DESC FixBorders: Dilate Iteration 11
vec4 hook() { if ((FixBorders * 2) < 11) return textureLod(DILATE_10_raw, DILATE_10_pos, 0.0);
vec2 tex_pt = DILATE_10_pt; vec4 max_alpha_color = textureLod(DILATE_10_raw, DILATE_10_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_10_raw, DILATE_10_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_11
//!SAVE DILATE_12
//!DESC FixBorders: Dilate Iteration 12
vec4 hook() { if ((FixBorders * 2) < 12) return textureLod(DILATE_11_raw, DILATE_11_pos, 0.0);
vec2 tex_pt = DILATE_11_pt; vec4 max_alpha_color = textureLod(DILATE_11_raw, DILATE_11_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_11_raw, DILATE_11_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_12
//!SAVE DILATE_13
//!DESC FixBorders: Dilate Iteration 13
vec4 hook() { if ((FixBorders * 2) < 13) return textureLod(DILATE_12_raw, DILATE_12_pos, 0.0);
vec2 tex_pt = DILATE_12_pt; vec4 max_alpha_color = textureLod(DILATE_12_raw, DILATE_12_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_12_raw, DILATE_12_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_13
//!SAVE DILATE_14
//!DESC FixBorders: Dilate Iteration 14
vec4 hook() { if ((FixBorders * 2) < 14) return textureLod(DILATE_13_raw, DILATE_13_pos, 0.0);
vec2 tex_pt = DILATE_13_pt; vec4 max_alpha_color = textureLod(DILATE_13_raw, DILATE_13_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_13_raw, DILATE_13_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_14
//!SAVE DILATE_15
//!DESC FixBorders: Dilate Iteration 15
vec4 hook() { if ((FixBorders * 2) < 15) return textureLod(DILATE_14_raw, DILATE_14_pos, 0.0);
vec2 tex_pt = DILATE_14_pt; vec4 max_alpha_color = textureLod(DILATE_14_raw, DILATE_14_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_14_raw, DILATE_14_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_15
//!SAVE DILATE_16
//!DESC FixBorders: Dilate Iteration 16
vec4 hook() { if ((FixBorders * 2) < 16) return textureLod(DILATE_15_raw, DILATE_15_pos, 0.0);
vec2 tex_pt = DILATE_15_pt; vec4 max_alpha_color = textureLod(DILATE_15_raw, DILATE_15_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_15_raw, DILATE_15_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_16
//!SAVE DILATE_17
//!DESC FixBorders: Dilate Iteration 17
vec4 hook() { if ((FixBorders * 2) < 17) return textureLod(DILATE_16_raw, DILATE_16_pos, 0.0);
vec2 tex_pt = DILATE_16_pt; vec4 max_alpha_color = textureLod(DILATE_16_raw, DILATE_16_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_16_raw, DILATE_16_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_17
//!SAVE DILATE_18
//!DESC FixBorders: Dilate Iteration 18
vec4 hook() { if ((FixBorders * 2) < 18) return textureLod(DILATE_17_raw, DILATE_17_pos, 0.0);
vec2 tex_pt = DILATE_17_pt; vec4 max_alpha_color = textureLod(DILATE_17_raw, DILATE_17_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_17_raw, DILATE_17_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}
//!HOOK MAIN
//!BIND DILATE_18
//!SAVE DILATE_19
//!DESC FixBorders: Dilate Iteration 19
vec4 hook() { if ((FixBorders * 2) < 19) return textureLod(DILATE_18_raw, DILATE_18_pos, 0.0);
vec2 tex_pt = DILATE_18_pt; vec4 max_alpha_color = textureLod(DILATE_18_raw, DILATE_18_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_18_raw, DILATE_18_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}

//!HOOK MAIN
//!BIND DILATE_19
//!SAVE PASS8
//!DESC FixBorders: Dilate Iteration 20
vec4 hook() { if ((FixBorders * 2) < 20) return textureLod(DILATE_19_raw, DILATE_19_pos, 0.0);
vec2 tex_pt = DILATE_19_pt; vec4 max_alpha_color = textureLod(DILATE_19_raw, DILATE_19_pos, 0.0);
for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) { vec4 n = textureLod(DILATE_19_raw, DILATE_19_pos + vec2(x,y)*tex_pt, 0.0);
if (n.a > max_alpha_color.a) max_alpha_color = n; } return max_alpha_color;
}

//!HOOK MAIN
//!BIND PASS_REFINED
//!SAVE FinalBlurH
//!DESC (Inward Feather Step 1) Standard Horizontal Blur
#define get_weight(x, sigma) (exp(-(x) * (x) / (2.0 * sigma * sigma)))
vec4 hook() {
    float RADIUS = floor(max(input_size.x, input_size.y) * 0.05 * FinalMatteBlur);
    if (RADIUS < 1.0) return textureLod(PASS_REFINED_raw, PASS_REFINED_pos, 0.0);
    float SIGMA = RADIUS / 3.0;
    vec4 csum = textureLod(PASS_REFINED_raw, PASS_REFINED_pos, 0.0) * get_weight(0.0, SIGMA);
    float wsum = get_weight(0.0, SIGMA);
    for(float i = 1.0; i <= RADIUS; ++i) {
        float weight = get_weight(i, SIGMA);
        csum += (textureLod(PASS_REFINED_raw, PASS_REFINED_pos + PASS_REFINED_pt * vec2(-i, 0.0), 0.0) + textureLod(PASS_REFINED_raw, PASS_REFINED_pos + PASS_REFINED_pt * vec2(i, 0.0), 0.0)) * weight;
        wsum += 2.0 * weight;
    }
    return csum / wsum;
}

//!HOOK MAIN
//!BIND FinalBlurH
//!SAVE FinalBlurV
//!DESC (Inward Feather Step 2) Standard Vertical Blur
#define get_weight(x, sigma) (exp(-(x) * (x) / (2.0 * sigma * sigma)))
vec4 hook() {
    float RADIUS = floor(max(input_size.x, input_size.y) * 0.05 * FinalMatteBlur);
    if (RADIUS < 1.0) return textureLod(FinalBlurH_raw, FinalBlurH_pos, 0.0);
    float SIGMA = RADIUS / 3.0;
    vec4 csum = textureLod(FinalBlurH_raw, FinalBlurH_pos, 0.0) * get_weight(0.0, SIGMA);
    float wsum = get_weight(0.0, SIGMA);
    for(float i = 1.0; i <= RADIUS; ++i) {
        float weight = get_weight(i, SIGMA);
        csum += (textureLod(FinalBlurH_raw, FinalBlurH_pos + FinalBlurH_pt * vec2(0.0, -i), 0.0) + textureLod(FinalBlurH_raw, FinalBlurH_pos + FinalBlurH_pt * vec2(0.0, i), 0.0)) * weight;
        wsum += 2.0 * weight;
    }
    return csum / wsum;
}

//!HOOK MAIN
//!BIND PASS_REFINED
//!BIND FinalBlurV
//!SAVE PASS10
//!DESC (Inward Feather Step 3) Stencil blurred matte with original matte.
vec4 hook() {
    float stencil_matte = textureLod(PASS_REFINED_raw, PASS_REFINED_pos, 0.0).r;
    float blurred_matte = textureLod(FinalBlurV_raw, FinalBlurV_pos, 0.0).r;
    float final_matte = blurred_matte * stencil_matte;
    return vec4(final_matte);
}

//!HOOK MAIN
//!BIND PASS10
//!BIND PASS_REFINED
//!BIND PASS8
//!BIND OCOL
//!BIND CLEAN_PLATE_FINAL
//!DESC Final Pass
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
vec3 rgb2ycbcr_U(vec3 rgb){
    float Y  = dot(rgb, vec3(0.299, 0.587, 0.114));
    float Cb = 0.564 * (rgb.b - Y);
    float Cr = 0.713 * (rgb.r - Y);
    return vec3(Y, Cb, Cr);
}
vec3 ycbcr2rgb_U(vec3 ycc){
    float Y=ycc.x, Cb=ycc.y, Cr=ycc.z;
    float R = Y + Cr/0.713;
    float B = Y + Cb/0.564;
    float G = (Y - 0.299*R - 0.114*B)/0.587;
    return clamp(vec3(R,G,B), 0.0, 1.0);
}
vec3 suppress_spill_simple(vec3 color, vec3 spill_color, float strength, float tolerance) { vec3 hsl_color; { float max_c = max(color.r, max(color.g, color.b));
float min_c = min(color.r, min(color.g, color.b)); float delta = max_c - min_c; float h = 0.0;
if (delta > 0.0) { if (max_c == color.r) { h = mod((color.g - color.b) / delta, 6.0);
} else if (max_c == color.g) { h = (color.b - color.r) / delta + 2.0;
} else { h = (color.r - color.g) / delta + 4.0; } h /= 6.0;
} float l = (max_c + min_c) / 2.0; float s = (l > 0.0 && l < 1.0) ?
delta / (1.0 - abs(2.0 * l - 1.0)) : 0.0; hsl_color = vec3(h, s, l); } vec3 hsl_spill;
{ float max_c = max(spill_color.r, max(spill_color.g, spill_color.b)); float min_c = min(spill_color.r, min(spill_color.g, spill_color.b)); float delta = max_c - min_c;
float h = 0.0; if (delta > 0.0) { if (max_c == spill_color.r) { h = mod((spill_color.g - spill_color.b) / delta, 6.0);
} else if (max_c == spill_color.g) { h = (spill_color.b - spill_color.r) / delta + 2.0;
} else { h = (spill_color.r - spill_color.g) / delta + 4.0; } h /= 6.0;
} float l = (max_c + min_c) / 2.0; float s = (l > 0.0 && l < 1.0) ?
delta / (1.0 - abs(2.0 * l - 1.0)) : 0.0; hsl_spill = vec3(h, s, l);
} float hue_dist = abs(hsl_color.r - hsl_spill.r); hue_dist = min(hue_dist, 1.0 - hue_dist); float width = 1.0 / tolerance;
float suppression_factor = (1.0 - smoothstep(0.0, width, hue_dist)) * strength; return mix(color, vec3(luma(color)), suppression_factor);
}
vec4 hook() {
    vec4 original_pixel = textureLod(OCOL_raw, OCOL_pos, 0.0);
    vec4 trueAlpha = textureLod(PASS10_raw, PASS10_pos, 0.0)*textureLod(PASS_REFINED_raw, PASS_REFINED_pos, 0.0)*vec4(original_pixel.a);
    vec4 pCol = textureLod(PASS8_raw, PASS8_pos, 0.0);
    
    vec3 color = pCol.rgb;
    vec3 SpillColor = vec3(KeyColorR,KeyColorG,KeyColorB);
    float fAlpha = clamp(trueAlpha.a,0.0,1.0);
    if (ColorUnmix == 1) {
        vec3 C = color;
        float a = fAlpha;
        vec3 key_color = vec3(KeyColorR, KeyColorG, KeyColorB);
        vec3 key_ycbcr = rgb2ycbcr_U(key_color);
        vec3 C_ycbcr   = rgb2ycbcr_U(C);
        vec3 B_ycbcr   = vec3(C_ycbcr.x, key_ycbcr.y, key_ycbcr.z);
        vec3 B         = ycbcr2rgb_U(B_ycbcr);
        float eps = 1e-5;
        vec3 F = (C - (1.0 - a) * B) / max(a, eps);
        color = mix(color,clamp(F, 0.0, 1.0),Unmix);
    }
    
    vec3 Despill;
    if (DespillMode == 0) {
        Despill = suppress_spill_simple(color, SpillColor, SpillSuppressor, SpillTolerance);
    } else {
        vec3 key_color = vec3(KeyColorR, KeyColorG, KeyColorB);
        if (key_color.g > key_color.r && key_color.g > key_color.b) {
            float limit = max(color.r, color.b);
            float spill_amount = max(0.0, color.g - limit) * SpillSuppressor;
            Despill = vec3(
                color.r + spill_amount * 0.5,
                color.g - spill_amount,
                color.b + spill_amount * 0.5
            );
        } else if (key_color.b > key_color.r && key_color.b > key_color.g) {
            float limit = max(color.r, color.g);
            float spill_amount = max(0.0, color.b - limit) * SpillSuppressor;
            Despill = vec3(
                color.r + spill_amount * 0.5,
                color.g + spill_amount * 0.5,
                color.b - spill_amount
            );
        } else {
            float limit = max(color.g, color.b);
            float spill_amount = max(0.0, color.r - limit) * SpillSuppressor;
            Despill = vec3(
                color.r - spill_amount,
                color.g + spill_amount * 0.5,
                color.b + spill_amount * 0.5
            );
        }
    }

    vec4 outCol = vec4(Despill*fAlpha, fAlpha);
    if(ViewMatte == 1u){
        outCol = vec4(vec3(fAlpha), 1.0);
    }else if(ViewCleanPlate == 1u){        
        outCol = textureLod(CLEAN_PLATE_FINAL_raw, CLEAN_PLATE_FINAL_pos, 0.0);
    } else if (TestBackground == 1u) {
        vec3 test_bg_color = vec3(TestBgR, TestBgG, TestBgB);
        vec3 final_rgb = outCol.rgb + test_bg_color * (1.0 - outCol.a);
        outCol = vec4(final_rgb, 1.0);
    }

    float split;
    if(SplitView <= 1.0){
        split = SplitView;
        if(OCOL_pos.x > split){ outCol = outCol;
        } else { outCol = original_pixel; }
    }else{
        split = SplitView - 1.0;
        if(split > OCOL_pos.x){ outCol = outCol; } else { outCol = original_pixel;
        }
    }
    
    return outCol;
}