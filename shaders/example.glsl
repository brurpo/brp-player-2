// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Exemplo
// @brp pt.param KeyColorR Cor da chave | Intensidade do efeito.
// @brp pt.param BColorR Cor de fundo | Intensidade do efeito.
// @brp pt.param MIX Mistura | Intensidade do efeito.
//!PARAM KeyColorR
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM KeyColorG
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM KeyColorB
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM BColorR
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM BColorG
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0
//!PARAM BColorB
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0
//!PARAM MIX
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0



//!HOOK MAIN
//!BIND HOOKED

vec4 hook() {
    // Pega a cor original do pixel atual
    vec4 original_color = HOOKED_tex(HOOKED_pos);
    vec3 mixed = mix((vec3(KeyColorR, KeyColorG, KeyColorB)),(vec3(BColorR, BColorG, BColorB)),MIX);

    return vec4(mixed, original_color.a);
}
