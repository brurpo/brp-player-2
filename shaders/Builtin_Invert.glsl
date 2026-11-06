//!PARAM OnOff
//!DESC Intensity of the effect
//!TYPE uint
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM intensity
//!DESC Intensity of the effect
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM intensity2
//!DESC Intensity of the effect
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 100
100

//!HOOK MAIN
//!BIND HOOKED

vec4 hook() {
    // Pega a cor original do pixel atual
    vec4 original_color = HOOKED_tex(HOOKED_pos);

    // Inverte os canais de cor (RGB)
    vec3 inverted_color = vec3(1.0) - original_color.rgb;

    // Mistura a cor original com a invertida usando o parâmetro 'mix'
    vec3 final_color = mix(original_color.rgb, inverted_color, intensity * float(OnOff) * (float(intensity2) / 100.0));

    // Retorna a cor final, mantendo o canal alfa original
    return vec4(final_color, original_color.a);
}
