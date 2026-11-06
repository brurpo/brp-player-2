//!PARAM BlurAmount
//!DESC Number of samples along the motion path. Higher is more blur.
//!TYPE float
//!MINIMUM 2.0
//!MAXIMUM 64.0
16.0

//!PARAM SearchRadius
//!DESC How many pixels to search for motion. Higher finds faster motion but is slower.
//!TYPE int
//!MINIMUM 4
//!MAXIMUM 32
32

//!TEXTURE LastFrameTex
//!SIZE 3840 2160 
//!FORMAT rgba32f
//!STORAGE

//!HOOK MAIN
//!BIND HOOKED
//!BIND LastFrameTex
//!DESC GPU Motion Blur - VECTOR DEBUG VIEW

vec2 block_match(ivec2 pixel_coord, sampler2D current_tex, layout(rgba16f) readonly image2D prev_tex_storage, int SR) {
    float best_match = 1e5;
    vec2 motion_vec = vec2(0.0);
    int search_radius = int(SR);

    vec3 current_block[9];
    int idx = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            current_block[idx++] = texelFetch(current_tex, pixel_coord + ivec2(i, j), 0).rgb;
        }
    }

    for (int y = -search_radius; y <= search_radius; y++) {
        for (int x = -search_radius; x <= search_radius; x++) {
            ivec2 search_coord = pixel_coord + ivec2(x, y);
            float sad = 0.0;
            idx = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    vec3 prev_pixel = imageLoad(prev_tex_storage, search_coord + ivec2(i, j)).rgb;
                    sad += abs(current_block[idx].r - prev_pixel.r) + abs(current_block[idx].g - prev_pixel.g) + abs(current_block[idx].b - prev_pixel.b);
                    idx++;
                }
            }
            if (sad < best_match) {
                best_match = sad;
                motion_vec = vec2(x, y);
            }
        }
    }
    return motion_vec;
}

vec4 hook() {
    ivec2 current_pos_ivec = ivec2(HOOKED_pos * HOOKED_size);
    vec4 final_color;
    int SSR = SearchRadius;

    if (frame == 0) {
        // No primeiro frame, não há movimento. Retorna a cor cinza que representa o vetor (0,0).
        final_color = vec4(0.5, 0.5, 0.0, 1.0);

    } else {
        // Calcula o vetor de movimento em pixels.
        vec2 motion_px = block_match(current_pos_ivec, HOOKED_raw, LastFrameTex, SSR);

        // --- INÍCIO DA MODIFICAÇÃO: VISUALIZAÇÃO DO VETOR ---
        
        // Normaliza o vetor de movimento. O intervalo de motion_px é [-SearchRadius, +SearchRadius].
        // Mapeamos este intervalo para [0.0, 1.0] para que possa ser exibido como uma cor.
        vec2 normalized_motion = (motion_px / SearchRadius) * 0.5 + 0.5;

        // Atribui o vetor normalizado aos canais Vermelho e Verde.
        final_color = vec4(normalized_motion.x, normalized_motion.y, 0.0, 1.0);

        // --- FIM DA MODIFICAÇÃO ---
    }

    // O LOOP DE FEEDBACK continua essencial para que a estimação funcione no próximo frame.
    // Ele salva o frame atual (sempre o original) para a próxima iteração.
    imageStore(LastFrameTex, current_pos_ivec, HOOKED_texOff(vec2(0.0)));

    return final_color;
}


