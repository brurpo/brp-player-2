// Portuguese names for the BRP Player effects panel (plain comments: mpv ignores them).
// @brp pt.title Glitch
// @brp pt.param MasterIntensity Intensidade geral | Força geral de todos os efeitos de glitch juntos.
// @brp pt.param Speed Velocidade | Velocidade geral das animações e da evolução do glitch.
// @brp pt.param Probability Probabilidade | Limiar de ativação do glitch, sobre uma onda de ruído suave. Valores maiores dão glitches mais frequentes.
// @brp pt.param Seed Semente | Muda o padrão aleatório dos glitches. Qualquer valor dá uma nova sequência.
// @brp pt.param BlockSize Tamanho do bloco | Tamanho dos blocos do glitch. Valores menores = blocos maiores.
// @brp pt.param SliceHeight Altura da fatia | Altura das fatias do glitch. Valores menores = fatias mais finas.
// @brp pt.param SliceShiftAmount Deslocamento da fatia | Distância horizontal máxima do deslocamento das fatias.
// @brp pt.param BlockGlitch Glitch de blocos | Intensidade do glitch de macroblocos e do deslocamento de blocos.
// @brp pt.param SliceGlitch Glitch de fatias | Intensidade do deslocamento horizontal das fatias.
// @brp pt.param ColorShift Troca de cor | Probabilidade de um bloco ter a cor (matiz) trocada.
// @brp pt.param ColorShiftAmount Quantidade da troca | Troca máxima de matiz num bloco com glitch.
// @brp pt.param BlockRgbShift Separação RGB em blocos | Intensidade da separação dos canais RGB por blocos.
// @brp pt.param ScanlineRgbShift Separação RGB em linhas | Intensidade da separação dos canais RGB por linhas.
// @brp pt.param Noise Ruído | Quantidade de ruído digital sobre a imagem.
// @brp pt.param Scanlines Linhas de varredura | Intensidade do efeito de linhas horizontais.
//!PARAM MasterIntensity
//!DESC Controls the overall strength of all glitch effects combined.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM Speed
//!DESC Controls the overall speed of the glitch animations and evolution.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
1.0

//!PARAM Probability
//!DESC Threshold for glitch activation based on a smooth noise wave. Higher values mean more frequent glitches.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.8

//!PARAM Seed
//!DESC Changes the random pattern of the glitches. Set to any value to get a new sequence.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 100.0
1.0


//!PARAM BlockSize
//!DESC Controls the size of the glitch blocks. Smaller values = larger blocks.
//!TYPE float
//!MINIMUM 4.0
//!MAXIMUM 256.0
32.0

//!PARAM SliceHeight
//!DESC Controls the height of the glitch slices. Smaller values = thinner slices.
//!TYPE float
//!MINIMUM 0.005
//!MAXIMUM 0.5
0.05

//!PARAM SliceShiftAmount
//!DESC Controls the maximum horizontal shift distance for the slice glitch.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.5
0.2


//!PARAM BlockGlitch
//!DESC Intensity of the macroblocking and block displacement glitch.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM SliceGlitch
//!DESC Intensity of the horizontal slice displacement glitch.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM ColorShift
//!DESC Probability of a block having its color (hue) shifted.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!PARAM ColorShiftAmount
//!DESC The maximum amount of hue shift to apply on a glitched block.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM BlockRgbShift
//!DESC Intensity of the block-based RGB channel separation.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.3

//!PARAM ScanlineRgbShift
//!DESC Intensity of the scanline-based RGB channel separation.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.2

//!PARAM Noise
//!DESC Amount of digital noise to overlay on the image.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.5
0.1

//!PARAM Scanlines
//!DESC Intensity of the horizontal scanline effect.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.1


//!HOOK MAIN
//!BIND HOOKED
//!DESC Simulates various digital video glitch artifacts with fine-grained control.

// --- Helper Functions ---
float glitch_rand(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 rgb2hsv(vec3 c) { vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0); vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g)); vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r)); float d = q.x - min(q.w, q.y); float e = 1.0e-10; return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x); }
vec3 hsv2rgb(vec3 c) { vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0); vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www); return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y); }
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }
float snoise(vec2 v) { const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439); vec2 i  = floor(v + dot(v, C.yy) ); vec2 x0 = v -   i + dot(i, C.xx); vec2 i1; i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0); vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1; i = mod289(i); vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 )); vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0); m = m*m; m = m*m; vec3 x = 2.0 * fract(p * C.www) - 1.0; vec3 h = abs(x) - 0.5; vec3 ox = floor(x + 0.5); vec3 a0 = x - ox; m *= (1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h )); vec3 g; g.x  = a0.x  * x0.x  + h.x  * x0.y; g.yz = a0.yz * x12.xz + h.yz * x12.yw; return 130.0 * dot(m, g); }


vec4 hook() {
    // --- Initial Setup ---
    vec2 uv = HOOKED_pos;
    vec4 original_color = texture(HOOKED_raw, uv);
    float time = (frame * 0.01 * Speed) + Seed;

    // --- Master Probability Check (Continuous) ---
    float noise_wave = (snoise(vec2(time, time * 0.7)) + 1.0) * 0.5;
    float continuous_intensity = smoothstep(1.0 - Probability, 1.0, noise_wave);
    if (continuous_intensity <= 0.0) {
        return original_color;
    }

    // --- Glitch Logic ---
    vec3 color = vec3(0.0);
    float final_master_intensity = MasterIntensity * continuous_intensity;
    
    // Read intensities from parameters
    float block_intensity = BlockGlitch * final_master_intensity;
    float slice_intensity = SliceGlitch * final_master_intensity;
    float color_shift_intensity = ColorShift * final_master_intensity;
    float color_shift_amount = ColorShiftAmount * final_master_intensity;
    float block_rgb_shift_intensity = BlockRgbShift * final_master_intensity;
    float scanline_rgb_shift_intensity = ScanlineRgbShift * final_master_intensity;
    float noise_intensity = Noise * final_master_intensity;
    float scan_intensity = Scanlines * final_master_intensity;

    // 1. Block and Slice UV Displacement
    vec2 block_uv = floor(uv * BlockSize) / BlockSize;
    if (glitch_rand(block_uv + time) < block_intensity) {
        uv.x -= snoise(vec2(time * 2.0, block_uv.y)) * 0.1 * block_intensity;
        uv.y -= snoise(vec2(block_uv.x, time * 2.0)) * 0.1 * block_intensity;
    }
    
    float slice_y = floor(uv.y / SliceHeight) * SliceHeight;
    if (glitch_rand(vec2(slice_y, time)) < slice_intensity) {
        float shift_amount = snoise(vec2(time, slice_y)) * SliceShiftAmount;
        uv.x += shift_amount;
    }
    
    // 2. Color Sampling (with two types of RGB Shift)
    float r_block_shift = snoise(vec2(time, block_uv.y + 10.0)) * 0.05 * block_rgb_shift_intensity;
    float g_block_shift = snoise(vec2(time, block_uv.y + 20.0)) * 0.05 * block_rgb_shift_intensity;
    float scanline_shift_offset = snoise(vec2(uv.y * 4.0, time)) * 0.02 * scanline_rgb_shift_intensity;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        color = vec3(0.0);
    } else {
        color.r = texture(HOOKED_raw, uv + vec2(r_block_shift + scanline_shift_offset, 0.0)).r;
        color.g = texture(HOOKED_raw, uv + vec2(g_block_shift, 0.0)).g;
        color.b = texture(HOOKED_raw, uv - vec2(scanline_shift_offset, 0.0)).b;
    }

    // 3. Color Shifting by Blocks (Hue Shift)
    // ** FIX IS HERE ** Broke the 'if' condition into two lines to avoid parsing errors.
    float hue_shift_check = glitch_rand(block_uv + 0.5 + time);
    if (hue_shift_check < color_shift_intensity) {
        vec3 hsv = rgb2hsv(color);
        float smooth_hue_shift = snoise(vec2(block_uv.x, time)) * color_shift_amount;
        hsv.x = fract(hsv.x + smooth_hue_shift);
        color = hsv2rgb(hsv);
    }

    // 4. Noise and Scanline Overlay
    color += (glitch_rand(uv + time) - 0.5) * noise_intensity;
    float scanline_val = sin(uv.y * 800.0) * 0.5 + 0.5;
    color *= mix(1.0, scanline_val, scan_intensity);

    // --- Final Output ---
    return vec4(clamp(color, 0.0, 1.0), original_color.a);
}