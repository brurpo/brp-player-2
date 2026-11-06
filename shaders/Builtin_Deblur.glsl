//!PARAM Iterations
//!DESC Number of deconvolution iterations. Higher is sharper but much slower.
//!TYPE int
//!MINIMUM 1
//!MAXIMUM 10
5

//!PARAM BlurRadius
//!DESC Estimated radius of the blur in the source footage. This is the most critical parameter.
//!TYPE float
//!MINIMUM 0.5
//!MAXIMUM 10.0
2.0

//!PARAM Regularization
//!DESC Prevents noise amplification and artifacts. Increase if you see ringing or worms.
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 0.1
0.001

//!HOOK MAIN
//!BIND HOOKED
//!SAVE ITER_1_CONV
//!DESC Richardson-Lucy Deblur - Iteration 1 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 1) { return vec4(0.0); } return convolve(HOOKED_raw, HOOKED_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_1_CONV
//!SAVE ITER_1_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 1 Ratio
vec4 hook() { if (Iterations < 1) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_1_CONV_raw, ITER_1_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_1_RATIO
//!SAVE ITER_1_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 1 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 1) { return vec4(0.0); } return convolve(ITER_1_RATIO_raw, ITER_1_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_1_CORRECTION
//!SAVE ITER_1_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 1 Final
vec4 hook() { if (Iterations < 1) { return texture(HOOKED_raw, HOOKED_pos); } vec4 current_est = texture(HOOKED_raw, HOOKED_pos); vec4 update_factor = texture(ITER_1_CORRECTION_raw, ITER_1_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_1_FINAL
//!SAVE ITER_2_CONV
//!DESC Richardson-Lucy Deblur - Iteration 2 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 2) { return vec4(0.0); } return convolve(ITER_1_FINAL_raw, ITER_1_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_2_CONV
//!SAVE ITER_2_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 2 Ratio
vec4 hook() { if (Iterations < 2) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_2_CONV_raw, ITER_2_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_2_RATIO
//!SAVE ITER_2_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 2 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 2) { return vec4(0.0); } return convolve(ITER_2_RATIO_raw, ITER_2_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_1_FINAL
//!BIND ITER_2_CORRECTION
//!SAVE ITER_2_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 2 Final
vec4 hook() { if (Iterations < 2) { return texture(ITER_1_FINAL_raw, ITER_1_FINAL_pos); } vec4 current_est = texture(ITER_1_FINAL_raw, ITER_1_FINAL_pos); vec4 update_factor = texture(ITER_2_CORRECTION_raw, ITER_2_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_2_FINAL
//!SAVE ITER_3_CONV
//!DESC Richardson-Lucy Deblur - Iteration 3 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 3) { return vec4(0.0); } return convolve(ITER_2_FINAL_raw, ITER_2_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_3_CONV
//!SAVE ITER_3_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 3 Ratio
vec4 hook() { if (Iterations < 3) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_3_CONV_raw, ITER_3_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_3_RATIO
//!SAVE ITER_3_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 3 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 3) { return vec4(0.0); } return convolve(ITER_3_RATIO_raw, ITER_3_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_2_FINAL
//!BIND ITER_3_CORRECTION
//!SAVE ITER_3_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 3 Final
vec4 hook() { if (Iterations < 3) { return texture(ITER_2_FINAL_raw, ITER_2_FINAL_pos); } vec4 current_est = texture(ITER_2_FINAL_raw, ITER_2_FINAL_pos); vec4 update_factor = texture(ITER_3_CORRECTION_raw, ITER_3_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_3_FINAL
//!SAVE ITER_4_CONV
//!DESC Richardson-Lucy Deblur - Iteration 4 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 4) { return vec4(0.0); } return convolve(ITER_3_FINAL_raw, ITER_3_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_4_CONV
//!SAVE ITER_4_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 4 Ratio
vec4 hook() { if (Iterations < 4) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_4_CONV_raw, ITER_4_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_4_RATIO
//!SAVE ITER_4_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 4 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 4) { return vec4(0.0); } return convolve(ITER_4_RATIO_raw, ITER_4_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_3_FINAL
//!BIND ITER_4_CORRECTION
//!SAVE ITER_4_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 4 Final
vec4 hook() { if (Iterations < 4) { return texture(ITER_3_FINAL_raw, ITER_3_FINAL_pos); } vec4 current_est = texture(ITER_3_FINAL_raw, ITER_3_FINAL_pos); vec4 update_factor = texture(ITER_4_CORRECTION_raw, ITER_4_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_4_FINAL
//!SAVE ITER_5_CONV
//!DESC Richardson-Lucy Deblur - Iteration 5 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 5) { return vec4(0.0); } return convolve(ITER_4_FINAL_raw, ITER_4_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_5_CONV
//!SAVE ITER_5_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 5 Ratio
vec4 hook() { if (Iterations < 5) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_5_CONV_raw, ITER_5_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_5_RATIO
//!SAVE ITER_5_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 5 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 5) { return vec4(0.0); } return convolve(ITER_5_RATIO_raw, ITER_5_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_4_FINAL
//!BIND ITER_5_CORRECTION
//!SAVE ITER_5_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 5 Final
vec4 hook() { if (Iterations < 5) { return texture(ITER_4_FINAL_raw, ITER_4_FINAL_pos); } vec4 current_est = texture(ITER_4_FINAL_raw, ITER_4_FINAL_pos); vec4 update_factor = texture(ITER_5_CORRECTION_raw, ITER_5_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_5_FINAL
//!SAVE ITER_6_CONV
//!DESC Richardson-Lucy Deblur - Iteration 6 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 6) { return vec4(0.0); } return convolve(ITER_5_FINAL_raw, ITER_5_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_6_CONV
//!SAVE ITER_6_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 6 Ratio
vec4 hook() { if (Iterations < 6) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_6_CONV_raw, ITER_6_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_6_RATIO
//!SAVE ITER_6_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 6 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 6) { return vec4(0.0); } return convolve(ITER_6_RATIO_raw, ITER_6_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_5_FINAL
//!BIND ITER_6_CORRECTION
//!SAVE ITER_6_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 6 Final
vec4 hook() { if (Iterations < 6) { return texture(ITER_5_FINAL_raw, ITER_5_FINAL_pos); } vec4 current_est = texture(ITER_5_FINAL_raw, ITER_5_FINAL_pos); vec4 update_factor = texture(ITER_6_CORRECTION_raw, ITER_6_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_6_FINAL
//!SAVE ITER_7_CONV
//!DESC Richardson-Lucy Deblur - Iteration 7 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 7) { return vec4(0.0); } return convolve(ITER_6_FINAL_raw, ITER_6_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_7_CONV
//!SAVE ITER_7_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 7 Ratio
vec4 hook() { if (Iterations < 7) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_7_CONV_raw, ITER_7_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_7_RATIO
//!SAVE ITER_7_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 7 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 7) { return vec4(0.0); } return convolve(ITER_7_RATIO_raw, ITER_7_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_6_FINAL
//!BIND ITER_7_CORRECTION
//!SAVE ITER_7_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 7 Final
vec4 hook() { if (Iterations < 7) { return texture(ITER_6_FINAL_raw, ITER_6_FINAL_pos); } vec4 current_est = texture(ITER_6_FINAL_raw, ITER_6_FINAL_pos); vec4 update_factor = texture(ITER_7_CORRECTION_raw, ITER_7_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_7_FINAL
//!SAVE ITER_8_CONV
//!DESC Richardson-Lucy Deblur - Iteration 8 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 8) { return vec4(0.0); } return convolve(ITER_7_FINAL_raw, ITER_7_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_8_CONV
//!SAVE ITER_8_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 8 Ratio
vec4 hook() { if (Iterations < 8) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_8_CONV_raw, ITER_8_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_8_RATIO
//!SAVE ITER_8_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 8 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 8) { return vec4(0.0); } return convolve(ITER_8_RATIO_raw, ITER_8_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_7_FINAL
//!BIND ITER_8_CORRECTION
//!SAVE ITER_8_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 8 Final
vec4 hook() { if (Iterations < 8) { return texture(ITER_7_FINAL_raw, ITER_7_FINAL_pos); } vec4 current_est = texture(ITER_7_FINAL_raw, ITER_7_FINAL_pos); vec4 update_factor = texture(ITER_8_CORRECTION_raw, ITER_8_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_8_FINAL
//!SAVE ITER_9_CONV
//!DESC Richardson-Lucy Deblur - Iteration 9 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 9) { return vec4(0.0); } return convolve(ITER_8_FINAL_raw, ITER_8_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_9_CONV
//!SAVE ITER_9_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 9 Ratio
vec4 hook() { if (Iterations < 9) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_9_CONV_raw, ITER_9_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_9_RATIO
//!SAVE ITER_9_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 9 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 9) { return vec4(0.0); } return convolve(ITER_9_RATIO_raw, ITER_9_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_8_FINAL
//!BIND ITER_9_CORRECTION
//!SAVE ITER_9_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 9 Final
vec4 hook() { if (Iterations < 9) { return texture(ITER_8_FINAL_raw, ITER_8_FINAL_pos); } vec4 current_est = texture(ITER_8_FINAL_raw, ITER_8_FINAL_pos); vec4 update_factor = texture(ITER_9_CORRECTION_raw, ITER_9_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND ITER_9_FINAL
//!SAVE ITER_10_CONV
//!DESC Richardson-Lucy Deblur - Iteration 10 Convolution
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 10) { return vec4(0.0); } return convolve(ITER_9_FINAL_raw, ITER_9_FINAL_pos, BlurRadius); }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_10_CONV
//!SAVE ITER_10_RATIO
//!DESC Richardson-Lucy Deblur - Iteration 10 Ratio
vec4 hook() { if (Iterations < 10) { return vec4(0.0); } vec4 orig = texture(HOOKED_raw, HOOKED_pos); vec4 conv = texture(ITER_10_CONV_raw, ITER_10_CONV_pos); return orig / (conv + Regularization); }

//!HOOK MAIN
//!BIND ITER_10_RATIO
//!SAVE ITER_10_CORRECTION
//!DESC Richardson-Lucy Deblur - Iteration 10 Correction
float create_psf(vec2 uv, float radius) { float sigma = radius / 2.0; float dist_sq = dot(uv, uv); return exp(-dist_sq / (2.0 * sigma * sigma)) / (6.28318 * sigma * sigma); }
vec4 convolve(sampler2D tex, vec2 uv, float radius) { vec4 sum = vec4(0.0); vec2 pt = vec2(1.0) / vec2(textureSize(tex, 0)); int kernel_size = int(ceil(radius * 2.0)); float wsum = 0.0; for (int y = -kernel_size; y <= kernel_size; y++) { for (int x = -kernel_size; x <= kernel_size; x++) { vec2 offset = vec2(x, y); if(distance(offset, vec2(0.0)) > float(kernel_size)) continue; float weight = create_psf(offset * pt, radius); sum += texture(tex, uv + offset * pt) * weight; wsum += weight; } } return sum / wsum; }
vec4 hook() { if (Iterations < 10) { return vec4(0.0); } return convolve(ITER_10_RATIO_raw, ITER_10_RATIO_pos, BlurRadius); }

//!HOOK MAIN
//!BIND ITER_9_FINAL
//!BIND ITER_10_CORRECTION
//!SAVE ITER_10_FINAL
//!DESC Richardson-Lucy Deblur - Iteration 10 Final
vec4 hook() { if (Iterations < 10) { return texture(ITER_9_FINAL_raw, ITER_9_FINAL_pos); } vec4 current_est = texture(ITER_9_FINAL_raw, ITER_9_FINAL_pos); vec4 update_factor = texture(ITER_10_CORRECTION_raw, ITER_10_CORRECTION_pos); return current_est * update_factor; }

//!HOOK MAIN
//!BIND HOOKED
//!BIND ITER_1_FINAL
//!BIND ITER_2_FINAL
//!BIND ITER_3_FINAL
//!BIND ITER_4_FINAL
//!BIND ITER_5_FINAL
//!BIND ITER_6_FINAL
//!BIND ITER_7_FINAL
//!BIND ITER_8_FINAL
//!BIND ITER_9_FINAL
//!BIND ITER_10_FINAL
//!DESC Selects the result from the correct iteration
vec4 hook() {
    if (Iterations == 1) { return texture(ITER_1_FINAL_raw, ITER_1_FINAL_pos); }
    if (Iterations == 2) { return texture(ITER_2_FINAL_raw, ITER_2_FINAL_pos); }
    if (Iterations == 3) { return texture(ITER_3_FINAL_raw, ITER_3_FINAL_pos); }
    if (Iterations == 4) { return texture(ITER_4_FINAL_raw, ITER_4_FINAL_pos); }
    if (Iterations == 5) { return texture(ITER_5_FINAL_raw, ITER_5_FINAL_pos); }
    if (Iterations == 6) { return texture(ITER_6_FINAL_raw, ITER_6_FINAL_pos); }
    if (Iterations == 7) { return texture(ITER_7_FINAL_raw, ITER_7_FINAL_pos); }
    if (Iterations == 8) { return texture(ITER_8_FINAL_raw, ITER_8_FINAL_pos); }
    if (Iterations == 9) { return texture(ITER_9_FINAL_raw, ITER_9_FINAL_pos); }
    if (Iterations >= 10) { return texture(ITER_10_FINAL_raw, ITER_10_FINAL_pos); }
    return texture(HOOKED_raw, HOOKED_pos);
}