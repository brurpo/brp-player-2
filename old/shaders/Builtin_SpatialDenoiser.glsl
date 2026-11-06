//!PARAM Strength
//!DESC Denoise strength
//!TYPE float
//!MINIMUM 0.01
//!MAXIMUM 3.0
1.5

//!PARAM BIAS
//!DESC Denoise strength
//!TYPE float
//!MINIMUM 0.01
//!MAXIMUM 1.0
0.6

//!PARAM Tolerance
//!DESC Denoise strength
//!TYPE float
//!MINIMUM 2.0
//!MAXIMUM 50.0
20

//!HOOK MAIN
//!BIND HOOKED

#define SAMPLES 256  // HIGHER = NICER = SLOWER
float DISTRIBUTION_BIAS= BIAS; // between 0. and 1.
float PIXEL_MULTIPLIER = Strength; // between 1. and 3. (keep low)
float INVERSE_HUE_TOLERANCE = Tolerance; // (2. - 30.)

#define GOLDEN_ANGLE 2.3999632 //3PI-sqrt(5)PI

#define pow(a,b) pow(max(a,0.),b) // @morimea

mat2 sample2D = mat2(cos(GOLDEN_ANGLE),sin(GOLDEN_ANGLE),-sin(GOLDEN_ANGLE),cos(GOLDEN_ANGLE));

vec4 hook() {
    vec4 original_color = HOOKED_tex(HOOKED_pos);
    vec3 denoisedColor           = vec3(0.);
    vec2 resolution = vec2(1920.0, 1080.0);
    const float sampleRadius     = sqrt(float(SAMPLES));
    const float sampleTrueRadius = 0.5/(sampleRadius*sampleRadius);
    vec2        samplePixel      = vec2(1.0/resolution.x,1.0/resolution.y); 
    vec3        sampleCenter     = texture(HOOKED_raw, HOOKED_pos).rgb;
    vec3        sampleCenterNorm = normalize(sampleCenter);
    float       sampleCenterSat  = length(sampleCenter);
    
    float  influenceSum = 0.0;
    float brightnessSum = 0.0;
    
    vec2 pixelRotated = vec2(0.,1.);
    
    for (float x = 0.0; x <= float(SAMPLES); x++) {
        
        pixelRotated *= sample2D;
        
        vec2  pixelOffset    = PIXEL_MULTIPLIER*pixelRotated*sqrt(x)*0.5;
        float pixelInfluence = 1.0-sampleTrueRadius*pow(dot(pixelOffset,pixelOffset),DISTRIBUTION_BIAS);
        pixelOffset *= samplePixel;
            
        vec3 thisDenoisedColor = 
            texture(HOOKED_raw, HOOKED_pos + pixelOffset).rgb;

        pixelInfluence      *= pixelInfluence*pixelInfluence;
        /*
            HUE + SATURATION FILTER
        */
        pixelInfluence      *=   
            pow(0.5+0.5*dot(sampleCenterNorm,normalize(thisDenoisedColor)),INVERSE_HUE_TOLERANCE)
            * pow(1.0 - abs(length(thisDenoisedColor)-length(sampleCenterSat)),8.);
            
        influenceSum += pixelInfluence;
        denoisedColor += thisDenoisedColor*pixelInfluence;
    }
    
    vec3 final_color = denoisedColor/influenceSum;
    return vec4(final_color, original_color.a);
    
}


