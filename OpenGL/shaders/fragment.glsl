#version 460 core

in vec2 vUV;
in vec3 vPosW;
in vec3 vNormalW;
in vec4 vPosLightClip;

uniform int uVisMode;
// World space light & camera
uniform vec3 uLightPosW;
uniform vec3 uCamPosW;

// Spotlight
uniform vec3 uLightDirW;
uniform float uSpotCosInner;
uniform float uSpotCosOuter;

// Shadow Map
uniform sampler2DShadow uShadowMap;

// Material properties from .mtl file
uniform vec3 uKa;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uTf;
uniform float uNs;
uniform float uNi;
uniform int uIllum;

uniform sampler2D uDiffuseTex;
uniform bool uHasDiffuseTex;
uniform sampler2D uSpecularTex;
uniform bool uHasSpecularTex;

// Envrionment mapping
uniform samplerCube uEnvMap;
uniform float uReflectStrength;

out vec4 FragColor;

float ComputeSpotFactor(vec3 L) // L = normalize(lightPos - pos) (from fragment to light)
{
    // From light to fragment:
    vec3 lightToFrag = normalize(-L);
    float cosTheta = dot(normalize(uLightDirW), lightToFrag);
    return smoothstep(uSpotCosOuter, uSpotCosInner, cosTheta);
}

float ComputeShadow(vec3 N, vec3 L)
{
    // Project into shadow map UVZ
    vec3 proj = vPosLightClip.xyz / vPosLightClip.w;
    proj = proj * 0.5 + 0.5;

    // Outside light frustum => no shadowing
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z < 0.0 || proj.z > 1.0)
        return 1.0;

    // Slope-scale depth bias (helps reduce acne)
    float ndotl = max(dot(N, L), 0.0);
    float bias = max(0.0015 * (1.0 - ndotl), 0.0005);

    // 3x3 PCF
    ivec2 sz = textureSize(uShadowMap, 0);
    vec2 texel = 1.0 / vec2(sz);

    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            vec2 uv = proj.xy + vec2(x, y) * texel;
            sum += texture(uShadowMap, vec3(uv, proj.z - bias));
        }
    }
    return sum / 9.0;
}

void main()
{
    vec3 N = normalize(vNormalW);
    if (!gl_FrontFacing) 
        N = -N;
    vec3 L = normalize(uLightPosW - vPosW);
    vec3 V = normalize(uCamPosW - vPosW);                 // camera at origin in view space
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 kdTex = uHasDiffuseTex ? texture(uDiffuseTex, vUV).rgb : vec3(1.0);
    vec3 ksTex = uHasSpecularTex ? texture(uSpecularTex, vUV).rgb : vec3(1.0);

    // Simple Blinn-Phong from .mtl
    vec3 Ka = uKa;
    vec3 Kd = uHasDiffuseTex ? kdTex : uKd;
    vec3 Ks = uKs * ksTex;
    float shininess = max(uNs, 1.0);

    vec3 albedo = uHasDiffuseTex ? kdTex : uKd;
    vec3 ambient  = 0.1 * albedo;
    vec3 diffuse  = albedo * NdotL;

    vec3 specular = vec3(0.0);
    if (uIllum >= 2 && NdotL > 0.0)
    {
        specular = Ks * pow(NdotH, shininess);
    }

    float spot = ComputeSpotFactor(L);
    float shadow = ComputeShadow(N, L);

    vec3 blinn = ambient + (diffuse + specular) * spot * shadow;

    // Reflection (world position)
    vec3 R = reflect(-V, N);
    vec3 envColor = texture(uEnvMap, R).rgb;

    // Fresnel
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 5.0);
    float reflW = mix(uReflectStrength, 1.0, fresnel) * uReflectStrength;

    // Viewing mode
    vec3 color;
    if (uVisMode == 1) 
        color = ambient;
    else if (uVisMode == 2) 
        color = diffuse * spot * shadow;
    else if (uVisMode == 3) 
        color = specular * spot * shadow;
    else if (uVisMode == 4) 
        color = N * 0.5 + 0.5;
    else 
        color = mix(blinn, envColor, clamp(reflW, 0.0, 1.0));

    FragColor = vec4(color, 1.0);
}