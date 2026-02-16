#version 460 core

in vec2 vUV;
in vec3 vPosV;
in vec3 vNormalV;

uniform int uVisMode;
uniform vec3 uLightPosV;

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

out vec4 FragColor;

void main()
{
    vec3 N = normalize(vNormalV);
    vec3 L = normalize(uLightPosV - vPosV);
    vec3 V = normalize(-vPosV);                 // camera at origin in view space
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    vec3 kdTex = uHasDiffuseTex ? texture(uDiffuseTex, vUV).rgb : vec3(1.0);
    vec3 ksTex = uHasSpecularTex ? texture(uSpecularTex, vUV).rgb : vec3(1.0);

    // Simple Blinn-Phong defaults from .mtl
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

    vec3 color;
    if (uVisMode == 1) 
        color = ambient;
    else if (uVisMode == 2) 
        color = diffuse;
    else if (uVisMode == 3) 
        color = specular;
    else if (uVisMode == 4) 
        color = N * 0.5 + 0.5;
    else 
        color = ambient + diffuse + specular;

    FragColor = vec4(color, 1.0);
}