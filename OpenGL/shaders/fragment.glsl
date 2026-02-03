#version 460 core

in vec3 vNormalV;
in vec3 vPosV;

uniform int uVisMode;
uniform vec3 uLightPosV;

out vec4 FragColor;

void main()
{
    vec3 N = normalize(vNormalV);

    if (uVisMode == 4)
    {
        FragColor = vec4(max(N, vec3(0.0)), 1.0);
        return;
    }

    // Hard-coded material and light
    vec3 Ka = vec3(0.08);               // ambient
    vec3 Kd = vec3(0.75, 0.75, 0.75);   // diffuse
    vec3 Ks = vec3(0.35);               // specular
    float glossiness = 128.0;

    //vec3 lightPosV = vec3(2.0, 2.0, 2.0);   // light position in View space
    vec3 lightColor = vec3(1.0, 0.0, 0.0);

    vec3 V = normalize(-vPosV);                 // camera at origin in view space
    vec3 L = normalize(uLightPosV - vPosV);
    vec3 H = normalize(L + V);

    vec3 ambient = Ka * lightColor;

    float ndotl = max(dot(N, L), 0.0);
    vec3 diffuse = Kd * ndotl * lightColor;

    float ndoth = max(dot(N, H), 0.0);
    float specPow = (ndotl > 0.0) ? pow(ndoth, glossiness) : 0.0;
    vec3 specular = Ks * specPow * lightColor;

    vec3 color;
    if (uVisMode == 1) color = ambient;
    else if (uVisMode == 2) color = diffuse;
    else if (uVisMode == 3) color = specular;
    else color = ambient + diffuse + specular;

    FragColor = vec4(color, 1.0);
}