#version 460 core

layout(location=0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uM;
uniform mat4 uV;
uniform mat4 uP;

uniform mat4 uLightVP;

out vec2 vUV;
out vec3 vPosW;
out vec3 vNormalW;
out vec4 vPosLightClip;

void main()
{
	vec4 posW = uM * vec4(aPos, 1.0);
    vPosW = posW.xyz;
    mat3 normalMat = transpose(inverse(mat3(uM)));
    vNormalW = normalMat * aNormal;
    vUV = vec2(aUV.x, 1.0 - aUV.y);     // Flip V coordinate for OpenGL
    //vUV = aUV;

    vPosLightClip = uLightVP * posW;    // For Shadow Mapping

    gl_Position = uP * uV * posW;
}