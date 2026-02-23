#version 460 core

layout(location=0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uM;
uniform mat4 uV;
uniform mat4 uP;

out vec2 vUV;
out vec3 vPosW;
out vec3 vNormalW;

void main()
{
	vec4 posW = uM * vec4(aPos, 1.0);
    vPosW = posW.xyz;
    mat3 normalMat = transpose(inverse(mat3(uM)));
    vNormalW = normalMat * aNormal;
    vUV = vec2(aUV.x, 1.0 - aUV.y);     // Flip V coordinate for OpenGL
    //vUV = aUV;

    gl_Position = uP * uV * posW;
}