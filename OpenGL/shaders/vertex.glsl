#version 460 core

layout(location=0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec2 vUV;
out vec3 vPosV;
out vec3 vNormalV;

void main()
{
	vec4 posV = uMV * vec4(aPos, 1.0);
    vPosV = posV.xyz;
    mat3 normalMat = transpose(inverse(mat3(uMV)));
    vNormalV = normalMat * aNormal;
    vUV = aUV;
    //vUV = vec2(aUV.x, 1.0 - aUV.y);

    gl_Position = uMVP * vec4(aPos, 1.0);
}