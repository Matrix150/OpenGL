#version 460 core

layout(location=0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uMV;

out vec3 vNormalV;
out vec3 vPosV;

void main()
{
	vec4 posV = uMV * vec4(aPos, 1.0);
    vPosV = posV.xyz;

    mat3 N = transpose(inverse(mat3(uMV)));
    vNormalV = normalize(N * aNormal);

    gl_Position = uMVP * vec4(aPos, 1.0);
}