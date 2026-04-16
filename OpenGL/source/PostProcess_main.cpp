#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cyGL.h"
#include "cyTriMesh.h"
#include "cyMatrix.h"

// Properties
// Mouse status
static bool g_leftDown = false;
static bool g_rightDown = false;
static double g_lastX = 0.0, g_lastY = 0.0;

// Camera parameter (object -> render to texture)
static float g_yaw = 0.0f;
static float g_pitch = 0.0f;
static float g_dist = 2.0f;
static cy::Vec3f g_camTarget(0.0f, 0.0f, 0.0f);

// Object
static cy::Vec3f g_objCenter(0.0f, 0.0f, 0.0f);
static float g_objScale = 1.0f;

// Perspective or Orthographic
static bool g_usePerspective = true;
static float g_orthoScale = 1.5f;

// Depth
static bool g_showDepth = false;
// FXAA
static bool g_enableFXAA = true;
// Motion Blur
static bool g_enableMotionBlur = true;
static bool g_hasPrevFrame = false;
static cy::Matrix4f g_prevVP;
// Blooming
static bool g_enableBloom = true;
static float g_bloomThreshold = 0.80f;
static float g_bloomStrength = 0.90f;
// Tone Mapping and HDR
static bool g_enableToneMapping = true;
static float g_exposure = 1.0f;
static int g_toneMapMode = 1; // 0 = Reinhard, 1 = ACES Approx
//Color Grading
static bool g_enableColorGrading = true;
static float g_gradeSaturation = 1.10f;
static float g_gradeContrast = 1.05f;
static float g_gradeBrightness = 0.00f;
static cy::Vec3f g_colorFilter(1.0f, 1.0f, 1.0f);

// Light properties
static float g_lightYaw = 0.7f;
static float g_lightPitch = 0.4f;
static float g_lightRadius = 3.0f;

static const char* kFullscreenVS = R"GLSL(
    #version 460 core
    layout(location=0) in vec2 aPos;
    layout(location=1) in vec2 aUV;
    out vec2 vUV;
    void main()
    {
        vUV = aUV;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)GLSL";
// ------------------------------


// Shader
struct LitShader
{
    cy::GLSLProgram prog;
    bool built = false;

    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aNormal;

        uniform mat4 uM;
        uniform mat4 uV;
        uniform mat4 uP;

        out vec3 vWorldPos;
        out vec3 vWorldNormal;

        void main()
        {
            vec4 worldPos = uM * vec4(aPos, 1.0);
            vWorldPos = worldPos.xyz;

            mat3 normalMat = transpose(inverse(mat3(uM)));
            vWorldNormal = normalize(normalMat * aNormal);

            gl_Position = uP * uV * worldPos;
        }
    )GLSL";

    const char* fs = R"GLSL(
        #version 460 core
        in vec3 vWorldPos;
        in vec3 vWorldNormal;

        out vec4 FragColor;

        uniform vec3 uCamPosW;
        uniform vec3 uLightPosW;

        uniform vec3 uBaseColor;
        uniform vec3 uAmbientColor;
        uniform vec3 uLightColor;
        uniform float uShininess;

        void main()
        {
            vec3 N = normalize(vWorldNormal);
            vec3 L = normalize(uLightPosW - vWorldPos);
            vec3 V = normalize(uCamPosW - vWorldPos);
            vec3 H = normalize(L + V);

            float diff = max(dot(N, L), 0.0);
            float spec = pow(max(dot(N, H), 0.0), uShininess);

            vec3 ambient = uAmbientColor * uBaseColor;
            vec3 diffuse = diff * uLightColor * uBaseColor;
            vec3 specular = spec * uLightColor * 0.80;

            vec3 color = ambient + diffuse + specular;
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct MotionBlurShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uSceneColor;
        uniform sampler2D uSceneDepth;
        uniform int uEnableMotionBlur;
        uniform mat4 uCurrInvVP;
        uniform mat4 uPrevVP;

        vec3 ReconstructWorldPos(vec2 uv, float depth)
        {
            vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
            vec4 world = uCurrInvVP * ndc;
            return world.xyz / world.w;
        }

        vec2 ComputeMotionVector(vec2 uv, float depth)
        {
            vec3 worldPos = ReconstructWorldPos(uv, depth);

            vec4 prevClip = uPrevVP * vec4(worldPos, 1.0);
            vec2 prevNDC = prevClip.xy / prevClip.w;
            vec2 prevUV = prevNDC * 0.5 + 0.5;

            return uv - prevUV;
        }

        vec3 ApplyMotionBlur(vec2 uv)
        {
            float depth = texture(uSceneDepth, uv).r;
            vec3 centerColor = texture(uSceneColor, uv).rgb;

            if (depth >= 0.999999)
                return centerColor;

            vec2 velocity = ComputeMotionVector(uv, depth);
            float speed = length(velocity);

            if (speed < 1e-5)
                return centerColor;

            const int sampleCount = 8;
            vec3 sum = vec3(0.0);

            for (int i = 0; i < sampleCount; ++i)
            {
                float t = float(i) / float(sampleCount - 1) - 0.5;
                vec2 sampleUV = uv + velocity * t * 1.5;
                sampleUV = clamp(sampleUV, vec2(0.0), vec2(1.0));
                sum += texture(uSceneColor, sampleUV).rgb;
            }

            return sum / float(sampleCount);
        }

        void main()
        {
            vec3 color = texture(uSceneColor, vUV).rgb;
            if (uEnableMotionBlur == 1)
                color = ApplyMotionBlur(vUV);
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct BrightExtractShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uInputTex;
        uniform float uThreshold;
        uniform int uEnableBloom;

        void main()
        {
            if (uEnableBloom == 0)
            {
                FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }

            vec3 color = texture(uInputTex, vUV).rgb;
            float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
            vec3 bright = (brightness > uThreshold) ? color : vec3(0.0);
            FragColor = vec4(bright, 1.0);
        }
    )GLSL";
};

struct BlurShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uInputTex;
        uniform vec2 uTexelSize;
        uniform int uHorizontal;

        void main()
        {
            float weights[5] = float[](0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f);
            vec2 dir = (uHorizontal == 1) ? vec2(uTexelSize.x, 0.0) : vec2(0.0, uTexelSize.y);

            vec3 result = texture(uInputTex, vUV).rgb * weights[0];
            for (int i = 1; i < 5; ++i)
            {
                result += texture(uInputTex, vUV + dir * float(i)).rgb * weights[i];
                result += texture(uInputTex, vUV - dir * float(i)).rgb * weights[i];
            }

            FragColor = vec4(result, 1.0);
        }
    )GLSL";
};

struct CombineShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uSceneTex;
        uniform sampler2D uBloomTex;
        uniform int uEnableBloom;
        uniform float uBloomStrength;

        void main()
        {
            vec3 scene = texture(uSceneTex, vUV).rgb;
            vec3 bloom = texture(uBloomTex, vUV).rgb;

            vec3 color = scene;
            if (uEnableBloom == 1)
                color += bloom * uBloomStrength;

            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct ToneMapShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uInputTex;
        uniform int uEnableToneMapping;
        uniform float uExposure;
        uniform int uToneMapMode;

        vec3 ToneMapReinhard(vec3 color)
        {
            return color / (color + vec3(1.0));
        }

        vec3 ToneMapACES(vec3 x)
        {
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }

        void main()
        {
            vec3 color = texture(uInputTex, vUV).rgb;
            color *= uExposure;

            if (uEnableToneMapping == 1)
            {
                if (uToneMapMode == 0)
                    color = ToneMapReinhard(color);
                else
                    color = ToneMapACES(color);
            }

            color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct ColorGradingShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uInputTex;
        uniform int uEnableColorGrading;
        uniform float uSaturation;
        uniform float uContrast;
        uniform float uBrightness;
        uniform vec3 uColorFilter;

        vec3 ApplySaturation(vec3 color, float saturation)
        {
            float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
            vec3 gray = vec3(luma);
            return mix(gray, color, saturation);
        }

        vec3 ApplyContrast(vec3 color, float contrast)
        {
            return (color - 0.5) * contrast + 0.5;
        }

        void main()
        {
            vec3 color = texture(uInputTex, vUV).rgb;

            if (uEnableColorGrading == 1)
            {
                color *= uColorFilter;
                color += vec3(uBrightness);
                color = ApplySaturation(color, uSaturation);
                color = ApplyContrast(color, uContrast);
            }

            FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        }
    )GLSL";
};

struct FXAAShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uInputTex;
        uniform int uEnableFXAA;
        uniform vec2 uInvScreenSize;

        float Luma(vec3 color)
        {
            return dot(color, vec3(0.299, 0.587, 0.114));
        }

        vec3 ApplyFXAA(vec2 uv)
        {
            vec2 px = uInvScreenSize;

            vec3 rgbM  = texture(uInputTex, uv).rgb;
            vec3 rgbNW = texture(uInputTex, uv + vec2(-px.x, -px.y)).rgb;
            vec3 rgbNE = texture(uInputTex, uv + vec2( px.x, -px.y)).rgb;
            vec3 rgbSW = texture(uInputTex, uv + vec2(-px.x,  px.y)).rgb;
            vec3 rgbSE = texture(uInputTex, uv + vec2( px.x,  px.y)).rgb;

            float lumaM  = Luma(rgbM);
            float lumaNW = Luma(rgbNW);
            float lumaNE = Luma(rgbNE);
            float lumaSW = Luma(rgbSW);
            float lumaSE = Luma(rgbSE);

            float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
            float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

            float range = lumaMax - lumaMin;
            if (range < max(0.0312, lumaMax * 0.125))
                return rgbM;

            vec2 dir;
            dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
            dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

            float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.0312, 1.0 / 128.0);
            float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
            dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;

            vec3 rgbA = 0.5 * (
                texture(uInputTex, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                texture(uInputTex, uv + dir * (2.0 / 3.0 - 0.5)).rgb
            );

            vec3 rgbB = rgbA * 0.5 + 0.25 * (
                texture(uInputTex, uv + dir * -0.5).rgb +
                texture(uInputTex, uv + dir *  0.5).rgb
            );

            float lumaB = Luma(rgbB);
            if (lumaB < lumaMin || lumaB > lumaMax)
                return rgbA;

            return rgbB;
        }

        void main()
        {
            vec3 color = texture(uInputTex, vUV).rgb;
            if (uEnableFXAA == 1)
                color = ApplyFXAA(vUV);
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct DepthPreviewShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = kFullscreenVS;

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uSceneDepth;

        float LinearizeDepth(float z, float nearPlane, float farPlane)
        {
            float ndc = z * 2.0 - 1.0;
            return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - ndc * (farPlane - nearPlane));
        }

        void main()
        {
            float z = texture(uSceneDepth, vUV).r;
            float linearDepth = LinearizeDepth(z, 0.1, 100.0);
            float normalizedDepth = clamp(linearDepth / 8.0, 0.0, 1.0);
            FragColor = vec4(vec3(normalizedDepth), 1.0);
        }
    )GLSL";
};

// Build Shader
template <typename ShaderType>
bool BuildShader(ShaderType& shader, const char* errorMessage)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << errorMessage << std::endl;
        return false;
    }
    shader.built = true;
    return true;
}
// ------------------------------


// Helping tools
static float DegToRad(float deg)
{
    return deg * 3.1415926535f / 180.0f;
}

static cy::Matrix4f MakeOrthographic(float l, float r, float b, float t, float n, float f)
{
    cy::Matrix4f M;
    M.SetIdentity();
    const float invRL = 1.0f / (r - l);
    const float invTB = 1.0f / (t - b);
    const float invFN = 1.0f / (f - n);
    M.SetRow(0, 2.0f * invRL, 0.0f, 0.0f, -(r + l) * invRL);
    M.SetRow(1, 0.0f, 2.0f * invTB, 0.0f, -(t + b) * invTB);
    M.SetRow(2, 0.0f, 0.0f, -2.0f * invFN, -(f + n) * invFN);
    M.SetRow(3, 0.0f, 0.0f, 0.0f, 1.0f);

    return M;
}

static cy::Matrix4f MakeProjection(int fbW, int fbH, bool usePerspective, float orthoScale)
{
    float aspect = (fbH > 0) ? (float)fbW / (float)fbH : 1.0f;
    if (usePerspective)
    {
        return cy::Matrix4f::Perspective(DegToRad(60.0f), aspect, 0.1f, 100.0f);       // Perspective
    }

    float halfH = orthoScale;
    float halfW = orthoScale * aspect;
    return MakeOrthographic(-halfW, halfW, -halfH, halfH, 0.1f, 200.0f);
}

static cy::Matrix4f MakeView(float yaw, float pitch, float dist)
{
    cy::Matrix4f R_yaw = cy::Matrix4f::RotationY(yaw);
    cy::Matrix4f R_pitch = cy::Matrix4f::RotationX(pitch);
    cy::Matrix4f Tcam = cy::Matrix4f::Translation(cy::Vec3f(0.0f, 0.0f, -dist));
    cy::Matrix4f Ttarget = cy::Matrix4f::Translation(-g_camTarget);

    return Tcam * R_pitch * R_yaw * Ttarget;
}

static cy::Vec3f ComputeLightPosWorld()
{
    float cp = cosf(g_lightPitch);
    float sp = sinf(g_lightPitch);
    float cyaw = cosf(g_lightYaw);
    float syaw = sinf(g_lightYaw);
    return cy::Vec3f(g_lightRadius * cp * syaw, g_lightRadius * sp, g_lightRadius * cp * cyaw);
}

static void DrawFullscreenQuad(GLuint fsQuadVAO)
{
    glBindVertexArray(fsQuadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
// ------------------------------


// Render Targets
struct SceneRenderTarget
{
	GLuint fbo = 0;
	GLuint colorTex = 0;
	GLuint depthTex = 0;
	int width = 0;
	int height = 0;
};

struct ColorRenderTarget
{
    GLuint fbo = 0;
    GLuint colorTex = 0;
    int width = 0;
    int height = 0;
};

static void DestroySceneRenderTarget(SceneRenderTarget& renderTarget)
{
    if (renderTarget.colorTex)
		glDeleteTextures(1, &renderTarget.colorTex);
    if (renderTarget.depthTex)
        glDeleteTextures(1, &renderTarget.depthTex);
    if (renderTarget.fbo)
        glDeleteFramebuffers(1, &renderTarget.fbo);
	renderTarget = {};
}

static bool CreateSceneRenderTarget(SceneRenderTarget& renderTarget, int width, int height)
{
	DestroySceneRenderTarget(renderTarget);

	renderTarget.width = width;
	renderTarget.height = height;

    glCreateFramebuffers(1, &renderTarget.fbo);

    glCreateTextures(GL_TEXTURE_2D, 1, &renderTarget.colorTex);
    glTextureStorage2D(renderTarget.colorTex, 1, GL_RGBA16F, width, height);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glCreateTextures(GL_TEXTURE_2D, 1, &renderTarget.depthTex);
    glTextureStorage2D(renderTarget.depthTex, 1, GL_DEPTH_COMPONENT24, width, height);
    glTextureParameteri(renderTarget.depthTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(renderTarget.depthTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(renderTarget.depthTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(renderTarget.depthTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glNamedFramebufferTexture(renderTarget.fbo, GL_COLOR_ATTACHMENT0, renderTarget.colorTex, 0);
    glNamedFramebufferTexture(renderTarget.fbo, GL_DEPTH_ATTACHMENT, renderTarget.depthTex, 0);

	const GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glNamedFramebufferDrawBuffers(renderTarget.fbo, 1, drawBuffers);

    GLenum status = glCheckNamedFramebufferStatus(renderTarget.fbo, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "ERROR: Framebuffer is not complete: " << status << "\n";
        DestroySceneRenderTarget(renderTarget);
        return false;
	}

    return true;
}

static void DestroyColorRenderTarget(ColorRenderTarget& renderTarget)
{
    if (renderTarget.colorTex)
        glDeleteTextures(1, &renderTarget.colorTex);
    if (renderTarget.fbo)
        glDeleteFramebuffers(1, &renderTarget.fbo);
    renderTarget = {};
}

static bool CreateColorRenderTarget(ColorRenderTarget& renderTarget, int width, int height)
{
    DestroyColorRenderTarget(renderTarget);

    renderTarget.width = width;
    renderTarget.height = height;

    glCreateFramebuffers(1, &renderTarget.fbo);

    glCreateTextures(GL_TEXTURE_2D, 1, &renderTarget.colorTex);
    glTextureStorage2D(renderTarget.colorTex, 1, GL_RGBA16F, width, height);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(renderTarget.colorTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glNamedFramebufferTexture(renderTarget.fbo, GL_COLOR_ATTACHMENT0, renderTarget.colorTex, 0);

    const GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glNamedFramebufferDrawBuffers(renderTarget.fbo, 1, drawBuffers);

    GLenum status = glCheckNamedFramebufferStatus(renderTarget.fbo, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "ERROR: Framebuffer is not complete: " << status << "\n";
        DestroyColorRenderTarget(renderTarget);
        return false;
    }

    return true;
}
// ------------------------------


// Event Callback
static void framebuffer_size_callback(GLFWwindow*, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        g_leftDown = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
        g_rightDown = (action == GLFW_PRESS);
    if (action == GLFW_PRESS)
        glfwGetCursorPos(window, &g_lastX, &g_lastY);
}

static void cursor_pos_callback(GLFWwindow* window, double x, double y)
{
    double dx = x - g_lastX;
    double dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    const float rotSpeed = 0.005f;
    const float zoomSpeed = 0.02f;
    bool ctrlDown = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) || (glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);

    if (g_leftDown)
    {
        if (ctrlDown)
        {
            g_lightYaw += (float)dx * rotSpeed;
            g_lightPitch += (float)dy * rotSpeed;
            if (g_lightPitch > 1.55f) g_lightPitch = 1.55f;
            if (g_lightPitch < -1.55f) g_lightPitch = -1.55f;
        }
        else
        {
            g_yaw += (float)dx * rotSpeed;
            g_pitch += (float)dy * rotSpeed;
            if (g_pitch > 1.55f) g_pitch = 1.55f;
            if (g_pitch < -1.55f) g_pitch = -1.55f;
        }
    }
    if (g_rightDown)
    {
        g_dist += (float)dy * zoomSpeed;
        if (g_dist < 0.8f) g_dist = 0.8f;
        if (g_dist > 8.0f) g_dist = 8.0f;
    }
}

static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        g_usePerspective = !g_usePerspective;
        std::cout << "[P] Projection = " << (g_usePerspective ? "Perspective" : "Orthographic") << std::endl;
    }
    if (key == GLFW_KEY_Z && action == GLFW_PRESS)
    {
        g_showDepth = !g_showDepth;
        std::cout << "[Z] Show Depth = " << (g_showDepth ? "ON" : "OFF") << std::endl;
    }
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
    {
        g_enableFXAA = !g_enableFXAA;
        std::cout << "[F] FXAA = " << (g_enableFXAA ? "ON" : "OFF") << std::endl;
	}
    if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        g_enableMotionBlur = !g_enableMotionBlur;
        std::cout << "[M] Motion Blur = " << (g_enableMotionBlur ? "ON" : "OFF") << std::endl;
	}
    if (key == GLFW_KEY_B && action == GLFW_PRESS)
    {
        g_enableBloom = !g_enableBloom;
        std::cout << "[B] Bloom = " << (g_enableBloom ? "ON" : "OFF") << std::endl;
    }    
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
    {
        g_enableToneMapping = !g_enableToneMapping;
        std::cout << "[T] Tone Mapping = " << (g_enableToneMapping ? "ON" : "OFF") << std::endl;
    }
    if (key == GLFW_KEY_G && action == GLFW_PRESS)
    {
        g_enableColorGrading = !g_enableColorGrading;
        std::cout << "[G] Color Grading = " << (g_enableColorGrading ? "ON" : "OFF") << std::endl;
    }
    if (key == GLFW_KEY_LEFT_BRACKET && action == GLFW_PRESS)
    {
        g_exposure = max(0.1f, g_exposure - 0.1f);
        std::cout << "[Exposure] " << g_exposure << std::endl;
    }
    if (key == GLFW_KEY_RIGHT_BRACKET && action == GLFW_PRESS)
    {
        g_exposure += 0.1f;
        std::cout << "[Exposure] " << g_exposure << std::endl;
    }
    if (key == GLFW_KEY_COMMA && action == GLFW_PRESS)
    {
        g_bloomStrength = max(0.0f, g_bloomStrength - 0.1f);
        std::cout << "[Bloom Strength] " << g_bloomStrength << std::endl;
    }
    if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS)
    {
        g_bloomStrength += 0.1f;
        std::cout << "[Bloom Strength] " << g_bloomStrength << std::endl;
    }
}
// ------------------------------


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mesh.obj>\n";
        return -1;
    }

    cy::TriMesh mesh;
    if (!mesh.LoadFromFileObj(argv[1], true, &std::cout))
    {
        std::cerr << "ERROR: failed to load obj: " << argv[1] << "\n";
        return -1;
    }

	// Get bound & xcenter & scale
    cy::Vec3f bbMin(1e30f, 1e30f, 1e30f);
	cy::Vec3f bbMax(-1e30f, -1e30f, -1e30f);
    for (unsigned int i = 0; i < mesh.NV(); ++i)
    {
		cy::Vec3f p = mesh.V((int)i);
        bbMin.x = min(bbMin.x, p.x);
        bbMin.y = min(bbMin.y, p.y);
        bbMin.z = min(bbMin.z, p.z);
        bbMax.x = max(bbMax.x, p.x);
        bbMax.y = max(bbMax.y, p.y);
        bbMax.z = max(bbMax.z, p.z);
    }

    g_objCenter = (bbMin + bbMax) * 0.5f;
    cy::Vec3f extent = bbMax - bbMin;
    float maxExtent = max(extent.x, max(extent.y, extent.z));
    g_objScale = (maxExtent > 1e-8f) ? (2.0f / maxExtent) : 1.0f;

    float diag = sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z) * g_objScale;
    g_dist = max(2.0f, diag * 1.2f);

	mesh.ComputeNormals();

    std::vector<float> positions;
    std::vector<float> normals;
    positions.reserve(mesh.NF() * 3 * 3);
    normals.reserve(mesh.NF() * 3 * 3);

    for (unsigned int fi = 0; fi < mesh.NF(); ++fi)
    {
		auto f = mesh.F((int)fi);
		auto fn = mesh.FN((int)fi);

        for (int c = 0; c < 3; ++c)
        {
			int vi = f.v[c];
            int ni = fn.v[c];
			cy::Vec3f p = mesh.V(vi);
			cy::Vec3f n = (mesh.NVN() > 0 && ni >= 0) ? mesh.VN(ni) : cy::Vec3f(0, 1, 0);

            positions.push_back(p.x);
            positions.push_back(p.y);
            positions.push_back(p.z);
            normals.push_back(n.x);
            normals.push_back(n.y);
			normals.push_back(n.z);
        }
    }

    // GLFW
    if (!glfwInit())
    {
        std::cerr << "ERROR: glfwInit failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    // Initialize viewport size
    const int initW = 1280;
    const int initH = 720;
    GLFWwindow* window = glfwCreateWindow(initW, initH, "OpenGL Multi-Pass Post Process", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "ERROR: glfwCreateWindow failed\n";
        glfwTerminate();
        return -1;
    }

	// Binding Event Callback
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "ERROR: gladLoadGLLoader failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
    std::cout << "Controls:\n";
    std::cout << "Controls:\n";
    std::cout << "  LMB drag        : rotate camera\n";
    std::cout << "  RMB drag        : zoom camera\n";
    std::cout << "  Ctrl + LMB drag : rotate light\n";
    std::cout << "  W/A/S/D         : move camera (XZ)\n";
    std::cout << "  Q/E             : move camera (Y)\n";
    std::cout << "  Z               : toggle depth preview\n";
    std::cout << "  F               : toggle FXAA\n";
    std::cout << "  M               : toggle motion blur\n";
    std::cout << "  B               : toggle bloom\n";
    std::cout << "  , / .           : bloom strength - / +\n";
    std::cout << "  T               : toggle tone mapping\n";
    std::cout << "  [ / ]           : exposure - / +\n";
    std::cout << "  G               : toggle color grading\n";
    std::cout << "  P               : perspective / orthographic\n";

    // Sahder
    LitShader litShader;
    MotionBlurShader motionShader;
    BrightExtractShader brightShader;
    BlurShader blurShader;
    CombineShader combineShader;
    ToneMapShader toneMapShader;
    ColorGradingShader gradingShader;
    FXAAShader fxaaShader;
    DepthPreviewShader depthShader;

    if (!BuildShader(litShader, "Failed to build lit shader.") ||
        !BuildShader(motionShader, "Failed to build motion blur shader.") ||
        !BuildShader(brightShader, "Failed to build bright extract shader.") ||
        !BuildShader(blurShader, "Failed to build blur shader.") ||
        !BuildShader(combineShader, "Failed to build combine shader.") ||
        !BuildShader(toneMapShader, "Failed to build tone mapping shader.") ||
        !BuildShader(gradingShader, "Failed to build color grading shader.") ||
        !BuildShader(fxaaShader, "Failed to build FXAA shader.") ||
        !BuildShader(depthShader, "Failed to build depth preview shader."))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Mesh VAO
	GLuint meshVAO = 0, posVBO = 0, normVBO = 0;
    glCreateVertexArrays(1, &meshVAO);
    glCreateBuffers(1, &posVBO);
    glCreateBuffers(1, &normVBO);

    glNamedBufferData(posVBO, (GLsizeiptr)(positions.size() * sizeof(float)), positions.data(), GL_STATIC_DRAW);
    glNamedBufferData(normVBO, (GLsizeiptr)(normals.size() * sizeof(float)), normals.data(), GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(meshVAO, 0, posVBO, 0, 3 * sizeof(float));
    glVertexArrayVertexBuffer(meshVAO, 1, normVBO, 0, 3 * sizeof(float));

    glEnableVertexArrayAttrib(meshVAO, 0);
    glVertexArrayAttribFormat(meshVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(meshVAO, 0, 0);

    glEnableVertexArrayAttrib(meshVAO, 1);
    glVertexArrayAttribFormat(meshVAO, 1, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(meshVAO, 1, 1);

	// Fullscreen Quad
    GLuint fsQuadVAO = 0, fsQuadVBO = 0;
    glCreateVertexArrays(1, &fsQuadVAO);
    glCreateBuffers(1, &fsQuadVBO);

    const float fsQuadVerts[] = {
        // pos      // uv
        -1.f, -1.f, 0.f, 0.f,
         1.f, -1.f, 1.f, 0.f,
         1.f,  1.f, 1.f, 1.f,

        -1.f, -1.f, 0.f, 0.f,
         1.f,  1.f, 1.f, 1.f,
        -1.f,  1.f, 0.f, 1.f
    };

    glNamedBufferData(fsQuadVBO, sizeof(fsQuadVerts), fsQuadVerts, GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(fsQuadVAO, 0, fsQuadVBO, 0, 4 * sizeof(float));

    glEnableVertexArrayAttrib(fsQuadVAO, 0);
    glVertexArrayAttribFormat(fsQuadVAO, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(fsQuadVAO, 0, 0);

    glEnableVertexArrayAttrib(fsQuadVAO, 1);
    glVertexArrayAttribFormat(fsQuadVAO, 1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float));
    glVertexArrayAttribBinding(fsQuadVAO, 1, 0);

    int fbW = 0, fbH = 0;
    glfwGetFramebufferSize(window, &fbW, &fbH);

	// Render Target
    SceneRenderTarget sceneRT;
    ColorRenderTarget motionRT;
    ColorRenderTarget bloomBrightRT;
    ColorRenderTarget bloomBlurRT1;
    ColorRenderTarget bloomBlurRT2;
    ColorRenderTarget combineRT;
    ColorRenderTarget toneMapRT;
    ColorRenderTarget gradeRT;

    if (!CreateSceneRenderTarget(sceneRT, fbW, fbH) ||
        !CreateColorRenderTarget(motionRT, fbW, fbH) ||
        !CreateColorRenderTarget(bloomBrightRT, fbW, fbH) ||
        !CreateColorRenderTarget(bloomBlurRT1, fbW, fbH) ||
        !CreateColorRenderTarget(bloomBlurRT2, fbW, fbH) ||
        !CreateColorRenderTarget(combineRT, fbW, fbH) ||
        !CreateColorRenderTarget(toneMapRT, fbW, fbH) ||
        !CreateColorRenderTarget(gradeRT, fbW, fbH))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

	glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window))
    {
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (fbW != sceneRT.width || fbH != sceneRT.height)
        {
            if (!CreateSceneRenderTarget(sceneRT, fbW, fbH) ||
                !CreateColorRenderTarget(motionRT, fbW, fbH) ||
                !CreateColorRenderTarget(bloomBrightRT, fbW, fbH) ||
                !CreateColorRenderTarget(bloomBlurRT1, fbW, fbH) ||
                !CreateColorRenderTarget(bloomBlurRT2, fbW, fbH) ||
                !CreateColorRenderTarget(combineRT, fbW, fbH) ||
                !CreateColorRenderTarget(toneMapRT, fbW, fbH) ||
                !CreateColorRenderTarget(gradeRT, fbW, fbH))
            {
                std::cerr << "ERROR: failed to resize render targets\n";
                break;
            }
        }

        // Camera Moverment
		const float moveSpeed = 0.2f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) g_camTarget.z -= moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) g_camTarget.z += moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) g_camTarget.x -= moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) g_camTarget.x += moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) g_camTarget.y -= moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) g_camTarget.y += moveSpeed;

        cy::Matrix4f P = MakeProjection(fbW, fbH, g_usePerspective, g_orthoScale);
        cy::Matrix4f V = MakeView(g_yaw, g_pitch, g_dist);

        cy::Matrix4f currentVP = P * V;
        cy::Matrix4f currentInvVP = currentVP.GetInverse();

        cy::Matrix4f Vinv = V.GetInverse();
        cy::Vec4f camPos4 = Vinv * cy::Vec4f(0, 0, 0, 1);
        cy::Vec3f camPosW(camPos4.x, camPos4.y, camPos4.z);

        cy::Vec3f lightPosW = ComputeLightPosWorld();

        cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
        cy::Matrix4f S;
        S.SetScale(g_objScale);
        cy::Matrix4f M = S * Tcenter;

		// Pass1: Scene Render to Scene Render Target
        glBindFramebuffer(GL_FRAMEBUFFER, sceneRT.fbo);
        glViewport(0, 0, fbW, fbH);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        litShader.prog.Bind();
        litShader.prog.SetUniformMatrix4("uM", M.cell);
        litShader.prog.SetUniformMatrix4("uV", V.cell);
        litShader.prog.SetUniformMatrix4("uP", P.cell);

        litShader.prog.SetUniform("uCamPosW", camPosW.x, camPosW.y, camPosW.z);
        litShader.prog.SetUniform("uLightPosW", lightPosW.x, lightPosW.y, lightPosW.z);

        litShader.prog.SetUniform("uBaseColor", 0.82f, 0.58f, 0.32f);
        litShader.prog.SetUniform("uAmbientColor", 0.16f, 0.16f, 0.18f);
        litShader.prog.SetUniform("uLightColor", 1.0f, 0.96f, 0.90f);
        litShader.prog.SetUniform("uShininess", 64.0f);

        glBindVertexArray(meshVAO);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh.NF() * 3));

        if (g_showDepth)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            depthShader.prog.Bind();
            depthShader.prog.SetUniform("uSceneDepth", 0);
            glBindTextureUnit(0, sceneRT.depthTex);
            DrawFullscreenQuad(fsQuadVAO);
        }
        else
        {
			// Pass2: Motion Blur to Motion Render Target
            glBindFramebuffer(GL_FRAMEBUFFER, motionRT.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            motionShader.prog.Bind();
            motionShader.prog.SetUniform("uSceneColor", 0);
            motionShader.prog.SetUniform("uSceneDepth", 1);
            motionShader.prog.SetUniform("uEnableMotionBlur", (g_enableMotionBlur && g_hasPrevFrame) ? 1 : 0);
            motionShader.prog.SetUniformMatrix4("uCurrInvVP", currentInvVP.cell);
            motionShader.prog.SetUniformMatrix4("uPrevVP", g_prevVP.cell);
            glBindTextureUnit(0, sceneRT.colorTex);
            glBindTextureUnit(1, sceneRT.depthTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass3: Bright Extract to Bloom Brightness Render Target
            glBindFramebuffer(GL_FRAMEBUFFER, bloomBrightRT.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            brightShader.prog.Bind();
            brightShader.prog.SetUniform("uInputTex", 0);
            brightShader.prog.SetUniform("uThreshold", g_bloomThreshold);
            brightShader.prog.SetUniform("uEnableBloom", g_enableBloom ? 1 : 0);
            glBindTextureUnit(0, motionRT.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass4: Blur (Horizontal) to Bloom Blur Render Target 1
            glBindFramebuffer(GL_FRAMEBUFFER, bloomBlurRT1.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            blurShader.prog.Bind();
            blurShader.prog.SetUniform("uInputTex", 0);
            blurShader.prog.SetUniform("uTexelSize", 1.0f / (float)fbW, 1.0f / (float)fbH);
            blurShader.prog.SetUniform("uHorizontal", 1);
            glBindTextureUnit(0, bloomBrightRT.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass5: Blur (Vertical) to Bloom Blur Render Target 2
            glBindFramebuffer(GL_FRAMEBUFFER, bloomBlurRT2.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            blurShader.prog.Bind();
            blurShader.prog.SetUniform("uInputTex", 0);
            blurShader.prog.SetUniform("uTexelSize", 1.0f / (float)fbW, 1.0f / (float)fbH);
            blurShader.prog.SetUniform("uHorizontal", 0);
            glBindTextureUnit(0, bloomBlurRT1.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass6: Combine Scene + Bloom to Combine Render Target
            glBindFramebuffer(GL_FRAMEBUFFER, combineRT.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            combineShader.prog.Bind();
            combineShader.prog.SetUniform("uSceneTex", 0);
            combineShader.prog.SetUniform("uBloomTex", 1);
            combineShader.prog.SetUniform("uEnableBloom", g_enableBloom ? 1 : 0);
            combineShader.prog.SetUniform("uBloomStrength", g_bloomStrength);
            glBindTextureUnit(0, motionRT.colorTex);
            glBindTextureUnit(1, bloomBlurRT2.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass7: Tone Mapping to ToneMap Render Target
            glBindFramebuffer(GL_FRAMEBUFFER, toneMapRT.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            toneMapShader.prog.Bind();
            toneMapShader.prog.SetUniform("uInputTex", 0);
            toneMapShader.prog.SetUniform("uEnableToneMapping", g_enableToneMapping ? 1 : 0);
            toneMapShader.prog.SetUniform("uExposure", g_exposure);
            toneMapShader.prog.SetUniform("uToneMapMode", g_toneMapMode);
            glBindTextureUnit(0, combineRT.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

            // Pass8: Color Grading to Grade Render Target
            glBindFramebuffer(GL_FRAMEBUFFER, gradeRT.fbo);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            gradingShader.prog.Bind();
            gradingShader.prog.SetUniform("uInputTex", 0);
            gradingShader.prog.SetUniform("uEnableColorGrading", g_enableColorGrading ? 1 : 0);
            gradingShader.prog.SetUniform("uSaturation", g_gradeSaturation);
            gradingShader.prog.SetUniform("uContrast", g_gradeContrast);
            gradingShader.prog.SetUniform("uBrightness", g_gradeBrightness);
            gradingShader.prog.SetUniform("uColorFilter", g_colorFilter.x, g_colorFilter.y, g_colorFilter.z);
            glBindTextureUnit(0, toneMapRT.colorTex);
            DrawFullscreenQuad(fsQuadVAO);

			// Pass9: FXAA to Screen
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbW, fbH);
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT);

            fxaaShader.prog.Bind();
            fxaaShader.prog.SetUniform("uInputTex", 0);
            fxaaShader.prog.SetUniform("uEnableFXAA", g_enableFXAA ? 1 : 0);
            fxaaShader.prog.SetUniform("uInvScreenSize", 1.0f / (float)fbW, 1.0f / (float)fbH);
            glBindTextureUnit(0, gradeRT.colorTex);
            DrawFullscreenQuad(fsQuadVAO);
        }

		// Set Current VP as Previous VP for next frame
        g_prevVP = currentVP;
        g_hasPrevFrame = true;

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

	// Destroy Render Targets
    DestroyColorRenderTarget(gradeRT);
    DestroyColorRenderTarget(toneMapRT);
    DestroyColorRenderTarget(combineRT);
    DestroyColorRenderTarget(bloomBlurRT2);
    DestroyColorRenderTarget(bloomBlurRT1);
    DestroyColorRenderTarget(bloomBrightRT);
    DestroyColorRenderTarget(motionRT);
    DestroySceneRenderTarget(sceneRT);

    glDeleteBuffers(1, &fsQuadVBO);
    glDeleteVertexArrays(1, &fsQuadVAO);
    glDeleteBuffers(1, &normVBO);
    glDeleteBuffers(1, &posVBO);
    glDeleteVertexArrays(1, &meshVAO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}