#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cyGL.h"
#include "cyTriMesh.h"
#include "cyMatrix.h"
#include "lodepng.h"


// Properties
// Mouse status
static bool g_leftDown = false;
static bool g_rightDown = false;
static double g_lastX = 0.0, g_lastY = 0.0;

// Camera parameter (object -> render to texture)
static float g_yaw = 0.0f;
static float g_pitch = 0.0f;
static float g_dist = 2.0f;

// Camera parameter (plane view -> default framebuffer, hold ALT/Option to control)
static float g_planeYaw = 0.0f;
static float g_planePitch = 0.0f;
static float g_planeDist = 2.0f;

// Object
static cy::Vec3f g_objCenter(0.0f, 0.0f, 0.0f);
static float g_objScale = 1.0f;

// Perspective or Orthographic
static bool g_usePerspective = true;
static float g_orthoScale = 1.5f;

// Visualization & shading mode (used by key_callback)
// 0: full shading, 1: ambient, 2: diffuse, 3: specular, 4: normal-as-color
static int g_visMode = 0;
static bool g_showTriangulation = false;

static bool g_useDisplacement = false;
static float g_tessLevel = 8.0f;
static float g_dispScale = 0.0f;

// Light properties
static float g_lightYaw = 0.7f;
static float g_lightPitch = 0.4f;
static float g_lightRadius = 3.0f;


// Shader
struct Shader
{
    cy::GLSLProgram prog;
    bool reloadShaders = false;

    // Read shader from glsl
    std::string vsPath = "";
    std::string fsPath = "";

    // Fallback
    const char* vsFallback = R"GLSL(
    #version 460 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aNormal;
    layout(location=2) in vec2 aUV;
    layout(location=3) in vec3 aTangent;
    layout(location=4) in vec3 aBitangent;

    uniform mat4 uM;
    uniform mat4 uV;
    uniform mat4 uP;

    out vec3 vPosW;
    out vec2 vUV;
    out mat3 vTBN;

    void main()
    {
        vec4 posW = uM * vec4(aPos, 1.0);
        vPosW = posW.xyz;
        vUV = aUV;

        mat3 normalMat = mat3(transpose(inverse(uM)));
        vec3 N = normalize(normalMat * aNormal);
        vec3 T = normalize(normalMat * aTangent);
        T = normalize(T - dot(T, N) * N);
        vec3 B = normalize(cross(N, T));

        vTBN = mat3(T, B, N);

        gl_Position = uP * uV * posW;
    }
    )GLSL";

    const char* fsFallback = R"GLSL(
    #version 460 core

    in vec3 vPosW;
    in vec2 vUV;
    in mat3 vTBN;

    uniform sampler2D uNormalMap;
    uniform vec3 uLightPosW;
    uniform vec3 uCamPosW;
    uniform vec3 uKa;
    uniform vec3 uKd;
    uniform vec3 uKs;
    uniform float uNs;
    uniform int uVisMode;

    out vec4 FragColor;

    void main()
    {
        vec3 nTex = texture(uNormalMap, vUV).xyz;
        nTex = nTex * 2.0 - 1.0;

        vec3 N = normalize(vTBN * nTex);
        vec3 L = normalize(uLightPosW - vPosW);
        vec3 V = normalize(uCamPosW - vPosW);
        vec3 H = normalize(L + V);

        float diff = max(dot(N, L), 0.0);
        float spec = 0.0;
        if (diff > 0.0)
        {
            spec = pow(max(dot(N, H), 0.0), uNs);
        }

        vec3 ambient  = uKa;
        vec3 diffuse  = uKd * diff;
        vec3 specular = uKs * spec;

        vec3 color;
        if (uVisMode == 1) color = ambient;
        else if (uVisMode == 2) color = diffuse;
        else if (uVisMode == 3) color = specular;
        else if (uVisMode == 4) color = N * 0.5 + 0.5;
        else color = ambient + diffuse + specular;

        FragColor = vec4(color, 1.0);
    }
    )GLSL";
};

// Plane shader (render to a quad)
struct PlaneShader
{
    cy::GLSLProgram prog;
    bool built = false;

    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec2 aUV;
        uniform mat4 uMVP;
        out vec2 vUV;
        void main()
        {
            vUV = vec2(aUV.x, 1.0 - aUV.y);
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )GLSL";
    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        uniform sampler2D uTex;
        uniform vec3 uAdd;
        out vec4 FragColor;
        void main()
        {
            vec3 c = texture(uTex, vUV).rgb + uAdd;   // small constant to separate from background
            FragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
        }
    )GLSL";
};

struct SkyboxShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        out vec3 vDirW;
        uniform mat4 uProj;
        uniform mat4 uView; 
        void main()
        {
            vDirW = aPos;
            vec4 pos = uProj * uView * vec4(aPos, 1.0);
            gl_Position = pos.xyww;
        }
    )GLSL";
    const char* fs = R"GLSL(
        #version 460 core
        in vec3 vDirW;
        out vec4 FragColor;
        uniform samplerCube uEnv;
        void main()
        {
            vec3 dir = normalize(vDirW);
            vec3 c = texture(uEnv, dir).rgb;
            FragColor = vec4(c, 1.0);
        }
    )GLSL";
};

struct ReflectShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        layout(location=2) in vec2 aUV;
        uniform mat4 uM;
        uniform mat4 uV;
        uniform mat4 uP;
        out vec4 vWorldPos;
        void main()
        {
            vWorldPos = uM * vec4(aPos, 1.0);
            gl_Position = uP * uV * vWorldPos;
        }
    )GLSL";
    const char* fs = R"GLSL(
        #version 460 core
        in vec4 vWorldPos;
        out vec4 FragColor;
        uniform mat4 uVref;
        uniform mat4 uPref;
        uniform sampler2D uReflectionTex;
        uniform vec3 uCamPosW;
        uniform vec3 uFadeCenterW;
        uniform float uFadeRadius;
        uniform float uReflectOpacity;
        void main()
        {
            vec4 clip = uPref * uVref * vWorldPos;
            vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
            float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
            vec3 planar = texture(uReflectionTex, clamp(uv, 0.0, 1.0)).rgb;
            // Fresnel based on plane normal (0,1,0)
            vec3 N = vec3(0,1,0);
            vec3 V = normalize(uCamPosW - vWorldPos.xyz);
            float F = pow(1.0 - max(dot(N, V), 0.0), 5.0);
            // Distance Fade on XZ
            float d = distance(vWorldPos.xz, uFadeCenterW.xz);
            float fade = 1.0 - smoothstep(0.0, uFadeRadius, d);
            float alpha = uReflectOpacity * inside * (0.15 + 0.85*F) * fade;
            FragColor = vec4(planar, alpha);
        }
    )GLSL";
};

struct ShadowDepthShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uLightMVP;
        void main()
        {
            gl_Position = uLightMVP * vec4(aPos, 1.0);
        }
    )GLSL";

    const char* fs = R"GLSL(
        #version 460 core
        void main() { }
    )GLSL";
};

struct LightMarkerShader
{
    cy::GLSLProgram prog;
    bool built = false;
    const char* vs = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uM;
        uniform mat4 uV;
        uniform mat4 uP;
        void main()
        {
            gl_Position = uP * uV * uM * vec4(aPos, 1.0);
        }
    )GLSL";

    const char* fs = R"GLSL(
        #version 460 core
        uniform vec3 uColor;
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(uColor, 1.0);
        }
    )GLSL";
};
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
    return Tcam * R_pitch * R_yaw;
}

static bool LoadPNGTexture(const std::string& pngPath, std::vector<unsigned char>& outRGBA, unsigned& outW, unsigned& outH)
{
    outRGBA.clear();
    outW = outH = 0;
    unsigned err = lodepng::decode(outRGBA, outW, outH, pngPath);
    if (err != 0)
    {
        std::cerr << "ERROR: lodepng decode failed: " << pngPath << " (" << err << ": " << lodepng_error_text(err) << ")\n";
        return false;
    }
    return true;
}

static GLuint CreateTexture2D(unsigned w, unsigned h, const unsigned char* rgba)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)w, (GLsizei)h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static GLuint CompileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        std::string log((size_t)max(1, logLen), '\0');
        glGetShaderInfoLog(shader, logLen, nullptr, log.data());
        std::cerr << "Shader compile failed:\n" << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint LinkProgram(const std::vector<GLuint>& shaders)
{
    GLuint prog = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(prog, s);
    glLinkProgram(prog);

    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        GLint logLen = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
        std::string log((size_t)max(1, logLen), '\0');
        glGetProgramInfoLog(prog, logLen, nullptr, log.data());
        std::cerr << "Program link failed:\n" << log << std::endl;
        glDeleteProgram(prog);
        return 0;
    }
    for (GLuint s : shaders) glDetachShader(prog, s);
    return prog;
}

static GLuint BuildProgramVF(const char* vs, const char* fs)
{
    GLuint v = CompileShader(GL_VERTEX_SHADER, vs);
    GLuint f = CompileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) return 0;
    GLuint p = LinkProgram({ v, f });
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static GLuint BuildProgramVGF(const char* vs, const char* gs, const char* fs)
{
    GLuint v = CompileShader(GL_VERTEX_SHADER, vs);
    GLuint g = CompileShader(GL_GEOMETRY_SHADER, gs);
    GLuint f = CompileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !g || !f)
    {
        if (v) glDeleteShader(v);
        if (g) glDeleteShader(g);
        if (f) glDeleteShader(f);
        return 0;
    }
    GLuint p = LinkProgram({ v, g, f });
    glDeleteShader(v);
    glDeleteShader(g);
    glDeleteShader(f);
    return p;
}

static GLuint BuildProgramVTETF(const char* vs, const char* tcs, const char* tes, const char* fs)
{
    GLuint v = CompileShader(GL_VERTEX_SHADER, vs);
    GLuint c = CompileShader(GL_TESS_CONTROL_SHADER, tcs);
    GLuint e = CompileShader(GL_TESS_EVALUATION_SHADER, tes);
    GLuint f = CompileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !c || !e || !f)
    {
        if (v) glDeleteShader(v);
        if (c) glDeleteShader(c);
        if (e) glDeleteShader(e);
        if (f) glDeleteShader(f);
        return 0;
    }
    GLuint p = LinkProgram({ v, c, e, f });
    glDeleteShader(v);
    glDeleteShader(c);
    glDeleteShader(e);
    glDeleteShader(f);
    return p;
}

static GLuint BuildProgramVTETGF(const char* vs, const char* tcs, const char* tes, const char* gs, const char* fs)
{
    GLuint v = CompileShader(GL_VERTEX_SHADER, vs);
    GLuint c = CompileShader(GL_TESS_CONTROL_SHADER, tcs);
    GLuint e = CompileShader(GL_TESS_EVALUATION_SHADER, tes);
    GLuint g = CompileShader(GL_GEOMETRY_SHADER, gs);
    GLuint f = CompileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !c || !e || !g || !f)
    {
        if (v) glDeleteShader(v);
        if (c) glDeleteShader(c);
        if (e) glDeleteShader(e);
        if (g) glDeleteShader(g);
        if (f) glDeleteShader(f);
        return 0;
    }
    GLuint p = LinkProgram({ v, c, e, g, f });
    glDeleteShader(v);
    glDeleteShader(c);
    glDeleteShader(e);
    glDeleteShader(g);
    glDeleteShader(f);
    return p;
}

// Event call back
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
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        g_usePerspective = !g_usePerspective;
        std::cout << "[P] Projection = " << (g_usePerspective ? "Perspective" : "Orthographic") << std::endl;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        g_showTriangulation = !g_showTriangulation;
        std::cout << "[Space] Triangulation = " << (g_showTriangulation ? "ON" : "OFF") << std::endl;
    }
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_LEFT)
        {
            g_tessLevel -= 1.0f;
            if (g_tessLevel < 1.0f) g_tessLevel = 1.0f;
            std::cout << "[Left] Tess level = " << g_tessLevel << std::endl;
        }
        if (key == GLFW_KEY_RIGHT)
        {
            g_tessLevel += 1.0f;
            if (g_tessLevel > 64.0f) g_tessLevel = 64.0f;
            std::cout << "[Right] Tess level = " << g_tessLevel << std::endl;
        }
        if (key == GLFW_KEY_UP)
        {
            g_dispScale += 0.02f;
            std::cout << "[Up] Displacement scale = " << g_dispScale << std::endl;
        }
        if (key == GLFW_KEY_DOWN)
        {
            g_dispScale -= 0.02f;
            if (g_dispScale < 0.0f) g_dispScale = 0.0f;
            std::cout << "[Down] Displacement scale = " << g_dispScale << std::endl;
        }
        if (key == GLFW_KEY_0) g_visMode = 0;
        if (key == GLFW_KEY_1) g_visMode = 1;
        if (key == GLFW_KEY_2) g_visMode = 2;
        if (key == GLFW_KEY_3) g_visMode = 3;
        if (key == GLFW_KEY_N) g_visMode = 4;
    }
}
// ------------------------------


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <normal_map.png>\n";
        return -1;
    }

    const std::string normalMapPath = argv[1];

    std::string dispMapPath;
    if (argc >= 3)
    {
        dispMapPath = argv[2];
        g_useDisplacement = true;
    }

    if (!glfwInit())
    {
        std::cerr << "ERROR: glfwInit failed\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Displacement Mapping", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "ERROR: glfwCreateWindow failed\n";
        glfwTerminate();
        return -1;
    }

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

    std::vector<unsigned char> normalRGBA;
    unsigned normalW = 0, normalH = 0;
    if (!LoadPNGTexture(normalMapPath, normalRGBA, normalW, normalH))
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    GLuint normalTex = CreateTexture2D(normalW, normalH, normalRGBA.data());

    GLuint dispTex = 0;
    if (g_useDisplacement)
    {
        std::vector<unsigned char> dispRGBA;
        unsigned dispW = 0, dispH = 0;
        if (!LoadPNGTexture(dispMapPath, dispRGBA, dispW, dispH))
        {
            glfwDestroyWindow(window);
            glfwTerminate();
            return -1;
        }
        dispTex = CreateTexture2D(dispW, dispH, dispRGBA.data());
    }

    const char* flatVS = R"GLSL(
    #version 460 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aNormal;
    layout(location=2) in vec2 aUV;
    layout(location=3) in vec3 aTangent;
    layout(location=4) in vec3 aBitangent;

    uniform mat4 uM;
    uniform mat4 uV;
    uniform mat4 uP;

    out vec3 vPosW;
    out vec2 vUV;
    out mat3 vTBN;

    void main()
    {
        vec4 posW = uM * vec4(aPos, 1.0);
        vPosW = posW.xyz;
        vUV = aUV;

        mat3 normalMat = mat3(transpose(inverse(uM)));
        vec3 N = normalize(normalMat * aNormal);
        vec3 T = normalize(normalMat * aTangent);
        T = normalize(T - dot(T, N) * N);
        vec3 B = normalize(cross(N, T));
        vTBN = mat3(T, B, N);

        gl_Position = uP * uV * posW;
    }
    )GLSL";

    const char* litFS = R"GLSL(
    #version 460 core
    in vec3 vPosW;
    in vec2 vUV;
    in mat3 vTBN;

    uniform sampler2D uNormalMap;
    uniform vec3 uLightPosW;
    uniform vec3 uCamPosW;
    uniform vec3 uKa;
    uniform vec3 uKd;
    uniform vec3 uKs;
    uniform float uNs;
    uniform int uVisMode;

    out vec4 FragColor;

    void main()
    {
        vec3 nTex = texture(uNormalMap, vUV).xyz * 2.0 - 1.0;
        vec3 N = normalize(vTBN * nTex);
        vec3 L = normalize(uLightPosW - vPosW);
        vec3 V = normalize(uCamPosW - vPosW);
        vec3 H = normalize(L + V);

        float diff = max(dot(N, L), 0.0);
        float spec = diff > 0.0 ? pow(max(dot(N, H), 0.0), uNs) : 0.0;

        vec3 ambient = uKa;
        vec3 diffuse = uKd * diff;
        vec3 specular = uKs * spec;

        vec3 color;
        if (uVisMode == 1) color = ambient;
        else if (uVisMode == 2) color = diffuse;
        else if (uVisMode == 3) color = specular;
        else if (uVisMode == 4) color = N * 0.5 + 0.5;
        else color = ambient + diffuse + specular;

        FragColor = vec4(color, 1.0);
    }
    )GLSL";

    const char* triVS = R"GLSL(
    #version 460 core
    layout(location=0) in vec3 aPos;

    uniform mat4 uM;
    uniform mat4 uV;
    uniform mat4 uP;
    uniform float uLineOffset;

    void main()
    {
        vec4 posW = uM * vec4(aPos, 1.0);
        vec4 posV = uV * posW;
        posV.z += uLineOffset;
        gl_Position = uP * posV;
    }
    )GLSL";

    const char* triGS = R"GLSL(
    #version 460 core
    layout(triangles) in;
    layout(line_strip, max_vertices = 6) out;

    void EmitEdge(int a, int b)
    {
        gl_Position = gl_in[a].gl_Position; EmitVertex();
        gl_Position = gl_in[b].gl_Position; EmitVertex();
        EndPrimitive();
    }

    void main()
    {
        EmitEdge(0,1);
        EmitEdge(1,2);
        EmitEdge(2,0);
    }
    )GLSL";

    const char* triFS = R"GLSL(
    #version 460 core
    uniform vec3 uColor;
    out vec4 FragColor;
    void main() { FragColor = vec4(uColor, 1.0); }
    )GLSL";

    const char* patchVS = R"GLSL(
    #version 460 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec2 aUV;
    out vec2 vUV;
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        vUV = aUV;
    }
    )GLSL";

    const char* patchTCS = R"GLSL(
    #version 460 core
    layout(vertices = 4) out;

    in vec2 vUV[];
    out vec2 tcUV[];

    uniform float uTessLevel;

    void main()
    {
        gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
        tcUV[gl_InvocationID] = vUV[gl_InvocationID];

        if (gl_InvocationID == 0)
        {
            gl_TessLevelOuter[0] = uTessLevel;
            gl_TessLevelOuter[1] = uTessLevel;
            gl_TessLevelOuter[2] = uTessLevel;
            gl_TessLevelOuter[3] = uTessLevel;
            gl_TessLevelInner[0] = uTessLevel;
            gl_TessLevelInner[1] = uTessLevel;
        }
    }
    )GLSL";

    const char* patchTES = R"GLSL(
    #version 460 core
    layout(quads, fractional_odd_spacing, ccw) in;

    in vec2 tcUV[];
    out vec3 tePosW;
    out vec2 teUV;
    out mat3 teTBN;

    uniform mat4 uM;
    uniform mat4 uV;
    uniform mat4 uP;
    uniform sampler2D uDispMap;
    uniform float uDispScale;

    vec3 EvalPos(float u, float v)
    {
        vec3 p00 = gl_in[0].gl_Position.xyz;
        vec3 p10 = gl_in[1].gl_Position.xyz;
        vec3 p01 = gl_in[2].gl_Position.xyz;
        vec3 p11 = gl_in[3].gl_Position.xyz;

        vec3 p0 = mix(p00, p10, u);
        vec3 p1 = mix(p01, p11, u);
        return mix(p0, p1, v);
    }

    vec2 EvalUV(float u, float v)
    {
        vec2 uv00 = tcUV[0];
        vec2 uv10 = tcUV[1];
        vec2 uv01 = tcUV[2];
        vec2 uv11 = tcUV[3];

        vec2 t0 = mix(uv00, uv10, u);
        vec2 t1 = mix(uv01, uv11, u);
        return mix(t0, t1, v);
    }

    void main()
    {
        float u = gl_TessCoord.x;
        float v = gl_TessCoord.y;
        vec3 pos = EvalPos(u, v);
        vec2 uv = EvalUV(u, v);

        float h = texture(uDispMap, uv).r;
        pos.y += h * uDispScale;

        const float eps = 1.0 / 1024.0;
        float hU = texture(uDispMap, clamp(uv + vec2(eps, 0.0), 0.0, 1.0)).r;
        float hV = texture(uDispMap, clamp(uv + vec2(0.0, eps), 0.0, 1.0)).r;

        vec3 dpdu = vec3(2.0, (hU - h) * uDispScale / eps, 0.0);
        vec3 dpdv = vec3(0.0, (hV - h) * uDispScale / eps, 2.0);
        vec3 Nobj = normalize(cross(dpdv, dpdu));
        vec3 Tobj = normalize(dpdu);
        vec3 Bobj = normalize(dpdv);

        mat3 normalMat = mat3(transpose(inverse(uM)));
        vec3 Nw = normalize(normalMat * Nobj);
        vec3 Tw = normalize(normalMat * Tobj);
        Tw = normalize(Tw - dot(Tw, Nw) * Nw);
        vec3 Bw = normalize(cross(Nw, Tw));

        teTBN = mat3(Tw, Bw, Nw);

        vec4 posW = uM * vec4(pos, 1.0);
        tePosW = posW.xyz;
        teUV = uv;
        gl_Position = uP * uV * posW;
    }
    )GLSL";

    const char* patchFS = R"GLSL(
    #version 460 core
    in vec3 tePosW;
    in vec2 teUV;
    in mat3 teTBN;

    uniform sampler2D uNormalMap;
    uniform vec3 uLightPosW;
    uniform vec3 uCamPosW;
    uniform vec3 uKa;
    uniform vec3 uKd;
    uniform vec3 uKs;
    uniform float uNs;
    uniform int uVisMode;

    out vec4 FragColor;

    void main()
    {
        vec3 nTex = texture(uNormalMap, teUV).xyz * 2.0 - 1.0;
        vec3 N = normalize(teTBN * nTex);
        vec3 L = normalize(uLightPosW - tePosW);
        vec3 V = normalize(uCamPosW - tePosW);
        vec3 H = normalize(L + V);

        float diff = max(dot(N, L), 0.0);
        float spec = diff > 0.0 ? pow(max(dot(N, H), 0.0), uNs) : 0.0;

        vec3 ambient = uKa;
        vec3 diffuse = uKd * diff;
        vec3 specular = uKs * spec;

        vec3 color;
        if (uVisMode == 1) color = ambient;
        else if (uVisMode == 2) color = diffuse;
        else if (uVisMode == 3) color = specular;
        else if (uVisMode == 4) color = N * 0.5 + 0.5;
        else color = ambient + diffuse + specular;

        FragColor = vec4(color, 1.0);
    }
    )GLSL";

    const char* patchTriTES = R"GLSL(
    #version 460 core
    layout(quads, fractional_odd_spacing, ccw) in;

    in vec2 tcUV[];

    uniform mat4 uM;
    uniform mat4 uV;
    uniform mat4 uP;
    uniform sampler2D uDispMap;
    uniform float uDispScale;
    uniform float uLineOffset;

    vec3 EvalPos(float u, float v)
    {
        vec3 p00 = gl_in[0].gl_Position.xyz;
        vec3 p10 = gl_in[1].gl_Position.xyz;
        vec3 p01 = gl_in[2].gl_Position.xyz;
        vec3 p11 = gl_in[3].gl_Position.xyz;

        vec3 p0 = mix(p00, p10, u);
        vec3 p1 = mix(p01, p11, u);
        return mix(p0, p1, v);
    }

    vec2 EvalUV(float u, float v)
    {
        vec2 uv00 = tcUV[0];
        vec2 uv10 = tcUV[1];
        vec2 uv01 = tcUV[2];
        vec2 uv11 = tcUV[3];

        vec2 t0 = mix(uv00, uv10, u);
        vec2 t1 = mix(uv01, uv11, u);
        return mix(t0, t1, v);
    }

    void main()
    {
        float u = gl_TessCoord.x;
        float v = gl_TessCoord.y;
        vec3 pos = EvalPos(u, v);
        vec2 uv = EvalUV(u, v);
        float h = texture(uDispMap, uv).r;
        pos.y += h * uDispScale;

        vec4 posW = uM * vec4(pos, 1.0);
        vec4 posV = uV * posW;
        posV.z += uLineOffset;
        gl_Position = uP * posV;
    }
    )GLSL";

    GLuint flatProg = BuildProgramVF(flatVS, litFS);
    GLuint triProg = BuildProgramVGF(triVS, triGS, triFS);
    GLuint patchProg = BuildProgramVTETF(patchVS, patchTCS, patchTES, patchFS);
    GLuint patchTriProg = BuildProgramVTETGF(patchVS, patchTCS, patchTriTES, triGS, triFS);
    if (!flatProg || !triProg || !patchProg || !patchTriProg)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    GLuint quadVAO = 0, quadVBO = 0;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    const float s = 1.0f;
    const float quadVerts[] = {
        // pos               // n         // uv   // tangent   // bitangent
        -s, 0.0f, -s,        0, 1, 0,     0, 0,   1, 0, 0,     0, 0, 1,
         s, 0.0f, -s,        0, 1, 0,     1, 0,   1, 0, 0,     0, 0, 1,
         s, 0.0f,  s,        0, 1, 0,     1, 1,   1, 0, 0,     0, 0, 1,
        -s, 0.0f, -s,        0, 1, 0,     0, 0,   1, 0, 0,     0, 0, 1,
         s, 0.0f,  s,        0, 1, 0,     1, 1,   1, 0, 0,     0, 0, 1,
        -s, 0.0f,  s,        0, 1, 0,     0, 1,   1, 0, 0,     0, 0, 1,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3); glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(8 * sizeof(float)));
    glEnableVertexAttribArray(4); glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(11 * sizeof(float)));
    glBindVertexArray(0);

    GLuint patchVAO = 0, patchVBO = 0;
    glGenVertexArrays(1, &patchVAO);
    glGenBuffers(1, &patchVBO);
    glBindVertexArray(patchVAO);
    glBindBuffer(GL_ARRAY_BUFFER, patchVBO);
    const float patchVerts[] = {
        // pos              // uv
        -s, 0.0f, -s,       0.0f, 0.0f,
         s, 0.0f, -s,       1.0f, 0.0f,
        -s, 0.0f,  s,       0.0f, 1.0f,
         s, 0.0f,  s,       1.0f, 1.0f,
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(patchVerts), patchVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window))
    {
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cy::Matrix4f P = MakeProjection(fbW, fbH, g_usePerspective, g_orthoScale);
        cy::Matrix4f V = MakeView(g_yaw, g_pitch, g_dist);
        cy::Matrix4f M;
        M.SetIdentity();

        cy::Matrix4f Vinv = V.GetInverse();
        cy::Vec4f camPos4 = Vinv * cy::Vec4f(0, 0, 0, 1);
        cy::Vec3f camPosW(camPos4.x, camPos4.y, camPos4.z);

        float ly = std::sinf(g_lightPitch) * g_lightRadius;
        float lxz = std::cosf(g_lightPitch) * g_lightRadius;
        float lx = std::sinf(g_lightYaw) * lxz;
        float lz = std::cosf(g_lightYaw) * lxz;
        cy::Vec3f lightPosW(lx, ly, lz);

        if (!g_useDisplacement)
        {
            glUseProgram(flatProg);
            glUniformMatrix4fv(glGetUniformLocation(flatProg, "uM"), 1, GL_FALSE, M.cell);
            glUniformMatrix4fv(glGetUniformLocation(flatProg, "uV"), 1, GL_FALSE, V.cell);
            glUniformMatrix4fv(glGetUniformLocation(flatProg, "uP"), 1, GL_FALSE, P.cell);
            glUniform3f(glGetUniformLocation(flatProg, "uLightPosW"), lightPosW.x, lightPosW.y, lightPosW.z);
            glUniform3f(glGetUniformLocation(flatProg, "uCamPosW"), camPosW.x, camPosW.y, camPosW.z);
            glUniform3f(glGetUniformLocation(flatProg, "uKa"), 0.08f, 0.08f, 0.08f);
            glUniform3f(glGetUniformLocation(flatProg, "uKd"), 0.75f, 0.75f, 0.75f);
            glUniform3f(glGetUniformLocation(flatProg, "uKs"), 0.40f, 0.40f, 0.40f);
            glUniform1f(glGetUniformLocation(flatProg, "uNs"), 64.0f);
            glUniform1i(glGetUniformLocation(flatProg, "uVisMode"), g_visMode);
            glUniform1i(glGetUniformLocation(flatProg, "uNormalMap"), 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, normalTex);
            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            if (g_showTriangulation)
            {
                glDisable(GL_DEPTH_TEST);
                glUseProgram(triProg);
                glUniformMatrix4fv(glGetUniformLocation(triProg, "uM"), 1, GL_FALSE, M.cell);
                glUniformMatrix4fv(glGetUniformLocation(triProg, "uV"), 1, GL_FALSE, V.cell);
                glUniformMatrix4fv(glGetUniformLocation(triProg, "uP"), 1, GL_FALSE, P.cell);
                glUniform1f(glGetUniformLocation(triProg, "uLineOffset"), 0.01f);
                glUniform3f(glGetUniformLocation(triProg, "uColor"), 1.0f, 1.0f, 0.0f);
                glBindVertexArray(quadVAO);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                glEnable(GL_DEPTH_TEST);
            }
        }
        else
        {
            glPatchParameteri(GL_PATCH_VERTICES, 4);
            glUseProgram(patchProg);
            glUniformMatrix4fv(glGetUniformLocation(patchProg, "uM"), 1, GL_FALSE, M.cell);
            glUniformMatrix4fv(glGetUniformLocation(patchProg, "uV"), 1, GL_FALSE, V.cell);
            glUniformMatrix4fv(glGetUniformLocation(patchProg, "uP"), 1, GL_FALSE, P.cell);
            glUniform3f(glGetUniformLocation(patchProg, "uLightPosW"), lightPosW.x, lightPosW.y, lightPosW.z);
            glUniform3f(glGetUniformLocation(patchProg, "uCamPosW"), camPosW.x, camPosW.y, camPosW.z);
            glUniform3f(glGetUniformLocation(patchProg, "uKa"), 0.08f, 0.08f, 0.08f);
            glUniform3f(glGetUniformLocation(patchProg, "uKd"), 0.75f, 0.75f, 0.75f);
            glUniform3f(glGetUniformLocation(patchProg, "uKs"), 0.40f, 0.40f, 0.40f);
            glUniform1f(glGetUniformLocation(patchProg, "uNs"), 64.0f);
            glUniform1f(glGetUniformLocation(patchProg, "uTessLevel"), g_tessLevel);
            glUniform1f(glGetUniformLocation(patchProg, "uDispScale"), g_dispScale);
            glUniform1i(glGetUniformLocation(patchProg, "uVisMode"), g_visMode);
            glUniform1i(glGetUniformLocation(patchProg, "uNormalMap"), 0);
            glUniform1i(glGetUniformLocation(patchProg, "uDispMap"), 1);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, normalTex);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, dispTex);
            glBindVertexArray(patchVAO);
            glDrawArrays(GL_PATCHES, 0, 4);

            if (g_showTriangulation)
            {
                glDisable(GL_DEPTH_TEST);
                glUseProgram(patchTriProg);
                glUniformMatrix4fv(glGetUniformLocation(patchTriProg, "uM"), 1, GL_FALSE, M.cell);
                glUniformMatrix4fv(glGetUniformLocation(patchTriProg, "uV"), 1, GL_FALSE, V.cell);
                glUniformMatrix4fv(glGetUniformLocation(patchTriProg, "uP"), 1, GL_FALSE, P.cell);
                glUniform1f(glGetUniformLocation(patchTriProg, "uTessLevel"), g_tessLevel);
                glUniform1f(glGetUniformLocation(patchTriProg, "uDispScale"), g_dispScale);
                glUniform1f(glGetUniformLocation(patchTriProg, "uLineOffset"), 0.01f);
                glUniform1i(glGetUniformLocation(patchTriProg, "uDispMap"), 1);
                glUniform3f(glGetUniformLocation(patchTriProg, "uColor"), 1.0f, 1.0f, 0.0f);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, dispTex);
                glBindVertexArray(patchVAO);
                glDrawArrays(GL_PATCHES, 0, 4);
                glEnable(GL_DEPTH_TEST);
            }
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteBuffers(1, &quadVBO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &patchVBO);
    glDeleteVertexArrays(1, &patchVAO);
    glDeleteTextures(1, &normalTex);
    if (dispTex) glDeleteTextures(1, &dispTex);
    glDeleteProgram(flatProg);
    glDeleteProgram(triProg);
    glDeleteProgram(patchProg);
    glDeleteProgram(patchTriProg);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}