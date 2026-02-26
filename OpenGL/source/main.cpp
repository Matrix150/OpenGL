#include <iostream>
#include <array>

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

// Light properties
static float g_lightYaw = 0.7f;
static float g_lightPitch = 0.4f;
static float g_lightRadius = 3.0f;

// Texture
struct TexturePaths
{
    std::string kd;     // Diffuse
	std::string ks;     // Specular
};

struct GPUMaterial
{
    cy::Vec3f Ka{ 0, 0, 0 };
    cy::Vec3f Kd{ 1, 1, 1 };
    cy::Vec3f Ks{ 0, 0, 0 };
    cy::Vec3f Tf{ 0, 0, 0 };
	float Ns = 0.0f;
	float Ni = 1.0f;
	int illum = 2;

    GLuint texKd = 0;
	GLuint texKs = 0;
	bool hasKd = false;
	bool hasKs = false;
};
// ------------------------------


// Shader
struct Shader
{
    cy::GLSLProgram prog;
    bool reloadShaders = false;

    // Read shader from glsl
    std::string vsPath = "shaders/vertex.glsl";
    std::string fsPath = "shaders/fragment.glsl";

    // Fallback
    const char* vsFallback = R"GLSL(
        #version 460 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uMVP;
        void main()
        {
            gl_Position = uMVP * vec4(aPos, 1.0);
            gl_PointSize = 2.0;
        }
    )GLSL";

    const char* fsFallback = R"GLSL(
        #version 460 core
        out vec4 FragColor;
        void main()
        {
            FragColor = vec4(1.0, 1.0, 1.0, 1.0); // constant color white
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

// Object MVP & MV
static cy::Matrix4f MakeObjectMVP(int fbW, int fbH, float yaw, float pitch, float dist)
{
    cy::Matrix4f P = MakeProjection(fbW, fbH, g_usePerspective, g_orthoScale);
    cy::Matrix4f V = MakeView(yaw, pitch, dist);
    cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
    cy::Matrix4f S; 
    S.SetScale(g_objScale);
    cy::Matrix4f M = S * Tcenter;
    return P * V * M;
}

static cy::Matrix4f MakeObjectMV(float yaw, float pitch, float dist)
{
    cy::Matrix4f V = MakeView(yaw, pitch, dist);
    cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
    cy::Matrix4f S; 
    S.SetScale(g_objScale);
	cy::Matrix4f M = S * Tcenter;
    return V * M;
}

// Plane MVP & MV
static cy::Matrix4f MakePlaneMVP(int fhW, int fbH, float yaw, float pitch, float dist)
{
    cy::Matrix4f P = MakeProjection(fhW, fbH, true, g_orthoScale);
    cy::Matrix4f V = MakeView(yaw, pitch, dist);
    cy::Matrix4f M;
	M.SetIdentity();
	return P * V;
}

static cy::Matrix4f MakePlaneMV(float yaw, float pitch, float dist)
{
    cy::Matrix4f V = MakeView(yaw, pitch, dist);
    cy::Matrix4f M;
    M.SetIdentity();
    return V * M;
}

static bool ReadTextFile(const std::string& path, std::string& outText)
{
    std::ifstream f(path, std::ios::in);
    if (!f.is_open()) 
        return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    outText = ss.str();
    return true;
}

static bool BuildShaders(Shader& shader)
{
    std::string vsText, fsText;
    const char* vs = shader.vsFallback;
    const char* fs = shader.fsFallback;

    bool vsOk = ReadTextFile(shader.vsPath, vsText);
    bool fsOk = ReadTextFile(shader.fsPath, fsText);

    if (vsOk && fsOk)
    {
        vs = vsText.c_str();
        fs = fsText.c_str();
        std::cout << "[F6] Reloading shaders from files:\n" << "  VS: " << shader.vsPath << "\n" << "  FS: " << shader.fsPath << "\n";
    }
    else
    {
        std::cout << "[F6] Shader file(s) not found. Using embedded fallback shaders.\n" << "  Expected:\n" << "    " << shader.vsPath << "\n" << "    " << shader.fsPath << "\n";
    }

    // cyGL: rebuild program
    // Build<files=false, parse=false> : source strings
    if (!shader.prog.Build<false, false>(vs, fs))
    {
        std::cerr << "[F6] Shader build failed. Keeping previous program (if any).\n";
        return false;
    }

    std::cout << "[F6] Shader build OK.\n";
    return true;
}

static bool BuildPlaneShader(PlaneShader& shader)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << "Plane shader build failed.\n";
        return false;
    }
    shader.built = true;
    std::cout << "Plane shader build OK.\n";
    return true;
}

static bool BuildSkyboxShader(SkyboxShader& shader)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << "Skybox shader build failed.\n";
        return false;
    }
    shader.built = true;
    std::cout << "Skybox shader build OK.\n";
    return true;
}

static bool BuildReflectShader(ReflectShader& shader)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << "Reflection shader build failed.\n";
        return false;
    }
    shader.built = true;
    std::cout << "Reflection shader build OK.\n";
    return true;
}

static cy::Vec3f ComputeLightPosViewSpace(float camYaw, float camPitch, float camDist, float lightYaw, float lightPitch, float lightRadius)
{
    // View Matrix (V)
    cy::Matrix4f R_yaw = cy::Matrix4f::RotationY(camYaw);
    cy::Matrix4f R_pitch = cy::Matrix4f::RotationX(camPitch);
    cy::Matrix4f Tcam = cy::Matrix4f::Translation(cy::Vec3f(0.0f, 0.0f, -camDist));
    cy::Matrix4f V = Tcam * R_pitch * R_yaw;

    // Light Position in world space
    float cpitch = cosf(lightPitch), spitch = sinf(lightPitch);
    float cyaw = cosf(lightYaw), syaw = sinf(lightYaw);

    cy::Vec3f lightPosW(lightRadius * cpitch * syaw, lightRadius * spitch, lightRadius * cpitch * cyaw);

    cy::Vec4f lpv4 = V * cy::Vec4f(lightPosW.x, lightPosW.y, lightPosW.z, 1.0f);
    return cy::Vec3f(lpv4.x, lpv4.y, lpv4.z);
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
    if (!rgba || w == 0 || h == 0)
        return 0;

    GLuint tex = 0;
	glCreateTextures(GL_TEXTURE_2D, 1, &tex);
	glTextureStorage2D(tex, 1, GL_RGBA8, (GLsizei)w, (GLsizei)h);
	glTextureSubImage2D(tex, 0, 0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glGenerateTextureMipmap(tex);
	return tex;
}

// OpenGL faces order: +X, -X, +Y, -Y, +Z, -Z
static GLuint CreateCubemapFromPNG(const std::array<std::string, 6>& faces)
{
    std::vector<unsigned char> rgba0;
    unsigned w0 = 0, h0 = 0;
    if (!LoadPNGTexture(faces[0], rgba0, w0, h0))
    {
        std::cerr << "Failed to load cubemap face: " << faces[0] << "\n";
        return 0;
    }

    GLuint tex = 0;
    glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &tex);
    glTextureStorage2D(tex, 1, GL_RGBA8, (GLsizei)w0, (GLsizei)h0);

    auto uploadFace = [&](int FaceIndex, const std::string& path) {
        std::vector<unsigned char> rgba;
        unsigned w = 0, h = 0;
        if (!LoadPNGTexture(path, rgba, w, h))
        {
            std::cerr << "Failed to load cubemap face: " << path << "\n";
            return false;
        };
        if (w != w0 || h != h0)
        {
            std::cerr << "Cubemap face size mismatch: " << path << " (expected " << w0 << "x" << h0 << ", got " << w << "x" << h << ")\n";
            return false;
        }
        glTextureSubImage3D(tex, 0, 0, 0, FaceIndex, (GLsizei)w, (GLsizei)h, 1, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
        return true;
        };

    for (int i = 0; i < 6; ++i)
    {
        if (!uploadFace(i, faces[i]))
        {
            glDeleteTextures(1, &tex);
			return 0;
        }
    }

	glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	return tex;
}

static std::string ResolveTexPath(const std::string& objPathStr, const char* rel)
{
    if (!rel || rel[0] == '\0') 
        return {};
    std::filesystem::path objPath(objPathStr);
    std::filesystem::path baseDir = objPath.has_parent_path() ? objPath.parent_path() : std::filesystem::path(".");
    std::filesystem::path p(rel);
    if (p.is_relative()) 
        p = baseDir / p;
    return p.lexically_normal().string();
}

static void SetupRTTextureFiltering(GLuint texId)
{
	glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	GLfloat maxAnisotropy = 1.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy);
    if (maxAnisotropy > 1.0f)
    {
		GLfloat anisotropy = min(16.0f, maxAnisotropy);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, anisotropy);
    }
	glBindTexture(GL_TEXTURE_2D, 0);
}
// ------------------------------

// Event call back
static void framebuffer_size_callback(GLFWwindow* /*window*/, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
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
	bool altDown = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) || (glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS);

    // Decide controlling camera
	float* yaw = altDown ? &g_planeYaw : &g_yaw;
	float* pitch = altDown ? &g_planePitch : &g_pitch;
    float* dist = altDown ? &g_planeDist : &g_dist;

    if (g_leftDown)
    {
        if (ctrlDown && !altDown)       // Light rotation
        {
            g_lightYaw += (float)dx * rotSpeed;
            g_lightPitch += (float)dy * rotSpeed;
            const float limit = 1.55f;
            if (g_lightPitch > limit) 
                g_lightPitch = limit;
            if (g_lightPitch < -limit) 
                g_lightPitch = -limit;
        }
        else        // Camera Rotation
        {
            *yaw += (float)dx * rotSpeed;
            *pitch += (float)dy * rotSpeed;
            const float limit = 1.55f;      // approximate 89 degrees
            if (*pitch > limit)
                *pitch = limit;
            if (*pitch < -limit)
                *pitch = -limit;
        }
    }
    if (g_rightDown)
    {
        *dist += (float)dy * zoomSpeed;
        if (*dist < 0.5)
            *dist = 0.5;
        if (*dist > 5.0)
            *dist = 5.0f;
    }
}
 
static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    // ESC to close window
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    // F6 to recompile shader
    if (key == GLFW_KEY_F6 && action == GLFW_PRESS)
    {
        auto* shader = reinterpret_cast<Shader*>(glfwGetWindowUserPointer(window));
        if (shader)
            shader->reloadShaders = true;
    }
    // P to switch between perspective and orthogonal transformation
    if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        g_usePerspective = !g_usePerspective;
        std::cout << "[P] Projection = " << (g_usePerspective ? "Perspective" : "Orthographic") << std::endl;
    }
    // 1-3 to for Blinn components, 0 for full shading, N for normal visualization
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_0)
        {
            g_visMode = 0;
            std::cout << "[0] Shading = full (ambient+diffuse+spec)" << std::endl;
        }
        if (key == GLFW_KEY_1) 
        {
            g_visMode = 1; 
            std::cout << "[1] Shading = ambient" << std::endl; 
        }
        if (key == GLFW_KEY_2) 
        {
            g_visMode = 2;
            std::cout << "[2] Shading = diffuse" << std::endl; 
        }
        if (key == GLFW_KEY_3) 
        { 
            g_visMode = 3; 
            std::cout << "[3] Shading = specular" << std::endl; 
        }
        if (key == GLFW_KEY_N) 
        {
            g_visMode = 4; 
            std::cout << "[N] Shading = normal-as-color" << std::endl; 
        }
    }
}
// ------------------------------


int main(int argc, char** argv)
{
    // Load obj file from assets folder
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

    // Get vertices & bounding box & center & scale
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
    cy::Vec3f ext = bbMax - bbMin;
    float maxExtent = max(ext.x, max(ext.y, ext.z));
    const float targetSize = 2.0f;
    g_objScale = (maxExtent > 1e-8f) ? (targetSize / maxExtent) : 1.0f;         // Auto scale
    std::cout << "NV=" << mesh.NV() << "  NF=" << mesh.NF() << "\n";
    std::cout << "AABB Min: (" << bbMin.x << ", " << bbMin.y << ", " << bbMin.z << ")\n";
    std::cout << "AABB Max: (" << bbMax.x << ", " << bbMax.y << ", " << bbMax.z << ")\n";
    std::cout << "Center  : (" << g_objCenter.x << ", " << g_objCenter.y << ", " << g_objCenter.z << ")\n";
    std::cout << "Scale   : " << g_objScale << " (maxExtent=" << maxExtent << ")\n";
    float diag = sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z) * g_objScale;
    g_dist = max(2.0f, diag * 0.1f);
    g_orthoScale = 1.5f;

    mesh.ComputeNormals();
    const bool hasUV = mesh.HasTextureVertices();

    std::vector<float> positions;
    positions.reserve(mesh.NF() * 3 * 3);
    std::vector<float> normals;
    normals.reserve(mesh.NF() * 3 * 3);
    std::vector<float> uvs;
    uvs.reserve(mesh.NF() * 3 * 2);

    for (unsigned int fi = 0; fi < mesh.NF(); ++fi)
    {
        auto f = mesh.F((int)fi);
        auto fn = mesh.FN((int)fi);
        auto ft = hasUV ? mesh.FT((int)fi) : cy::TriMesh::TriFace();

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

            float u = 0.0f, v = 0.0f;
            if (hasUV)
            {
                int ti = ft.v[c];
                if (ti >= 0 && (unsigned)ti < mesh.NVT())
                {
                    cy::Vec3f t = mesh.VT(ti);
                    u = t.x;
                    v = t.y;
                }
            }
            uvs.push_back(u);
            uvs.push_back(v);
        }
    }

    //const GLsizei drawVertexCount = (GLsizei)(mesh.NF() * 3);

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
    GLFWwindow* window = glfwCreateWindow(initW, initH, "Project 6 - Environment Mapping", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "ERROR: glfwCreateWindow failed\n";
        glfwTerminate();
        return -1;
    }

    // Callback
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetKeyCallback(window, key_callback);

    // GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "ERROR: gladLoadGLLoader failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << "\n";        // Show OpenGL Version
    std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    Shader shader;
    glfwSetWindowUserPointer(window, &shader);
    if (!BuildShaders(shader))
    {
        std::cerr << "ERROR: shader build failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    PlaneShader planeShader;
    if (!BuildPlaneShader(planeShader))
    {
        std::cerr << "ERROR: plane shader build failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    SkyboxShader skyboxShader;
    if (!BuildSkyboxShader(skyboxShader))
    {
        std::cerr << "ERROR: skybox shader build failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    ReflectShader reflectShader;
    if (!BuildReflectShader(reflectShader))
    {
        std::cerr << "ERROR: reflection shader build failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Cubemap Setup
    std::array<std::string, 6> cubemapFaces = {
        "assets/cubemap/cubemap_posx.png",
        "assets/cubemap/cubemap_negx.png",
        "assets/cubemap/cubemap_posy.png",
        "assets/cubemap/cubemap_negy.png",
        "assets/cubemap/cubemap_posz.png",
        "assets/cubemap/cubemap_negz.png"
    };
    GLuint cubemapTex = CreateCubemapFromPNG(cubemapFaces);
    if (!cubemapTex)
    {
        std::cerr << "ERROR: env cubemap creation failed\n";
        return -1;
    }

    // Skybox Cube setup (for rendering cubemap background)
    GLuint skyboxVAO = 0, skyboxVBO = 0;
    glCreateVertexArrays(1, &skyboxVAO);
    glCreateBuffers(1, &skyboxVBO);
    static const float skyboxVerts[] = {
        // ... 36 vertices (pos only)
        -1,  1, -1,  -1, -1, -1,   1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
        -1, -1,  1,  -1, -1, -1,  -1,  1, -1,  -1,  1, -1,  -1,  1,  1,  -1, -1,  1,
         1, -1, -1,   1, -1,  1,   1,  1,  1,   1,  1,  1,   1,  1, -1,   1, -1, -1,
        -1, -1,  1,  -1,  1,  1,   1,  1,  1,   1,  1,  1,   1, -1,  1,  -1, -1,  1,
        -1,  1, -1,   1,  1, -1,   1,  1,  1,   1,  1,  1,  -1,  1,  1,  -1,  1, -1,
        -1, -1, -1,  -1, -1,  1,   1, -1, -1,   1, -1, -1,  -1, -1,  1,   1, -1,  1
    };

    glNamedBufferData(skyboxVBO, sizeof(skyboxVerts), skyboxVerts, GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(skyboxVAO, 0, skyboxVBO, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(skyboxVAO, 0);
    glVertexArrayAttribFormat(skyboxVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(skyboxVAO, 0, 0);

    // Render to texture setup (color + depth)
    cy::GLRenderTexture<GL_TEXTURE_2D> renderTex;
    int fbW0 = 1024, fbH0 = 1024;
    glfwGetFramebufferSize(window, &fbW0, &fbH0);
    if (!renderTex.Initialize(true, 4, (GLsizei)fbW0, (GLsizei)fbH0, cy::GL::TYPE_UBYTE))
    {
        std::cerr << "ERROR: render texture initialization failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    SetupRTTextureFiltering(renderTex.GetTextureID());

    // Plane geometry setup (Render texture)
    GLuint planeVAO = 0, planeVBO = 0;
    glCreateVertexArrays(1, &planeVAO);
    glCreateBuffers(1, &planeVBO);

    const float planeVertices[] = {
        // positions        // UVs
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f
    };
    glNamedBufferData(planeVBO, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(planeVAO, 0, planeVBO, 0, 5 * sizeof(float));
    glEnableVertexArrayAttrib(planeVAO, 0);
    glVertexArrayAttribFormat(planeVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(planeVAO, 0, 0);
    glEnableVertexArrayAttrib(planeVAO, 1);
    glVertexArrayAttribFormat(planeVAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
    glVertexArrayAttribBinding(planeVAO, 1, 0);

    g_planeDist = 2.5f;

    // Environment Reflection Plane setup (underneath the object)
    GLuint reflPlaneVAO = 0, reflPlaneVBO = 0;
    glCreateVertexArrays(1, &reflPlaneVAO);
    glCreateBuffers(1, &reflPlaneVBO);

    float size = 1.0f;
    const float reflPlaneVerts[] = {
        // positions            normals        UV
        -size, 0, -size,        0,1,0,         0,0,
         size, 0, -size,        0,1,0,         1,0,
         size, 0,  size,        0,1,0,         1,1,

        -size, 0, -size,        0,1,0,         0,0,
         size, 0,  size,        0,1,0,         1,1,
        -size, 0,  size,        0,1,0,         0,1,
    };
    glNamedBufferData(reflPlaneVBO, sizeof(reflPlaneVerts), reflPlaneVerts, GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(reflPlaneVAO, 0, reflPlaneVBO, 0, 8 * sizeof(float));
    glEnableVertexArrayAttrib(reflPlaneVAO, 0);
    glVertexArrayAttribFormat(reflPlaneVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(reflPlaneVAO, 0, 0);
    glEnableVertexArrayAttrib(reflPlaneVAO, 1);
    glVertexArrayAttribFormat(reflPlaneVAO, 1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
    glVertexArrayAttribBinding(reflPlaneVAO, 1, 0);
    glEnableVertexArrayAttrib(reflPlaneVAO, 2);
    glVertexArrayAttribFormat(reflPlaneVAO, 2, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float));
    glVertexArrayAttribBinding(reflPlaneVAO, 2, 0);

    // Build GPU Materials
    std::vector<GPUMaterial> gpuMtls;
    if (mesh.NM() > 0)
    {
        gpuMtls.resize(mesh.NM());

        for (unsigned int mi = 0; mi < mesh.NM(); ++mi)
        {
            const auto& mtl = mesh.M((int)mi);
            GPUMaterial gpuMtl;
            gpuMtl.Ka = cy::Vec3f(mtl.Ka[0], mtl.Ka[1], mtl.Ka[2]);
            gpuMtl.Kd = cy::Vec3f(mtl.Kd[0], mtl.Kd[1], mtl.Kd[2]);
            gpuMtl.Ks = cy::Vec3f(mtl.Ks[0], mtl.Ks[1], mtl.Ks[2]);
            gpuMtl.Tf = cy::Vec3f(mtl.Tf[0], mtl.Tf[1], mtl.Tf[2]);
            gpuMtl.Ns = mtl.Ns;
            gpuMtl.Ni = mtl.Ni;
            gpuMtl.illum = mtl.illum;

            // map_kd
            std::string kdPath = ResolveTexPath(argv[1], mtl.map_Kd.data);
            if (!kdPath.empty())
            {
                std::vector<unsigned char> rgba;
                unsigned w = 0, h = 0;
                if (LoadPNGTexture(kdPath, rgba, w, h))
                {
                    gpuMtl.texKd = CreateTexture2D(w, h, rgba.data());
                    gpuMtl.hasKd = (gpuMtl.texKd != 0);
                    std::cout << "Material " << mi << " map_Kd: " << kdPath << "\n";
                }
            }

            // map_Ks
            std::string ksPath = ResolveTexPath(argv[1], mtl.map_Ks.data);
            if (!ksPath.empty())
            {
                std::vector<unsigned char> rgba;
                unsigned w = 0, h = 0;
                if (LoadPNGTexture(ksPath, rgba, w, h))
                {
                    gpuMtl.texKs = CreateTexture2D(w, h, rgba.data());
                    gpuMtl.hasKs = (gpuMtl.texKs != 0);
                    std::cout << "Material " << mi << " map_Ks: " << ksPath << "\n";
                }
            }

            gpuMtls[mi] = gpuMtl;
        }
    }
    else    // No materials in OBJ/MTL
    {
        gpuMtls.resize(1);
    }

    GLuint vao = 0, vbo = 0, nbo = 0, tbo = 0;
    glCreateVertexArrays(1, &vao);
    glCreateBuffers(1, &vbo);
    glCreateBuffers(1, &nbo);
    glCreateBuffers(1, &tbo);

    glNamedBufferData(vbo, (GLsizeiptr)(positions.size() * sizeof(float)), positions.data(), GL_STATIC_DRAW);
    glNamedBufferData(nbo, (GLsizeiptr)(normals.size() * sizeof(float)), normals.data(), GL_STATIC_DRAW);
    glNamedBufferData(tbo, (GLsizeiptr)(uvs.size() * sizeof(float)), uvs.data(), GL_STATIC_DRAW);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 3 * sizeof(float));
    // Normal buffer at binding=1
    glVertexArrayVertexBuffer(vao, 1, nbo, 0, 3 * sizeof(float));
    glVertexArrayVertexBuffer(vao, 2, tbo, 0, 2 * sizeof(float));

    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0);

    glEnableVertexArrayAttrib(vao, 1);
    glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 1, 1);

    glEnableVertexArrayAttrib(vao, 2);
    glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 2, 2);

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window))
    {
        /*// Automatically animate the background color
        const float t = static_cast<float>(glfwGetTime());      // Get seconds
        const float r = 0.5f + 0.5f * std::sin(t * 1.0f);
        const float g = 0.5f + 0.5f * std::sin(t * 1.3f + 2.0f);
        const float b = 0.5f + 0.5f * std::sin(t * 1.7f + 4.0f);

        //glClearColor(1.0f, 0.0f, 0.0f, 1.0f);       // Set background color to red(1, 0, 0)
        glClearColor(r, g, b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glfwSwapBuffers(window);
        glfwPollEvents();*/

        if (shader.reloadShaders)
        {
            shader.reloadShaders = false;
            BuildShaders(shader);
            BuildSkyboxShader(skyboxShader);
            BuildReflectShader(reflectShader);
        }

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        static int lastRTW = 0, lastRTH = 0;
        if (fbW != lastRTW || fbH != lastRTH)
        {
            lastRTW = fbW;
            lastRTH = fbH;
            renderTex.Resize(4, (GLsizei)fbW, (GLsizei)fbH, cy::GL::TYPE_UBYTE);
            SetupRTTextureFiltering(renderTex.GetTextureID());
        }

        // Matrices
        cy::Matrix4f P = MakeProjection(fbW, fbH, g_usePerspective, g_orthoScale);
        cy::Matrix4f V = MakeView(g_yaw, g_pitch, g_dist);

        // Camera world position
        cy::Matrix4f Vinv = V.GetInverse();
        cy::Vec4f camPos4 = Vinv * cy::Vec4f(0, 0, 0, 1);
        cy::Vec3f camPosW(camPos4.x, camPos4.y, camPos4.z);

        // Light world position
        cy::Vec3f lightPosW; {
            float ly = sinf(g_lightPitch) * g_lightRadius;
            float lxz = cosf(g_lightPitch) * g_lightRadius;
            float lx = sinf(g_lightYaw) * lxz;
            float lz = cosf(g_lightYaw) * lxz;
            lightPosW = cy::Vec3f(lx, ly, lz);
        }

        // Model matrix
        cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
        cy::Matrix4f Tup = cy::Matrix4f::Translation(cy::Vec3f(0.0f, 0.5f, 0.0f));      // Make teapot above the plane
        cy::Matrix4f S;
        S.SetScale(g_objScale);
        cy::Matrix4f M = Tup * S * Tcenter;

        // Mirror Matrix
        cy::Matrix4f MirrorY;
        MirrorY.SetIdentity();
        MirrorY(1, 1) = -1.0f;     // Mirror across XZ plane (Y=0)

        // Pass 1: render teapot (mirrored) -> render texture
        renderTex.Bind();
        //glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbW, fbH);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_CULL_FACE);

        cy::Matrix4f Vref = MakeView(g_yaw, -g_pitch, g_dist);

        shader.prog.Bind();
        shader.prog.SetUniformMatrix4("uM", M.cell);
        shader.prog.SetUniformMatrix4("uV", Vref.cell);
        shader.prog.SetUniformMatrix4("uP", P.cell);
        shader.prog.SetUniform("uCamPosW", camPosW.x, camPosW.y, camPosW.z);
        shader.prog.SetUniform("uLightPosW", lightPosW.x, lightPosW.y, lightPosW.z);
        shader.prog.SetUniform("uEnvMap", 2);
        glBindTextureUnit(2, cubemapTex);

        shader.prog.SetUniform("uReflectStrength", 0.2f);
        shader.prog.SetUniform("uRefractStrength", 0.0f);
        shader.prog.SetUniform("uVisMode", 0);
        shader.prog.SetUniform("uDiffuseTex", 0);
        shader.prog.SetUniform("uSpecularTex", 1);

        glBindVertexArray(vao);

        // Multiuple materials
        if (mesh.NM() > 0)
        {
            for (unsigned int mi = 0; mi < mesh.NM(); ++mi)
            {
                int firstFace = mesh.GetMaterialFirstFace((int)mi);
                int faceCount = mesh.GetMaterialFaceCount((int)mi);
                if (faceCount <= 0)
                    continue;

                const GPUMaterial& m = gpuMtls[mi];

                shader.prog.SetUniform("uKa", m.Ka.x, m.Ka.y, m.Ka.z);
                shader.prog.SetUniform("uKd", m.Kd.x, m.Kd.y, m.Kd.z);
                shader.prog.SetUniform("uKs", m.Ks.x, m.Ks.y, m.Ks.z);
                shader.prog.SetUniform("uTf", m.Tf.x, m.Tf.y, m.Tf.z);
                shader.prog.SetUniform("uNs", m.Ns);
                shader.prog.SetUniform("uNi", m.Ni);
                shader.prog.SetUniform("uIllum", m.illum);

                glBindTextureUnit(0, m.texKd);
                glBindTextureUnit(1, m.texKs);
                shader.prog.SetUniform("uHasDiffuseTex", m.hasKd);
                shader.prog.SetUniform("uHasSpecularTex", m.hasKs);

                glDrawArrays(GL_TRIANGLES, firstFace * 3, faceCount * 3);
            }
        }
        else
        {
            const GPUMaterial& m = gpuMtls[0];
            shader.prog.SetUniform("uKa", m.Ka.x, m.Ka.y, m.Ka.z);
            shader.prog.SetUniform("uKd", m.Kd.x, m.Kd.y, m.Kd.z);
            shader.prog.SetUniform("uKs", m.Ks.x, m.Ks.y, m.Ks.z);
            shader.prog.SetUniform("uTf", m.Tf.x, m.Tf.y, m.Tf.z);
            shader.prog.SetUniform("uNs", m.Ns);
            shader.prog.SetUniform("uNi", m.Ni);
            shader.prog.SetUniform("uIllum", m.illum);

            glBindTextureUnit(0, 0);
            glBindTextureUnit(1, 0);
            shader.prog.SetUniform("uHasDiffuseTex", false);
            shader.prog.SetUniform("uHasSpecularTex", false);

            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh.NF() * 3));
        }

        // Pass 2: Render scene (skybox + object)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbW, fbH);
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Draw skybox
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);

        cy::Matrix4f VnoT = V;
        VnoT(0, 3) = 0;
        VnoT(1, 3) = 0;
        VnoT(2, 3) = 0;

        skyboxShader.prog.Bind();
        skyboxShader.prog.SetUniformMatrix4("uProj", P.cell);
        skyboxShader.prog.SetUniformMatrix4("uView", VnoT.cell);
        skyboxShader.prog.SetUniform("uEnv", 0);
        glBindTextureUnit(0, cubemapTex);
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        // Object
        shader.prog.Bind();
        shader.prog.SetUniformMatrix4("uM", M.cell);
        shader.prog.SetUniformMatrix4("uV", V.cell);
        shader.prog.SetUniformMatrix4("uP", P.cell);
        shader.prog.SetUniform("uCamPosW", camPosW.x, camPosW.y, camPosW.z);
        shader.prog.SetUniform("uLightPosW", lightPosW.x, lightPosW.y, lightPosW.z);
        shader.prog.SetUniform("uEnvMap", 2);
        glBindTextureUnit(2, cubemapTex);
        shader.prog.SetUniform("uReflectStrength", 1.0f);
        shader.prog.SetUniform("uRefractStrength", 0.0f);
        shader.prog.SetUniform("uVisMode", g_visMode);

        // Texture units
        shader.prog.SetUniform("uDiffuseTex", 0);
        shader.prog.SetUniform("uSpecularTex", 1);
        glBindVertexArray(vao);

        // Support multiple materials
        if (mesh.NM() > 0)
        {
            for (unsigned int mi = 0; mi < mesh.NM(); ++mi)
            {
                int firstFace = mesh.GetMaterialFirstFace((int)mi);
                int faceCount = mesh.GetMaterialFaceCount((int)mi);
                if (faceCount <= 0)
                    continue;

                const GPUMaterial& m = gpuMtls[mi];

                // Set material properties
                shader.prog.SetUniform("uKa", m.Ka.x, m.Ka.y, m.Ka.z);
                shader.prog.SetUniform("uKd", m.Kd.x, m.Kd.y, m.Kd.z);
                shader.prog.SetUniform("uKs", m.Ks.x, m.Ks.y, m.Ks.z);
                shader.prog.SetUniform("uTf", m.Tf.x, m.Tf.y, m.Tf.z);
                shader.prog.SetUniform("uNs", m.Ns);
                shader.prog.SetUniform("uNi", m.Ni);
                shader.prog.SetUniform("uIllum", m.illum);

                // Binding Texture (unit0 = kd, unit1 = ks)
                glBindTextureUnit(0, m.texKd);
                glBindTextureUnit(1, m.texKs);
                shader.prog.SetUniform("uHasDiffuseTex", m.hasKd);
                shader.prog.SetUniform("uHasSpecularTex", m.hasKs);

                glDrawArrays(GL_TRIANGLES, firstFace * 3, faceCount * 3);
            }
        }
        else    // If no material
        {
            const GPUMaterial& m = gpuMtls[0];
            shader.prog.SetUniform("uKa", m.Ka.x, m.Ka.y, m.Ka.z);
            shader.prog.SetUniform("uKd", m.Kd.x, m.Kd.y, m.Kd.z);
            shader.prog.SetUniform("uKs", m.Ks.x, m.Ks.y, m.Ks.z);
            shader.prog.SetUniform("uTf", m.Tf.x, m.Tf.y, m.Tf.z);
            shader.prog.SetUniform("uNs", m.Ns);
            shader.prog.SetUniform("uNi", m.Ni);
            shader.prog.SetUniform("uIllum", m.illum);

            glBindTextureUnit(0, 0);
            glBindTextureUnit(1, 0);
            shader.prog.SetUniform("uHasDiffuseTex", false);
            shader.prog.SetUniform("uHasSpecularTex", false);

            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(mesh.NF() * 3));
        }

        // Pass 3: Draw Plane with reflection
        // Plane
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        glBindVertexArray(reflPlaneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDisable(GL_POLYGON_OFFSET_FILL);

        cy::Matrix4f Mplane;
        Mplane.SetIdentity();
        shader.prog.Bind();
        shader.prog.SetUniformMatrix4("uM", Mplane.cell);
        shader.prog.SetUniformMatrix4("uV", V.cell);
        shader.prog.SetUniformMatrix4("uP", P.cell);
        shader.prog.SetUniform("uCamPosW", camPosW.x, camPosW.y, camPosW.z);
        shader.prog.SetUniform("uLightPosW", lightPosW.x, lightPosW.y, lightPosW.z);
        shader.prog.SetUniform("uEnvMap", 2);
        glBindTextureUnit(2, cubemapTex);

        shader.prog.SetUniform("uReflectStrength", 1.0f);
        shader.prog.SetUniform("uRefractStrength", 0.0f);
        shader.prog.SetUniform("uHasDiffuseTex", false);
        shader.prog.SetUniform("uHasSpecularTex", false);
        shader.prog.SetUniform("uKa", 0.00f, 0.00f, 0.00f);
        shader.prog.SetUniform("uKd", 0.04f, 0.04f, 0.04f);
        shader.prog.SetUniform("uKs", 0.6f, 0.6f, 0.6f);
        shader.prog.SetUniform("uNs", 512.0f);
        shader.prog.SetUniform("uReflectStrength", 0.8f);

        glBindVertexArray(reflPlaneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Reflection
        glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW);

        // Draw reflective plane
        reflectShader.prog.Bind();
        reflectShader.prog.SetUniformMatrix4("uM", Mplane.cell);
        reflectShader.prog.SetUniformMatrix4("uV", V.cell);
        reflectShader.prog.SetUniformMatrix4("uP", P.cell);
        reflectShader.prog.SetUniformMatrix4("uVref", Vref.cell);
        reflectShader.prog.SetUniformMatrix4("uPref", P.cell);

        reflectShader.prog.SetUniform("uCamPosW", camPosW.x, camPosW.y, camPosW.z);
        //cy::Vec3f fadeCenterW(0.0f, 0.0f, 0.0f);
        //reflectShader.prog.SetUniform("uFadeCenterW", fadeCenterW.x, fadeCenterW.y, fadeCenterW.z);
        reflectShader.prog.SetUniform("uFadeRadius", 2.0f);
        reflectShader.prog.SetUniform("uReflectOpacity", 0.6f);
        reflectShader.prog.SetUniform("uReflectionTex", 0);

        glBindTextureUnit(0, renderTex.GetTextureID());
        glBindVertexArray(reflPlaneVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glDepthMask(GL_TRUE);
		glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up materials
    for (auto& m : gpuMtls)
    {
        if (m.texKd) glDeleteTextures(1, &m.texKd);
        if (m.texKs) glDeleteTextures(1, &m.texKs);
    }
    glDeleteBuffers(1, &planeVBO);
    glDeleteVertexArrays(1, &planeVAO);
    glDeleteBuffers(1, &tbo);
    glDeleteBuffers(1, &nbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}