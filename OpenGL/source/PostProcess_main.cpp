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

// Object
static cy::Vec3f g_objCenter(0.0f, 0.0f, 0.0f);
static float g_objScale = 1.0f;

// Perspective or Orthographic
static bool g_usePerspective = true;
static float g_orthoScale = 1.5f;
static bool g_showDepth = false;

// Light properties
static float g_lightYaw = 0.7f;
static float g_lightPitch = 0.4f;
static float g_lightRadius = 3.0f;
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
            vec3 specular = spec * uLightColor * 0.35;

            vec3 color = ambient + diffuse + specular;
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

struct PostProcessShader
{
    cy::GLSLProgram prog;
    bool built = false;

    const char* vs = R"GLSL(
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

    const char* fs = R"GLSL(
        #version 460 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform sampler2D uSceneColor;
        uniform sampler2D uSceneDepth;
        uniform int uShowDepth;

        float LinearizeDepth(float z, float nearPlane, float farPlane)
        {
            float ndc = z * 2.0 - 1.0;
            return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - ndc * (farPlane - nearPlane));
        }

        void main()
        {
            if (uShowDepth == 1)
            {
                float z = texture(uSceneDepth, vUV).r;
                float linearDepth = LinearizeDepth(z, 0.1, 100.0);

                // for visualization only
                float normalizedDepth = clamp(linearDepth / 8.0, 0.0, 1.0);
                FragColor = vec4(vec3(normalizedDepth), 1.0);
                return;
            }

            vec3 color = texture(uSceneColor, vUV).rgb;
            FragColor = vec4(color, 1.0);
        }
    )GLSL";
};

static bool BuildLitShader(LitShader& shader)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << "Failed to build lit shader." << std::endl;
		return false;
    }
	shader.built = true;
	return true;
}

static bool BuildPostProcessShader(PostProcessShader& shader)
{
    if (!shader.prog.Build<false, false>(shader.vs, shader.fs))
    {
        std::cerr << "Failed to build post-process shader." << std::endl;
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
    return Tcam * R_pitch * R_yaw;
}

static cy::Vec3f ComputeLightPosWorld()
{
	float cp = cosf(g_lightPitch);
	float sp = sinf(g_lightPitch);
	float cy = cosf(g_lightYaw);
	float sy = sinf(g_lightYaw);
	return cy::Vec3f(g_lightRadius * cp * sy, g_lightRadius * sp, g_lightRadius * cp * cy);
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

static void DestroySceneRenderTarget(SceneRenderTarget& renderTarget)
{
    if (renderTarget.colorTex)
		glDeleteTextures(1, &renderTarget.colorTex);
    if (renderTarget.depthTex)
        glDeleteTextures(1, &renderTarget.depthTex);
    if (renderTarget.fbo)
        glDeleteBuffers(1, &renderTarget.fbo);
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
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
    {
        g_showDepth = !g_showDepth;
        std::cout << "[D] Show Depth = " << (g_showDepth ? "ON" : "OFF") << std::endl;
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
    GLFWwindow* window = glfwCreateWindow(initW, initH, "OpenGL Post Process Base (Color + Depth)", nullptr, nullptr);
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
    std::cout << "  LMB drag        : rotate camera\n";
    std::cout << "  RMB drag        : zoom camera\n";
    std::cout << "  Ctrl + LMB drag : rotate light\n";
    std::cout << "  D               : toggle color/depth preview\n";
    std::cout << "  P               : perspective / orthographic\n";

    LitShader litShader;
    PostProcessShader postShader;
    if (!BuildLitShader(litShader) || !BuildPostProcessShader(postShader))
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

    SceneRenderTarget sceneRT;
    if (!CreateSceneRenderTarget(sceneRT, fbW, fbH))
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
            if (!CreateSceneRenderTarget(sceneRT, fbW, fbH))
            {
                std::cerr << "ERROR: failed to resize scene render targets\n";
                break;
            }
        }

        cy::Matrix4f P = MakeProjection(fbW, fbH, g_usePerspective, g_orthoScale);
        cy::Matrix4f V = MakeView(g_yaw, g_pitch, g_dist);

        cy::Matrix4f Vinv = V.GetInverse();
        cy::Vec4f camPos4 = Vinv * cy::Vec4f(0, 0, 0, 1);
        cy::Vec3f camPosW(camPos4.x, camPos4.y, camPos4.z);

        cy::Vec3f lightPosW = ComputeLightPosWorld();

        cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
        cy::Matrix4f S;
        S.SetScale(g_objScale);
        cy::Matrix4f M = S * Tcenter;

        // Pass1
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

		// Pass2
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbW, fbH);
        glDisable(GL_DEPTH_TEST);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);

        postShader.prog.Bind();
        postShader.prog.SetUniform("uSceneColor", 0);
        postShader.prog.SetUniform("uSceneDepth", 1);
        postShader.prog.SetUniform("uShowDepth", g_showDepth ? 1 : 0);

        glBindTextureUnit(0, sceneRT.colorTex);
        glBindTextureUnit(1, sceneRT.depthTex);

        glBindVertexArray(fsQuadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

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