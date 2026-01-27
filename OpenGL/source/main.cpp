#include <iostream>
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

// Camera parameter
static float g_yaw = 0.0f;
static float g_pitch = 0.0f;
static float g_dist = 2.0f;

// Object
static cy::Vec3f g_objCenter(0.0f, 0.0f, 0.0f);
static float g_objScale = 1.0f;

// Perspective or Orthographic
static bool g_usePerspective = true;
static float g_orthoScale = 1.5f;
// ------------------------------


// Shader
struct  Shader
{
    cy::GLSLProgram prog;
    bool reloadShaders = false;

    // Read shader from glsl
    std::string vsPath = "assets/shaders/vertex.glsl";
    std::string fsPath = "assets/shaders/fragment.glsl";

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
// ------------------------------

// Help tools
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

static cy::Matrix4f MakeMVP(int fbW, int fbH)
{
    float aspect = (fbH > 0) ? (float)fbW / (float)fbH : 1.0f;

    // Projection
    cy::Matrix4f P;
    if (g_usePerspective)
    {
        P = cy::Matrix4f::Perspective(DegToRad(60.0f), aspect, 0.1f, 100.0f);       // Perspective
    }
    else
    {
        float halfH = g_orthoScale;
        float halfW = g_orthoScale * aspect;
        P = MakeOrthographic(-halfW, halfW, -halfH, halfH, 0.1f, 200.0f);
    }

    cy::Matrix4f R_yaw = cy::Matrix4f::RotationY(g_yaw);
    cy::Matrix4f R_pitch = cy::Matrix4f::RotationX(g_pitch);
    cy::Matrix4f Tcam = cy::Matrix4f::Translation(cy::Vec3f(0.0f, 0.0f, -g_dist));
    // V = T * R_pitch * R_yaw
    cy::Matrix4f V = Tcam * R_pitch * R_yaw;

    // Center + auto Scale
    cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
    float scale = g_objScale;
    if (g_usePerspective)
    {
        float d = max(g_dist, 0.001f);
        scale *= (1.0f / d);
    }
    cy::Matrix4f S;
    S.SetScale(g_objScale);
    cy::Matrix4f M = S * Tcenter;

    return P * V * M;
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

static void cursor_pos_callback(GLFWwindow* /*window*/, double x, double y)
{
    double dx = x - g_lastX;
    double dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    const float rotSpeed = 0.005f;
    const float zoomSpeed = 0.02f;

    if (g_leftDown)
    {
        g_yaw += (float)dx * rotSpeed;
        g_pitch += (float)dy * rotSpeed;
        const float limit = 1.55f;      // approximate 89 degrees
        if (g_pitch > limit)
            g_pitch = limit;
        if (g_pitch < -limit)
            g_pitch = -limit;
    }
    if (g_rightDown)
    {
        g_dist += (float)dy * zoomSpeed;
        if (g_dist < 0.5)
            g_dist = 0.5;
        if (g_dist > 5.0)
            g_dist = 5.0f;
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

    // Get vertices & bounding box
    std::vector<float> positions;
    positions.reserve(mesh.NV() * 3);
    cy::Vec3f bbMin(1e30f, 1e30f, 1e30f);
    cy::Vec3f bbMax(-1e30f, -1e30f, -1e30f);
    for (unsigned int i = 0; i < mesh.NV(); ++i)
    {
        cy::Vec3f p = mesh.V((int)i);
        positions.push_back(p.x);
        positions.push_back(p.y);
        positions.push_back(p.z);
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
    const GLsizei vertexCount = (GLsizei)mesh.NV();
    
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
    GLFWwindow* window = glfwCreateWindow(initW, initH, "Project 2 - Transformations", nullptr, nullptr);
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
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    GLuint vao = 0, vbo = 0;
    glCreateVertexArrays(1, &vao);
    glCreateBuffers(1, &vbo);

    glNamedBufferData(
        vbo,
        (GLsizeiptr)(positions.size() * sizeof(float)),
        positions.data(),
        GL_STATIC_DRAW
    );

    glVertexArrayVertexBuffer(
        vao,        // VAO
        0,          // binding index
        vbo,        // buffer
        0,          // offset
        3 * sizeof(float) // stride
    );

    glEnableVertexArrayAttrib(vao, 0);
    glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0);
    glEnable(GL_PROGRAM_POINT_SIZE);

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
        }

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        cy::Matrix4f MVP = MakeMVP(fbW, fbH);

        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.prog.Bind();
        shader.prog.SetUniformMatrix4("uMVP", MVP.cell);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, vertexCount);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}