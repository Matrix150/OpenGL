#include <iostream>

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
struct  Shader
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

static cy::Matrix4f MakeMV()
{
    cy::Matrix4f R_yaw = cy::Matrix4f::RotationY(g_yaw);
    cy::Matrix4f R_pitch = cy::Matrix4f::RotationX(g_pitch);
    cy::Matrix4f Tcam = cy::Matrix4f::Translation(cy::Vec3f(0.0f, 0.0f, -g_dist));
    cy::Matrix4f V = Tcam * R_pitch * R_yaw;

    cy::Matrix4f Tcenter = cy::Matrix4f::Translation(-g_objCenter);
    cy::Matrix4f S;
    S.SetScale(g_objScale);
    cy::Matrix4f M = S * Tcenter;

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

    if (g_leftDown)
    {
        if (ctrlDown)       // Light rotation
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
            g_yaw += (float)dx * rotSpeed;
            g_pitch += (float)dy * rotSpeed;
            const float limit = 1.55f;      // approximate 89 degrees
            if (g_pitch > limit)
                g_pitch = limit;
            if (g_pitch < -limit)
                g_pitch = -limit;
        }
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
    GLFWwindow* window = glfwCreateWindow(initW, initH, "Project 4 - Textures", nullptr, nullptr);
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

    glNamedBufferData(
        vbo,
        (GLsizeiptr)(positions.size() * sizeof(float)),
        positions.data(),
        GL_STATIC_DRAW
    );

    glNamedBufferData(
        nbo,
        (GLsizeiptr)(normals.size() * sizeof(float)),
        normals.data(),
        GL_STATIC_DRAW
    );

    glNamedBufferData(
        tbo,
        (GLsizeiptr)(uvs.size() * sizeof(float)),
        uvs.data(),
        GL_STATIC_DRAW
    );

    glVertexArrayVertexBuffer(
        vao,        // VAO
        0,          // binding index
        vbo,        // buffer
        0,          // offset
        3 * sizeof(float) // stride
    );
    // Normal buffer at binding=1
    glVertexArrayVertexBuffer(
        vao,
        1,
        nbo,
        0,
        3 * sizeof(float)
    );
    glVertexArrayVertexBuffer(
        vao, 
        2, 
        tbo, 
        0, 
        2 * sizeof(float)
    );

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
        }

        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        cy::Matrix4f MVP = MakeMVP(fbW, fbH);
        cy::Matrix4f MV = MakeMV();

        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.prog.Bind();
        shader.prog.SetUniformMatrix4("uMVP", MVP.cell);
        shader.prog.SetUniformMatrix4("uMV", MV.cell);
        shader.prog.SetUniform("uVisMode", g_visMode);
        cy::Vec3f lightPosV = ComputeLightPosViewSpace(g_yaw, g_pitch, g_dist, g_lightYaw, g_lightPitch, g_lightRadius);
        shader.prog.SetUniform("uLightPosV", lightPosV.x, lightPosV.y, lightPosV.z);

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

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

	// Clean up materials
    for (auto& m : gpuMtls)
    {
        if (m.texKd) glDeleteTextures(1, &m.texKd);
        if (m.texKs) glDeleteTextures(1, &m.texKs);
    }
    glDeleteBuffers(1, &tbo);
    glDeleteBuffers(1, &nbo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}