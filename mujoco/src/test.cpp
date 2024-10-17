#include "mujoco.h"
#include "GLFW/glfw3.h"
#include <iostream>
#include <thread>
#include <chrono>

// Path to your MuJoCo model file (replace with the correct path)
// const char* model_path = "../models/xml/achilles.xml";
const char* model_path = "../models/xml/3D_SLIP.xml";

// Initialize MuJoCo model and simulation
mjModel* m = nullptr;
mjData* d = nullptr;
mjvScene scn;             // MuJoCo visualization scene
mjrContext con;            // MuJoCo rendering context
mjvOption opt;             // Visualization options

// GLFW error callback
void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
}

// Main function
int main() {
    // MuJoCo initialization
    // mj_activate("mjkey.txt");

    // Error buffer
    char error[1000] = "Could not load model";

    // Load the model
    m = mj_loadXML(model_path, nullptr, error, 1000);
    if (!m) {
        std::cerr << "Error loading model: " << error << std::endl;
        return 1;
    }

    // Create data
    d = mj_makeData(m);

    // Set initial conditions (optional)
    d->qpos[2] = 2.0;  // Set height of the robot to 2 meters (z-coordinate)

    // Init GLFW
    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW" << std::endl;
        return 1;
    }

    // Set GLFW error callback
    glfwSetErrorCallback(glfw_error_callback);

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "MuJoCo Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Could not create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    // Make context current
    glfwMakeContextCurrent(window);

    // Initialize visualization data structures
    mjv_defaultScene(&scn);
    mjv_makeScene(m, &scn, 1000);  // Allocate visualization objects
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, mjFONTSCALE_150);  // Initialize rendering context
    mjv_defaultOption(&opt);  // Initialize visualization options

    // Camera and scene setup (optional)
    mjvCamera cam;                     // Camera
    mjv_defaultCamera(&cam);
    cam.azimuth = 90;                  // Set an initial view angle
    cam.elevation = -30;
    cam.distance = 5.0;

    // Simulation and rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Step the simulation
        mj_step(m, d);

        // Get viewport (window size)
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        mjrRect viewport = {0, 0, width, height};

        // Update the scene for the current model state
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

        // Clear OpenGL buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render the scene in the current OpenGL context
        mjr_render(viewport, &scn, &con);

        // Swap OpenGL buffers (double buffering)
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

        // Simulate in real-time (timestep of 0.01 seconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Clean up MuJoCo
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    // mj_deactivate();

    // Terminate GLFW
    glfwTerminate();

    return 0;
}
