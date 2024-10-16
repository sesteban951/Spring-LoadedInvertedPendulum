#include "mujoco.h"
#include "GLFW/glfw3.h"
#include <iostream>
#include <thread>
#include <chrono>

// Path to your MuJoCo model file (replace with the correct path)
const char* model_path = "path_to_your_model.xml";

// Initialize MuJoCo model and simulation
mjModel* m = nullptr;
mjData* d = nullptr;

// GLFW error callback
void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error: " << description << std::endl;
}

// Main function
int main() {
    // MuJoCo initialization
    // mj_activate("mjkey.txt");

    // Load the model
    m = mj_loadXML(model_path, nullptr, nullptr, 0);
    if (!m) {
        std::cerr << "Could not load model" << std::endl;
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

    // Loop to simulate dropping the robot
    while (!glfwWindowShouldClose(window)) {
        // Step the simulation
        mj_step(m, d);

        // Clear buffer
        glClear(GL_COLOR_BUFFER_BIT);

        // Simulate in real-time (timestep of 0.01 seconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Swap buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Clean up MuJoCo
    mj_deleteData(d);
    mj_deleteModel(m);
    // mj_deactivate();

    // Terminate GLFW
    glfwTerminate();

    return 0;
}
