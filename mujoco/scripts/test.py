import mujoco
import numpy as np
import glfw
import time

# Path to your MuJoCo XML model
xml_path = "../models/xml/2D_SLIP.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set up the GLFW window
def init_glfw():
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
    glfw.make_context_current(window)
    return window

window = init_glfw()

# Create a camera to render the scene
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()

# Set up scene and context for rendering
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Frame skipping parameters
frame_skip = 10  # Only render every so number of steps
step_counter = 0

# Set the initial configuration of the robot
qpos = np.array([0, 1.0, 0, 0])
qvel = np.array([0, 0, 0, 0])
data.qpos[:] = qpos
data.qvel[:] = qvel

# Function to compute the center of mass (CoM) of the model
def compute_com():
    total_mass = 0
    com = np.zeros(3)
    
    # Loop through all bodies in the model
    for i in range(model.nbody):
        # Get the mass of the body
        mass = model.body_mass[i]
        
        # Get the global position of the body's CoM
        xpos = data.xipos[i]
        
        # Accumulate the weighted CoM
        com += mass * xpos
        total_mass += mass
    
    # Divide by the total mass to get the overall CoM
    com /= total_mass
    return com

# Function to update the camera position to track the center of mass (CoM)
def update_camera_to_com():
    # Calculate the overall center of mass
    com_pos = compute_com()

    # Set camera parameters to track the CoM
    cam.lookat[:] = com_pos[:3]  # Make the camera look at the CoM
    cam.distance = 3.0  # Distance from the CoM (adjust as needed)
    cam.elevation = -10  # Camera elevation angle (adjust as needed)
    cam.azimuth = 90  # Camera azimuth angle (adjust as needed)

# Simulation loop
def run_simulation():
    global step_counter
    while not glfw.window_should_close(window):
        # Step the simulation
        mujoco.mj_step(model, data)

        # Update the camera to track the center of mass
        update_camera_to_com()

        # Do control (optional)
        # data.ctrl[0] = 0.2 * np.sin(data.time)

        step_counter += 1
        if step_counter % frame_skip == 0:
            # Get framebuffer size and create viewport for rendering
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            # Update scene for rendering
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)

            # Swap buffers and poll events for window
            glfw.swap_buffers(window)

        # Poll for window events like keypress or close
        glfw.poll_events()

# Main execution
try:
    run_simulation()
finally:
    glfw.terminate()
