import mujoco
import numpy as np
import glfw

# Path to your MuJoCo XML model
xml_path = "../models/xml/3D_SLIP.xml"

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
qpos = np.array([0, 0, 1.5, # px, py, pz
                 0.0,  0.0,     # leg roll, leg pitch
                 0])        # prismatic position
qvel = np.array([1.0, 1.0, 0,   # vx, vy, vz
                 0.0,  0.0,     # leg roll, leg pitch
                 0])        # prismatic position
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

# Function to print contact information
def get_contact_info():
    
    if data.ncon == 0:
        return None
    else:
        for i in range(data.ncon):  # Loop through all contacts
            contact = data.contact[i]
            body_a = contact.geom1  # First body involved in the contact
            body_b = contact.geom2  # Second body involved in the contact
            contact_point_pos = contact.pos  # Contact point position
            
            # body_a_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, body_a)
            # body_b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, body_b)
            
            # print(f"Contact {i + 1}:")
            # print(f"  Body A: {body_a_name}, Body B: {body_b_name}")
            # print(f"  Contact Point: {contact_point_pos}")

            return contact_point_pos

# Function to compute the hip torque
def compute_hip_torque(qpos, qvel):

    # get the current contact point if any
    contact_point = get_contact_info()

    # return 0 torque when you are in contact
    if contact_point is not None:
        return 0.0, 0.0

    # otherwise, do the Raibert controller in the air
    else:
        # get the current leg state
        pos_leg = [qpos[3], qpos[4]]  # roll, pitch
        vel_leg = [qvel[3], qvel[4]]  # roll, pitch

        # get the current state in the air
        pos_com = [qpos[0], qpos[1]]  # px, py
        vel_com = [qvel[0], qvel[1]]  # vx, vy

        # desired state
        pos_com_des = [0.0, 0.0]  # desired px, py
        vel_com_des = [0.0, 0.0]  # desired vx, vy

        # calculate the desired leg angles
        kp_raibert = 0.03
        kd_raibert = 0.15
        roll_des = kp_raibert * (pos_com[1] - pos_com_des[1]) + kd_raibert * (vel_com[1] - vel_com_des[1])
        pitch_des = kp_raibert * (pos_com[0] - pos_com_des[0]) + kd_raibert * (vel_com[0] - vel_com_des[0])
        
        # compute torque to achieve the desired leg angle
        kp_tau = 0.1
        kd_tau = 0.01
        tau_roll = kp_tau * (roll_des - pos_leg[0]) + kd_tau * (0.0 - vel_leg[0])
        tau_pitch = kp_tau * (pitch_des - pos_leg[1]) + kd_tau * (0.0 - vel_leg[1])

        return tau_roll, tau_pitch

# Simulation loop
def run_simulation():
    global step_counter
    while not glfw.window_should_close(window):

        # Update the camera to track the center of mass
        update_camera_to_com()

        # Do simple Raibert Controller
        data.ctrl[0], data.ctrl[1] = compute_hip_torque(data.qpos, data.qvel)
        
        # Step the simulation
        mujoco.mj_step(model, data)

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
