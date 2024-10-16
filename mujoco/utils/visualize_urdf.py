#!/usr/bin/env python3

##
#
# Use Drake's ModelVisualizer to show the model, collision geometries, inertias,
# etc. in MeshCat.
#
##

from pydrake.all import ModelVisualizer, MultibodyPlant, Parser, StartMeshcat 
from pydrake.all import *

# load model
urdf_path = "../models/urdf/3D_SLIP.urdf"
plant = MultibodyPlant(0)
Parser(plant).AddModels(urdf_path)
plant.Finalize()

########################################################################################

# model instance index
model_instance_idx = plant.GetModelInstanceByName("3D_SLIP")

# BODIES
print("-"*50)
print("BODIES")
print("Model Bodies:")
print("Model has ({}) bodies:".format(plant.num_bodies()))
body_idx = plant.GetBodyIndices(model_instance_idx)
for i in body_idx:
    print("  {}: {}".format(i, plant.get_body(i).name()))

# JOINTS
print("-"*50)
print("JOINTS")
print("Model has ({}) joints:".format(plant.num_joints()))
joint_idx = plant.GetJointIndices(model_instance_idx)
for i in joint_idx:
    print("  {}: {}".format(i, plant.get_joint(i).name()))

# FRAMES
print("-"*50)
print("FRAMES")
print("Model has ({}) frames:".format(plant.num_frames()))
frame_idx = plant.GetFrameIndices(model_instance_idx)
for i in frame_idx:
    print("  {}: {}".format(i, plant.get_frame(i).name()))

# ACTUATORS
print("-"*50)
print("ACTUATORS")
print("Model has ({}) actuators:".format(plant.num_actuators()))
actuator_idx = plant.GetJointActuatorIndices(model_instance_idx)
for i in actuator_idx:
    print("  {}: {}".format(i, plant.get_joint_actuator(i).name()))

# CONFIGURATION
print("-"*50)
print("CONFIGURATION")
print("Model has ({}) configuration variables.".format(plant.num_positions()))
pos_names = plant.GetPositionNames(model_instance_idx)
for i in range(plant.num_positions()):
    print("  {}: {}".format(i, pos_names[i]))

# VELOCITY
print("-"*50)
print("VELOCITY")
print("Model has ({}) velocity variables.".format(plant.num_velocities()))
vel_names = plant.GetVelocityNames(model_instance_idx)
for i in range(plant.num_velocities()):
    print("  {}: {}".format(i, vel_names[i]))

########################################################################################

# Print total mass of the robot
plant_context = plant.CreateDefaultContext()
mass = plant.CalcTotalMass(plant_context)
print("\nTotal mass of the robot: {} kg\n".format(mass))

# VISUALIZER
# drake > illustration > harpy
meshcat = StartMeshcat()
visualizer = ModelVisualizer(meshcat=meshcat, 
                             visualize_frames=True, triad_length=0.15, triad_radius=0.005)
visualizer.parser().AddModels(urdf_path)

visualizer.Run()
