cmake_minimum_required(VERSION 3.10)

# Project name
project("test")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)  # Need c++ 17 for inline variables
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

# Add the glfw3 package
find_package(glfw3 REQUIRED)

# Find OpenGL package
find_package(OpenGL REQUIRED)

# Find GLEW package
find_package(GLEW REQUIRED)

# Include directories for MuJoCo
include_directories($ENV{THOR_MUJOCO_DIR}/include)
include_directories($ENV{THOR_MUJOCO_DIR}/include/mujoco)
find_library(MUJOCO_LIB mujoco HINTS $ENV{THOR_MUJOCO_DIR}/lib REQUIRED)

# Add the executable
add_executable(test src/test.cpp)

# Link the executable to GLFW, OpenGL, GLEW, and MuJoCo
target_link_libraries(test glfw ${MUJOCO_LIB} ${OPENGL_LIBRARIES} GLEW::GLEW)
