# cmake version to use
cmake_minimum_required(VERSION 3.12)

# project name
project(mdrom)

# some c++ flags
set(CMAKE_CXX_STANDARD 17)  # Need c++ 17 for inline variables
set(CMAKE_CXX_STANDARD_REQUIRED True)

# optimize for speed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -fopenmp -march=native -mtune=native -ffast-math -funroll-loops -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")  # Disable debugging in release builds

# where to look for packages and libraries
SET(yaml-cpp_DIR /home/sergio/repos/yaml-cpp_install/lib/cmake/yaml-cpp/)

# include headers
include_directories(inc)

# find the relevant packages
find_package(Eigen3)
find_package(yaml-cpp REQUIRED)

# add the exectuable files
add_executable(a src/main.cpp src/dynamics.cpp src/control.cpp)

# link your executables
target_link_libraries(a Eigen3::Eigen)
target_link_libraries(a yaml-cpp::yaml-cpp)
