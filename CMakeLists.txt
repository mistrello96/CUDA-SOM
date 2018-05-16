# Minimum version for CUDA
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# CUDA required
project(CUDA-SOM LANGUAGES C CXX CUDA)

# Root dir
set(PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})

# Binaries directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT_DIR})

# Update sources directory
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
)

# Gengetopt
add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/cmdline.c ${PROJECT_SOURCE_DIR}/cmdline.h
    DEPENDS ${PROJECT_SOURCE_DIR}/lineparser.ggo
    COMMAND gengetopt -i ${PROJECT_SOURCE_DIR}/lineparser.ggo --output-dir=${PROJECT_SOURCE_DIR}
)

# Source files
add_executable(
    CUDA-SOM
    ${PROJECT_SOURCE_DIR}/SOM.cu
	${PROJECT_SOURCE_DIR}/cmdline.c
	${PROJECT_SOURCE_DIR}/cmdline.h
	${PROJECT_SOURCE_DIR}/distance_kernels.cu
	${PROJECT_SOURCE_DIR}/distance_kernels.h
	${PROJECT_SOURCE_DIR}/update_kernels.cu
	${PROJECT_SOURCE_DIR}/update_kernels.h
	${PROJECT_SOURCE_DIR}/utility_functions.cu
	${PROJECT_SOURCE_DIR}/utility_functions.h
)

# C++11
target_compile_features(CUDA-SOM PUBLIC cxx_std_11)

# Enable separable compilation
set_target_properties(CUDA-SOM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
