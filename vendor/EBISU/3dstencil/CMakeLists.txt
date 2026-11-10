cmake_minimum_required(VERSION 3.10)
project(EBISU_3DSTENCIL VERSION 1.0)
cmake_policy(SET CMP0077 NEW)

set(
    EBISU_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin"
    CACHE PATH
    "Runtime output directory for EBISU executables."
)
set(
    EBISU_CUDA_ARCHITECTURES
    "80;90"
    CACHE STRING
    "CUDA architectures used to build EBISU targets."
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EBISU_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_CUDA_ARCHITECTURES "${EBISU_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --static-global-template-stub=false")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)


#start
add_subdirectory(3d7pt)
add_subdirectory(3d13pt)

#box
add_subdirectory(3d17pt)
add_subdirectory(3d27pt)
add_subdirectory(poisson)
