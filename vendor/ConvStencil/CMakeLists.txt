cmake_minimum_required(VERSION 3.18)

project(ConvStencil VERSION 0.01 LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(
    CONV_STENCIL_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin"
    CACHE PATH
    "Runtime output directory for ConvStencil executables."
)
set(
    CONV_STENCIL_CUDA_ARCHITECTURES
    "80;90"
    CACHE STRING
    "CUDA architectures used to build ConvStencil targets."
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CONV_STENCIL_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_CUDA_ARCHITECTURES "${CONV_STENCIL_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/usr/local/cuda/lib64)

find_path(
    CUDNN_INCLUDE_DIR
    cudnn.h
    HINTS
        /usr/include
        /usr/local/include
        /usr/local/cuda/include
        /usr/local/cuda-13.0/include
)
find_library(
    CUDNN_LIBRARY
    cudnn
    HINTS
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib64
        /usr/local/cuda/lib64
        /usr/local/cuda-13.0/lib64
)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    include_directories(${CUDNN_INCLUDE_DIR})
    set(CONV_STENCIL_HAVE_CUDNN ON)
else()
    set(CONV_STENCIL_HAVE_CUDNN OFF)
    message(WARNING "cuDNN was not found. cuDNN benchmark targets will be skipped.")
endif()

function(configure_convstencil_target target)
    set(options NEEDS_CUDNN)
    cmake_parse_arguments(CONV "${options}" "" "" ${ARGN})

    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CONV_STENCIL_CUDA_ARCHITECTURES}"
        RUNTIME_OUTPUT_DIRECTORY "${CONV_STENCIL_RUNTIME_OUTPUT_DIRECTORY}")
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--use_fast_math>)

    if(CONV_NEEDS_CUDNN)
        target_link_libraries(${target} PRIVATE "${CUDNN_LIBRARY}")
    endif()
endfunction()

add_executable(
    convstencil_1d
    src/1d/main.cu
    src/1d/gpu_1r.cu
    src/1d/gpu_2r.cu
    src/1d/1d_utils.h
)
configure_convstencil_target(convstencil_1d)

if(CONV_STENCIL_HAVE_CUDNN)
    add_executable(cudnn_1d3p src/cudnn/conv_1d3p.cu)
    configure_convstencil_target(cudnn_1d3p NEEDS_CUDNN)

    add_executable(cudnn_1d5p src/cudnn/conv_1d5p.cu)
    configure_convstencil_target(cudnn_1d5p NEEDS_CUDNN)

    add_executable(cudnn_box2d49p src/cudnn/conv_box2d49p.cu)
    configure_convstencil_target(cudnn_box2d49p NEEDS_CUDNN)

    add_executable(cudnn_box2d9p src/cudnn/conv_box2d9p.cu)
    configure_convstencil_target(cudnn_box2d9p NEEDS_CUDNN)

    add_executable(cudnn_box3d27p src/cudnn/conv_box3d27p.cu)
    configure_convstencil_target(cudnn_box3d27p NEEDS_CUDNN)
endif()

add_executable(
    convstencil_2d
    src/2d/main.cu
    src/2d/gpu.cu
    src/2d/2d_utils.h
)
configure_convstencil_target(convstencil_2d)

add_executable(
    convstencil_3d
    src/3d/main.cu
    src/3d/gpu_box.cu
    src/3d/gpu_star.cu
    src/3d/3d_utils.h
)
configure_convstencil_target(convstencil_3d)
