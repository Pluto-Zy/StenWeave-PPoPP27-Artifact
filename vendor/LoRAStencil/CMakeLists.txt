cmake_minimum_required(VERSION 3.18)

project(LoRAStencil LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(
    LORA_STENCIL_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin"
    CACHE PATH
    "Runtime output directory for LoRAStencil executables."
)
set(
    LORA_STENCIL_CUDA_ARCHITECTURES
    "80;90"
    CACHE STRING
    "CUDA architectures used to build LoRAStencil targets."
)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LORA_STENCIL_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_CUDA_ARCHITECTURES "${LORA_STENCIL_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

link_directories(${CUDA_LIBRARY_DIRS})
link_directories(/usr/local/cuda/lib64)

function(configure_lorastencil_target target)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${LORA_STENCIL_CUDA_ARCHITECTURES}"
        RUNTIME_OUTPUT_DIRECTORY "${LORA_STENCIL_RUNTIME_OUTPUT_DIRECTORY}")
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--use_fast_math;--ptxas-options=-v>)
endfunction()

add_executable(
    lorastencil_1d
    src/1d/main.cu
    src/1d/gpu_1r.cu
    src/1d/gpu_2r.cu
    src/1d/1d_utils.h
)
configure_lorastencil_target(lorastencil_1d)

add_executable(
    lorastencil_2d
    src/2d/main.cu
    src/2d/gpu.cu
    src/2d/2d_utils.h
)
configure_lorastencil_target(lorastencil_2d)

add_executable(
    lorastencil_3d
    src/3d/main.cu
    src/3d/gpu_box.cu
    src/3d/gpu_star.cu
    src/3d/3d_utils.h
)
configure_lorastencil_target(lorastencil_3d)
