add_library(
    stencil_model
    src/config/batch.cpp
    src/artifact/catalog.cpp
    src/artifact/hash.cpp
    src/execution/plan.cpp
    src/execution/input.cpp
    src/stencil/model.cpp
    src/report/report.cpp
)

target_include_directories(stencil_model PUBLIC src)
target_link_libraries(
    stencil_model PUBLIC
    nlohmann_json::nlohmann_json
    tomlplusplus::tomlplusplus
    Threads::Threads
    fmt::fmt
)

find_path(NVML_INCLUDE_DIR nvml.h HINTS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)
find_library(
    NVML_LIBRARY NAMES nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs REQUIRED
)

add_executable(
    stencil
    src/main.cpp
    src/cuda/driver.cpp
    src/cuda/inspect.cpp
    src/execution/executor.cpp
)
target_include_directories(stencil PRIVATE ${NVML_INCLUDE_DIR})
target_link_libraries(
    stencil PRIVATE
    stencil_model
    CUDA::cuda_driver
    CLI11::CLI11
    fmt::fmt
    ${NVML_LIBRARY}
)

if (BUILD_TESTING)
    add_executable(runner_tests tests/runner_tests.cpp)
    target_link_libraries(runner_tests PRIVATE stencil_model)
    add_test(NAME runner_pure COMMAND runner_tests)
    add_test(
        NAME runner_copy_contract
        COMMAND ${CMAKE_COMMAND} -DRUNNER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/copy_contract.cmake
    )
    add_test(
        NAME runner_graph_contract
        COMMAND ${CMAKE_COMMAND} -DRUNNER_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/graph_contract.cmake
    )
endif ()

# Artifacts are generated offline; GPU integration is opt-in for an existing SM80 artifact.
set(STENCIL_GPU_ARTIFACT_DIR "" CACHE PATH "Artifact root for banded GPU integration")
set(STENCIL_GPU_DEVICE 0 CACHE STRING "CUDA device ordinal for banded GPU integration")
if (BUILD_TESTING AND STENCIL_GPU_ARTIFACT_DIR)
    add_test(
        NAME runner_banded_gpu
        COMMAND stencil run ${CMAKE_CURRENT_SOURCE_DIR}/tests/banded_gpu.toml
            --artifact-dir ${STENCIL_GPU_ARTIFACT_DIR} --device ${STENCIL_GPU_DEVICE}
            --output ${CMAKE_CURRENT_BINARY_DIR}/banded_gpu.jsonl --overwrite
            --input-threads 4 --no-color
    )
endif ()
