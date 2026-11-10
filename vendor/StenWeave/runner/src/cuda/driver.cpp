#include "cuda/driver.hpp"

#include <cuda.h>

#include <fmt/format.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace stencil {
void cuda_check(CUresult result) {
    if (result == CUDA_SUCCESS) {
        return;
    }

    char const *name = nullptr, *description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    throw std::runtime_error(
        fmt::format("{}: {}", name ? name : "CUDA error", description ? description : "unknown")
    );
}

auto device_info(int ordinal) -> DeviceInfo {
    cuda_check(cuInit(0));
    CUdevice device;
    cuda_check(cuDeviceGet(&device, ordinal));

    char name[256] {};
    cuda_check(cuDeviceGetName(name, sizeof(name), device));

    int major = 0, minor = 0;
    cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    std::string const arch = fmt::format("sm{}{}", major, minor);

    // clang-format off
    return {
        device,
        arch,
        {
            { "id", ordinal },
            { "name", name },
            { "arch", arch },
            { "compute_capability", fmt::format("{}.{}", major, minor) },
        },
    };
    // clang-format on
}

Session::Session(CUdevice device) {
    cuda_check(cuCtxGetCurrent(&previous_));
    cuda_check(cuCtxCreate(&context_, 0, device));
}

Session::~Session() {
    if (context_) {
        cuCtxDestroy(context_);
        cuCtxSetCurrent(previous_);
    }
}

Stream::Stream() {
    cuda_check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
}

Stream::~Stream() {
    if (stream_) {
        cuStreamDestroy(stream_);
    }
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) {
    cuda_check(cuMemAlloc(&pointer_, bytes));
}

DeviceBuffer::~DeviceBuffer() {
    if (pointer_) {
        cuMemFree(pointer_);
    }
}

Event::Event() {
    cuda_check(cuEventCreate(&event_, CU_EVENT_DEFAULT));
}

Event::~Event() {
    if (event_) {
        cuEventDestroy(event_);
    }
}
}  // namespace stencil
