#pragma once

#include <cuda.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

namespace stencil {
void cuda_check(CUresult result);

struct DeviceInfo {
    CUdevice device;
    std::string arch;
    nlohmann::json metadata;
};

auto device_info(int ordinal) -> DeviceInfo;

class Session {
public:
    explicit Session(CUdevice device);
    ~Session();

    Session(Session const &) = delete;
    auto operator=(Session const &) -> Session & = delete;

private:
    CUcontext context_ = nullptr;
    CUcontext previous_ = nullptr;
};

class Stream {
public:
    Stream();
    ~Stream();

    Stream(Stream const &) = delete;
    auto operator=(Stream const &) -> Stream & = delete;

    auto get() const -> CUstream {
        return stream_;
    }

private:
    CUstream stream_ = nullptr;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes);
    ~DeviceBuffer();

    DeviceBuffer(DeviceBuffer const &) = delete;
    auto operator=(DeviceBuffer const &) -> DeviceBuffer & = delete;

    auto get() const -> CUdeviceptr {
        return pointer_;
    }

private:
    CUdeviceptr pointer_ = 0;
};

class Event {
public:
    Event();
    ~Event();

    Event(Event const &) = delete;
    auto operator=(Event const &) -> Event & = delete;

    auto get() const -> CUevent {
        return event_;
    }

private:
    CUevent event_ = nullptr;
};
}  // namespace stencil
