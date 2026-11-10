#include "cuda/inspect.hpp"

#include "cuda/driver.hpp"

#include <cuda.h>
#include <nvml.h>

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <vector>

namespace stencil {
namespace {
class NVMLSession {
public:
    NVMLSession() {
        nvmlReturn_t const status = nvmlInit_v2();
        if (status != NVML_SUCCESS) {
            throw std::runtime_error(nvmlErrorString(status));
        }
    }

    ~NVMLSession() {
        nvmlShutdown();
    }

    NVMLSession(NVMLSession const &) = delete;
    auto operator=(NVMLSession const &) -> NVMLSession & = delete;
};

auto unavailable(nvmlReturn_t status) -> nlohmann::json {
    // clang-format off
    return { { "unavailable", nvmlErrorString(status) } };
    // clang-format on
}

auto processes(nvmlDevice_t device, bool graphics) -> nlohmann::json {
    auto const query = [&](unsigned *count, nvmlProcessInfo_t *data) -> nvmlReturn_t {
        return graphics ? nvmlDeviceGetGraphicsRunningProcesses(device, count, data)
                        : nvmlDeviceGetComputeRunningProcesses(device, count, data);
    };

    unsigned count = 0;
    nvmlReturn_t status = query(&count, nullptr);
    if (status != NVML_SUCCESS && status != NVML_ERROR_INSUFFICIENT_SIZE) {
        return unavailable(status);
    }

    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        std::vector<nvmlProcessInfo_t> data(count + 8);
        count = static_cast<unsigned>(data.size());
        status = query(&count, data.data());

        if (status == NVML_ERROR_INSUFFICIENT_SIZE) {
            continue;
        }

        if (status != NVML_SUCCESS) {
            return unavailable(status);
        }

        nlohmann::json result = nlohmann::json::array();
        for (unsigned i = 0; i < count; ++i) {
            // clang-format off
            nlohmann::json process { { "pid", data[i].pid } };
            // clang-format on
            if (data[i].usedGpuMemory != NVML_VALUE_NOT_AVAILABLE) {
                process["used_gpu_memory_bytes"] = data[i].usedGpuMemory;
            }

            result.push_back(process);
        }

        return result;
    }

    // clang-format off
    return { { "unavailable", "process list changed repeatedly during snapshot" } };
    // clang-format on
}
}  // namespace

auto inspect_device(int ordinal) -> nlohmann::json {
    DeviceInfo const info = device_info(ordinal);
    NVMLSession const session;

    // Match CUDA's selected device by PCI address, independent of NVML enumeration order.
    char pci[32] {};
    cuda_check(cuDeviceGetPCIBusId(pci, sizeof(pci), info.device));

    nvmlDevice_t device;
    nvmlReturn_t status = nvmlDeviceGetHandleByPciBusId_v2(pci, &device);
    if (status != NVML_SUCCESS) {
        throw std::runtime_error(nvmlErrorString(status));
    }

    nlohmann::json json = info.metadata;
    char text[256] {};
    auto const string_field = [&](char const *key, nvmlReturn_t result) {
        json[key] = result == NVML_SUCCESS ? nlohmann::json(text) : unavailable(result);
    };

    string_field("uuid", nvmlDeviceGetUUID(device, text, sizeof(text)));
    string_field("driver_version", nvmlSystemGetDriverVersion(text, sizeof(text)));

    auto const number = [&](char const *key, auto query) {
        unsigned value = 0;
        nvmlReturn_t const status = query(&value);
        json[key] = status == NVML_SUCCESS ? nlohmann::json(value) : unavailable(status);
    };

    number("sm_clock_mhz", [&](unsigned *value) -> nvmlReturn_t {
        return nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, value);
    });
    number("memory_clock_mhz", [&](unsigned *value) -> nvmlReturn_t {
        return nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, value);
    });

    number("temperature_c", [&](unsigned *value) -> nvmlReturn_t {
        return nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, value);
    });

    number("power_usage_mw", [&](unsigned *value) -> nvmlReturn_t {
        return nvmlDeviceGetPowerUsage(device, value);
    });
    number("power_limit_mw", [&](unsigned *value) -> nvmlReturn_t {
        return nvmlDeviceGetPowerManagementLimit(device, value);
    });

    nvmlComputeMode_t mode;
    status = nvmlDeviceGetComputeMode(device, &mode);
    json["compute_mode"] =
        status == NVML_SUCCESS ? nlohmann::json(static_cast<unsigned>(mode)) : unavailable(status);

    nvmlPstates_t state;
    status = nvmlDeviceGetPerformanceState(device, &state);
    json["performance_state"] =
        status == NVML_SUCCESS ? nlohmann::json(static_cast<unsigned>(state)) : unavailable(status);

    unsigned long long reasons = 0;
    status = nvmlDeviceGetCurrentClocksEventReasons(device, &reasons);
    json["active_clock_event_reasons_mask"] =
        status == NVML_SUCCESS ? nlohmann::json(reasons) : unavailable(status);

    json["compute_processes"] = processes(device, false);
    json["graphics_processes"] = processes(device, true);

    return json;
}
}  // namespace stencil
