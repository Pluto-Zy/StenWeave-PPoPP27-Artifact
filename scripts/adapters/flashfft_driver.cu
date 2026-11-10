#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if DIM == 2
#include "src/2D/common/fft_overlap_save/fft_2d.cuh"
using Plan = FlashFFT2DPlan;
static cudaError_t create_plan(Plan** p, const double* c, int r) {
    return flashfft2dCreatePlan(p, c, r);
}
static cudaError_t execute(const Plan* p, const double* in, double* out, int w) {
    return flashfft2dExecute(p, in, out, w);
}
static cudaError_t destroy_plan(Plan* p) { return flashfft2dDestroyPlan(p); }
#elif DIM == 3
#include "src/3D/common/fft_overlap_save/fft_3d.cuh"
using Plan = FlashFFT3DPlan;
static cudaError_t create_plan(Plan** p, const double* c, int r) {
    return flashfft3dCreatePlan(p, c, r);
}
static cudaError_t execute(const Plan* p, const double* in, double* out, int w) {
    return flashfft3dExecute(p, in, out, w);
}
static cudaError_t destroy_plan(Plan* p) { return flashfft3dDestroyPlan(p); }
#else
#error DIM must be 2 or 3
#endif

namespace {
void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
        std::exit(2);
    }
}

struct Kernel {
    int radius{};
    std::vector<double> coefficients;
};

struct Options {
    std::string shape = "star";
    int radius = 1;
    int width = 64;
    int steps = 1;
    int fusion = 1;
    int warmups = 1;
    int repeats = 3;
    bool check = false;
    bool asymmetric = false;
    bool uniform = false;
    std::string input = "zero";
};

Options parse_options(int argc, char** argv) {
    Options o;
    bool input_specified = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) {
                std::cerr << "missing value for " << key << '\n';
                std::exit(2);
            }
            return argv[i];
        };
        if (key == "--shape") o.shape = value();
        else if (key == "--radius") o.radius = std::stoi(value());
        else if (key == "--width") o.width = std::stoi(value());
        else if (key == "--steps") o.steps = std::stoi(value());
        else if (key == "--fusion") o.fusion = std::stoi(value());
        else if (key == "--warmups") o.warmups = std::stoi(value());
        else if (key == "--repeats") o.repeats = std::stoi(value());
        else if (key == "--check") o.check = true;
        else if (key == "--asymmetric") o.asymmetric = true;
        else if (key == "--uniform") o.uniform = true;
        else if (key == "--input") { o.input = value(); input_specified = true; }
        else {
            std::cerr << "unknown option: " << key << '\n';
            std::exit(2);
        }
    }
    if (o.check && !input_specified) o.input = "random";
    if ((o.shape != "star" && o.shape != "box") || o.radius < 1 || o.radius > 3
        || o.width < 64 || o.steps < 1 || o.fusion < 1 || o.fusion > o.steps
        || o.warmups < 0 || o.repeats < 1 || (o.uniform && o.asymmetric)
        || (o.input != "random" && o.input != "zero" && o.input != "one")) {
        std::cerr << "invalid arguments; width >= 64, radius 1..3, steps >= fusion >= 1\n";
        std::exit(2);
    }
    const int max_effective_radius = DIM == 2 ? 15 : 7;
    if (o.radius * o.fusion > max_effective_radius) {
        std::cerr << "fusion exceeds the transform's valid halo radius\n";
        std::exit(2);
    }
    return o;
}

std::size_t volume(int width) {
    std::size_t result = 1;
    for (int d = 0; d < DIM; ++d) {
        if (result > static_cast<std::size_t>(-1) / static_cast<std::size_t>(width)) {
            std::cerr << "volume overflow\n";
            std::exit(2);
        }
        result *= static_cast<std::size_t>(width);
    }
    return result;
}

std::size_t kernel_volume(int radius) {
    std::size_t result = 1;
    for (int d = 0; d < DIM; ++d) result *= static_cast<std::size_t>(2 * radius + 1);
    return result;
}

std::vector<int> decode(std::size_t index, int radius) {
    const int side = 2 * radius + 1;
    std::vector<int> offset(DIM);
    for (int d = DIM - 1; d >= 0; --d) {
        offset[d] = static_cast<int>(index % side) - radius;
        index /= side;
    }
    return offset;
}

std::size_t encode(const std::vector<int>& offset, int radius) {
    const std::size_t side = static_cast<std::size_t>(2 * radius + 1);
    std::size_t index = 0;
    for (int d = 0; d < DIM; ++d) {
        index = index * side + static_cast<std::size_t>(offset[d] + radius);
    }
    return index;
}

Kernel make_basic_kernel(const Options& o) {
    Kernel kernel{o.radius, std::vector<double>(kernel_volume(o.radius), 0.0)};
    const double box_values[] = {20.567, 18.123, 22.901, 19.345, 21.789, 17.234, 23.456};
    const double star_values[] = {18.123, 22.901, 19.345, 21.789, 21.123, 18.789};
    const double star_divisor = 20.0 + o.radius * DIM * 2 * 20.0;
    const double box_divisor = 20.0 * static_cast<double>(kernel.coefficients.size());
    for (std::size_t i = 0; i < kernel.coefficients.size(); ++i) {
        const auto offset = decode(i, o.radius);
        int nonzero_axes = 0;
        int active_axis = -1;
        for (int d = 0; d < DIM; ++d) {
            if (offset[d] != 0) {
                ++nonzero_axes;
                active_axis = d;
            }
        }
        if (o.shape == "box") {
            kernel.coefficients[i] = box_values[offset[0] + o.radius] / box_divisor;
        } else if (nonzero_axes == 0) {
            kernel.coefficients[i] = 20.567 / star_divisor;
        } else if (nonzero_axes == 1) {
            int direction_index = 0;
            if (DIM == 2) {
                direction_index = active_axis == 0 ? (offset[active_axis] < 0 ? 0 : 1)
                                                   : (offset[active_axis] < 0 ? 2 : 3);
            } else {
                direction_index = active_axis == 0 ? (offset[active_axis] < 0 ? 0 : 1)
                    : active_axis == 2 ? (offset[active_axis] < 0 ? 2 : 3)
                                       : (offset[active_axis] < 0 ? 4 : 5);
            }
            // StenWeave repeats the same directional weights at every distance.
            kernel.coefficients[i] = star_values[direction_index] / star_divisor;
        }
    }
    if (o.uniform) {
        const auto count = std::count_if(kernel.coefficients.begin(), kernel.coefficients.end(),
                                        [](double c) { return c != 0.0; });
        for (double& c : kernel.coefficients) if (c != 0.0) c = 1.0 / count;
    }
    if (o.asymmetric) {
        double sum = 0.0;
        for (std::size_t i = 0; i < kernel.coefficients.size(); ++i) {
            if (kernel.coefficients[i] != 0.0) {
                kernel.coefficients[i] = 1.0 + 0.01 * static_cast<double>(i + 1);
                sum += kernel.coefficients[i];
            }
        }
        for (double& coefficient : kernel.coefficients) coefficient /= sum;
    }
    return kernel;
}

Kernel convolve(const Kernel& a, const Kernel& b) {
    Kernel result{a.radius + b.radius,
                  std::vector<double>(kernel_volume(a.radius + b.radius), 0.0)};
    for (std::size_t ia = 0; ia < a.coefficients.size(); ++ia) {
        if (a.coefficients[ia] == 0.0) continue;
        const auto oa = decode(ia, a.radius);
        for (std::size_t ib = 0; ib < b.coefficients.size(); ++ib) {
            if (b.coefficients[ib] == 0.0) continue;
            const auto ob = decode(ib, b.radius);
            std::vector<int> offset(DIM);
            for (int d = 0; d < DIM; ++d) offset[d] = oa[d] + ob[d];
            result.coefficients[encode(offset, result.radius)] +=
                a.coefficients[ia] * b.coefficients[ib];
        }
    }
    return result;
}

Kernel fused_kernel(const Kernel& basic, int fusion) {
    Kernel result{0, {1.0}};
    for (int i = 0; i < fusion; ++i) result = convolve(result, basic);
    return result;
}

std::vector<double> apply_reference(const Kernel& kernel, const std::vector<double>& input,
                                    int width) {
    const std::size_t points = volume(width);
    std::vector<double> output(points, 0.0);
    std::vector<int> coordinate(DIM);
    for (std::size_t p = 0; p < points; ++p) {
        std::size_t rest = p;
        for (int d = DIM - 1; d >= 0; --d) {
            coordinate[d] = static_cast<int>(rest % width);
            rest /= width;
        }
        double value = 0.0;
        for (std::size_t q = 0; q < kernel.coefficients.size(); ++q) {
            const double coefficient = kernel.coefficients[q];
            if (coefficient == 0.0) continue;
            const auto offset = decode(q, kernel.radius);
            std::size_t source = 0;
            for (int d = 0; d < DIM; ++d) {
                int x = (coordinate[d] + offset[d]) % width;
                if (x < 0) x += width;
                source = source * static_cast<std::size_t>(width) +
                         static_cast<std::size_t>(x);
            }
            value += coefficient * input[source];
        }
        output[p] = value;
    }
    return output;
}

void execute_steps(const Options& o, const Plan* full_plan, const Plan* tail_plan,
                   double* first, double* second, const double** final) {
    const int full_groups = o.steps / o.fusion;
    const int tail = o.steps % o.fusion;
    const double* input = first;
    double* output = second;
    for (int i = 0; i < full_groups; ++i) {
        cuda_check(execute(full_plan, input, output, o.width), "FlashFFT execute");
        input = output;
        output = (output == second) ? first : second;
    }
    if (tail != 0) {
        cuda_check(execute(tail_plan, input, output, o.width), "FlashFFT tail execute");
        input = output;
    }
    *final = input;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n & 1) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}
} // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const Kernel basic = make_basic_kernel(options);
    const Kernel full_fused = fused_kernel(basic, options.fusion);
    const int tail = options.steps % options.fusion;
    const Kernel tail_fused = tail ? fused_kernel(basic, tail) : full_fused;

    Plan* full_plan = nullptr;
    Plan* tail_plan = nullptr;
    cuda_check(create_plan(&full_plan, full_fused.coefficients.data(), full_fused.radius),
               "Create full-fusion plan");
    if (tail != 0 && tail != options.fusion) {
        cuda_check(create_plan(&tail_plan, tail_fused.coefficients.data(), tail_fused.radius),
                   "Create tail plan");
    } else {
        tail_plan = full_plan;
    }

    const std::size_t points = volume(options.width);
    std::vector<double> host_input(points);
    std::mt19937_64 random(0x5eed + DIM * 100 + options.radius);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (double& x : host_input)
        x = options.input == "random" ? distribution(random) : options.input == "one" ? 1.0 : 0.0;

    double *device_a = nullptr, *device_b = nullptr;
    cuda_check(cudaMalloc(&device_a, points * sizeof(double)), "cudaMalloc input");
    cuda_check(cudaMalloc(&device_b, points * sizeof(double)), "cudaMalloc output");

    if (options.check) {
        cuda_check(cudaMemcpy(device_a, host_input.data(), points * sizeof(double),
                              cudaMemcpyHostToDevice), "copy correctness input");
        const double* final = nullptr;
        execute_steps(options, full_plan, tail_plan, device_a, device_b, &final);
        cuda_check(cudaDeviceSynchronize(), "synchronize correctness run");
        std::vector<double> actual(points);
        cuda_check(cudaMemcpy(actual.data(), final, points * sizeof(double),
                              cudaMemcpyDeviceToHost), "copy correctness result");
        std::vector<double> expected = host_input;
        for (int step = 0; step < options.steps; ++step) expected = apply_reference(basic, expected,
                                                                                      options.width);
        double max_abs = 0.0, max_rel = 0.0;
        bool within_tolerance = true;
        std::size_t worst = 0;
        for (std::size_t i = 0; i < points; ++i) {
            const double error = std::abs(actual[i] - expected[i]);
            const double relative = error / std::max(1.0, std::abs(expected[i]));
            if (error > max_abs) { max_abs = error; worst = i; }
            max_rel = std::max(max_rel, relative);
            within_tolerance &= error <= 1.0e-9 + 1.0e-9 * std::abs(expected[i]);
        }
        std::cout << std::setprecision(12) << "check_max_abs=" << max_abs
                  << " check_max_rel=" << max_rel << " worst_index=" << worst << '\n';
        if (!within_tolerance) {
            std::cerr << "correctness check failed\n";
            return 3;
        }
    }

    if (options.repeats > 0) {
        cudaEvent_t start{}, stop{};
        cuda_check(cudaEventCreate(&start), "create start event");
        cuda_check(cudaEventCreate(&stop), "create stop event");
        for (int i = 0; i < options.warmups; ++i) {
            cuda_check(cudaMemcpy(device_a, host_input.data(), points * sizeof(double),
                                  cudaMemcpyHostToDevice), "copy warmup input");
            const double* final = nullptr;
            execute_steps(options, full_plan, tail_plan, device_a, device_b, &final);
            cuda_check(cudaDeviceSynchronize(), "synchronize warmup");
        }
        std::vector<double> times;
        for (int i = 0; i < options.repeats; ++i) {
            cuda_check(cudaMemcpy(device_a, host_input.data(), points * sizeof(double),
                                  cudaMemcpyHostToDevice), "copy timed input");
            cuda_check(cudaDeviceSynchronize(), "synchronize timed input");
            const double* final = nullptr;
            cuda_check(cudaEventRecord(start), "record start event");
            execute_steps(options, full_plan, tail_plan, device_a, device_b, &final);
            cuda_check(cudaEventRecord(stop), "record stop event");
            cuda_check(cudaEventSynchronize(stop), "synchronize stop event");
            float elapsed = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed time");
            times.push_back(elapsed);
        }
        const double selected_ms = median(times);
        const double updates = static_cast<double>(points) * options.steps;
        const double gstencils = updates / (selected_ms * 1.0e6);
        int device = 0;
        cudaDeviceProp properties{};
        cuda_check(cudaGetDevice(&device), "get device");
        cuda_check(cudaGetDeviceProperties(&properties, device), "get properties");
        std::cout << std::setprecision(10) << "device=" << properties.name
                  << " arch=sm" << properties.major << properties.minor << " DIM=" << DIM
                  << " shape=" << options.shape << " radius=" << options.radius
                  << " fusion=" << options.fusion << " steps=" << options.steps
                  << " coefficients=" << (options.asymmetric ? "asymmetric-test" : options.uniform ? "uniform" : "stenweave")
                  << " input=" << options.input
                  << " width=" << options.width << '\n'
                  << "Time = " << selected_ms << "[ms]\n"
                  << "GStencil/s = " << gstencils << '\n'
                  << "trials_ms=";
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << times[i];
        }
        std::cout << '\n';
    }

    cuda_check(cudaFree(device_a), "free input");
    cuda_check(cudaFree(device_b), "free output");
    if (tail_plan != full_plan) cuda_check(destroy_plan(tail_plan), "destroy tail plan");
    cuda_check(destroy_plan(full_plan), "destroy plan");
    return 0;
}
