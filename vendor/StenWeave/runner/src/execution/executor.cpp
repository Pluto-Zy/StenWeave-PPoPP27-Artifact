#include "execution/executor.hpp"

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "cuda/driver.hpp"
#include "execution/input.hpp"
#include "execution/launch.hpp"
#include "execution/plan.hpp"
#include "report/report.hpp"
#include "stencil/model.hpp"

#include <cuda.h>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace stencil {
namespace {
class PreparedKernel {
public:
    explicit PreparedKernel(ArtifactKernel const &kernel) {
        cuda_check(cuModuleLoad(&module_, kernel.cubin_path.c_str()));

        try {
            cuda_check(cuModuleGetFunction(&function_, module_, kernel.entry_symbol.c_str()));

            CUdeviceptr storage;
            std::size_t bytes = 0;
            cuda_check(
                cuModuleGetGlobal(&storage, &bytes, module_, kernel.coefficient_symbol.c_str())
            );

            std::vector<double> const coefficient_values = coefficients(kernel);
            if (bytes != coefficient_values.size() * sizeof(double)) {
                throw std::runtime_error("Coefficient storage size mismatch");
            }

            cuda_check(cuMemcpyHtoD(storage, coefficient_values.data(), bytes));
        } catch (...) {
            cuModuleUnload(module_);

            throw;
        }
    }

    ~PreparedKernel() {
        cuModuleUnload(module_);
    }

    PreparedKernel(PreparedKernel const &) = delete;
    auto operator=(PreparedKernel const &) -> PreparedKernel & = delete;

    auto function() const -> CUfunction {
        return function_;
    }

private:
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
};

struct Workspace {
    DeviceBuffer baseline;
    DeviceBuffer work_a;
    DeviceBuffer work_b;

    explicit Workspace(std::size_t bytes) : baseline(bytes), work_a(bytes), work_b(bytes) { }

    void reset(std::size_t bytes, CUstream stream) const {
        cuda_check(cuMemcpyDtoDAsync(work_a.get(), baseline.get(), bytes, stream));
    }
};

class BandExecution {
public:
    BandExecution(
        ResolvedCase const &resolved_case,
        CUfunction function,
        CUdeviceptr work_a,
        CUdeviceptr work_b
    ) {
        cuda_check(cuGraphCreate(&graph_, 0));

        try {
            // A100 measurements favor this DAG: empty fan-in/fan-out nodes reduce
            // concurrent kernel execution at large band counts (ARTIFACT_VALIDATION.md).
            std::vector<CUgraphNode> previous;

            for (unsigned t = 0; t < resolved_case.run.time_steps;
                 t += resolved_case.kernel->time_block_size) {
                std::vector<CUgraphNode> current;

                for (ResolvedBand const &band : resolved_case.bands) {
                    CUdeviceptr input = work_a + band.byte_offset;
                    CUdeviceptr output = work_b + band.byte_offset;
                    launch::Params launch_params = band.params;
                    bool condition = false;

                    std::vector<void *> args { &input, &output };
                    if (resolved_case.kernel->backend == Backend::CUDA) {
                        launch::cuda::Params &params =
                            std::get<launch::cuda::Params>(launch_params);
                        args.push_back(&params.height);

                        if (resolved_case.kernel->profile.name == "breakdown") {
                            args.push_back(&condition);
                        }
                    } else {
                        launch::sass::Params &params =
                            std::get<launch::sass::Params>(launch_params);
                        args.insert(
                            args.end(),
                            {
                                &params.row_pitch,
                                &params.plane_stride,
                                &params.cta_stride_x,
                                &params.cta_stride_y,
                                &params.write_begin_x,
                                &params.write_end_x,
                                &params.write_begin_y,
                                &params.write_end_y,
                                &params.pipeline_iterations,
                                &params.pipeline_remainder,
                                &params.store_head,
                                &params.store_tail,
                            }
                        );
                    }

                    CUDA_KERNEL_NODE_PARAMS params {};
                    params.func = function;
                    params.gridDimX = band.grid.x;
                    params.gridDimY = band.grid.y.value_or(1);
                    params.gridDimZ = 1;

                    params.blockDimX = resolved_case.run.block.x;
                    params.blockDimY = resolved_case.run.block.y.value_or(1);
                    params.blockDimZ = 1;

                    params.sharedMemBytes = resolved_case.dynamic_smem_bytes;
                    params.kernelParams = args.data();

                    CUgraphNode node;
                    cuda_check(cuGraphAddKernelNode(
                        &node,
                        graph_,
                        previous.data(),
                        previous.size(),
                        &params
                    ));

                    current.push_back(node);
                }

                previous = std::move(current);
                std::swap(work_a, work_b);
            }

            output_ = work_a;
            cuda_check(cuGraphInstantiateWithFlags(&instance_, graph_, 0));
        } catch (...) {
            cuGraphDestroy(graph_);

            throw;
        }
    }

    ~BandExecution() {
        cuGraphExecDestroy(instance_);
        cuGraphDestroy(graph_);
    }

    BandExecution(BandExecution const &) = delete;
    auto operator=(BandExecution const &) -> BandExecution & = delete;

    auto launch(CUstream stream) const -> CUdeviceptr {
        cuda_check(cuGraphLaunch(instance_, stream));

        return output_;
    }

private:
    CUgraph graph_ = nullptr;
    CUgraphExec instance_ = nullptr;
    CUdeviceptr output_ = 0;
};

void validate_device(ExecutionPlan const &plan, CUdevice device) {
    auto const attribute = [&](CUdevice_attribute attr) -> unsigned int {
        int value = 0;
        cuda_check(cuDeviceGetAttribute(&value, attr, device));

        return static_cast<unsigned>(value);
    };

    unsigned const max_threads = attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    unsigned const max_x = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    unsigned const max_y = attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    unsigned const grid_x = attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    unsigned const grid_y = attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    unsigned const smem = attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);

    for (ResolvedCase const &resolved_case : plan.cases) {
        if (checked_product(resolved_case.run.block.x, resolved_case.run.block.y.value_or(1))
                > max_threads
            || resolved_case.run.block.x > max_x || resolved_case.run.block.y.value_or(1) > max_y
            || std::any_of(
                resolved_case.bands.begin(),
                resolved_case.bands.end(),
                [&](ResolvedBand const &band) -> bool {
                    return band.grid.x > grid_x || band.grid.y.value_or(1) > grid_y;
                }
            )
            || resolved_case.dynamic_smem_bytes > smem) {
            throw std::runtime_error(
                fmt::format("Case exceeds GPU launch limits: {}", resolved_case.name)
            );
        }
    }
}

using ReferenceCache = std::map<std::string, std::vector<double>>;

auto validate_case(
    ResolvedCase const &resolved_case,
    DeviceInfo const &device,
    Workspace const &workspace,
    std::vector<double> const &input,
    ReferenceCache &references
) -> nlohmann::json {
    std::size_t const case_bytes = resolved_case.workspace_elements * sizeof(double);

    // Isolate validation from previous launches. Fresh shared-memory contents are
    // not guaranteed by CUDA; the cubin halo assumption is checked on available hardware.
    // Copy the shared device baseline directly; never upload another host input.
    std::vector<double> actual(resolved_case.workspace_elements);
    {
        CUcontext source = nullptr;
        cuda_check(cuCtxGetCurrent(&source));

        Session const validation_session(device.device);
        CUcontext destination = nullptr;
        cuda_check(cuCtxGetCurrent(&destination));

        DeviceBuffer const work_a(case_bytes);
        DeviceBuffer const work_b(case_bytes);
        cuda_check(
            cuMemcpyPeer(work_a.get(), destination, workspace.baseline.get(), source, case_bytes)
        );

        Stream const validation_stream;
        PreparedKernel const validation_kernel(*resolved_case.kernel);

        cuda_check(cuFuncSetAttribute(
            validation_kernel.function(),
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(resolved_case.dynamic_smem_bytes)
        ));

        BandExecution const
            execution(resolved_case, validation_kernel.function(), work_a.get(), work_b.get());
        CUdeviceptr const output = execution.launch(validation_stream.get());
        cuda_check(cuStreamSynchronize(validation_stream.get()));

        cuda_check(cuMemcpyDtoH(actual.data(), output, case_bytes));
    }

    std::string const key = resolved_case.name;
    if (!references.count(key)) {
        if (resolved_case.kernel->backend == Backend::CUDA) {
            std::vector<double> expected;
            expected.reserve(resolved_case.workspace_elements);

            for (ResolvedBand const &band : resolved_case.bands) {
                unsigned const width = band.execution_extent.x;
                unsigned const depth = band.execution_extent.y.value_or(1);
                std::size_t const elements =
                    checked_product(checked_product(width, depth), band.stream);
                Domain const domain { width, depth, band.stream, elements };

                std::size_t const offset = band.byte_offset / sizeof(double);
                std::vector<double> const
                    strip_input(input.data() + offset, input.data() + offset + elements);

                std::vector<double> const result = compute_reference(
                    *resolved_case.kernel,
                    domain,
                    resolved_case.run.time_steps,
                    strip_input
                );
                expected.insert(expected.end(), result.begin(), result.end());
            }

            references.emplace(key, std::move(expected));
        } else {
            references.emplace(key, compute_reference(resolved_case, input));
        }
    }

    std::vector<double> const &expected = references.at(key);
    std::size_t mismatches = 0;
    double max_error = 0;
    nlohmann::json examples = nlohmann::json::array();

    for (std::size_t i = 0; i < actual.size(); ++i) {
        double const error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);

        if (!std::isfinite(actual[i]) || error > resolved_case.run.tolerance) {
            ++mismatches;

            if (examples.size() < 5) {
                // clang-format off
                examples.push_back({
                    { "index", i },
                    { "actual", actual[i] },
                    { "expected", expected[i] },
                });
                // clang-format on
            }
        }
    }

    // clang-format off
    return {
        { "status", mismatches ? "failed" : "passed" },
        { "mismatches", mismatches },
        { "max_abs_error", max_error },
        { "examples", examples },
        { "isolation", "fresh_cuda_context" },
        { "scope", resolved_case.kernel->backend == Backend::SASS
            ? "execution_domain" : "cuda_strips" },
    };
    // clang-format on
}

void warmup_case(
    ResolvedCase const &resolved_case,
    BandExecution const &execution,
    Workspace const &workspace,
    CUstream stream
) {
    std::size_t const case_bytes = resolved_case.workspace_elements * sizeof(double);

    for (unsigned i = 0; i < resolved_case.run.warmup_runs; ++i) {
        workspace.reset(case_bytes, stream);

        execution.launch(stream);
        cuda_check(cuStreamSynchronize(stream));
    }
}

auto measure_case(
    ResolvedCase const &resolved_case,
    BandExecution const &execution,
    Workspace const &workspace,
    CUstream stream,
    Event const &start,
    Event const &stop
) -> std::vector<SampleResult> {
    std::size_t const case_bytes = resolved_case.workspace_elements * sizeof(double);
    std::vector<SampleResult> samples;

    for (unsigned i = 0; i < resolved_case.run.repeat_runs; ++i) {
        if (resolved_case.run.cooldown_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(resolved_case.run.cooldown_ms));
        }

        warmup_case(resolved_case, execution, workspace, stream);
        workspace.reset(case_bytes, stream);

        cuda_check(cuEventRecord(start.get(), stream));
        execution.launch(stream);
        cuda_check(cuEventRecord(stop.get(), stream));

        cuda_check(cuEventSynchronize(stop.get()));

        float elapsed_ms = 0;
        cuda_check(cuEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        samples.push_back({ elapsed_ms, throughput(resolved_case.stencil_updates, elapsed_ms) });
    }

    return samples;
}

void report_case(
    ResolvedCase const &resolved_case,
    CaseResult const &result,
    std::size_t completed,
    std::size_t case_count,
    Reporter &reporter,
    bool no_color
) {
    nlohmann::json const record = case_json(resolved_case, result);
    reporter.write(record);

    std::string const status =
        resolved_case.run.check ? result.validation["status"].get<std::string>() : "skipped";
    fmt::text_style const status_style = no_color
        ? fmt::text_style {}
        : (status == "passed" ? fmt::fg(fmt::color::green) : fmt::fg(fmt::color::yellow));

    fmt::println(
        "[{}/{}] {}  {}  {}x{}x{}  {} {:.6f} ms  {:.4f} GStencils/s  check={}",
        completed,
        case_count,
        resolved_case.name,
        resolved_case.kernel->id,
        resolved_case.domain.width,
        resolved_case.domain.depth,
        resolved_case.domain.height,
        record["summary"]["selected"]["statistic"].get<std::string>(),
        record["summary"]["selected"]["time_ms"].get<double>(),
        record["summary"]["selected"]["gstencils_per_s"].get<double>(),
        fmt::styled(status, status_style)
    );
}

using PreparedCache = std::map<std::string, std::unique_ptr<PreparedKernel>>;

auto prepare_kernels(ExecutionPlan const &plan) -> PreparedCache {
    PreparedCache prepared;

    for (ResolvedCase const &resolved_case : plan.cases) {
        if (!prepared.count(resolved_case.kernel->id)) {
            prepared.emplace(
                resolved_case.kernel->id,
                std::make_unique<PreparedKernel>(*resolved_case.kernel)
            );
        }

        CUfunction const function = prepared.at(resolved_case.kernel->id)->function();
        int function_threads = 0;
        cuda_check(
            cuFuncGetAttribute(&function_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function)
        );

        if (checked_product(resolved_case.run.block.x, resolved_case.run.block.y.value_or(1))
            > static_cast<unsigned>(function_threads)) {
            throw std::runtime_error(
                fmt::format("Case exceeds cubin thread limit: {}", resolved_case.name)
            );
        }

        cuda_check(cuFuncSetAttribute(
            function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(resolved_case.dynamic_smem_bytes)
        ));
    }

    return prepared;
}
}  // namespace

auto execute(
    ExecutionPlan const &plan,
    BatchSpec const &batch,
    ArtifactCatalog const &catalog,
    DeviceInfo const &device,
    unsigned input_threads,
    Reporter &reporter,
    bool no_color
) -> int {
    // clang-format off
    reporter.write({
        { "type", "run" },
        { "schema_version", 2 },
        { "sampling_protocol", "cooldown_warmup_per_repeat" },
        { "seed", batch.seed },
        { "artifact", catalog.metadata() },
        { "device", device.metadata },
        { "input", {
            { "max_elements", plan.max_input_elements },
            { "algorithm", "splitmix64-index-u53-range-200" },
        } },
        { "case_count", plan.cases.size() },
    });
    // clang-format on

    validate_device(plan, device.device);
    Session const session(device.device);
    Stream const stream;

    std::size_t const bytes = plan.max_input_elements * sizeof(double);
    std::size_t free = 0;
    std::size_t total = 0;
    cuda_check(cuMemGetInfo(&free, &total));

    bool const checks = std::any_of(
        plan.cases.begin(),
        plan.cases.end(),
        [](ResolvedCase const &resolved_case) -> bool { return resolved_case.run.check; }
    );
    if (bytes > free / (checks ? 5 : 3)) {
        throw std::runtime_error(
            "Planned workspace including validation scratch exceeds available device memory"
        );
    }

    std::vector<double> const input =
        generate_input(batch.seed, plan.max_input_elements, input_threads);

    Workspace const workspace(bytes);
    cuda_check(cuMemcpyHtoD(workspace.baseline.get(), input.data(), bytes));

    PreparedCache const prepared = prepare_kernels(plan);

    Event const start;
    Event const stop;
    ReferenceCache references;

    unsigned failures = 0;
    std::size_t completed = 0;

    for (ResolvedCase const &resolved_case : plan.cases) {
        CUfunction const function = prepared.at(resolved_case.kernel->id)->function();

        cuda_check(cuFuncSetAttribute(
            function,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(resolved_case.dynamic_smem_bytes)
        ));

        CaseResult result;

        if (resolved_case.run.check) {
            result.validation = validate_case(resolved_case, device, workspace, input, references);
            failures += result.validation["status"] == "failed" ? 1 : 0;
        }

        BandExecution const
            execution(resolved_case, function, workspace.work_a.get(), workspace.work_b.get());
        result.samples =
            measure_case(resolved_case, execution, workspace, stream.get(), start, stop);

        report_case(resolved_case, result, ++completed, plan.cases.size(), reporter, no_color);
    }

    // clang-format off
    reporter.write({
        { "type", "end" },
        { "status", "completed" },
        { "case_count", completed },
        { "validation_failures", failures },
    });
    // clang-format on

    return failures ? 1 : 0;
}
}  // namespace stencil
