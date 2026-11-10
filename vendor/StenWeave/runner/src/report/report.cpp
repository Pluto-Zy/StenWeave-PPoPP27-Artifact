#include "report/report.hpp"

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "execution/plan.hpp"

#include <fmt/base.h>
#include <fmt/os.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace stencil {
auto throughput(std::uint64_t updates, double time_ms) -> double {
    if (!std::isfinite(time_ms) || time_ms <= 0) {
        throw std::runtime_error("Invalid CUDA Event duration");
    }

    return static_cast<double>(updates) / time_ms / 1e6;
}

auto statistics(std::vector<double> values) -> nlohmann::json {
    if (values.empty()) {
        throw std::runtime_error("Cannot summarize zero samples");
    }

    std::sort(values.begin(), values.end());
    std::size_t const count = values.size();
    double const mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(count);

    double variance = 0;
    for (double const value : values) {
        variance += (value - mean) * (value - mean);
    }

    // clang-format off
    return {
        { "min", values.front() },
        { "max", values.back() },
        { "mean", mean },
        { "median", (values[(count - 1) / 2] + values[count / 2]) / 2 },
        { "stddev", std::sqrt(variance / static_cast<double>(count)) },
    };
    // clang-format on
}

auto case_json(ResolvedCase const &resolved_case, CaseResult const &result) -> nlohmann::json {
    auto const cross = [](CrossSection const &value) -> nlohmann::json {
        // clang-format off
        nlohmann::json json { { "x", value.x } };
        // clang-format on
        if (value.y) {
            json["y"] = *value.y;
        }

        return json;
    };

    RunSpec const &run = resolved_case.run;
    // clang-format off
    nlohmann::json padding { { "x", run.padding.x }, { "y", run.padding.y } };
    // clang-format on
    if (resolved_case.kernel->dimensions == 3) {
        padding["z"] = run.padding.z.value_or(0);
    }

    nlohmann::json samples = nlohmann::json::array();
    std::vector<double> times, rates;
    unsigned const measurement_runs =
        resolved_case.run.measurement_runs.value_or(resolved_case.run.repeat_runs);
    std::size_t const measurement_count =
        std::min<std::size_t>(measurement_runs, result.samples.size());
    std::size_t const measurement_begin = result.samples.size() - measurement_count;

    for (std::size_t i = 0; i < result.samples.size(); ++i) {
        SampleResult const &sample = result.samples[i];
        // clang-format off
        samples.push_back({
            { "time_ms", sample.time_ms },
            { "gstencils_per_s", sample.gstencils_per_s },
        });
        // clang-format on

        if (i < measurement_begin) {
            continue;
        }

        times.push_back(sample.time_ms);
        rates.push_back(sample.gstencils_per_s);
    }

    nlohmann::json bands = nlohmann::json::array();
    for (ResolvedBand const &band : resolved_case.bands) {
        // clang-format off
        nlohmann::json group {
            { "stream", band.stream },
            { "grid", cross(band.grid) },
            { "actual_overlap", cross(band.overlap) },
            { "execution_extent", cross(band.execution_extent) },
        };
        // clang-format on
        if (!bands.empty()) {
            nlohmann::json previous = bands.back();
            previous.erase("repeat");

            if (previous == group) {
                bands.back()["repeat"] = bands.back()["repeat"].get<unsigned>() + 1;
                continue;
            }
        }

        group["repeat"] = 1;
        bands.push_back(group);
    }

    nlohmann::json const time_summary = statistics(times);
    nlohmann::json const rate_summary = statistics(rates);
    std::string const statistic =
        std::string(measurement_statistic_name(run.measurement_statistic));

    // clang-format off
    return {
        { "type", "case" },
        { "name", resolved_case.name },
        { "kernel", kernel_json(*resolved_case.kernel) },
        { "run", {
            { "block", cross(run.block) },
            { "bands", bands },
            { "band_count", resolved_case.band_count },
            { "time_steps", run.time_steps },
            { "warmup_runs", run.warmup_runs },
            { "repeat_runs", run.repeat_runs },
            { "measurement_runs", measurement_runs },
            { "measurement_statistic", statistic },
            { "conditioning_runs", run.repeat_runs - measurement_runs },
            { "check", run.check },
            { "tolerance", run.tolerance },
            { "cooldown_ms", run.cooldown_ms },
        } },
        { "domain", {
            { "width", resolved_case.domain.width },
            { "depth", resolved_case.domain.depth },
            { "height", resolved_case.domain.height },
            { "elements", resolved_case.domain.elements },
        } },
        { "padding", padding },
        { "execution_domain", {
            { "width", resolved_case.execution_domain.width },
            { "depth", resolved_case.execution_domain.depth },
            { "height", resolved_case.execution_domain.height },
            { "elements", resolved_case.execution_domain.elements },
        } },
        { "workspace_elements", resolved_case.workspace_elements },
        { "dynamic_smem_bytes", resolved_case.dynamic_smem_bytes },
        { "stencil_updates", resolved_case.stencil_updates },
        { "execution_stencil_updates", resolved_case.execution_stencil_updates },
        { "samples", samples },
        { "summary", {
            { "time_ms", time_summary },
            { "gstencils_per_s", rate_summary },
            { "selected", {
                { "statistic", statistic },
                { "time_ms", time_summary[statistic] },
                { "gstencils_per_s", rate_summary[statistic] },
            } },
        } },
        { "validation", result.validation },
    };
    // clang-format on
}

namespace {
auto result_path(std::filesystem::path const &path, bool overwrite) -> std::string {
    if (!overwrite && std::filesystem::exists(path)) {
        throw std::runtime_error("Result file exists; use --overwrite");
    }

    return path.string();
}
}  // namespace

Reporter::Reporter(std::filesystem::path const &path, bool overwrite) :
    file_(result_path(path, overwrite).c_str(), overwrite ? "w" : "wx") { }

void Reporter::write(nlohmann::json const &record) {
    fmt::println(file_.get(), "{}", record.dump());

    if (std::fflush(file_.get()) != 0) {
        throw std::system_error(errno, std::generic_category(), "Cannot flush result record");
    }
}
}  // namespace stencil
