#pragma once

#include "execution/plan.hpp"

#include <fmt/os.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace stencil {
struct SampleResult {
    double time_ms;
    double gstencils_per_s;
};

struct CaseResult {
    std::vector<SampleResult> samples;

    // clang-format off
    nlohmann::json validation { { "status", "not_requested" } };
    // clang-format on
};

auto throughput(std::uint64_t updates, double time_ms) -> double;

auto statistics(std::vector<double> values) -> nlohmann::json;

auto case_json(ResolvedCase const &resolved_case, CaseResult const &result) -> nlohmann::json;

class Reporter {
public:
    Reporter(std::filesystem::path const &path, bool overwrite);

    void write(nlohmann::json const &record);

private:
    fmt::buffered_file file_;
};
}  // namespace stencil
