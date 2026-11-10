#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace stencil {
struct CrossSection {
    unsigned x = 1;
    std::optional<unsigned> y;
};

struct KernelSelector {
    std::optional<std::string> id;

    std::optional<std::string> shape;
    std::optional<std::string> backend;
    std::optional<std::string> profile;
    std::optional<unsigned> dimensions;
    std::optional<unsigned> radius;
    std::optional<unsigned> time_block_size;
    std::optional<unsigned> stage;

    std::optional<unsigned> leading_load_count;
    std::optional<unsigned> min_comp_stall;
    std::optional<unsigned> launch_bounds;

    std::optional<CrossSection> minimum_overlap;
    std::optional<CrossSection> group;
};

struct LogicalDomain {
    unsigned x;
    unsigned y;
    std::optional<unsigned> z;
};

struct DomainPadding {
    unsigned x = 0;
    unsigned y = 0;
    std::optional<unsigned> z;
};

struct BandGroup {
    CrossSection grid;
    unsigned repeat = 1;
    std::optional<unsigned> stream;
};

enum class MeasurementStatistic {
    Mean,
    Median,
};

auto measurement_statistic_name(MeasurementStatistic statistic) -> std::string_view;

struct RunSpec {
    CrossSection block;
    std::optional<LogicalDomain> domain;
    DomainPadding padding;
    std::vector<BandGroup> bands;

    unsigned time_steps = 0;

    unsigned warmup_runs = 2;
    unsigned repeat_runs = 10;
    std::optional<unsigned> measurement_runs;
    MeasurementStatistic measurement_statistic = MeasurementStatistic::Median;

    bool check = false;
    double tolerance = 1e-8;
    unsigned cooldown_ms = 0;
};

struct CaseSpec {
    std::string name;
    KernelSelector selector;
    RunSpec run;
};

struct BatchSpec {
    std::uint64_t seed = 0;
    std::vector<CaseSpec> cases;
};

auto parse_batch(std::filesystem::path const &path) -> BatchSpec;
}  // namespace stencil
