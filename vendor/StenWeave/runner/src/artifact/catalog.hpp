#pragma once

#include "config/batch.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace stencil {
enum class Shape {
    Star,
    Box
};

enum class Backend {
    CUDA,
    SASS
};

struct Profile {
    std::string name;
    unsigned stage = 0;
};

struct SASSTuning {
    unsigned leading_load_count;
    unsigned min_comp_stall;
};

struct PipelineContract {
    unsigned head_rows;
    unsigned tail_rows;
    unsigned planes_per_iteration;
    unsigned stream_halo;
    unsigned halo_head_extension;
};

struct ArtifactKernel {
    std::string id;
    Shape shape;
    unsigned dimensions;
    Backend backend;
    std::string arch;

    unsigned radius;
    unsigned time_block_size;
    CrossSection minimum_overlap;
    std::optional<CrossSection> group;

    Profile profile;
    std::optional<SASSTuning> sass;
    std::optional<unsigned> launch_bounds;
    std::string dtype;

    unsigned abi_version;
    PipelineContract pipeline;
    std::string entry_symbol;
    std::string coefficient_symbol;
    unsigned coefficient_count;
    std::filesystem::path cubin_path;
    std::string cubin_sha256;
};

auto shape_name(Shape shape) -> std::string;

auto backend_name(Backend backend) -> std::string;

auto kernel_json(ArtifactKernel const &kernel) -> nlohmann::json;

class ArtifactCatalog {
public:
    explicit ArtifactCatalog(std::filesystem::path root);

    auto select(
        KernelSelector const &selector,
        std::string const &arch,
        std::string const &name
    ) const -> ArtifactKernel const &;

    auto records() const -> std::vector<ArtifactKernel> const & {
        return records_;
    }

    auto metadata() const -> nlohmann::json const & {
        return metadata_;
    }

private:
    std::vector<ArtifactKernel> records_;
    nlohmann::json metadata_;
};
}  // namespace stencil
