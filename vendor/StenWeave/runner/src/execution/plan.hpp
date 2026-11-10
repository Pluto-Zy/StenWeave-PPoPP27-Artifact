#pragma once

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "execution/launch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stencil {
struct Domain {
    unsigned width;
    unsigned depth;
    unsigned height;
    std::size_t elements;
};

struct ResolvedBand {
    CrossSection grid;
    CrossSection overlap;
    unsigned stream;
    unsigned stream_offset;

    CrossSection execution_extent;
    std::size_t byte_offset = 0;
    launch::Params params;
};

struct ResolvedCase {
    std::string name;
    ArtifactKernel const *kernel;
    RunSpec run;

    Domain domain;
    Domain execution_domain;
    std::vector<ResolvedBand> bands;
    std::size_t workspace_elements;
    unsigned band_count;

    unsigned dynamic_smem_bytes;
    std::uint64_t stencil_updates;
    std::uint64_t execution_stencil_updates;
};

struct ExecutionPlan {
    std::vector<ResolvedCase> cases;
    std::size_t max_input_elements = 0;
};

auto make_plan(BatchSpec const &batch, ArtifactCatalog const &catalog, std::string const &arch)
    -> ExecutionPlan;

auto checked_product(std::uint64_t a, std::uint64_t b) -> std::uint64_t;
}  // namespace stencil
