#pragma once

#include "artifact/catalog.hpp"
#include "execution/plan.hpp"

#include <vector>

namespace stencil {
auto coefficients(ArtifactKernel const &kernel) -> std::vector<double>;

auto shared_memory(ArtifactKernel const &kernel, CrossSection block) -> unsigned;

auto compute_reference(
    ArtifactKernel const &kernel,
    Domain const &domain,
    unsigned time_steps,
    std::vector<double> const &input
) -> std::vector<double>;

auto compute_reference(ResolvedCase const &resolved_case, std::vector<double> const &input)
    -> std::vector<double>;
}  // namespace stencil
