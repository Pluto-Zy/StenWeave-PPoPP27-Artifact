#pragma once

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "cuda/driver.hpp"
#include "execution/plan.hpp"
#include "report/report.hpp"

namespace stencil {
auto execute(
    ExecutionPlan const &plan,
    BatchSpec const &batch,
    ArtifactCatalog const &catalog,
    DeviceInfo const &device,
    unsigned input_threads,
    Reporter &reporter,
    bool no_color
) -> int;
}  // namespace stencil
