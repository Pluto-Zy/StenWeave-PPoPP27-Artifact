#include "execution/plan.hpp"

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "execution/launch.hpp"
#include "stencil/model.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stencil {
auto checked_product(std::uint64_t a, std::uint64_t b) -> std::uint64_t {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::runtime_error("Geometry/update count overflow");
    }

    return a * b;
}

namespace {
void rank(CrossSection const &cross_section, unsigned dimensions, bool nonzero) {
    if ((dimensions == 3) != cross_section.y.has_value()
        || (nonzero && (!cross_section.x || (cross_section.y && !*cross_section.y)))) {
        throw std::runtime_error("Invalid named dimension rank or zero geometry");
    }
}

auto extent(unsigned grid, unsigned block, unsigned group, unsigned overlap) -> unsigned {
    std::uint64_t const tile = checked_product(block, group);
    if (tile <= overlap) {
        throw std::runtime_error("Overlap consumes entire block tile");
    }

    std::uint64_t const full = checked_product(grid, tile);
    std::uint64_t const redundant = checked_product(overlap, grid - 1);
    if (full <= redundant || full - redundant > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("Invalid domain extent");
    }

    return static_cast<unsigned>(full - redundant);
}

auto padded_extent(unsigned extent, unsigned padding) -> unsigned {
    if (padding > std::numeric_limits<unsigned>::max() - extent) {
        throw std::runtime_error("Padding extent overflow");
    }

    return extent + padding;
}

auto actual_overlap(
    unsigned domain,
    unsigned grid,
    unsigned footprint,
    unsigned minimum,
    unsigned group
) -> unsigned {
    if (grid == 1) {
        if (domain != footprint) {
            throw std::runtime_error("Single CTA domain must equal its footprint");
        }

        return minimum;
    }

    std::uint64_t const full = checked_product(grid, footprint);
    if (domain > full || (full - domain) % (grid - 1)) {
        throw std::runtime_error("Non-integral actual overlap");
    }

    std::uint64_t const overlap = (full - domain) / (grid - 1);
    if (overlap < minimum || overlap >= footprint || overlap % (2 * group)) {
        throw std::runtime_error("Actual overlap violates minimum/footprint/ownership alignment");
    }

    return static_cast<unsigned>(overlap);
}
}  // namespace

auto make_plan(BatchSpec const &batch, ArtifactCatalog const &catalog, std::string const &arch)
    -> ExecutionPlan {
    ExecutionPlan plan;
    std::set<std::string> names;

    for (CaseSpec const &case_spec : batch.cases) {
        if (case_spec.name.empty() || !names.insert(case_spec.name).second) {
            throw std::runtime_error(fmt::format("Duplicate/empty case name: {}", case_spec.name));
        }

        ArtifactKernel const &kernel = catalog.select(case_spec.selector, arch, case_spec.name);
        RunSpec run = case_spec.run;
        unsigned const measurement_runs = run.measurement_runs.value_or(run.repeat_runs);
        run.measurement_runs = measurement_runs;

        if ((kernel.dimensions != 2 && kernel.dimensions != 3) || !kernel.radius
            || kernel.radius > 3
            || (kernel.shape == Shape::Box && kernel.dimensions == 3 && kernel.radius != 1)
            || !kernel.time_block_size || (kernel.arch != "sm80" && kernel.arch != "sm90")) {
            throw std::runtime_error("Unsupported artifact geometry/architecture");
        }

        if ((kernel.profile.name == "default" && kernel.profile.stage != 0)
            || (kernel.profile.name != "default"
                && (kernel.profile.name != "breakdown" || kernel.profile.stage < 1
                    || kernel.profile.stage > 4 || kernel.backend != Backend::CUDA
                    || kernel.dimensions != 2))) {
            throw std::runtime_error("Unsupported artifact profile ABI");
        }

        rank(run.block, kernel.dimensions, true);
        if (kernel.backend == Backend::CUDA
            && checked_product(run.block.x, run.block.y.value_or(1)) != kernel.launch_bounds) {
            throw std::runtime_error(
                fmt::format(
                    "case {}: block thread count must equal CUDA launch_bounds ({})",
                    case_spec.name,
                    kernel.launch_bounds.value_or(0)
                )
            );
        }

        rank(kernel.minimum_overlap, kernel.dimensions, false);
        if (kernel.backend == Backend::SASS && !kernel.group) {
            throw std::runtime_error("SASS artifact requires group");
        }
        if (kernel.backend == Backend::CUDA && kernel.group) {
            throw std::runtime_error("CUDA artifact cannot have group");
        }

        if (kernel.group) {
            rank(*kernel.group, kernel.dimensions, true);
        }

        if (run.bands.empty() || !run.time_steps || !run.repeat_runs || !measurement_runs
            || measurement_runs > run.repeat_runs || run.time_steps % kernel.time_block_size
            || !std::isfinite(run.tolerance) || run.tolerance < 0) {
            throw std::runtime_error(
                fmt::format(
                    "Invalid runtime bands/time_steps/repeats/measurement/tolerance: {}",
                    case_spec.name
                )
            );
        }

        std::uint64_t const halo = checked_product(2, kernel.pipeline.stream_halo);
        if (kernel.minimum_overlap.x < halo
            || (kernel.minimum_overlap.y && *kernel.minimum_overlap.y < halo)) {
            throw std::runtime_error("Artifact overlap smaller than temporal halo");
        }

        std::uint64_t band_count = 0;
        bool const explicit_stream = run.bands.front().stream.has_value();
        for (BandGroup const &group : run.bands) {
            rank(group.grid, kernel.dimensions, true);
            if (!group.repeat || group.stream.has_value() != explicit_stream) {
                throw std::runtime_error(
                    "Band repeat must be positive; partial stream omission is invalid"
                );
            }

            band_count += group.repeat;
            if (band_count > std::numeric_limits<unsigned>::max()) {
                throw std::runtime_error("Band count overflow");
            }
        }

        if (!explicit_stream && !run.domain) {
            throw std::runtime_error("Domain omitted requires stream in every band group");
        }

        if (kernel.dimensions == 2 && run.padding.z) {
            throw std::runtime_error("2D padding accepts only x and y");
        }

        bool const padded = run.padding.x || run.padding.y || run.padding.z.value_or(0);
        if (padded && !run.domain) {
            throw std::runtime_error("Nonzero padding requires an explicit domain");
        }

        Domain domain { 0, 1, 0, 0 };
        Domain execution_domain { 0, 1, 0, 0 };
        if (run.domain) {
            LogicalDomain const &requested = *run.domain;
            if ((kernel.dimensions == 3) != requested.z.has_value() || !requested.x || !requested.y
                || (requested.z && !*requested.z)) {
                throw std::runtime_error("Invalid logical domain rank/extent");
            }

            domain = {
                requested.x,
                kernel.dimensions == 3 ? requested.y : 1,
                requested.z.value_or(requested.y),
                0,
            };

            execution_domain = {
                padded_extent(domain.width, run.padding.x),
                kernel.dimensions == 3 ? padded_extent(domain.depth, run.padding.y) : 1,
                padded_extent(
                    domain.height,
                    kernel.dimensions == 3 ? run.padding.z.value_or(0) : run.padding.y
                ),
                0,
            };

            if (!explicit_stream && execution_domain.height % band_count) {
                throw std::runtime_error("Domain streaming extent is not divisible by band count");
            }
        }

        unsigned const group_x = kernel.group ? kernel.group->x : 1;
        unsigned const group_y = kernel.group ? kernel.group->y.value_or(1) : 1;
        if (kernel.minimum_overlap.x % (2 * group_x)
            || (kernel.minimum_overlap.y && *kernel.minimum_overlap.y % (2 * group_y))) {
            throw std::runtime_error("Artifact minimum overlap violates ownership alignment");
        }

        unsigned const footprint_x = extent(1, run.block.x, group_x, kernel.minimum_overlap.x);
        unsigned const footprint_y = kernel.dimensions == 3
            ? extent(1, *run.block.y, group_y, *kernel.minimum_overlap.y)
            : 1;

        PipelineContract const &pipeline = kernel.pipeline;
        unsigned const head_rows = pipeline.head_rows;
        unsigned const tail_rows = pipeline.tail_rows;
        unsigned const unroll = pipeline.planes_per_iteration;

        struct BandGeometry {
            CrossSection grid;
            CrossSection overlap;
            unsigned stream;
            unsigned stream_offset;
            CrossSection execution_extent;
        };

        std::vector<BandGeometry> geometries;
        std::uint64_t stream_sum = 0;
        for (BandGroup const &group : run.bands) {
            unsigned const stream = explicit_stream
                ? *group.stream
                : static_cast<unsigned>(execution_domain.height / band_count);
            if (stream < head_rows + tail_rows) {
                throw std::runtime_error("Stream too small for kernel temporal pipeline");
            }

            CrossSection overlap = kernel.minimum_overlap;
            unsigned const width = extent(group.grid.x, run.block.x, group_x, overlap.x);
            unsigned const depth = kernel.dimensions == 3
                ? extent(*group.grid.y, *run.block.y, group_y, *overlap.y)
                : 1;

            if (!run.domain) {
                if (geometries.empty()) {
                    execution_domain.width = width;
                    execution_domain.depth = depth;
                } else if (execution_domain.width != width || execution_domain.depth != depth) {
                    throw std::runtime_error(
                        "Bands imply inconsistent cross sections; explicit domain required"
                    );
                }
            } else if (kernel.backend == Backend::SASS) {
                overlap.x = actual_overlap(
                    execution_domain.width,
                    group.grid.x,
                    footprint_x,
                    kernel.minimum_overlap.x,
                    group_x
                );

                if (kernel.dimensions == 3) {
                    overlap.y = actual_overlap(
                        execution_domain.depth,
                        *group.grid.y,
                        footprint_y,
                        *kernel.minimum_overlap.y,
                        group_y
                    );
                }
            }

            CrossSection const execution_extent {
                width,
                kernel.dimensions == 3 ? std::optional<unsigned>(depth) : std::nullopt,
            };

            std::uint64_t const group_stream = checked_product(group.repeat, stream);
            if (group_stream > std::numeric_limits<unsigned>::max() - stream_sum) {
                throw std::runtime_error("Streaming extent overflow");
            }

            for (unsigned i = 0; i < group.repeat; ++i) {
                BandGeometry const band {
                    group.grid,
                    overlap,
                    stream,
                    static_cast<unsigned>(stream_sum + checked_product(i, stream)),
                    execution_extent,
                };
                geometries.push_back(band);
            }

            stream_sum += checked_product(group.repeat, stream);
            if (stream_sum > std::numeric_limits<unsigned>::max()) {
                throw std::runtime_error("Streaming extent overflow");
            }
        }

        if (run.domain && stream_sum != execution_domain.height) {
            throw std::runtime_error("Band stream sum differs from execution domain");
        }

        execution_domain.height = static_cast<unsigned>(stream_sum);

        std::uint64_t const wide_plane_stride =
            checked_product(execution_domain.width, execution_domain.depth);
        if (kernel.backend == Backend::SASS
            && wide_plane_stride > std::numeric_limits<unsigned>::max()) {
            throw std::runtime_error("Plane stride exceeds kernel ABI range");
        }

        auto const plane_stride = static_cast<unsigned>(wide_plane_stride);

        std::vector<ResolvedBand> bands;
        std::size_t workspace_elements = 0;
        for (BandGeometry const &band : geometries) {
            if (kernel.backend == Backend::CUDA) {
                std::uint64_t const byte_offset =
                    checked_product(workspace_elements, sizeof(double));
                std::uint64_t const strip_elements = checked_product(
                    checked_product(band.execution_extent.x, band.execution_extent.y.value_or(1)),
                    band.stream
                );
                if (strip_elements > std::numeric_limits<std::size_t>::max() / (3 * sizeof(double))
                        - workspace_elements) {
                    throw std::runtime_error("CUDA strip workspace overflow");
                }

                bands.push_back(
                    {
                        band.grid,
                        band.overlap,
                        band.stream,
                        band.stream_offset,
                        band.execution_extent,
                        byte_offset,
                        launch::cuda::Params { band.stream },
                    }
                );
                workspace_elements += strip_elements;
                continue;
            }

            CrossSection const execution_extent {
                execution_domain.width,
                kernel.dimensions == 3 ? std::optional<unsigned>(execution_domain.depth)
                                       : std::nullopt,
            };

            unsigned const end = band.stream_offset + band.stream;
            unsigned const input_begin =
                band.stream_offset ? band.stream_offset - pipeline.stream_halo : 0;
            unsigned const input_end =
                end == execution_domain.height ? end : end + pipeline.stream_halo;

            unsigned const store_head = band.stream_offset == 0;
            unsigned const store_tail = end == execution_domain.height;
            unsigned const extension = store_head ? 0 : pipeline.halo_head_extension;
            if (input_end - input_begin < head_rows + tail_rows + extension) {
                throw std::runtime_error(
                    "Band stream is too short for static halo head and pipeline"
                );
            }

            unsigned const rows = input_end - input_begin - head_rows - tail_rows - extension;
            unsigned const address_group = kernel.dimensions == 2 ? group_x : 1;
            std::uint64_t const byte_offset =
                checked_product(checked_product(input_begin, plane_stride), sizeof(double));

            launch::sass::Params const params {
                execution_domain.width / address_group,
                plane_stride,
                (footprint_x - band.overlap.x) / address_group,
                kernel.dimensions == 3 ? footprint_y - *band.overlap.y : 0,
                band.overlap.x / (2 * group_x),
                run.block.x - band.overlap.x / (2 * group_x),
                band.overlap.y.value_or(0) / (2 * group_y),
                run.block.y.value_or(1) - band.overlap.y.value_or(0) / (2 * group_y),
                rows / unroll,
                rows % unroll,
                store_head,
                store_tail,
            };

            bands.push_back(
                {
                    band.grid,
                    band.overlap,
                    band.stream,
                    band.stream_offset,
                    execution_extent,
                    byte_offset,
                    params,
                }
            );
        }

        std::uint64_t const count = checked_product(
            checked_product(execution_domain.width, execution_domain.depth),
            execution_domain.height
        );
        if (count > std::numeric_limits<std::size_t>::max()
            || (kernel.backend == Backend::SASS
                && count > std::numeric_limits<std::size_t>::max() / (3 * sizeof(double)))) {
            throw std::runtime_error("Workspace capacity overflow");
        }

        execution_domain.elements = static_cast<std::size_t>(count);
        if (kernel.backend == Backend::SASS) {
            workspace_elements = execution_domain.elements;
        }

        if (!run.domain) {
            domain = execution_domain;
        } else {
            domain.elements = static_cast<std::size_t>(
                checked_product(checked_product(domain.width, domain.depth), domain.height)
            );
        }

        if (coefficients(kernel).size() != kernel.coefficient_count) {
            throw std::runtime_error("Manifest coefficient count mismatch");
        }

        plan.cases.push_back(
            {
                case_spec.name,
                &kernel,
                run,
                domain,
                execution_domain,
                std::move(bands),
                workspace_elements,
                static_cast<unsigned>(band_count),
                shared_memory(kernel, run.block),
                checked_product(domain.elements, run.time_steps),
                checked_product(execution_domain.elements, run.time_steps),
            }
        );
        plan.max_input_elements = std::max(plan.max_input_elements, workspace_elements);
    }

    return plan;
}
}  // namespace stencil
