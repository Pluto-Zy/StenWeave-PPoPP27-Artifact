#include "stencil/model.hpp"

#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "execution/plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stencil {
auto coefficients(ArtifactKernel const &kernel) -> std::vector<double> {
    unsigned const radius = kernel.radius;
    std::vector<double> result;

    if (kernel.shape == Shape::Star) {
        double const divisor = 20.0 + radius * kernel.dimensions * 2 * 20.0;
        result.push_back(20.567 / divisor);

        for (unsigned i = 0; i < radius; ++i) {
            for (double const value : { 18.123, 22.901, 19.345, 21.789 }) {
                result.push_back(value / divisor);
            }

            if (kernel.dimensions == 3) {
                result.push_back(21.123 / divisor);
                result.push_back(18.789 / divisor);
            }
        }
    } else {
        unsigned const side = 2 * radius + 1;
        unsigned const count = side * side * (kernel.dimensions == 3 ? side : 1);
        double const values[] { 20.567, 18.123, 22.901, 19.345, 21.789, 17.234, 23.456 };

        for (unsigned i = 0; i < count; ++i) {
            result.push_back(values[i % side] / (count * 20.0));
        }
    }

    return result;
}

auto shared_memory(ArtifactKernel const &kernel, CrossSection block) -> unsigned {
    std::uint64_t const gx = kernel.group ? kernel.group->x : 1;
    std::uint64_t const gy = kernel.group ? kernel.group->y.value_or(1) : 1;
    unsigned const radius = kernel.radius;

    std::uint64_t const interior_x = gx > 2 * radius ? gx - 2 * radius : 0;
    std::uint64_t const interior_y = gy > 2 * radius ? gy - 2 * radius : 0;
    std::uint64_t const slot = kernel.dimensions == 2 ? std::min<std::uint64_t>(gx, 2 * radius)
                                                      : gx * gy - interior_x * interior_y;
    std::uint64_t chunk = checked_product(slot, kernel.time_block_size);
    if (chunk % 2 == 0) {
        ++chunk;
    }

    std::uint64_t const halo = (radius + gx - 1) / gx;
    std::uint64_t slots = 0;
    if (kernel.dimensions == 2) {
        slots = 2ULL * block.x + halo * 3;
        if (kernel.shape == Shape::Box && kernel.profile.name == "breakdown") {
            slots = 2ULL * block.x + radius + 2ULL * radius * (2 * radius + 1);
        }
    } else {
        slots = 2 * (block.y.value_or(1) * (halo + block.x)) + halo;
        if (kernel.shape == Shape::Box) {
            slots += 2 * (halo + 1);
        }
    }

    std::uint64_t const bytes = checked_product(checked_product(slots, chunk), sizeof(double));
    if (bytes > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("Shared memory size overflow");
    }

    return static_cast<unsigned>(bytes);
}

auto compute_reference(
    ArtifactKernel const &kernel,
    Domain const &domain,
    unsigned time_steps,
    std::vector<double> const &input
) -> std::vector<double> {
    std::vector<double> const coefficient_values = coefficients(kernel);
    auto const input_end = input.begin() + static_cast<std::ptrdiff_t>(domain.elements);
    std::vector<double> current(input.begin(), input_end);
    std::vector<double> next(domain.elements);

    int const radius = static_cast<int>(kernel.radius);
    int const side = 2 * radius + 1;

    for (unsigned step = 0; step < time_steps; ++step) {
        auto const at = [&](std::int64_t x, std::int64_t y, std::int64_t z) -> double {
            if (x < 0 || y < 0 || z < 0 || x >= domain.width || y >= domain.depth
                || z >= domain.height) {
                return 0;
            }

            return current[(z * domain.depth + y) * domain.width + x];
        };

        for (unsigned z = 0; z < domain.height; ++z) {
            for (unsigned y = 0; y < domain.depth; ++y) {
                for (unsigned x = 0; x < domain.width; ++x) {
                    double value = 0;

                    if (kernel.shape == Shape::Star) {
                        unsigned const stride = 2 * kernel.dimensions;

                        for (int offset = radius; offset > 0; --offset) {
                            value += coefficient_values[offset * stride - stride + 1]
                                * at(x, y, static_cast<std::int64_t>(z) - offset);
                        }

                        value += coefficient_values[0] * at(x, y, z);
                        value += coefficient_values[2] * at(x, y, z + 1);

                        for (int i = 0; i < radius; ++i) {
                            int const offset = kernel.dimensions == 2 ? radius - i : i + 1;

                            value += coefficient_values[(offset - 1) * stride + 3]
                                * at(static_cast<std::int64_t>(x) - offset, y, z);
                            value += coefficient_values[(offset - 1) * stride + 4]
                                * at(x + offset, y, z);

                            if (kernel.dimensions == 3) {
                                value += coefficient_values[(offset - 1) * stride + 5]
                                    * at(x, static_cast<std::int64_t>(y) - offset, z);
                                value += coefficient_values[(offset - 1) * stride + 6]
                                    * at(x, y + offset, z);
                            }
                        }

                        for (int offset = 2; offset <= radius; ++offset) {
                            value += coefficient_values[(offset - 1) * stride + 2]
                                * at(x, y, z + offset);
                        }
                    } else {
                        for (int dz = -radius; dz <= radius; ++dz) {
                            auto const add = [&](int dx, int dy) {
                                unsigned const index = kernel.dimensions == 2
                                    ? dz + radius
                                    : (dy + radius) * side * side + (dx + radius) * side + dz
                                        + radius;

                                value += at(static_cast<std::int64_t>(x) + dx,
                                            static_cast<std::int64_t>(y) + dy,
                                            static_cast<std::int64_t>(z) + dz)
                                    * coefficient_values[index];
                            };

                            add(0, 0);
                            if (kernel.dimensions == 2) {
                                for (int offset = 1; offset <= radius; ++offset) {
                                    add(-offset, 0);
                                    add(offset, 0);
                                }
                            } else {
                                for (int pair_index = 0; pair_index < side * side / 2;
                                     ++pair_index) {
                                    int const dx = pair_index % side - radius;
                                    int const dy = pair_index / side - radius;

                                    add(dx, dy);
                                    add(-dx, -dy);
                                }
                            }
                        }
                    }

                    next[(static_cast<std::size_t>(z) * domain.depth + y) * domain.width + x] =
                        value;
                }
            }
        }

        current.swap(next);
    }

    return current;
}

auto compute_reference(ResolvedCase const &resolved_case, std::vector<double> const &input)
    -> std::vector<double> {
    return compute_reference(
        *resolved_case.kernel,
        resolved_case.execution_domain,
        resolved_case.run.time_steps,
        input
    );
}
}  // namespace stencil
