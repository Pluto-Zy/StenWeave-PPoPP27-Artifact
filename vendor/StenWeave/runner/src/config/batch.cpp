#include "config/batch.hpp"

#include <fmt/format.h>

#include <toml++/toml.hpp>

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace stencil {
auto measurement_statistic_name(MeasurementStatistic statistic) -> std::string_view {
    switch (statistic) {
    case MeasurementStatistic::Mean:
        return "mean";
    case MeasurementStatistic::Median:
        return "median";
    }

    throw std::runtime_error("Unknown measurement statistic");
}

namespace {
auto require_table(toml::node const &node) -> toml::table const & {
    toml::table const *const result = node.as_table();
    if (!result) {
        throw std::runtime_error("Expected TOML table");
    }

    return *result;
}

void validate_fields(toml::table const &table, std::set<std::string> const &allowed) {
    for (auto const &[key, value] : table) {
        if (!allowed.count(std::string(key.str()))) {
            throw std::runtime_error(fmt::format("Unknown TOML field: {}", key.str()));
        }
    }
}

template <class T>
auto value(toml::node const &node) -> T {
    std::optional<T> const result = node.value<T>();
    if (!result) {
        throw std::runtime_error("Invalid TOML value type");
    }

    return *result;
}

auto measurement_statistic(toml::node const &node) -> MeasurementStatistic {
    std::string const name = value<std::string>(node);
    if (name == "mean") {
        return MeasurementStatistic::Mean;
    }

    if (name == "median") {
        return MeasurementStatistic::Median;
    }

    throw std::runtime_error(
        fmt::format("measurement_statistic must be 'mean' or 'median', got '{}'", name)
    );
}

auto number(toml::node const &node) -> unsigned {
    std::int64_t const number_value = value<std::int64_t>(node);
    if (number_value < 0
        || static_cast<std::uint64_t>(number_value) > std::numeric_limits<unsigned>::max()) {
        throw std::runtime_error("Integer outside unsigned runtime range");
    }

    return static_cast<unsigned>(number_value);
}

auto cross(toml::node const &node) -> CrossSection {
    toml::table const &table = require_table(node);

    validate_fields(table, { "x", "y" });
    if (!table.contains("x")) {
        throw std::runtime_error("Named dimensions require x");
    }

    CrossSection result { number(*table.get("x")), std::nullopt };
    if (table.contains("y")) {
        result.y = number(*table.get("y"));
    }

    return result;
}

void runtime(toml::table const &table, RunSpec &run, std::set<std::string> &present) {
    validate_fields(
        table,
        {
            "block",
            "domain",
            "padding",
            "bands",
            "time_steps",
            "warmup_runs",
            "repeat_runs",
            "measurement_runs",
            "measurement_statistic",
            "check",
            "tolerance",
            "cooldown_ms",
        }
    );

    for (auto const &[key, node] : table) {
        std::string const field_name = std::string(key.str());
        present.insert(field_name);

        if (field_name == "block") {
            run.block = cross(node);
        } else if (field_name == "domain") {
            toml::table const &domain = require_table(node);
            validate_fields(domain, { "x", "y", "z" });
            if (!domain.contains("x") || !domain.contains("y")) {
                throw std::runtime_error("Domain requires x and y");
            }

            run.domain = LogicalDomain {
                number(*domain.get("x")),
                number(*domain.get("y")),
                domain.contains("z") ? std::optional<unsigned>(number(*domain.get("z")))
                                     : std::nullopt,
            };
        } else if (field_name == "padding") {
            toml::table const &padding = require_table(node);
            validate_fields(padding, { "x", "y", "z" });

            auto const component = [&](char const *axis) -> unsigned {
                toml::node const *const extent = padding.get(axis);
                if (!extent) {
                    return 0;
                }

                if (!extent->is_integer()) {
                    throw std::runtime_error("Padding components must be nonnegative integers");
                }

                return number(*extent);
            };

            run.padding = DomainPadding {
                component("x"),
                component("y"),
                padding.contains("z") ? std::optional<unsigned>(component("z")) : std::nullopt,
            };
        } else if (field_name == "bands") {
            toml::array const *const groups = node.as_array();
            if (!groups || groups->empty()) {
                throw std::runtime_error("bands requires a nonempty array");
            }

            run.bands.clear();
            for (toml::node const &entry : *groups) {
                toml::table const &group = require_table(entry);
                validate_fields(group, { "repeat", "stream", "grid" });
                if (!group.contains("grid")) {
                    throw std::runtime_error("Band group requires grid");
                }

                BandGroup band { cross(*group.get("grid")), 1, std::nullopt };
                if (group.contains("repeat")) {
                    band.repeat = number(*group.get("repeat"));
                }

                if (group.contains("stream")) {
                    band.stream = number(*group.get("stream"));
                }

                run.bands.push_back(band);
            }
        } else if (field_name == "check") {
            run.check = value<bool>(node);
        } else if (field_name == "tolerance") {
            run.tolerance = value<double>(node);
        } else if (field_name == "measurement_statistic") {
            run.measurement_statistic = measurement_statistic(node);
        } else {
            unsigned const number_value = number(node);

            if (field_name == "time_steps") {
                run.time_steps = number_value;
            } else if (field_name == "warmup_runs") {
                run.warmup_runs = number_value;
            } else if (field_name == "repeat_runs") {
                run.repeat_runs = number_value;
            } else if (field_name == "measurement_runs") {
                run.measurement_runs = number_value;
            } else if (field_name == "cooldown_ms") {
                run.cooldown_ms = number_value;
            }
        }
    }
}

auto selector(toml::table const &table) -> KernelSelector {
    validate_fields(
        table,
        {
            "id",
            "shape",
            "dimensions",
            "backend",
            "radius",
            "time_block_size",
            "minimum_overlap",
            "group",
            "profile",
            "leading_load_count",
            "min_comp_stall",
            "launch_bounds",
        }
    );

    KernelSelector kernel_selector;
    if (table.contains("id")) {
        if (table.size() != 1) {
            throw std::runtime_error("id cannot be combined with semantic fields");
        }

        kernel_selector.id = value<std::string>(*table.get("id"));

        return kernel_selector;
    }

    for (auto const &[key, node] : table) {
        std::string_view const field_name = key.str();

        if (field_name == "shape") {
            kernel_selector.shape = value<std::string>(node);
        } else if (field_name == "backend") {
            kernel_selector.backend = value<std::string>(node);
        } else if (field_name == "dimensions") {
            kernel_selector.dimensions = number(node);
        } else if (field_name == "radius") {
            kernel_selector.radius = number(node);
        } else if (field_name == "time_block_size") {
            kernel_selector.time_block_size = number(node);
        } else if (field_name == "leading_load_count") {
            kernel_selector.leading_load_count = number(node);
        } else if (field_name == "min_comp_stall") {
            kernel_selector.min_comp_stall = number(node);
        } else if (field_name == "launch_bounds") {
            kernel_selector.launch_bounds = number(node);
        } else if (field_name == "minimum_overlap") {
            kernel_selector.minimum_overlap = cross(node);
        } else if (field_name == "group") {
            kernel_selector.group = cross(node);
        } else if (field_name == "profile") {
            toml::table const &profile_table = require_table(node);
            validate_fields(profile_table, { "name", "stage" });
            if (profile_table.contains("name")) {
                kernel_selector.profile = value<std::string>(*profile_table.get("name"));
            }

            if (profile_table.contains("stage")) {
                kernel_selector.stage = number(*profile_table.get("stage"));
            }
        }
    }

    return kernel_selector;
}
}  // namespace

auto parse_batch(std::filesystem::path const &path) -> BatchSpec {
    toml::table const table = toml::parse_file(path.string());

    validate_fields(table, { "schema_version", "seed", "defaults", "case" });
    if (!table.contains("schema_version") || number(*table.get("schema_version")) != 2) {
        throw std::runtime_error("Unsupported batch schema_version");
    }

    std::random_device entropy;
    BatchSpec batch {
        static_cast<std::uint64_t>(entropy()) << 32U | static_cast<std::uint64_t>(entropy()),
        {},
    };
    if (table.contains("seed")) {
        std::int64_t const seed = value<std::int64_t>(*table.get("seed"));
        if (seed < 0) {
            throw std::runtime_error("seed must be nonnegative");
        }

        batch.seed = static_cast<std::uint64_t>(seed);
    }

    RunSpec defaults;
    std::set<std::string> default_present;
    if (table.contains("defaults")) {
        toml::table const &defaults_table = require_table(*table.get("defaults"));
        validate_fields(defaults_table, { "run" });
        if (defaults_table.contains("run")) {
            runtime(require_table(*defaults_table.get("run")), defaults, default_present);
        }
    }

    toml::array const *const cases = table["case"].as_array();
    if (!cases || cases->empty()) {
        throw std::runtime_error("Batch requires at least one [[case]]");
    }

    for (toml::node const &node : *cases) {
        toml::table const &case_table = require_table(node);
        validate_fields(case_table, { "name", "kernel", "run" });
        if (!case_table.contains("name") || !case_table.contains("kernel")) {
            throw std::runtime_error("case requires name and kernel");
        }

        CaseSpec spec {
            value<std::string>(*case_table.get("name")),
            selector(require_table(*case_table.get("kernel"))),
            defaults,
        };

        std::set<std::string> present = default_present;
        if (case_table.contains("run")) {
            runtime(require_table(*case_table.get("run")), spec.run, present);
        }

        for (char const *const required : { "block", "bands", "time_steps" }) {
            if (!present.count(required)) {
                throw std::runtime_error(fmt::format("Missing run field: {}", required));
            }
        }

        batch.cases.push_back(std::move(spec));
    }

    return batch;
}
}  // namespace stencil
