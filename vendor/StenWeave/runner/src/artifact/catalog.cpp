#include "artifact/catalog.hpp"

#include "artifact/hash.hpp"
#include "config/batch.hpp"

#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stencil {
auto shape_name(Shape shape) -> std::string {
    return shape == Shape::Star ? "star" : "box";
}

auto backend_name(Backend backend) -> std::string {
    return backend == Backend::CUDA ? "cuda" : "sass";
}

namespace {
auto cross(nlohmann::json const &json) -> CrossSection {
    CrossSection cross_section { json.at("x").get<unsigned>(), std::nullopt };
    if (json.contains("y")) {
        cross_section.y = json.at("y").get<unsigned>();
    }

    if (json.size() != (cross_section.y ? 2 : 1)) {
        throw std::runtime_error("Malformed manifest cross section");
    }

    return cross_section;
}

auto cross_json(CrossSection const &cross_section) -> nlohmann::json {
    // clang-format off
    nlohmann::json json { { "x", cross_section.x } };
    // clang-format on
    if (cross_section.y) {
        json["y"] = *cross_section.y;
    }

    return json;
}

template <class T>
auto matches(std::optional<T> const &selector, T const &value) -> bool {
    return !selector || *selector == value;
}

auto matches_cross(std::optional<CrossSection> const &selector, CrossSection const &value) -> bool {
    return !selector || (selector->x == value.x && (!selector->y || selector->y == value.y));
}
}  // namespace

auto kernel_json(ArtifactKernel const &kernel) -> nlohmann::json {
    // clang-format off
    nlohmann::json json {
        { "id", kernel.id },
        { "shape", shape_name(kernel.shape) },
        { "dimensions", kernel.dimensions },
        { "backend", backend_name(kernel.backend) },
        { "arch", kernel.arch },
        { "radius", kernel.radius },
        { "time_block_size", kernel.time_block_size },
        { "minimum_overlap", cross_json(kernel.minimum_overlap) },
        { "profile", { { "name", kernel.profile.name } } },
        { "dtype", kernel.dtype },
        { "cubin", {
            { "path", kernel.cubin_path.string() },
            { "sha256", kernel.cubin_sha256 },
        } },
        { "entry_symbol", kernel.entry_symbol },
        { "abi_version", kernel.abi_version },
        { "pipeline", {
            { "head_rows", kernel.pipeline.head_rows },
            { "tail_rows", kernel.pipeline.tail_rows },
            { "planes_per_iteration", kernel.pipeline.planes_per_iteration },
            { "stream_halo", kernel.pipeline.stream_halo },
            { "halo_head_extension", kernel.pipeline.halo_head_extension },
        } },
        { "coefficients", {
            { "symbol", kernel.coefficient_symbol },
            { "count", kernel.coefficient_count },
            { "dtype", "f64" },
        } },
    };
    // clang-format on

    if (kernel.group) {
        json["group"] = cross_json(*kernel.group);
    }

    if (kernel.profile.stage) {
        json["profile"]["stage"] = kernel.profile.stage;
    }

    if (kernel.sass) {
        json["leading_load_count"] = kernel.sass->leading_load_count;
        json["min_comp_stall"] = kernel.sass->min_comp_stall;
    }

    if (kernel.launch_bounds) {
        json["launch_bounds"] = *kernel.launch_bounds;
    }

    return json;
}

ArtifactCatalog::ArtifactCatalog(std::filesystem::path root) {
    root = std::filesystem::absolute(root);

    std::ifstream stream(root / "manifest.json");
    if (!stream) {
        throw std::runtime_error("Cannot open artifact manifest");
    }

    nlohmann::json const manifest = nlohmann::json::parse(stream);
    if (manifest.at("schema_version") != 2) {
        throw std::runtime_error("Unsupported artifact schema");
    }

    metadata_ = manifest.at("artifact");
    metadata_["root"] = root.string();
    metadata_["provenance"] = manifest.value("provenance", nlohmann::json::object());

    std::set<std::string> ids;
    for (nlohmann::json const &json : manifest.at("kernels")) {
        ArtifactKernel kernel {};
        kernel.id = json.at("id");
        if (!ids.insert(kernel.id).second || kernel.id.empty()) {
            throw std::runtime_error("Duplicate/empty artifact ID");
        }

        std::string const shape = json.at("shape").get<std::string>();
        std::string const backend = json.at("backend").get<std::string>();
        if ((shape != "star" && shape != "box") || (backend != "cuda" && backend != "sass")) {
            throw std::runtime_error("Unsupported artifact family");
        }

        kernel.shape = shape == "star" ? Shape::Star : Shape::Box;
        kernel.backend = backend == "cuda" ? Backend::CUDA : Backend::SASS;

        kernel.dimensions = json.at("dimensions");
        kernel.arch = json.at("arch");
        kernel.radius = json.at("radius");
        kernel.time_block_size = json.at("time_block_size");
        kernel.minimum_overlap = cross(json.at("minimum_overlap"));
        if (json.contains("group_size")) {
            kernel.group = cross(json.at("group_size"));
        }

        kernel.profile = { json.at("profile").at("name"), json.at("profile").value("stage", 0U) };
        if (kernel.backend == Backend::SASS) {
            kernel.sass = SASSTuning {
                json.at("sass").at("leading_load_count"),
                json.at("sass").at("min_comp_stall"),
            };
        } else {
            nlohmann::json const &bound = json.at("cuda").at("launch_bounds");
            if (!bound.is_number_integer() || bound < 1 || bound > 1024) {
                throw std::runtime_error("CUDA launch_bounds must be an integer in 1..1024");
            }

            kernel.launch_bounds = bound.get<unsigned>();
        }

        kernel.dtype = json.at("dtype");
        kernel.abi_version = json.at("entry").at("abi_version");
        if (kernel.abi_version != (kernel.backend == Backend::CUDA ? 1U : 2U)) {
            throw std::runtime_error("Kernel ABI version does not match backend");
        }

        nlohmann::json const &pipeline = json.at("pipeline");
        kernel.pipeline = {
            pipeline.at("head_rows"),
            pipeline.at("tail_rows"),
            pipeline.at("planes_per_iteration"),
            pipeline.at("stream_halo"),
            pipeline.at("halo_head_extension"),
        };

        PipelineContract const &contract = kernel.pipeline;
        if (!contract.head_rows || !contract.tail_rows || !contract.planes_per_iteration
            || !contract.stream_halo || contract.halo_head_extension < contract.stream_halo
            || contract.halo_head_extension % contract.planes_per_iteration
            || contract.halo_head_extension - contract.stream_halo
                >= contract.planes_per_iteration) {
            throw std::runtime_error("Invalid artifact pipeline contract");
        }

        kernel.entry_symbol = json.at("entry").at("symbol");
        kernel.coefficient_symbol = json.at("coefficients").at("symbol");
        kernel.coefficient_count = json.at("coefficients").at("count");
        if (kernel.dtype != "f64" || json.at("coefficients").at("dtype") != "f64") {
            throw std::runtime_error("Only FP64 artifacts are supported");
        }

        std::filesystem::path const relative(json.at("cubin").at("path").get<std::string>());
        if (relative.empty() || relative.is_absolute()) {
            throw std::runtime_error("Cubin path must be relative");
        }

        for (std::filesystem::path const &part : relative) {
            if (part == "..") {
                throw std::runtime_error("Cubin path escapes artifact root");
            }
        }

        kernel.cubin_path = root / relative;
        std::uintmax_t const expected_size = json.at("cubin").at("size").get<std::uintmax_t>();
        if (std::filesystem::file_size(kernel.cubin_path) != expected_size) {
            throw std::runtime_error("Cubin byte size mismatch");
        }

        kernel.cubin_sha256 = json.at("cubin").at("sha256");
        if (sha256_file(kernel.cubin_path) != kernel.cubin_sha256) {
            throw std::runtime_error(fmt::format("Cubin SHA-256 mismatch: {}", relative.string()));
        }

        records_.push_back(std::move(kernel));
    }
}

auto ArtifactCatalog::select(
    KernelSelector const &selector,
    std::string const &arch,
    std::string const &name
) const -> ArtifactKernel const & {
    std::vector<ArtifactKernel const *> candidates;

    for (ArtifactKernel const &kernel : records_) {
        if (selector.id) {
            if (kernel.id == *selector.id) {
                if (kernel.arch != arch) {
                    throw std::runtime_error(
                        fmt::format("case {}: ID incompatible with GPU {}", name, arch)
                    );
                }

                candidates.push_back(&kernel);
            }
            continue;
        }

        if (kernel.arch != arch || !matches(selector.shape, shape_name(kernel.shape))
            || !matches(selector.backend, backend_name(kernel.backend))
            || !matches(selector.dimensions, kernel.dimensions)
            || !matches(selector.radius, kernel.radius)
            || !matches(selector.time_block_size, kernel.time_block_size)
            || !matches(selector.profile, kernel.profile.name)
            || !matches(selector.stage, kernel.profile.stage)
            || !matches_cross(selector.minimum_overlap, kernel.minimum_overlap)
            || (selector.group && (!kernel.group || !matches_cross(selector.group, *kernel.group)))
            || (selector.leading_load_count
                && (!kernel.sass
                    || *selector.leading_load_count != kernel.sass->leading_load_count))
            || (selector.min_comp_stall
                && (!kernel.sass || *selector.min_comp_stall != kernel.sass->min_comp_stall))
            || (selector.launch_bounds && selector.launch_bounds != kernel.launch_bounds)) {
            continue;
        }

        candidates.push_back(&kernel);
    }

    if (candidates.empty()) {
        throw std::runtime_error(
            fmt::format("case \"{}\": no matching artifact record (GPU {})", name, arch)
        );
    }

    if (candidates.size() != 1) {
        std::string message = fmt::format(
            "case \"{}\" is ambiguous: selector matched {} artifact records\ncandidates:\n",
            name,
            candidates.size()
        );
        std::map<std::string, std::set<std::string>> differences;

        for (ArtifactKernel const *const kernel : candidates) {
            nlohmann::json json = kernel_json(*kernel);
            message += fmt::format("  {}\n", json.dump());

            json.erase("id");
            json.erase("cubin");
            nlohmann::json const flat = json.flatten();
            for (auto const &[key, value] : flat.items()) {
                differences[key].insert(value.dump());
            }
        }

        message += "differing selector fields:\n";
        for (auto const &[key, values] : differences) {
            if (values.size() < 2) {
                continue;
            }

            message += fmt::format("  {}:", key);
            for (std::string const &value : values) {
                message += fmt::format(" {}", value);
            }

            message += '\n';
        }

        throw std::runtime_error(message);
    }

    return *candidates.front();
}
}  // namespace stencil
