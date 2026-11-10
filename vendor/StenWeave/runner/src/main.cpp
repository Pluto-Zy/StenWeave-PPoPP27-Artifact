#include "artifact/catalog.hpp"
#include "config/batch.hpp"
#include "cuda/driver.hpp"
#include "cuda/inspect.hpp"
#include "execution/executor.hpp"
#include "execution/plan.hpp"
#include "report/report.hpp"

#include <CLI/CLI.hpp>

#include <fmt/base.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

auto main(int argc, char *const *argv) -> int {
    std::unique_ptr<stencil::Reporter> reporter;

    try {
        CLI::App app { "Offline stencil artifact batch experiments" };
        app.require_subcommand(1);
        CLI::App *const run = app.add_subcommand("run", "Execute a TOML batch");
        CLI::App *const list = app.add_subcommand("list", "Discover manifest records");
        CLI::App *const inspect = app.add_subcommand("inspect", "Read a GPU environment snapshot");

        std::string config, artifact, output, snapshot;
        int device = 0;
        int list_device = 0;
        unsigned threads = 0;
        bool overwrite = false;
        bool no_color = false;

        run->add_option("config", config)->required()->check(CLI::ExistingFile);
        run->add_option("--artifact-dir", artifact)->required()->check(CLI::ExistingDirectory);
        run->add_option("--device", device)->check(CLI::NonNegativeNumber);
        run->add_option("--output", output)->required();
        run->add_option("--input-threads", threads);
        run->add_flag("--force,--overwrite", overwrite);
        run->add_flag("--no-color", no_color, "Output is plain text");

        list->add_option("--artifact-dir", artifact)->required()->check(CLI::ExistingDirectory);
        CLI::Option const *const filter =
            list->add_option("--device", list_device)->check(CLI::NonNegativeNumber);

        inspect->add_option("--device", device)->check(CLI::NonNegativeNumber);
        inspect->add_option("--output", snapshot);

        try {
            app.parse(argc, argv);
        } catch (CLI::ParseError const &error) {
            return app.exit(error);
        }

        if (*inspect) {
            nlohmann::json const json = stencil::inspect_device(device);
            fmt::println("{}", json.dump(2));

            if (!snapshot.empty()) {
                stencil::Reporter writer(snapshot, false);
                writer.write(json);
            }

            return 0;
        }

        if (*list) {
            stencil::ArtifactCatalog const catalog(artifact);
            std::string const arch = filter->count() ? stencil::device_info(list_device).arch : "";
            for (stencil::ArtifactKernel const &kernel : catalog.records()) {
                if (arch.empty() || kernel.arch == arch) {
                    nlohmann::json json = stencil::kernel_json(kernel);
                    json.erase("cubin");
                    json.erase("coefficients");
                    json.erase("entry_symbol");

                    fmt::println("{}", json.dump());
                }
            }

            return 0;
        }

        stencil::BatchSpec const batch = stencil::parse_batch(config);
        stencil::DeviceInfo const device_info = stencil::device_info(device);
        stencil::ArtifactCatalog const catalog(artifact);
        stencil::ExecutionPlan const plan = stencil::make_plan(batch, catalog, device_info.arch);

        fmt::println(
            "Artifact: {}\nGPU: {} ({})\nSeed: {}\nCases: {}\nShared input: {} doubles",
            catalog.metadata().at("name").get<std::string>(),
            device_info.metadata.at("name").get<std::string>(),
            device_info.arch,
            batch.seed,
            plan.cases.size(),
            plan.max_input_elements
        );

        reporter = std::make_unique<stencil::Reporter>(output, overwrite);

        return stencil::execute(plan, batch, catalog, device_info, threads, *reporter, no_color);
    } catch (std::exception const &error) {
        if (reporter) {
            try {
                // clang-format off
                reporter->write({
                    { "type", "error" },
                    { "status", "failed" },
                    { "message", error.what() },
                });
                // clang-format on
            } catch (std::exception const &report_error) {
                fmt::println(stderr, "Could not persist error record: {}", report_error.what());
            }
        }

        fmt::println(stderr, "error: {}", error.what());

        return 1;
    }
}
