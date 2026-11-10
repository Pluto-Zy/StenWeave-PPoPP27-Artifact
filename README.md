# StenWeave — PPoPP 2027 Artifact

This artifact accompanies **StenWeave: Latency-Oriented Execution for
High-Throughput Stencils on Modern GPUs** (PPoPP 2027). It includes StenWeave,
five baseline implementations, and scripts to reproduce Figures 9–15 on NVIDIA
A100 and H100 GPUs. DRStencil's final performance results are provided directly
because repeating its tuning is time-consuming.

## Setup

Requirements:

- **System:** Linux with an NVIDIA Ampere (SM80) or Hopper (SM90) GPU.
  Our experiments used an **A100 PCIe 40 GB** and an **H100 PCIe 80 GB**.
- **CUDA:** CUDA 12.8 or later with a compatible driver, and Nsight Compute
  (`ncu`) with access to GPU performance counters. Put `nvcc` and `ncu` on `PATH`.
  Recommended NCU versions: **2025.1.1** or **2025.3.1**.
- **Build tools:** CMake, a C++17 compiler with OpenMP, Git, and `patch`.
- **Python:** Python 3.10 or later and the packages in `requirements.txt`.

Run the commands below in Bash from the artifact directory. First, prepare a
Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Set the architecture and target GPU in the terminal where you will run the
remaining commands:

```bash
ARCH=sm80       # sm80 for A100; sm90 for H100
DEVICE=0        # physical index of the target GPU for the experiments
```

Select the GPU through `DEVICE` and leave `CUDA_VISIBLE_DEVICES` unset. If your
environment already sets it, run `unset CUDA_VISIBLE_DEVICES` before continuing.
The first build requires internet access to download C++ dependencies.

## Quick check

```bash
python3 scripts/quickcheck.py --arch "$ARCH" --device "$DEVICE" --jobs 16 \
    --output-dir "runs/quickcheck-$ARCH"
```

The quick check builds and runs Box2D1R, performs CPU-reference checks, and
collects NCU metrics. It also generates a sample Figure 9 from **precomputed
data**, saved as `runs/quickcheck-$ARCH/figures/fig09.pdf`.

The check passes when the final summary shows `StenWeave quickcheck PASSED` and
all listed stages show `PASS`. You should also be able to open the sample PDF
above.

Quick check uses a timestamped output directory under `runs/` by default.

The short timings check that the implementations run; use the
[full experiments](#full-experiments) to reproduce performance results. A clean
A100 quick check takes about 7 minutes, depending on dependency download times.

## Full experiments

The workflow is to [build](#build), [run the experiments](#run-experiments),
and [generate figures](#generate-figures). Full reproduction uses both A100 and
H100; with one GPU type, you can reproduce the corresponding part of the results.

Use an idle GPU and set its graphics clock to **the maximum supported frequency**
for performance measurements. The maximum graphics clocks of our A100 PCIe
40 GB and H100 PCIe 80 GB GPUs are **1410 MHz** and **1755 MHz**, respectively.

### Build

Run the build on each machine used for evaluation: use `ARCH=sm80` on the A100
machine and `ARCH=sm90` on the H100 machine, as set in [Setup](#setup). If you
have only one GPU type, build only its architecture.

```bash
python3 scripts/build.py --arch "$ARCH" --jobs 16
```

### Run experiments

#### Experiments shared by A100 and H100

Run these experiments once on the A100 and once on the H100, using the
corresponding `ARCH` and `DEVICE` on each machine. If you have only one GPU type,
run them for that GPU to obtain partial results.

```bash
# Figures 9 and 14: baseline comparison and StenWeave variant speedups.
python3 scripts/run.py --arch "$ARCH" --device "$DEVICE" --suite benchmark
# Figure 11: roofline efficiency.
python3 scripts/run.py --arch "$ARCH" --device "$DEVICE" --suite roofline
# Figure 12: roofline efficiency across temporal blocking depths.
python3 scripts/run.py --arch "$ARCH" --device "$DEVICE" --suite time-block-sweep
# Figure 13: execution-model breakdown.
python3 scripts/run.py --arch "$ARCH" --device "$DEVICE" --suite breakdown
```

#### A100-only experiments

If you are using an A100, also run the following two experiments. Skip this
section on H100.

```bash
# Figure 10: performance across problem sizes.
python3 scripts/run.py --arch sm80 --device "$DEVICE" --suite size-sweep
# Figure 15: instruction mix, stalls, and resource utilization.
python3 scripts/run.py --arch sm80 --device "$DEVICE" --suite low-level-metrics
```

#### Resume an interrupted experiment

Each experiment saves its results in its default directory under `runs/`. To
continue an interrupted experiment, pass that directory to `--resume`, for
example:

```bash
python3 scripts/run.py --resume "runs/benchmark-$ARCH"
```

Completed tasks are reused; unfinished tasks are rerun. To start a separate
batch, choose a fresh path with `--output-dir`.

### Generate figures

The commands below generate PDFs in `figures/`, for example `figures/fig09.pdf`.
Add `--export-data` to also save the plotted values as CSV files in
`figures/details/`. Figure generation requires the Python packages but no GPU
or NCU installation.

#### With one GPU type

You can evaluate the artifact even if only one of the two GPU types is available:

- **A100 only:** reproduce all seven figures using A100 data; the H100 results
  are not included.
- **H100 only:** reproduce the H100 results in Figures 9, 11, 12, 13, and 14.
  Figures 10 and 15 require an A100.

Both options provide partial coverage of the paper.

After completing the [experiments](#run-experiments) for your available GPU,
generate every figure supported by its results:

```bash
python3 scripts/experiment.py --arch "$ARCH"
```

#### Combining A100 and H100 results

To combine results collected on separate machines, package the four complete
H100 result directories on the H100 machine:

```bash
tar -czf h100-results.tar.gz \
    runs/benchmark-sm90 runs/roofline-sm90 \
    runs/time-block-sweep-sm90 runs/breakdown-sm90
```

Copy this archive to the A100 machine's artifact directory and extract it:

```bash
tar -xzf h100-results.tar.gz
```

With both sets of result directories under `runs/`, generate all seven figures:

```bash
python3 scripts/experiment.py
```

The seven PDFs are saved as `figures/fig09.pdf` through `figures/fig15.pdf`,
including both GPUs in the figures that compare them.
Keep the full result directories, including `tasks/`, when transferring data.
You can also gather both sets of results on a third machine and run the same
figure-generation command there.

By default, `run.py` writes to `runs/<suite>-<arch>` (for example,
`runs/benchmark-sm80`) and will not overwrite an existing directory. Use
`--resume` to continue an interrupted batch, or `--output-dir` to collect a new
batch elsewhere. For custom result paths, pass each batch with a repeated
`--input` option to `experiment.py`.

### Command-line options

The table summarizes the main options. Required means that no default applies
for a new run.

| Script | Option | Meaning | Default |
| --- | --- | --- | --- |
| `build.py` | `--arch` | Target architecture (`sm80` or `sm90`). | Required |
|  | `--impl` | Implementations to build. | `all` |
|  | `--jobs` | Parallel jobs per implementation where supported. | `4` |
|  | `--nvcc` | CUDA compiler path. | `NVCC`, `CUDA_ROOT`, then `PATH` |
|  | `--suite` | StenWeave generator suites to build. | All applicable suites |
|  | `--cuassembler` | Override the bundled CuAssembler source. | Bundled source |
|  | `--dependency-root` | Reuse a CMake dependency source cache. | Download as needed |
|  | `--dry-run` | Print build commands without compiling. | Off |
| `run.py` | `--suite` | Experiment suite. | Required for a new batch |
|  | `--arch` | Target architecture. | Required for a new batch |
|  | `--device` | Physical GPU index. | Required for a new batch |
|  | `--impl` | Implementations to run. | All participating implementations |
|  | `--stencil` | Restrict to selected stencils; repeat as needed. | All configured stencils |
|  | `--case` | Select task IDs shown by `--dry-run`; repeat as needed. | All selected tasks |
|  | `--output-dir` | Result directory for a new batch. | `runs/<suite>-<arch>` |
|  | `--ncu` | Nsight Compute executable for profiling suites. | `NCU`, `PATH`, or CUDA installation |
|  | `--timeout` | Maximum seconds per task. | `7200` |
|  | `--resume` | Continue an interrupted batch and reuse verified tasks. | Off |
|  | `--dry-run` | Show the plan or inspect resume candidates. | Off |
| `experiment.py` | `--arch` | Architectures represented in the results. | `sm80 sm90` |
|  | `--figure` | Figures to generate. | All figures supported by `--arch` |
|  | `--input` | Custom batches or CSV/JSONL inputs; repeat to combine. | `runs/<suite>-<arch>` for selected figures |
|  | `--output-dir` | Figure and optional data destination. | `figures/` |
|  | `--export-data` | Also write figure CSVs and input manifests. | Off |
|  | `--tables-only` | Export data without PDFs; implies `--export-data`. | Off |
| `quickcheck.py` | `--arch`, `--device` | Target architecture and physical GPU index. | Required |
|  | `--jobs` | Parallel build jobs. | `4` |
|  | `--output-dir` | Quick-check result directory. | Timestamped under `runs/` |
|  | `--build-root` | Build output directory. | Fresh directory under `build/quickcheck/` |
|  | `--dependency-root` | Optional CMake dependency cache. | Download as needed |
|  | `--nvcc`, `--ncu` | CUDA compiler and Nsight Compute executable. | `PATH` / CUDA installation |
|  | `--resume` | Reuse verified quick-check builds. | Off |
|  | `--report` | Display a saved quick-check report. | Off |
|  | `--timeout` | Maximum seconds per step. | `1200` |
|  | `--dry-run` | Show steps without executing. | Off |

Run a script with `--help` for descriptions and less common diagnostic options.

## Reference wall times

| Step | A100 | H100 |
| --- | --- | --- |
| Full build (`--jobs 16`) | 4 min 28 s | 7 min 14 s |
| `benchmark` | 74 min 33 s | 51 min 24 s |
| `size-sweep` | 236 min 36 s | — |
| `roofline` | 5 min 44 s | 6 min 22 s |
| `time-block-sweep` | 3 min 19 s | 4 min 45 s |
| `breakdown` | 1 min 42 s | 1 min 54 s |
| `low-level-metrics` | 1 min 55 s | — |

Generating all seven figures from both sets of results takes about 33 seconds.

Times vary with the host and network, particularly dependency downloads during
the first build.

## Expected results

Compare the generated figures with Figures 9–15 in the paper. The expected
performance trends and numerical results are summarized below.

| Figure | Experiment | Expected behavior |
| --- | --- | --- |
| 9 | `benchmark` | SW-SASS-ILP should outperform every baseline in all evaluated cases, with speedups close to those reported in the paper. |
| 10 | `size-sweep` | SW-SASS and SW-SASS-ILP should consistently outperform the baselines at large domain sizes, with their throughput gradually reaching a plateau as the domain size increases. Results for small domains may differ more from the paper because they are more sensitive to clock fluctuations and warm-up effects. |
| 11 | `roofline` | SW-SASS and SW-SASS-ILP should achieve high roofline efficiency, with values close to those in the paper. |
| 12 | `time-block-sweep` | Each curve should reach a plateau at around 95% roofline efficiency once the temporal blocking depth is sufficiently large. The minimum depth needed to reach this plateau varies across curves. |
| 13 | `breakdown` | The cumulative execution-model stages should reproduce the performance growth shown in the paper, with throughput values close to the reported results. |
| 14 | `benchmark` | The geometric-mean speedup in the CUDA row should be substantially higher than that in the SASS row. The ILP row should show substantially larger speedups on H100 than on A100. |
| 15 | `low-level-metrics` | In panel (a), the SW bars should have a substantially smaller Addr fraction than the baseline bars. In panel (b), MIO Thr. should account for a substantial share of stalls for Star2D1R, and Math Thr. for Box2D1R. Panel (c)'s utilization values should be close to those in the paper. |

Figures 10 and 15 use A100 data; the other figures compare A100 and H100.

The paper's exact numerical results may not be reproduced in every environment.
Compiler behavior, GPU clock settings and fluctuations, power limits, and
CPU-side scheduling of CUDA Graph launches can all affect the measurements.
Compare both the trends above and the numerical values with the paper, allowing
for these sources of variation.

Use `--export-data` with the [figure-generation commands](#generate-figures)
to inspect the numerical values in `figures/details/fig09.csv` through
`figures/details/fig15.csv` for the selected figures.

## Reuse and adaptation (Reusable badge)

This section describes how to adapt StenWeave beyond the paper's workloads.
The reproduction commands above use the supplied configurations without changes.

### Configuration files

There are two configuration layers:

| File | Purpose | When to rebuild |
| --- | --- | --- |
| `configs/stenweave/generator/<suite>.toml` | Defines stencil shape/radius, architecture, backend, temporal blocking, overlap, ILP grouping, and code scheduling. | After changing a kernel specification. |
| `configs/stenweave/runner/<arch>/<suite>.toml` | Selects generated kernels and sets domains, thread blocks, bands, time steps, and measurement settings. | Runtime-only changes can reuse a compatible generated kernel. |

Work on copies of these files. In a runner file, common settings belong in
`[defaults.run]`; a `[case.run]` value overrides the default for that case.
`[case.kernel].id` selects a record from the generated artifact's `manifest.json`.
Keep that ID when reusing the same kernel. Changing it selects another kernel;
it does not modify the original one. An ID selector cannot be combined with
other kernel selector fields in the same `[case.kernel]` table.

### Changing the domain of an existing kernel

Start with a case for the same architecture, stencil, radius, and backend.
For a first experiment, retain its kernel ID, thread block, time steps, and
measurement settings, and change only the domain and the bands covering it.
The principal layout constraints are:

- `domain` uses positive `x, y` extents for 2D and `x, y, z` for 3D. The streaming
  dimension is `y` in 2D and `z` in 3D; the other dimensions are tiled.
- `bands` is a nonempty list. Each group's `repeat` and `grid` extents must be
  positive. Provide `stream` for every group or omit it from every group.
  With explicit streams, `sum(repeat × stream)` must equal the streaming domain
  extent, including any padding. Without them, that extent must be divisible
  by the total band count `sum(repeat)`.
- For SASS, in each tiled dimension let `F = block × group_size` be the tile
  footprint, `g` the band grid extent, and `D` the domain extent including
  padding. For `g > 1`, the actual overlap is `O = (g × F - D) / (g - 1)`.
  It must be an integer and satisfy `minimum_overlap ≤ O < F`, with `O` divisible
  by `2 × group_size`. For `g = 1`, `D` must equal `F`. Apply this per tiled
  dimension and per band group. The generated manifest records `group_size`
  and `minimum_overlap`.
- Each band's stream must be long enough for the kernel's temporal pipeline.
  The manifest's `pipeline` records the head/tail and halo sizes; the runner
  checks both the stream length and the additional halo head needed by interior
  bands. Retain a known-valid stream length when first changing other geometry.
- `time_steps` must be positive and a multiple of the kernel's `time_block_size`.
  For CUDA kernels, the thread count `block.x × block.y` (with `block.y = 1` in
  2D) must equal the generated `launch_bounds`. For SASS kernels, the thread
  block and ILP group determine the tile footprint used above. A changed block
  must also fit the GPU's thread and shared-memory limits.
- `repeat_runs` and `measurement_runs` must be positive, with
  `measurement_runs ≤ repeat_runs`. The latter selects the final samples used
  by `measurement_statistic` (`mean` or `median`).

The runner rejects incompatible layouts. For an overlap or alignment error,
recompute the grid and domain together rather than changing only the domain.
If the desired small domain cannot accommodate the temporal halo, select or
regenerate a kernel with a smaller fusion depth. Arbitrary domain/grid pairs
are not supported by a fixed SASS kernel. Layout validity also does not ensure
that a large input and its workspaces fit in GPU memory.

For a concrete SM80 example, build the benchmark first, then copy its runner
configuration:

```bash
cp configs/stenweave/runner/sm80/benchmark.toml /tmp/custom-runner.toml
```

In the copied file, find `star2d1r-sm80-sass-ilp`. Keep its kernel ID and
`block = { x = 256 }`, and replace just its `domain` and `bands` with:

```toml
domain = { x = 17742, y = 40950 }
bands = [{ repeat = 9, grid = { x = 24 }, stream = 4550 }]
```

This doubles the streaming extent and each band's stream from the supplied
case while preserving its tiled width. The kernel has `group_size = 3` and
`minimum_overlap = 30`: `(24 × 256 × 3 - 17742) / 23 = 30`, and
`9 × 4550 = 40950`. No new kernel generation is needed. Run the edited case:

```bash
python3 scripts/run/implementations/stenweave.py --arch sm80 \
    --device "$DEVICE" --config /tmp/custom-runner.toml \
    --case star2d1r-sm80-sass-ilp --output-dir runs/custom-domain
```

Results are written to `runs/custom-domain/results.jsonl`, with the native
records in `native.jsonl`. Use a fresh output directory for each trial. On
SM90, start from the SM90 runner case and its manifest, whose kernel settings
and geometry differ from this example.

### Generating a different kernel

To change fusion depth or instruction scheduling, copy the matching generator
configuration and edit a variant in `[[kernels.variants]]`. Key constraints are:

| Parameter | Constraint |
| --- | --- |
| `radius`, `time_block_size` | Radius is 1–3; 3D box currently supports radius 1. Fusion depth must be positive. |
| `minimum_overlap` | In each tiled dimension, at least `2 × radius × time_block_size`; the block/group footprint must leave a positive interior. |
| `group_size` | SASS only; positive per dimension, with `minimum_overlap` divisible by `2 × group_size`. Use a scalar for 2D and `{ x, y }` for 3D. |
| `leading_load_count`, `min_comp_stall` | SASS only; load count is 1 or 2, and minimum computation stall is a positive integer. |
| `launch_bounds` | CUDA only; thread count is 1–1024 and must match the runner block. CUDA 3D requires equal x/y minimum overlap. |

Build the copied configuration into a separate tree to preserve the paper's
build and generated artifacts:

```bash
cp configs/stenweave/generator/benchmark.toml /tmp/custom-generator.toml
# Edit the desired kernel variant in /tmp/custom-generator.toml.
python3 scripts/build/stenweave.py --arch "$ARCH" --component all \
    --suite benchmark --generator-config /tmp/custom-generator.toml \
    --build-root build/custom --jobs 16
```

Look up the generated kernel in
`build/custom/artifacts/StenWeave/<arch>/benchmark/manifest.json`, set its ID in
the copied runner case, and adjust the workload to satisfy the new kernel's
layout and time-step constraints. Then run with the same build tree:

```bash
python3 scripts/run/implementations/stenweave.py --arch "$ARCH" \
    --device "$DEVICE" --build-root build/custom \
    --config /tmp/custom-runner.toml --case "star2d1r-$ARCH-sass-ilp" \
    --output-dir runs/custom-kernel
```

Only stencil shapes and dimensions supported by the generator and runner can
be configured this way; adding a new stencil operator requires source changes.

### Performance tuning

The shipped settings are tuned for the paper's workloads and target GPUs.
A valid configuration is a starting point: optimal performance on another
input or GPU depends on tuning fusion depth, thread-block shape, ILP grouping,
band/grid layout, and scheduling parameters. Larger fusion depth or ILP groups
can increase overlap, register use, and shared-memory pressure, so increasing
them does not necessarily improve performance. Measure alternative valid
configurations on the target GPU with a consistent measurement protocol and
compare throughput. Keep each trial's generator and runner configurations with
its results; the shipped tuning choices need not be optimal for a new workload.
