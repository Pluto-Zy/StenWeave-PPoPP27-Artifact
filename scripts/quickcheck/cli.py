"""Build and exercise Box2D1R, profile one launch, and redraw Figure 9.

Dependencies must already be installed in the invoking Python environment.
"""

import argparse
import datetime
import importlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from run import profile
from run.process import run
from run.suites.common import ROOT, baseline_tasks, sha, stenweave_tasks, write_json

from .cache import build_lock, reusable
from .cache import inputs as build_inputs
from .output import (
    StepProgress,
    failure_details,
    print_report,
    print_toolchain,
    step_label,
)

BASELINES = ("an5d", "ebisu", "convstencil", "lorastencil", "flashfft")


def dependencies(nvcc, ncu):
    missing = []
    cxx = shlex.split(os.environ.get("CXX") or "c++") or ["c++"]
    toolchain = {
        "Python": {"version": platform.python_version(), "path": sys.executable}
    }
    for module in ("numpy", "matplotlib", "elftools", "sympy"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if sys.version_info < (3, 11):
        try:
            importlib.import_module("tomli")
        except ImportError:
            missing.append("tomli")
    for executable in ("cmake", "patch", cxx[0], "nvidia-smi", nvcc, ncu):
        if not shutil.which(executable):
            missing.append(executable)
    if not missing:
        for label, command in (
            ("CUDA / nvcc", [nvcc]),
            ("C++", cxx),
            ("CMake", ["cmake"]),
            ("NCU", [ncu]),
        ):
            executable = command[0]
            try:
                result = subprocess.run(
                    [*command, "--version"],
                    check=True,
                    timeout=30,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                lines = [
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                ]
                if not lines:
                    raise ValueError(f"No version information from {executable}")
                version = lines[0]
                if label == "CUDA / nvcc":
                    version = next(
                        (line for line in lines if "release " in line), version
                    )
                elif label == "NCU":
                    version = next(
                        (line for line in lines if "Version " in line), version
                    )
                toolchain[label] = {
                    "path": str(Path(shutil.which(executable)).absolute()),
                    "version": version,
                    "version_output": result.stdout,
                    "command": command,
                }
            except (OSError, subprocess.SubprocessError):
                missing.append(f"{executable} (cannot start in this environment)")
    if missing:
        raise ValueError(
            f"Missing dependencies: {', '.join(missing)}. "
            "Install dependencies manually before running quickcheck. "
            "Python packages are listed in requirements.txt."
        )
    return toolchain


def commands(args, folder, build_root, baselines):
    python = sys.executable
    result = []
    for name in (*baselines, "stenweave"):
        command = [
            python,
            str(ROOT / f"scripts/build/{name}.py"),
            "--arch",
            args.arch,
            "--nvcc",
            args.nvcc,
            "--build-root",
            str(build_root),
        ]
        if name == "flashfft":
            command += ["--dim", "2"]
        elif name == "stenweave":
            command += [
                "--suite",
                "benchmark",
                "--jobs",
                str(args.jobs),
                "--generator-config",
                str(ROOT / "configs/quickcheck/generator.toml"),
            ]
            if args.dependency_root:
                command += ["--dependency-root", str(args.dependency_root)]
        else:
            command += ["--stencil", "box2d1r", "--jobs", str(args.jobs)]
        result.append((f"build-{name}", command))
    for checking in (False, True):
        mode = "checks" if checking else "timing"
        for name in (*baselines, "stenweave"):
            command = [
                python,
                str(ROOT / f"scripts/run/implementations/{name}.py"),
                "--arch",
                args.arch,
                "--device",
                str(args.device),
                "--build-root",
                str(build_root),
                "--output-dir",
                str(folder / mode / name),
            ]
            if name == "stenweave":
                command += [
                    "--suite",
                    "benchmark",
                    "--config",
                    str(ROOT / f"configs/quickcheck/{args.arch}.toml"),
                ]
            else:
                command += ["--stencil", "box2d1r"]
            if checking:
                command += ["--check"]
            elif name == "an5d":
                command += ["--width", "4096", "--steps", "64", "--repeats", "3"]
            elif name == "flashfft":
                command += ["--width", "4096", "--steps", "60", "--repeats", "3"]
            elif name in ("convstencil", "lorastencil"):
                command += [
                    "--dimensions",
                    "4096",
                    "4096",
                    "--iterations",
                    "128",
                    "--repeats",
                    "3",
                ]
            elif name == "ebisu":
                command += ["--dimensions", "4096", "4096", "--repeats", "3"]
            result.append((f"{mode}-{name}", command))
    result.append(
        (
            "plot-fig09",
            [
                python,
                str(ROOT / "scripts/experiment.py"),
                "--figure",
                "9",
                "--input",
                str(ROOT / "precomputed/fig09.csv"),
                "--output-dir",
                str(folder / "figures"),
            ],
        )
    )
    return result


def profile_tasks(arch):
    tasks = baseline_tasks("an5d", arch, ("box2d1r",))
    tasks += baseline_tasks("ebisu", arch, ("box2d1r",))
    task = next(
        t
        for t in stenweave_tasks(arch, "benchmark", variants=("sass",))
        if t["stencil"] == "box2d1r"
    )
    config = ROOT / f"configs/quickcheck/{arch}.toml"
    task.update(config=str(config.relative_to(ROOT)), config_sha256=sha(config))
    return [*tasks, task]


def write_summary(folder, baselines, profile_count, scope):
    pdf = folder / "figures/fig09.pdf"
    if not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError("Figure 9 PDF was not generated")
    timings = []
    for name in (*baselines, "stenweave"):
        source = folder / "timing" / name / "results.jsonl"
        for line in source.read_text().splitlines():
            row = json.loads(line)
            timings.append(
                {
                    "implementation": row["implementation"],
                    "variant": row.get("variant", "native"),
                    "gstencils_per_s": row["gstencils_per_s"],
                    "source": str(source.relative_to(folder)),
                }
            )
    expected_count = len(baselines) + 3
    if len(timings) != expected_count:
        raise ValueError("Unexpected quick timing result count")
    write_json(
        folder / "summary.json",
        {
            "scope": scope,
            "timings": timings,
            "checks": len(baselines) + 3,
            "profiles": profile_count,
            "historical_plot": "figures/fig09.pdf",
            "historical_csv_sha256": sha(ROOT / "precomputed/fig09.csv"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("sm80", "sm90"))
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--report",
        type=Path,
        help="display saved results without rerunning experiments",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse verified builds from an earlier run; collect fresh results",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--nvcc",
    )
    parser.add_argument("--ncu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--build-root",
        type=Path,
        help="default: a fresh build/quickcheck/<run-id>/ tree",
    )
    parser.add_argument(
        "--dependency-root",
        type=Path,
        help="optional source cache for the runner's pinned CMake dependencies",
    )
    parser.add_argument("--timeout", type=int, default=1200, help="seconds per step")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.report and args.resume:
        parser.error("--report and --resume cannot be combined")
    if args.report:
        return 0 if print_report(args.report.resolve()) == "completed" else 1
    previous = None
    if args.resume:
        args.resume = args.resume.resolve()
        previous = json.loads((args.resume / "plan.json").read_text())
        old_state = json.loads((args.resume / "status.json").read_text())
        if old_state["status"] == "running":
            parser.error(
                "the source run is still marked running; stop it before resuming"
            )
        if args.arch is not None and args.arch != previous["arch"]:
            parser.error("--resume must use the source run's architecture")
        args.arch = previous["arch"]
        if args.device is None:
            args.device = previous["device"]
        if (
            args.build_root
            and args.build_root.resolve() != Path(previous["build_root"]).resolve()
        ):
            parser.error("--resume uses the source run's build tree")
        args.build_root = Path(previous["build_root"])
        old_build = next(
            step["command"]
            for step in previous["steps"]
            if step["id"].startswith("build-")
        )
        if args.nvcc is None:
            args.nvcc = old_build[old_build.index("--nvcc") + 1]
        if args.ncu is None:
            args.ncu = previous["profiles"][0]["command"][0]
    args.nvcc = (
        args.nvcc
        or os.environ.get("NVCC")
        or shutil.which("nvcc")
        or str(Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda")) / "bin/nvcc")
    )
    args.ncu = args.ncu or os.environ.get("NCU") or shutil.which("ncu") or "ncu"
    if args.arch is None or args.device is None:
        parser.error("--arch and --device are required unless using --report")
    if args.device < 0 or args.jobs < 1 or args.timeout < 1:
        parser.error("device must be nonnegative; jobs/timeout must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("unset CUDA_VISIBLE_DEVICES; --device selects the physical GPU")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = (
        args.output_dir or ROOT / "runs" / f"quickcheck-{args.arch}-{stamp}"
    ).resolve()
    build_root = (args.build_root or ROOT / "build/quickcheck" / folder.name).resolve()
    if args.dependency_root:
        args.dependency_root = args.dependency_root.resolve()
    baselines = [
        name
        for name in BASELINES
        if not (args.arch == "sm90" and name == "convstencil")
    ]
    steps = commands(args, folder, build_root, baselines)
    tasks = profile_tasks(args.arch)
    plan = {
        "arch": args.arch,
        "device": args.device,
        "build_root": str(build_root),
        "stencil": "box2d1r",
        "baselines": baselines,
        "variants": ["CUDA", "SASS", "SASS-ILP"],
        "steps": [{"id": name, "command": cmd} for name, cmd in steps],
        "profiles": [
            {
                "task": task,
                "command": profile.commands(
                    task,
                    args.arch,
                    args.device,
                    folder / "profiles" / task["id"],
                    args.ncu,
                    "low-level-metrics",
                    verify=False,
                    build_root=build_root,
                )[1],
            }
            for task in tasks
        ],
        "historical_plot_input": "precomputed/fig09.csv",
        "scope": "quick functional checks; timings are not paper performance measurements",
    }
    if args.resume:
        plan["resume_from"] = str(args.resume)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    toolchain = dependencies(args.nvcc, args.ncu)
    plan["build_inputs"] = {
        name: build_inputs(name, args.arch, toolchain)
        for name in (*baselines, "stenweave")
    }
    if folder.exists() or (build_root.exists() and not args.resume):
        raise ValueError("Use fresh output and build directories")
    # Read-only identity/occupancy checks, with no frequency requirements.
    from adapters.ebisu import device_info

    gpu = device_info(str(args.device), args.arch)
    occupied = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.device),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if occupied:
        raise ValueError(f"Selected GPU has resident compute processes: {occupied}")
    with build_lock(build_root):
        folder.mkdir(parents=True)
        (folder / "logs").mkdir()
        write_json(folder / "plan.json", plan)
        write_json(
            folder / "environment.json",
            {
                "gpu": gpu,
                "python": sys.version,
                "executable": sys.executable,
                "toolchain": toolchain,
            },
        )
        state = {"status": "running", "steps": [], "profiles": []}
        write_json(folder / "status.json", state)
        started = time.monotonic()
        current = ""
        print(
            f"\nStenWeave quickcheck | {args.arch} | GPU {args.device} | Box2D1R",
            flush=True,
        )
        print(f"Results: {folder}", flush=True)
        print_toolchain(toolchain)
        if args.arch == "sm90":
            print("SKIP ConvStencil on H100", flush=True)
        try:
            for name, command in steps:
                current = name
                if args.resume and name.startswith("build-"):
                    implementation = name.removeprefix("build-")
                    old_inputs = previous.get("build_inputs", {}).get(implementation)
                    if (
                        old_inputs is not None
                        and old_inputs != plan["build_inputs"][implementation]
                    ):
                        cached, reason = False, "build inputs or toolchain changed"
                    else:
                        cached, reason = reusable(
                            implementation,
                            args.arch,
                            build_root,
                            toolchain["CUDA / nvcc"]["version_output"],
                        )
                    if cached:
                        print(
                            f"  [{len(state['steps']) + 1:2}/{len(steps) + len(tasks)}] REUSE {step_label(name)} (verified)",
                            flush=True,
                        )
                        (folder / "logs" / f"{name}.log").write_text(
                            f"Reused verified build from {args.resume}\n"
                        )
                        state["steps"].append(
                            {"id": name, "seconds": 0.0, "reused": True}
                        )
                        write_json(folder / "status.json", state)
                        continue
                    print(f"  REBUILD {step_label(name)}: {reason}", flush=True)
                step_start = time.monotonic()
                with (
                    StepProgress(
                        len(state["steps"]) + 1,
                        len(steps) + len(tasks),
                        step_label(name),
                    ),
                    (folder / "logs" / f"{name}.log").open("w") as log,
                ):
                    run(
                        command,
                        cwd=ROOT,
                        timeout=args.timeout,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                state["steps"].append(
                    {"id": name, "seconds": time.monotonic() - step_start}
                )
                write_json(folder / "status.json", state)
            from experiments.low_level import calculate as low_level
            from experiments.roofline import calculate as roofline

            for task in tasks:
                current = f"profile-{task['id']}"
                target = folder / "profiles" / task["id"]
                step_start = time.monotonic()
                with StepProgress(
                    len(steps) + len(state["profiles"]) + 1,
                    len(steps) + len(tasks),
                    "NCU "
                    + {"an5d": "AN5D", "ebisu": "EBISU", "stenweave": "StenWeave SASS"}[
                        task["implementation"]
                    ],
                ):
                    rows = profile.collect(
                        task,
                        "low-level-metrics",
                        args.arch,
                        args.device,
                        target,
                        args.ncu,
                        args.timeout,
                        build_root=build_root,
                    )
                    for key, filename in (
                        ("report_path", "report.ncu-rep"),
                        ("raw_csv", "raw.csv"),
                        ("target_path", "target.json"),
                    ):
                        rows[0][key] = str((target / filename).relative_to(folder))
                    raw = (target / "raw.csv").read_text()
                    result = {
                        "id": task["id"],
                        "seconds": time.monotonic() - step_start,
                        "roofline": roofline(raw, args.arch),
                        "low_level": low_level(raw, args.arch),
                        "report": rows[0],
                    }
                    write_json(target / "analysis.json", result)
                    state["profiles"].append(result)
                write_json(folder / "status.json", state)
            current = "summary"
            write_summary(folder, baselines, len(tasks), plan["scope"])
            state.update(status="completed", seconds=time.monotonic() - started)
        except BaseException as error:
            state.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                failed_step=current,
                error=str(error),
                failure=failure_details(folder, plan, current, error),
                seconds=time.monotonic() - started,
            )
            if not isinstance(error, (Exception, KeyboardInterrupt)):
                raise
        finally:
            write_json(folder / "status.json", state)
            print_report(folder, show_toolchain=False)
        return (
            0
            if state["status"] == "completed"
            else 130
            if state["status"] == "interrupted"
            else 1
        )
