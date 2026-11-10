"""Build selected artifact implementations through their independent build scripts."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = {
    "an5d": "AN5D",
    "ebisu": "EBISU",
    "convstencil": "ConvStencil",
    "lorastencil": "LoRAStencil",
    "flashfft": "FlashFFTStencil",
    "stenweave": "StenWeave",
}
SUITES = ("benchmark", "size-sweep", "time-block-sweep", "breakdown")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  python3 scripts/build.py --arch sm80 --impl all --jobs 4\n"
            "  python3 scripts/build.py --arch sm90 --impl baselines\n"
            "  python3 scripts/build.py --arch sm80 --impl an5d ebisu --dry-run\n"
            "  python3 scripts/build.py --arch sm80 --impl stenweave --suite benchmark\n"
            "\nBuilds run sequentially. --jobs controls each implementation's own "
            "parallelism where supported; FlashFFTStencil has no jobs option. "
            "StenWeave builds its runner and selected generator suites. "
            "Roofline reuses benchmark cubins. Outputs remain under build/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", choices=("sm80", "sm90"), required=True)
    parser.add_argument(
        "--impl", nargs="+", choices=("all", "baselines", *IMPLEMENTATIONS),
        default=["all"], help="implementations to build (default: all)",
    )
    parser.add_argument("--jobs", type=int, default=4, help="per-implementation jobs (default: 4)")
    parser.add_argument(
        "--nvcc", help="CUDA compiler path or name; otherwise use NVCC, CUDA_ROOT, or PATH",
    )
    parser.add_argument(
        "--suite", nargs="+", choices=("all", *SUITES),
        help="StenWeave suites (default: all applicable suites; size-sweep is SM80 only)",
    )
    parser.add_argument(
        "--cuassembler", type=Path,
        help="StenWeave: override bundled CuAssembler with a checkout containing CuAsm/",
    )
    parser.add_argument(
        "--dependency-root", type=Path, help="StenWeave: existing CMake _deps cache",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without building")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    selected = []
    for name in args.impl:
        if name == "all":
            selected.extend(IMPLEMENTATIONS)
        elif name == "baselines":
            selected.extend(n for n in IMPLEMENTATIONS if n != "stenweave")
        else:
            selected.append(name)
    selected = list(dict.fromkeys(selected))
    if "stenweave" not in selected and any(
        value is not None for value in (args.suite, args.cuassembler, args.dependency_root)
    ):
        parser.error(
            "--suite, --cuassembler and --dependency-root require StenWeave to be selected"
        )
    suites = (
        list(SUITES) if args.suite is None or "all" in args.suite
        else list(dict.fromkeys(args.suite))
    )
    if "stenweave" in selected and args.arch == "sm90" and "size-sweep" in suites:
        suites.remove("size-sweep")
        if not suites:
            parser.error("size-sweep is SM80 only; no applicable StenWeave suite selected")
        print("Skipping StenWeave size-sweep: it is SM80 only.", flush=True)

    compiler = args.nvcc or os.environ.get("NVCC")
    if not compiler:
        cuda_root = os.environ.get("CUDA_ROOT")
        compiler = (
            str(Path(cuda_root).expanduser() / "bin/nvcc") if cuda_root
            else shutil.which("nvcc")
        )
    if not compiler and Path("/usr/local/cuda/bin/nvcc").is_file():
        compiler = "/usr/local/cuda/bin/nvcc"
    if compiler:
        compiler = os.path.expanduser(compiler)
        executable = shutil.which(compiler)
        if executable:
            compiler = str(Path(executable).resolve())
        elif not args.dry_run:
            parser.error(f"CUDA compiler is not executable: {compiler}")
        elif os.sep in compiler:
            compiler = str(Path(compiler).resolve())
    elif not args.dry_run:
        parser.error("nvcc not found; use --nvcc, NVCC, CUDA_ROOT, or PATH")

    commands = []
    for name in selected:
        script = ROOT / "scripts/build" / f"{name}.py"
        if not script.is_file():
            parser.error(f"build script not found: {script}")
        command = [sys.executable, str(script), "--arch", args.arch]
        if compiler:
            command += ["--nvcc", compiler]
        if name != "flashfft":
            command += ["--jobs", str(args.jobs)]
        if name == "stenweave":
            for suite in suites:
                command += ["--suite", suite]
            for option, path in (
                ("--cuassembler", args.cuassembler), ("--dependency-root", args.dependency_root),
            ):
                if path is not None:
                    command += [option, str(path.expanduser().resolve())]
        commands.append((name, command))

    for number, (name, command) in enumerate(commands, start=1):
        label = f"{IMPLEMENTATIONS[name]} ({args.arch})"
        print(f"[{number}/{len(commands)}] {label}\n  {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        try:
            result = subprocess.run(command, cwd=ROOT, check=False)
        except OSError as error:
            print(f"Could not start {label}: {error}", file=sys.stderr)
            return 1
        if result.returncode:
            logs = ROOT / "build/logs" / IMPLEMENTATIONS[name] / args.arch
            print(
                f"Build failed: {label}, exit code {result.returncode}. Logs: {logs}",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    action = "Planned" if args.dry_run else "Completed"
    print(f"{action} {len(commands)} implementation builds for {args.arch}.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Build interrupted.", file=sys.stderr)
        sys.exit(130)
