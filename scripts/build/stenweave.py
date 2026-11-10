"""Build the pinned StenWeave runner and experiment cubins independently."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("benchmark", "size-sweep", "time-block-sweep", "breakdown")


def cuassembler_path(override=None):
    if override is not None:
        directory = override.expanduser().resolve(strict=True)
    else:
        directory = ROOT / "vendor/CuAssembler"
        for line in (ROOT / "provenance/CuAssembler.sha256").read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            if sha(ROOT / path) != digest:
                raise RuntimeError(f"Source checksum mismatch: {path}")
    if not (directory / "CuAsm/CuAsmParser.py").is_file():
        raise ValueError(
            f"CuAssembler directory must contain CuAsm/CuAsmParser.py: {directory}"
        )
    return directory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_source(build_root=None):
    vendor = ROOT / "vendor/StenWeave"
    for line in (ROOT / "provenance/StenWeave.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"Source checksum mismatch: {path}")
    source = (build_root or ROOT / "build") / "src/StenWeave"
    source.parent.mkdir(parents=True, exist_ok=True)
    patchdir = ROOT / "provenance/patches/StenWeave"
    patches = (patchdir / "series").read_text().splitlines()
    with tempfile.TemporaryDirectory(
        dir=source.parent, prefix="stenweave-"
    ) as directory:
        stage = Path(directory) / "src"
        shutil.copytree(vendor, stage)
        for patch in patches:
            subprocess.run(
                ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patchdir / patch)],
                cwd=stage,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        for item in stage.rglob("*"):
            if item.is_file():
                dest = source / item.relative_to(stage)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.read_bytes() != item.read_bytes():
                    shutil.copyfile(item, dest)
                    dest.chmod(0o644)
    return source, {p: sha(patchdir / p) for p in patches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["sm80", "sm90"], required=True)
    parser.add_argument(
        "--component", choices=["all", "runner", "generator"], default="all"
    )
    parser.add_argument("--suite", choices=[*SUITES, "all"], action="append")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--nvcc", default=os.environ.get("NVCC"))
    parser.add_argument(
        "--cuassembler",
        type=Path,
        help="override bundled CuAssembler with a checkout containing CuAsm/",
    )
    parser.add_argument(
        "--dependency-root", type=Path, help="existing CMake _deps source cache"
    )
    parser.add_argument(
        "--build-root", type=Path, help="independent build tree (default: build/)"
    )
    parser.add_argument(
        "--generator-config", type=Path, help="alternative generator TOML for one suite"
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    suites = (
        list(SUITES) if "all" in (args.suite or []) else args.suite or ["benchmark"]
    )
    if args.arch == "sm90":
        if suites == ["size-sweep"]:
            parser.error("size-sweep is an SM80-only experiment")
        suites = [s for s in suites if s != "size-sweep"]
    suites = list(dict.fromkeys(suites))
    if args.generator_config and (len(suites) != 1 or args.component == "runner"):
        parser.error("--generator-config requires one generator suite")
    build_root = (args.build_root or ROOT / "build").resolve()
    nvcc = args.nvcc or shutil.which("nvcc")
    if not nvcc:
        cuda_root = Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda"))
        nvcc = str(cuda_root / "bin/nvcc")
    nvcc = Path(nvcc).resolve(strict=True)
    env = dict(os.environ)
    env["PATH"] = str(nvcc.parent) + os.pathsep + env.get("PATH", "")
    source, patches = prepare_source(build_root)
    imports = [str(source / "generator")]
    assembler = None
    if (
        args.component in ("all", "generator")
        and args.arch == "sm80"
        and any(s != "breakdown" for s in suites)
    ):
        assembler = cuassembler_path(args.cuassembler)
        imports.append(str(assembler))
    if env.get("PYTHONPATH"):
        imports.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(imports)
    logs = build_root / "logs/StenWeave" / args.arch
    logs.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, name):
        log = logs / f"{name}.log"
        print(f"Building {name}; log: {log}", flush=True)
        commands.append(command)
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Build failed ({result.returncode}); see {log}")

    info = {
        "arch": args.arch,
        "patches": patches,
        "source_manifest_sha256": sha(ROOT / "provenance/StenWeave.sha256"),
        "compiler": subprocess.check_output([str(nvcc), "--version"], text=True),
        "commands": commands,
    }
    if assembler is not None:
        info["cuassembler"] = {
            "path": str(assembler),
            "bundled": args.cuassembler is None,
            "source_manifest_sha256": sha(ROOT / "provenance/CuAssembler.sha256")
            if args.cuassembler is None
            else None,
        }
    if args.component in ("all", "runner"):
        directory = build_root / "StenWeave/runner"
        command = [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(directory),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCUDAToolkit_ROOT={nvcc.parent.parent}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
        if args.dependency_root:
            for name in ("cli11", "fmt", "json", "tomlplusplus"):
                dep = (args.dependency_root / f"{name}-src").resolve(strict=True)
                command.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={dep}")
        run(command, "configure-runner")
        run(
            ["cmake", "--build", str(directory), "--parallel", str(args.jobs)], "runner"
        )
        target = build_root / "bin/StenWeave"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / "runner/stencil", target / "stencil")
        info["binary_sha256"] = sha(target / "stencil")
        (target / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    if args.component in ("all", "generator"):
        if args.arch == "sm80" and any(s != "breakdown" for s in suites):
            subprocess.run(
                [sys.executable, "-c", "from CuAsm.CuAsmParser import CuAsmParser"],
                env=env,
                check=True,
            )
        for suite in suites:
            config = (
                args.generator_config
                or ROOT / f"configs/stenweave/generator/{suite}.toml"
            ).resolve(strict=True)
            output = build_root / f"artifacts/StenWeave/{args.arch}/{suite}"
            run(
                [
                    sys.executable,
                    "-m",
                    "codegen",
                    "--config",
                    str(config),
                    "--arch",
                    args.arch,
                    "--output",
                    str(output),
                    "--jobs",
                    str(args.jobs),
                ],
                f"{suite}-generator",
            )
            manifest = json.loads((output / "manifest.json").read_text())
            build = {
                **info,
                "config_sha256": sha(config),
                "manifest_sha256": sha(output / "manifest.json"),
                "kernel_count": len(manifest["kernels"]),
            }
            (output / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    print(f"Built StenWeave {args.arch}", flush=True)


if __name__ == "__main__":
    main()
