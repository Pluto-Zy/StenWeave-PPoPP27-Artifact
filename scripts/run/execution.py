"""Execute collection tasks and validate their successful measurement records."""

import json
import math
import subprocess
import sys

from .process import run
from .suites.common import IMPLEMENTATIONS, ROOT, baseline, sha


def command(task, suite, arch, device, folder):
    implementation = task["implementation"]
    argv = [
        sys.executable,
        str(ROOT / f"scripts/run/implementations/{implementation}.py"),
        "--arch",
        arch,
        "--device",
        str(device),
        "--output-dir",
        str(folder),
    ]
    if implementation == "stenweave":
        argv += ["--suite", task["native_suite"], "--case", task["case"]]
    else:
        argv += ["--stencil", task["stencil"], "--suite", suite]
        if "width" in task:
            if implementation == "an5d":
                argv += ["--width", str(task["width"])]
            else:
                dimensions = task.get("dimensions", [task["width"]] * task["rank"])
                argv += ["--dimensions", *map(str, dimensions)]
    return argv


def preflight(task, arch):
    """Check artifacts before starting a batch; never launch a GPU workload here."""
    implementation = task["implementation"]
    if sha(ROOT / task["config"]) != task["config_sha256"]:
        raise ValueError(f"Collection config changed: {task['config']}")
    if implementation == "stenweave":
        suite = task["native_suite"]
        generation = "benchmark" if suite == "roofline" else suite
        artifact = ROOT / f"build/artifacts/StenWeave/{arch}/{generation}"
        manifest_path = artifact / "manifest.json"
        build = json.loads((artifact / "build.json").read_text())
        if sha(manifest_path) != build["manifest_sha256"]:
            raise ValueError("StenWeave generator manifest changed")
        manifest = json.loads(manifest_path.read_text())
        kernel = next(k for k in manifest["kernels"] if k["id"] == task["kernel_id"])
        if kernel["arch"] != arch:
            raise ValueError("StenWeave architecture mismatch")
        cubin = artifact / kernel["cubin"]["path"]
        if (
            cubin.stat().st_size != kernel["cubin"]["size"]
            or sha(cubin) != kernel["cubin"]["sha256"]
        ):
            raise ValueError(f"StenWeave cubin differs from manifest: {cubin}")
        binary = ROOT / "build/bin/StenWeave/stencil"
        runner_build = json.loads((binary.parent / "build.json").read_text())
        if sha(binary) != runner_build["binary_sha256"]:
            raise ValueError("StenWeave runner binary changed")
        return {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                binary,
                binary.parent / "build.json",
                manifest_path,
                artifact / "build.json",
                cubin,
            )
        }
    config, path = baseline(implementation, arch)
    c = config["stencils"][task["stencil"]]
    directory = ROOT / f"build/bin/{IMPLEMENTATIONS[implementation]}/{arch}"
    manifest = json.loads((directory / "build.json").read_text())
    if manifest["arch"] != arch:
        raise ValueError("Build architecture mismatch")
    if "config_sha256" in manifest and manifest["config_sha256"] != sha(path):
        raise ValueError(f"Build/config mismatch: {implementation}")
    if implementation == "an5d":
        cap = c.get("maxrregcount", 0)
        name = c["variant"] + (f"-rreg{cap}" if cap else "")
        binary = directory / name
        expected = manifest["binaries"][name]["sha256"]
    elif implementation == "flashfft":
        binary = directory / f"flashfft{c['dim']}d"
        expected = manifest["binaries"][binary.name]
    else:
        b = manifest["binaries"][task["stencil"]]
        if b["config"] != c:
            raise ValueError(f"Frozen binary/config mismatch: {implementation}")
        entry = b if implementation == "ebisu" else b["performance"]
        binary, expected = ROOT / entry["path"], entry["sha256"]
    if not binary.is_file() or (expected is not None and sha(binary) != expected):
        raise ValueError(f"Missing or changed executable: {binary}")
    if "width" in task:
        width = task["width"]
        if implementation == "ebisu":
            sys.path.insert(0, str(ROOT / "scripts/adapters"))
            from ebisu import command as native_command

            native_command(
                binary, c, task.get("dimensions", [width] * c["dim"]), c["tstep"]
            )
        elif implementation == "lorastencil":
            if width % c["row"] or width % c["col"]:
                raise ValueError("Sweep width does not divide the selected LoRA tile")
    return {
        str(path.relative_to(ROOT)): sha(path)
        for path in (binary, directory / "build.json")
    }


def collect(task, suite, arch, device, folder, timeout):
    argv = command(task, suite, arch, device, folder)
    # Child runners create their own directory and refuse existing results.
    log_path = folder.parent / f"{task['id']}.log"
    with log_path.open("w") as log:
        run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return read_result(task, arch, folder)


def read_result(task, arch, folder):
    """Validate a task's native result without executing its workload."""
    rows = [
        json.loads(line)
        for line in (folder / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one measurement for {task['id']}, found {len(rows)}"
        )
    row = rows[0]
    if (
        row.get("arch") != arch
        or row.get("stencil") != task["stencil"]
        or row.get("implementation") != IMPLEMENTATIONS[task["implementation"]]
    ):
        raise ValueError(f"Unexpected native measurement identity: {task['id']}")
    rate = row.get("gstencils_per_s")
    if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid native measurement: {task['id']}")
    if "kernel_id" in task and row.get("config_id") != task["kernel_id"]:
        raise ValueError("Native measurement kernel ID mismatch")
    return rows
