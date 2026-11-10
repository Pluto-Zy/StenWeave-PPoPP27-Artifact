"""Native targets and explicit launch selection for NCU collection."""

import json

from ..suites.common import ROOT, baseline, selected_case_text, sha


def verified_binary(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Binary differs from build manifest: {path}")


def target(task, arch, device, folder, verify=True, build_root=None):
    build_root = build_root or ROOT / "build"
    implementation = task["implementation"]
    environment = {"OMP_NUM_THREADS": "8"}
    result = {"launch_skip": 0, "launch_count": 1, "sampling_overrides": {}}
    if implementation == "stenweave":
        suite = task["native_suite"]
        generation = "benchmark" if suite == "roofline" else suite
        artifact = build_root / f"artifacts/StenWeave/{arch}/{generation}"
        binary = build_root / "bin/StenWeave/stencil"
        text = selected_case_text(ROOT / task["config"], task["case"], profile=True)
        result.update(
            config_text=text,
            kernel_filter="regex:^stencil_kernel$",
            selection="first matching kernel launch from prepared workload; representative node",
            kernel_id=task["kernel_id"],
            artifact=str(artifact),
            sampling_overrides={
                "warmup_runs": 0,
                "repeat_runs": 1,
                "measurement_runs": 1,
                "check": False,
            },
        )
        command = [
            str(binary),
            "run",
            str(folder / "batch.toml"),
            "--artifact-dir",
            str(artifact),
            "--device",
            str(device),
            "--output",
            str(folder / "native.jsonl"),
            "--input-threads",
            "8",
            "--no-color",
        ]
        if verify:
            build = json.loads((binary.parent / "build.json").read_text())
            verified_binary(binary, build["binary_sha256"])
            generator = json.loads((artifact / "build.json").read_text())
            if sha(artifact / "manifest.json") != generator["manifest_sha256"]:
                raise ValueError("Generator manifest differs from build record")
            manifest = json.loads((artifact / "manifest.json").read_text())
            kernel = next(
                k for k in manifest["kernels"] if k["id"] == task["kernel_id"]
            )
            if kernel["arch"] != arch:
                raise ValueError("Profile kernel architecture mismatch")
            result.update(
                build=build,
                generator_build=generator,
                manifest_sha256=sha(artifact / "manifest.json"),
            )
    else:
        config, path = baseline(implementation, arch)
        c = config["stencils"][task["stencil"]]
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        if implementation == "an5d":
            cap = c.get("maxrregcount", 0)
            name = c["variant"] + (f"-rreg{cap}" if cap else "")
            binary = build_root / f"bin/AN5D/{arch}/{name}"
            steps, bt = config["steps"], c["bt"]
            # Generated hosts use this parity correction in the full-depth tile loop.
            correction = int(
                (steps % 2 != ((steps + bt - 1) // bt) % 2) and steps % bt < 2
            )
            skip = steps // bt - correction
            if skip < 1:
                raise ValueError("No full-depth AN5D temporal tile is available")
            source = ROOT / f"vendor/AN5D/compiled/double/{c['variant']}_host.cu"
            if "__c0Len / __side0Len - __c0Padr" not in source.read_text():
                raise ValueError("Unrecognized AN5D full-depth launch loop")
            command = [str(binary), "-s", str(c["width"]), "-t", str(steps), "-n", "1"]
            result.update(
                kernel_filter=f"regex:^kernel0_{bt}$",
                launch_skip=skip,
                selection="first full-depth tile in measured invocation, after native warmup",
                sampling_overrides={"repeats": 1},
                parameters=c,
                steps=steps,
            )
        elif implementation == "ebisu":
            binary = build_root / f"bin/EBISU/{arch}/{task['stencil']}/{c['target']}"
            command = [
                str(binary),
                *map(str, c["dimensions"]),
                f"--iter={c['tstep']}",
                "--bdim=256",
                "--verbose",
            ]
            result.update(
                kernel_filter=(
                    "regex:^kernel_temporal_traditional"
                    if c["dim"] == 2
                    else "regex:^kernel3d_temporal"
                ),
                selection="single temporal launch; native warmup disabled",
                sampling_overrides={"warmup": False, "repeats": 1},
                parameters=c,
                steps=c["tstep"],
            )
        else:
            raise ValueError(f"No NCU target adapter for {implementation}")
        if verify:
            manifest = json.loads(
                (
                    binary.parent / "build.json"
                    if implementation == "an5d"
                    else binary.parent.parent / "build.json"
                ).read_text()
            )
            if manifest["arch"] != arch:
                raise ValueError("Profile build architecture mismatch")
            key = name if implementation == "an5d" else task["stencil"]
            b = manifest["binaries"][key]
            verified_binary(binary, b["sha256"])
            if implementation == "ebisu" and (
                manifest["config_sha256"] != sha(path) or b["config"] != c
            ):
                raise ValueError("EBISU profile build/config mismatch")
            if implementation == "an5d" and b.get("maxrregcount", cap) != cap:
                raise ValueError("AN5D register cap mismatch")
            result.update(build=manifest, binary_sha256=b["sha256"])
    result.update(command=command, environment=environment)
    return result


def validate_native(task, native_target, folder, arch):
    """Profiled event times are only checked for completion, never used as performance."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts/adapters"))
    if task["implementation"] == "stenweave":
        from stenweave import read_batch

        cases = read_batch(folder / "native.jsonl", [task["case"]], arch)
        if cases[0]["kernel"]["id"] != task["kernel_id"]:
            raise ValueError("Profiled native kernel ID mismatch")
    else:
        output = (folder / "profile.log").read_text()
        c = native_target["parameters"]
        if task["implementation"] == "an5d":
            from an5d import parse_output

            parse_output(output, 1, c["width"], native_target["steps"], task["stencil"])
        else:
            from ebisu import parse_output

            parse_output(output, c, c["dimensions"], native_target["steps"])
