"""Run prepared StenWeave suites, preserving native records and measurement windows."""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "adapters"))
from stenweave import normalize, read_batch

ROOT = Path(__file__).resolve().parents[3]
SUITES = ("benchmark", "size-sweep", "roofline", "time-block-sweep", "breakdown")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["sm80", "sm90"], required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--suite", choices=SUITES, default="benchmark")
    parser.add_argument(
        "--case", action="append", help="exact case name; repeat to select several"
    )
    parser.add_argument(
        "--config", type=Path, help="explicit alternative native batch TOML"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-threads", type=int, default=8)
    parser.add_argument(
        "--check", action="store_true", help="small isolated CPU-reference checks"
    )
    parser.add_argument(
        "--build-root", type=Path, help="independent build tree (default: build/)"
    )
    args = parser.parse_args()
    if args.device < 0 or args.input_threads < 0:
        parser.error("device and input-threads must be nonnegative")
    if args.suite == "size-sweep" and args.arch != "sm80":
        parser.error("size-sweep is an SM80-only experiment")
    build_root = (args.build_root or ROOT / "build").resolve()
    config = args.config or ROOT / f"configs/stenweave/runner/{args.arch}/{args.suite}.toml"
    config = config.resolve(strict=True)
    text = config.read_text()
    batch = tomllib.loads(text)
    names = [case["name"] for case in batch["case"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate native case names")
    chosen = set(args.case or names)
    if chosen - set(names):
        parser.error(f"Unknown cases: {sorted(chosen - set(names))}")
    names = [name for name in names if name in chosen]
    generation_suite = "benchmark" if args.suite == "roofline" else args.suite
    artifact = build_root / f"artifacts/StenWeave/{args.arch}/{generation_suite}"
    manifest = json.loads((artifact / "manifest.json").read_text())
    kernels = {k["id"]: k for k in manifest["kernels"]}
    generator_build = json.loads((artifact / "build.json").read_text())
    if sha(artifact / "manifest.json") != generator_build["manifest_sha256"]:
        raise ValueError("Generator manifest does not match its build record")
    expected_ids = {
        case["name"]: case["kernel"]["id"] for case in batch["case"] if case["name"] in chosen
    }
    for kernel_id in expected_ids.values():
        if kernels[kernel_id]["arch"] != args.arch:
            raise ValueError("Selected kernel does not match the requested architecture")
    binary = build_root / "bin/StenWeave/stencil"
    build = json.loads((binary.parent / "build.json").read_text())
    if sha(binary) != build["binary_sha256"]:
        raise ValueError("Runner binary does not match its build record")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = args.output_dir or ROOT / "runs" / f"stenweave-{args.arch}-{args.suite}-{stamp}"
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    # Retain the original TOML syntax and exact selected case blocks.
    blocks = text.split("[[case]]")
    selected_text = blocks[0] + "".join(
        "[[case]]" + block
        for block in blocks[1:]
        if tomllib.loads("[[case]]" + block)["case"][0]["name"] in chosen
    )
    if args.check:
        selected_text = f"schema_version = 2\nseed = {batch.get('seed', 42)}\n"
        for case in batch["case"]:
            if case["name"] not in chosen:
                continue
            kernel = kernels[case["kernel"]["id"]]
            block = case["run"].get("block", batch.get("defaults", {}).get("run", {}).get("block"))
            block_text = ", ".join(f"{axis} = {value}" for axis, value in block.items())
            grid = "x = 1" + (", y = 1" if kernel["dimensions"] == 3 else "")
            selected_text += (
                f"\n[[case]]\nname = {json.dumps(case['name'])}\n[case.kernel]\n"
                f"id = {json.dumps(kernel['id'])}\n[case.run]\nblock = {{ {block_text} }}\n"
                f"bands = [{{ grid = {{ {grid} }}, stream = 256 }}]\n"
                f"time_steps = {kernel['time_block_size']}\ncheck = true\n"
                "tolerance = 1e-7\nwarmup_runs = 0\nrepeat_runs = 1\n"
                'measurement_runs = 1\nmeasurement_statistic = "mean"\ncooldown_ms = 0\n'
            )
    run_config = folder / "batch.toml"
    run_config.write_text(selected_text)
    command = [
        str(binary),
        "run",
        str(run_config),
        "--artifact-dir",
        str(artifact),
        "--device",
        str(args.device),
        "--output",
        str(folder / "native.jsonl"),
        "--input-threads",
        str(args.input_threads),
        "--no-color",
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (folder / "inspect.log").open("w") as log:
        subprocess.run(
            [
                str(binary),
                "inspect",
                "--device",
                str(args.device),
                "--output",
                str(folder / "gpu.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    environment = {
        "utc": stamp,
        "arch": args.arch,
        "suite": args.suite,
        "check": args.check,
        "config": str(config),
        "config_sha256": sha(config),
        "effective_config_sha256": sha(run_config),
        "build": build,
        "manifest_sha256": sha(artifact / "manifest.json"),
        "generator_build": generator_build,
    }
    (folder / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    log_path = folder / "runner.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise SystemExit(f"StenWeave failed ({result.returncode}); see {log_path}")
    cases = read_batch(folder / "native.jsonl", names, args.arch)
    if any(case["kernel"]["id"] != expected_ids[case["name"]] for case in cases):
        raise ValueError("Native results do not match the selected kernel IDs")
    if not args.check:
        rows = [
            normalize(case, args.suite, str(config), command, str(folder / "native.jsonl"))
            for case in cases
        ]
        with (folder / "results.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        f"Completed {len(cases)} {'checks' if args.check else 'cases'}: {folder}",
        flush=True,
    )


if __name__ == "__main__":
    main()
