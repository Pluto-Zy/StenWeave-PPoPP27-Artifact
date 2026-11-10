"""Read complete profile collections and verify their report/case associations."""

import hashlib
import re
from pathlib import Path

from run import profile
from run.implementations.profile_targets import target as native_target
from run.suites import low_level_metrics, roofline, time_block_sweep
from run.suites.common import IMPLEMENTATIONS

from .catalog import ROOT, tomllib
from .data import read_rows
from .roofline import calculate

SUITES = {
    "roofline": roofline,
    "time-block-sweep": time_block_sweep,
    "low-level-metrics": low_level_metrics,
}


def expected(arches, read, suite="roofline"):
    result = {}
    for arch in arches:
        if arch not in SUITES[suite].ARCHES:
            continue
        for task in SUITES[suite].plan(arch):
            read(ROOT / task["config"])
            result[(arch, task["id"])] = task
    return result


def read_input(path, inputs, suites=("roofline",)):
    directory = path if path.is_dir() else path.parent
    source = directory / "profiles.jsonl" if path.is_dir() else path
    rows = read_rows(source, inputs)
    if (
        path.is_dir()
        or (directory / "plan.json").exists()
        or (directory / "status.json").exists()
    ):
        plan = inputs.json(directory / "plan.json")
        state = inputs.json(directory / "status.json")
        tasks = plan.get("tasks", [])
        ids = [task["id"] for task in tasks]
        if (
            plan.get("schema_version") != 1
            or plan.get("suite") not in suites
            or plan.get("kind") != "profile"
            or state.get("status") != "completed"
            or not ids
            or len(set(ids)) != len(ids)
            or state.get("completed") != len(ids)
            or state.get("task_count") != len(ids)
            or state.get("tasks") != dict.fromkeys(ids, "completed")
            or source != directory / "profiles.jsonl"
        ):
            raise ValueError(
                f"Not a complete requested profile collection: {directory}"
            )
        found = [row.get("task_id") for _, row in rows]
        if len(found) != len(ids) or set(found) != set(ids):
            raise ValueError("Profile index disagrees with collection task list")
        by_id = {task["id"]: task for task in tasks}
        for _, row in rows:
            if (
                row.get("arch") != plan["arch"]
                or row.get("run_id") != plan["run_id"]
                or any(
                    row.get("collection_case", {}).get(k) != v
                    for k, v in by_id[row["task_id"]].items()
                    if k not in ("command", "export_command", "launch_selection")
                )
            ):
                raise ValueError("Profile row disagrees with collection plan")
    if not rows:
        raise ValueError(f"Empty profile source: {source}")
    return source, rows


def profile_file(row, field, source):
    """Resolve portable indexes and older absolute indexes after a batch is copied."""
    recorded = Path(row[field])
    if not recorded.is_absolute():
        return source.parent / recorded
    # Prefer the accompanying files, even if the original machine's path exists.
    local = source.parent / "tasks" / row["task_id"] / recorded.name
    if local.is_file():
        return local
    return recorded


def validate_options(command, native):
    """Check collection semantics while allowing executable paths and extra options."""
    required = {
        "--replay-mode": "kernel",
        "--graph-profiling": "node",
        "--clock-control": "base",
        "--cache-control": "none",
        "--kernel-name": native["kernel_filter"],
        "--launch-skip": str(native["launch_skip"]),
        "--launch-count": "1",
    }
    for option, expected_value in required.items():
        values = []
        for index, argument in enumerate(command):
            if argument == option and index + 1 < len(command):
                values.append(command[index + 1])
            elif argument.startswith(option + "="):
                values.append(argument.split("=", 1)[1])
        if not values or any(value != expected_value for value in values):
            raise ValueError(f"Profile collection option mismatch: {option}")


def normalize(row, source, line, inputs, suites=("roofline",)):
    arch = row.get("arch")
    suite = row.get("experiment")
    task = row.get("collection_case", {})
    current = (
        {t["id"]: t for t in SUITES[suite].plan(arch)}
        if suite in suites and suite in SUITES and arch in SUITES[suite].ARCHES
        else {}
    )
    selected = current.get(row.get("task_id"))
    if (
        row.get("kind") != "profile"
        or row.get("suite") != suite
        or suite not in suites
        or selected is None
        or any(task.get(k) != v for k, v in selected.items())
        or row.get("implementation") != IMPLEMENTATIONS[selected["implementation"]]
        or row.get("stencil") != selected["stencil"]
    ):
        raise ValueError(f"Profile identity/configuration mismatch: {source}:{line}")
    config_bytes = inputs.read(ROOT / selected["config"])
    digest = hashlib.sha256(config_bytes).hexdigest()
    if digest != selected["config_sha256"]:
        raise ValueError("Profile uses a different frozen configuration")
    for field, checksum in (
        ("report_path", "report_sha256"),
        ("raw_csv", "raw_csv_sha256"),
    ):
        data = inputs.read(profile_file(row, field, source))
        if hashlib.sha256(data).hexdigest() != row[checksum]:
            raise ValueError(f"Profile hash mismatch: {row[field]}")
    target_path = profile_file(row, "target_path", source)
    target = inputs.json(target_path)
    # Paths, device ordinals, environment and executable locations are provenance,
    # not requirements on the machine performing offline analysis.
    native = native_target(selected, arch, 0, target_path.parent, verify=False)
    validate_options(target["profile_command"], native)
    metrics = (
        profile.LOW_LEVEL_METRICS
        if suite == "low-level-metrics"
        else profile.ROOFLINE_METRICS
    )
    for metadata in (row, target):
        if not set(metrics).issubset(metadata.get("metrics", [])):
            raise ValueError("Profile is missing required metrics")
    for field in (
        "launch_skip",
        "launch_count",
        "kernel_id",
        "parameters",
        "sampling_overrides",
    ):
        if target.get(field) != native.get(field):
            raise ValueError(f"Profile workload mismatch: {field}")
    if "config_text" in native:
        content = inputs.read(target_path.parent / "batch.toml")
        if tomllib.loads(content.decode()) != tomllib.loads(native["config_text"]):
            raise ValueError("Profile effective StenWeave configuration mismatch")
        if hashlib.sha256(content).hexdigest() != target.get("effective_config_sha256"):
            raise ValueError("Profile effective configuration hash mismatch")
    calculator = calculate
    if suite == "low-level-metrics":
        from .low_level import calculate as calculator

    result = calculator(
        inputs.read(profile_file(row, "raw_csv", source)).decode(), arch
    )
    if suite == "low-level-metrics":
        result["collection"] = selected["collection"]
    if (
        row.get("report_ids") != [result["report_id"]]
        or row.get("kernel_names") != [result["kernel_name"]]
        or not re.search(
            native["kernel_filter"].removeprefix("regex:"), result["kernel_name"]
        )
    ):
        raise ValueError("Profile report ID/kernel association mismatch")
    if suite == "time-block-sweep":
        config = tomllib.loads(config_bytes.decode())
        case = next(c for c in config["case"] if c["name"] == selected["case"])
        depth = case["run"]["time_steps"]
        if not selected["case"].endswith(f"-tb{depth:02d}"):
            raise ValueError("Temporal depth disagrees with frozen case name")
        result["time_block_size"] = depth
    return {
        **result,
        "experiment": suite,
        "arch": arch,
        "implementation": row["implementation"],
        "stencil": row["stencil"],
        "variant": {"sass": "SASS", "sass-ilp": "SASS-ILP"}.get(
            selected.get("variant"), "native"
        ),
        "config_id": selected.get("kernel_id"),
        "task_id": row["task_id"],
        "report_path": str(profile_file(row, "report_path", source).resolve()),
        "report_sha256": row["report_sha256"],
        "raw_csv": str(profile_file(row, "raw_csv", source).resolve()),
        "raw_csv_sha256": row["raw_csv_sha256"],
        "selection": native["selection"],
        "source_file": str(source),
        "source_row": line,
    }
