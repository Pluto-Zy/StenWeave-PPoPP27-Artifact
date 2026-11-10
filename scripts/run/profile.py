"""Collect native NCU reports and raw counters without computing derived metrics."""

import csv
import io
import math
import os
import re
import subprocess

from .implementations.profile_targets import target, validate_native
from .process import run
from .suites.common import IMPLEMENTATIONS, ROOT, sha, write_json

ROOFLINE_METRICS = (
    "sass__thread_inst_executed_true_per_opcode",
    "sass__thread_inst_executed_true_per_opcode_with_modifier_all",
    "gpc__cycles_elapsed.max",
    "gpc__cycles_elapsed.avg.per_second",
    "launch__sm_count",
    "device__attribute_compute_capability_major",
    "device__attribute_compute_capability_minor",
)
STALL_REASONS = (
    "barrier",
    "branch_resolving",
    "dispatch_stall",
    "drain",
    "imc_miss",
    "lg_throttle",
    "long_scoreboard",
    "math_pipe_throttle",
    "membar",
    "mio_throttle",
    "misc",
    "no_instructions",
    "not_selected",
    "selected",
    "short_scoreboard",
    "sleeping",
    "tex_throttle",
    "wait",
)
LOW_LEVEL_METRICS = (
    ROOFLINE_METRICS
    + (
        "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
        "l1tex__throughput.avg.pct_of_peak_sustained_active",
        "smsp__pcsamp_sample_count",
        "smsp__pcsamp_dropped_bytes",
        "smsp__pcsamp_interval_cycles",
        "gpu__time_duration.sum",
    )
    + tuple(
        f"smsp__pcsamp_warps_issue_stalled_{reason}_not_issued"
        for reason in STALL_REASONS
    )
)


def commands(task, arch, device, folder, ncu, suite, verify=True, build_root=None):
    native = target(task, arch, device, folder, verify=verify, build_root=build_root)
    metrics = LOW_LEVEL_METRICS if suite == "low-level-metrics" else ROOFLINE_METRICS
    command = [
        ncu,
        "--config-file",
        "0",
        "--replay-mode",
        "kernel",
        "--graph-profiling",
        "node",
        "--clock-control",
        "base",
        "--cache-control",
        "none",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        native["kernel_filter"],
        "--launch-skip",
        str(native["launch_skip"]),
        "--launch-count",
        "1",
        "--kill",
        "no",
        "--metrics",
        ",".join(metrics),
        "--export",
        str(folder / "report"),
        *native["command"],
    ]
    export = [
        ncu,
        "--config-file",
        "0",
        "--import",
        str(folder / "report.ncu-rep"),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--print-metric-instances",
        "details",
        "--print-kernel-base",
        "function",
    ]
    return native, command, export, metrics


def read_raw(text, low_level=False, arch=None):
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if header is None:
        raise ValueError("NCU raw CSV has no metric header")
    csv.field_size_limit(100_000_000)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    rows = [row for row in rows if row.get("ID", "").isdigit()]
    ids = {row["ID"] for row in rows}
    if len(ids) != 1 or len(rows) != 1:
        raise ValueError(
            f"Expected one profiled launch, found report IDs {sorted(ids)}"
        )
    row = rows[0]

    def number(name):
        # --print-metric-instances details appends instances to scalar aggregates too.
        value = float(row.get(name, "").split("(", 1)[0].strip().replace(",", ""))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite NCU counter: {name}")
        return value

    for name in ROOFLINE_METRICS[2:5]:
        if number(name) <= 0:
            raise ValueError(f"Invalid NCU counter: {name}")
    for name in ROOFLINE_METRICS[:2]:
        value = row.get(name, "").strip()
        start = value.find("(")
        if start < 0 or not value.endswith(")"):
            raise ValueError(f"Missing detailed opcode instances: {name}")
        instances = [
            entry.strip() for entry in value[start + 1 : -1].split(";") if entry.strip()
        ]
        if not instances:
            raise ValueError(f"Empty opcode instances: {name}")
        for entry in instances:
            label, count = entry.rsplit(":", 1)
            count = float(count.replace(",", ""))
            if not label.strip() or not math.isfinite(count) or count < 0:
                raise ValueError(f"Invalid opcode instance: {entry}")
    capability = tuple(int(number(name)) for name in ROOFLINE_METRICS[5:])
    if arch and capability != {"sm80": (8, 0), "sm90": (9, 0)}[arch]:
        raise ValueError(f"NCU report architecture mismatch: {capability}")
    if low_level:
        if number("smsp__pcsamp_dropped_bytes") != 0:
            raise ValueError("NCU PC sampling dropped data")
        if number("smsp__pcsamp_sample_count") <= 0:
            raise ValueError("NCU PC sampling collected no samples")
        for name in LOW_LEVEL_METRICS[len(ROOFLINE_METRICS) :]:
            if number(name) < 0:
                raise ValueError(f"Negative NCU counter: {name}")
    name = row.get("Kernel Name", "").strip()
    if not name:
        raise ValueError("NCU report has no kernel name")
    return sorted(ids), [name]


def collect(task, suite, arch, device, folder, ncu, timeout, build_root=None):
    native, command, export, metrics = commands(
        task, arch, device, folder, ncu, suite, build_root=build_root
    )
    folder.mkdir(parents=True, exist_ok=False)
    if "config_text" in native:
        (folder / "batch.toml").write_text(native["config_text"])
    metadata = {k: v for k, v in native.items() if k != "config_text"}
    if "config_text" in native:
        metadata["effective_config_sha256"] = sha(folder / "batch.toml")
    metadata.update(profile_command=command, export_command=export, metrics=metrics)
    write_json(folder / "target.json", metadata)
    environment = dict(os.environ, **native["environment"])
    with (folder / "profile.log").open("w") as log:
        run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    report = folder / "report.ncu-rep"
    if not report.is_file() or not report.stat().st_size:
        raise ValueError(f"NCU did not produce a report: {folder}")
    validate_native(task, native, folder, arch)
    with (
        (folder / "raw.csv").open("w") as output,
        (folder / "export.log").open("w") as log,
    ):
        run(export, cwd=ROOT, stdout=output, stderr=log, timeout=timeout)
    return read_result(task, suite, arch, folder)


def read_result(task, suite, arch, folder):
    """Revalidate completed native output, report and counters without NCU."""
    import json

    native = json.loads((folder / "target.json").read_text())
    report = folder / "report.ncu-rep"
    if not report.is_file() or not report.stat().st_size:
        raise ValueError(f"Missing NCU report: {folder}")
    validate_native(task, native, folder, arch)
    ids, names = read_raw(
        (folder / "raw.csv").read_text(), suite == "low-level-metrics", arch
    )
    if any(
        not re.search(native["kernel_filter"].removeprefix("regex:"), name)
        for name in names
    ):
        raise ValueError(f"NCU reported an unexpected kernel: {names}")
    return [
        {
            "implementation": IMPLEMENTATIONS[task["implementation"]],
            "stencil": task["stencil"],
            "arch": arch,
            "kind": "profile",
            "report_path": f"tasks/{task['id']}/report.ncu-rep",
            "report_sha256": sha(report),
            "raw_csv": f"tasks/{task['id']}/raw.csv",
            "raw_csv_sha256": sha(folder / "raw.csv"),
            "report_ids": ids,
            "kernel_names": names,
            "target_path": f"tasks/{task['id']}/target.json",
            "selection": native["selection"],
            "metrics": native["metrics"],
        }
    ]
