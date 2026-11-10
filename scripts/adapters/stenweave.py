"""Validate complete native StenWeave JSONL batches and normalize measurements."""

import json
import math
import statistics
from pathlib import Path


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def read_batch(path: Path, expected: list[str], arch: str) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != len(expected) + 2:
        raise ValueError("Incomplete or unexpected native JSONL record count")
    header, *cases, end = records
    if (
        header.get("type") != "run"
        or header.get("schema_version") != 2
        or header.get("sampling_protocol") != "cooldown_warmup_per_repeat"
        or header.get("case_count") != len(expected)
    ):
        raise ValueError("Unsupported native run header or sampling protocol")
    if (
        end.get("type") != "end"
        or end.get("status") != "completed"
        or end.get("case_count") != len(expected)
        or end.get("validation_failures") != 0
    ):
        raise ValueError("Native run did not complete successfully")
    for case, name in zip(cases, expected, strict=True):
        if case.get("type") != "case" or case.get("name") != name or case["kernel"]["arch"] != arch:
            raise ValueError(f"Unexpected native case: {name}")
        run = case["run"]
        samples = case["samples"]
        count = run["measurement_runs"]
        if not (len(samples) == run["repeat_runs"] and 0 < count <= len(samples)):
            raise ValueError(f"Invalid sample count: {name}")
        for sample in samples:
            ms, rate = sample["time_ms"], sample["gstencils_per_s"]
            if (
                not positive(ms)
                or not positive(rate)
                or not math.isclose(rate, case["stencil_updates"] / ms / 1e6, rel_tol=1e-10)
            ):
                raise ValueError(f"Invalid timing/work accounting: {name}")
        selected = case["summary"]["selected"]
        statistic = selected["statistic"]
        if statistic not in ("mean", "median") or statistic != run["measurement_statistic"]:
            raise ValueError(f"Unexpected primary statistic: {name}")
        summarize = statistics.mean if statistic == "mean" else statistics.median
        for key in ("time_ms", "gstencils_per_s"):
            if not math.isclose(
                selected[key],
                summarize(s[key] for s in samples[-count:]),
                rel_tol=1e-10,
            ):
                raise ValueError(f"Invalid measurement window summary: {name}")
        if run["check"] and case["validation"].get("status") != "passed":
            raise ValueError(f"Native CPU comparison did not pass: {name}")
    return cases


def normalize(case: dict, suite: str, config: str, command: list[str], log: str) -> dict:
    kernel = case["kernel"]
    if kernel["profile"]["name"] == "breakdown":
        variant = f"breakdown-stage{kernel['profile']['stage']}"
    elif kernel["backend"] == "cuda":
        variant = "CUDA"
    else:
        group = kernel["group"]
        variant = "SASS-ILP" if any(v > 1 for v in group.values()) else "SASS"
    return {
        "experiment": suite,
        "implementation": "StenWeave",
        "variant": variant,
        "arch": kernel["arch"],
        "stencil": f"{kernel['shape']}{kernel['dimensions']}d{kernel['radius']}r",
        "config_id": kernel["id"],
        "case": case["name"],
        "config": config,
        "gstencils_per_s": case["summary"]["selected"]["gstencils_per_s"],
        "statistic": case["summary"]["selected"]["statistic"],
        "command": command,
        "log_path": log,
        "native": case,
    }
