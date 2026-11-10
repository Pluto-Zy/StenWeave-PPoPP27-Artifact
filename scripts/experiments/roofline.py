"""Instruction-result/shared-memory roofline for regular FP64 stencils.

DMUL and DFMA each count as one executed result; DADD is excluded. Shared
traffic counts predicated-on LDS/STS/LDGSTS threads at the SASS access width.
"""

import csv
import io
import math

OPCODES = "sass__thread_inst_executed_true_per_opcode"
MODIFIERS = "sass__thread_inst_executed_true_per_opcode_with_modifier_all"
FP64_PER_CYCLE = {"sm80": 32, "sm90": 64}
SHARED_PER_CYCLE = 128


def number(row, name, *, allow_zero=False):
    value = float(row.get(name, "").split("(", 1)[0].strip().replace(",", ""))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"Invalid NCU counter: {name}")
    return value


def instances(row, metric):
    text = row.get(metric, "").strip()
    start = text.find("(")
    if start < 0 or not text.endswith(")"):
        raise ValueError(f"Missing detailed opcode instances: {metric}")
    result = {}
    for entry in text[start + 1 : -1].split(";"):
        if not entry.strip():
            continue
        name, count = entry.rsplit(":", 1)
        name, count = name.strip().upper(), float(count.replace(",", ""))
        if not name or name in result or not math.isfinite(count) or count < 0:
            raise ValueError(f"Invalid or duplicate opcode instance: {entry}")
        result[name] = count
    if not result:
        raise ValueError(f"Empty opcode instances: {metric}")
    return result


def raw_row(text):
    """Read exactly one launch from a detailed NCU raw CSV export."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError("NCU CSV has no raw header")
    csv.field_size_limit(100_000_000)
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    rows = [row for row in rows if row.get("ID", "").isdigit()]
    if len(rows) != 1:
        raise ValueError("Expected exactly one profiled launch")
    return rows[0]


def calculate(text, arch):
    row = raw_row(text)
    capability = tuple(
        number(row, f"device__attribute_compute_capability_{axis}", allow_zero=True)
        for axis in ("major", "minor")
    )
    if capability != {"sm80": (8, 0), "sm90": (9, 0)}[arch]:
        raise ValueError("NCU compute capability disagrees with requested architecture")
    sm_count = number(row, "launch__sm_count")
    if not sm_count.is_integer():
        raise ValueError("NCU SM count must be integral")
    cycles = number(row, "gpc__cycles_elapsed.max")
    clock = number(row, "gpc__cycles_elapsed.avg.per_second")
    counts, modifiers = instances(row, OPCODES), instances(row, MODIFIERS)
    work = counts.get("DMUL", 0) + counts.get("DFMA", 0)
    shared = 0.0
    for opcode, count in modifiers.items():
        parts = opcode.split(".")
        base = parts[0]
        if base in ("LDSM", "STSM", "ATOMS", "REDS"):
            raise ValueError(f"Unsupported shared-memory opcode: {opcode}")
        if base in ("LDS", "STS", "LDGSTS"):
            width = 32
            for part in parts[1:]:
                if part in ("8", "16", "32", "64", "128"):
                    width = int(part)
            shared += count * width / 8
    if work <= 0 or shared <= 0:
        raise ValueError("Roofline requires positive DMUL/DFMA work and shared traffic")
    seconds = cycles / clock
    intensity = work / shared
    compute_peak = FP64_PER_CYCLE[arch] * sm_count * clock
    shared_peak = SHARED_PER_CYCLE * sm_count * clock
    ceiling = min(compute_peak, intensity * shared_peak)
    result = {
        "report_id": row["ID"],
        "kernel_name": row["Kernel Name"],
        "sm_count": int(sm_count),
        "elapsed_cycles": cycles,
        "sm_clock_hz": clock,
        "elapsed_seconds": seconds,
        "dmul_results": counts.get("DMUL", 0),
        "dfma_results": counts.get("DFMA", 0),
        "fp64_results": work,
        "shared_bytes": shared,
        "arithmetic_intensity": intensity,
        "achieved_results_per_second": work / seconds,
        "compute_peak_results_per_second": compute_peak,
        "shared_peak_bytes_per_second": shared_peak,
        "roofline_results_per_second": ceiling,
        "efficiency_percent": 100 * (work / seconds) / ceiling,
        "ridge_point": compute_peak / shared_peak,
        "bound": "compute bound"
        if compute_peak <= intensity * shared_peak
        else "shared memory bound",
    }
    if any(
        not math.isfinite(value)
        for value in result.values()
        if isinstance(value, (int, float))
    ):
        raise ValueError("Non-finite derived roofline value")
    return result
