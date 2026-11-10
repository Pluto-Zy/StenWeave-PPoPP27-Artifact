"""Instruction and sampled not-issued-state fractions from a single launch.

Each opcode counts as one predicated-on thread instruction, including DFMA.
Stall fractions use the sum of all 18 not-issued reason counters, rather than
kernel wall time. Pipe utilization retains the NCU active-cycle denominator.
"""

from run.profile import STALL_REASONS, read_raw

from .roofline import OPCODES, instances, number, raw_row

INSTRUCTION_GROUPS = {
    "FP64 Instr": ("DADD", "DMUL", "DFMA"),
    "INT Instr": (
        "IMAD",
        "IADD3",
        "UIADD3",
        "UIMAD",
        "ISETP",
        "UISETP",
        "SHF",
        "USHF",
        "PLOP3",
    ),
    "SMEM Instr": ("LDS", "STS", "LDGSTS"),
    "GMEM Instr": ("LDG", "STG"),
    "Sync Instr": ("BAR",),
}
STALL_GROUPS = {
    "Long Scoreboard Stall": "long_scoreboard",
    "Short Scoreboard Stall": "short_scoreboard",
    "Math Throttle Stall": "math_pipe_throttle",
    "MIO Throttle Stall": "mio_throttle",
    "Wait Stall": "wait",
    "Barrier Stall": "barrier",
}
UTILIZATION_METRICS = {
    "FP64 Pipe": "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "L1/TEX": "l1tex__throughput.avg.pct_of_peak_sustained_active",
}
FIELDS = (
    *INSTRUCTION_GROUPS,
    "Other Instr",
    *STALL_GROUPS,
    "Other Stall",
    *UTILIZATION_METRICS,
)


def calculate(text, arch):
    ids, names = read_raw(text, low_level=True, arch=arch)
    row = raw_row(text)
    counts = instances(row, OPCODES)
    instruction_total = sum(counts.values())
    if instruction_total <= 0:
        raise ValueError("Instruction mix requires a positive total")
    groups = {
        field: sum(counts.get(opcode, 0) for opcode in opcodes)
        for field, opcodes in INSTRUCTION_GROUPS.items()
    }
    groups["Other Instr"] = instruction_total - sum(groups.values())
    fractions = {
        field: 100 * count / instruction_total for field, count in groups.items()
    }
    stalls = {
        reason: number(
            row,
            f"smsp__pcsamp_warps_issue_stalled_{reason}_not_issued",
            allow_zero=True,
        )
        for reason in STALL_REASONS
    }
    stall_total = sum(stalls.values())
    if stall_total <= 0:
        raise ValueError("Stall mix requires positive not-issued samples")
    groups = {field: stalls[reason] for field, reason in STALL_GROUPS.items()}
    groups["Other Stall"] = stall_total - sum(groups.values())
    fractions.update(
        {field: 100 * count / stall_total for field, count in groups.items()}
    )
    fractions.update(
        {
            field: number(row, metric, allow_zero=True)
            for field, metric in UTILIZATION_METRICS.items()
        }
    )
    return {
        **fractions,
        "report_id": ids[0],
        "kernel_name": names[0],
        "instruction_total": instruction_total,
        "not_issued_total": stall_total,
        "pc_samples": number(row, "smsp__pcsamp_sample_count"),
        "pc_dropped_bytes": number(row, "smsp__pcsamp_dropped_bytes", allow_zero=True),
        "pc_interval_cycles": number(row, "smsp__pcsamp_interval_cycles"),
        "elapsed_cycles": number(row, "gpc__cycles_elapsed.max"),
        "sm_clock_hz": number(row, "gpc__cycles_elapsed.avg.per_second"),
        "duration_ns": number(row, "gpu__time_duration.sum"),
    }
