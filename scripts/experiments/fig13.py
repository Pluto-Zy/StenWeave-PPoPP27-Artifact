"""Figure 13: throughput across the four cumulative execution-model stages."""

import math

from .catalog import ROOT, tomllib
from .data import write_csv

SUITE = "breakdown"
STENCILS = ("star2d1r", "box2d1r")
STAGES = (
    "baseline",
    "segmented SMEM",
    "Comm-Centric SMEM Scheduling",
    "layered point update",
)


def expected(arches, read):
    result = {}
    for arch in arches:
        path = ROOT / f"configs/stenweave/runner/{arch}/breakdown.toml"
        config = tomllib.loads(read(path).decode())
        cases = config.get("case", [])
        if len(cases) != 8:
            raise ValueError(f"Expected eight frozen breakdown cases for {arch}")
        for case in cases:
            stencil, suffix = case["name"].split(f"-{arch}-", 1)
            if stencil not in STENCILS:
                raise ValueError(f"Unexpected breakdown stencil: {stencil}")
            if suffix not in STAGES:
                raise ValueError(f"Invalid breakdown stage in {case['name']}")
            stage = STAGES.index(suffix) + 1
            identity = (arch, "StenWeave", stencil, f"breakdown-stage{stage}")
            if identity in result:
                raise ValueError(f"Duplicate frozen breakdown stage: {identity}")
            result[identity] = {
                "config_id": case["kernel"]["id"],
                "statistic": config.get("defaults", {})
                .get("run", {})
                .get("measurement_statistic", "mean"),
                "case": case["name"],
            }
        for stencil in STENCILS:
            found = sum(key[0] == arch and key[2] == stencil for key in result)
            if found != 4:
                raise ValueError(
                    f"Incomplete frozen breakdown stages: {arch}/{stencil}"
                )
    return result


def normalize(row, source, number, specifications):
    arch, stencil = row.get("arch"), row.get("stencil")
    variant = row.get("variant")
    identity = (arch, "StenWeave", stencil, variant)
    specification = specifications.get(identity)
    if (
        row.get("experiment") != SUITE
        or row.get("implementation") != "StenWeave"
        or specification is None
        or row.get("case") != specification["case"]
        or row.get("config_id") != specification["config_id"]
        or row.get("statistic") != specification["statistic"]
    ):
        raise ValueError(
            f"Breakdown measurement does not match frozen case: {source}:{number}"
        )
    try:
        rate = float(row["gstencils_per_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid breakdown throughput: {source}:{number}") from error
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"Invalid breakdown throughput: {source}:{number}")
    return {
        "experiment": SUITE,
        "arch": arch,
        "implementation": "StenWeave",
        "stencil": stencil,
        "variant": variant,
        "case": row["case"],
        "config_id": row["config_id"],
        "gstencils_per_s": rate,
        "statistic": row["statistic"],
        "task_id": row.get("task_id"),
        "run_id": row.get("run_id"),
        "source_file": str(source),
        "source_row": number,
    }


def analyze(measurements, arches):
    return sorted(
        measurements.values(),
        key=lambda row: (
            STENCILS.index(row["stencil"]),
            arches.index(row["arch"]),
            int(row["variant"].removeprefix("breakdown-stage")),
        ),
    )


def export(result, arches, output):
    write_csv(output / "fig13.csv", result, list(result[0]))


def plot(result, arches, output):
    from .plotting import draw_fig13

    data = {}
    for row in result:
        stage = int(row["variant"].removeprefix("breakdown-stage"))
        data[row["stencil"], row["arch"], stage] = row["gstencils_per_s"]
    draw_fig13(data, arches, output)
