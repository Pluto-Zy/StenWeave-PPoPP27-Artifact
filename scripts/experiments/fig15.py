"""Figure 15: A100 instruction mix, sampled stalls and resource utilization."""

import json
from statistics import mean

from .data import write_csv
from .low_level import FIELDS

ARCHES = ("sm80",)
SUITE = "low-level-metrics"
STENCILS = ("star2d1r", "box2d1r")
IMPLEMENTATIONS = ("AN5D", "EBISU", "StenWeave")


def analyze(measurements, arches):
    if not arches:
        raise ValueError("Figure 15 requires SM80/A100")
    result = []
    for stencil in STENCILS:
        for implementation in IMPLEMENTATIONS:
            rows = sorted(
                (
                    r
                    for r in measurements.values()
                    if r["stencil"] == stencil and r["implementation"] == implementation
                ),
                key=lambda r: r["collection"],
            )
            if [r["collection"] for r in rows] != [1, 2]:
                raise ValueError(
                    f"Expected two independent collections for {implementation}/{stencil}"
                )
            name = implementation + ("-SASS" if implementation == "StenWeave" else "")
            row = {
                "Implementation": name,
                "Stencil": stencil,
                **{field: mean(r[field] for r in rows) for field in FIELDS},
                "arch": "sm80",
                "collections": 2,
                "aggregation": "arithmetic mean of per-collection percentages",
            }
            # Retain individual fractions, raw denominators and report associations.
            row["collection_details"] = json.dumps(
                rows, sort_keys=True, allow_nan=False
            )
            result.append(row)
    return result


def export(result, arches, output):
    write_csv(output / "fig15.csv", result, list(result[0]))


def plot(result, arches, output):
    from .plotting import draw_fig15

    data = {(r["Implementation"], r["Stencil"]): r for r in result}
    draw_fig15(data, output)
