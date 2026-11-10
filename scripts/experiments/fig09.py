"""Figure 9: StenWeave SASS-ILP relative to each available baseline."""

import statistics

from .catalog import BASELINES, STENCILS
from .data import write_csv


def analyze(measurements, arches):
    matrices, absolute = {}, []
    for arch in arches:
        table = []
        for baseline in BASELINES:
            row = {"baseline": baseline}
            ratios = []
            for stencil in STENCILS:
                reference = measurements.get((arch, baseline, stencil, "native"))
                if reference is None:
                    continue
                target = measurements[(arch, "StenWeave", stencil, "SASS-ILP")]
                speedup = target["gstencils_per_s"] / reference["gstencils_per_s"]
                row[stencil] = speedup
                ratios.append(speedup)
            row.update(
                geomean=statistics.geometric_mean(ratios), valid_count=len(ratios)
            )
            table.append(row)
        matrices[arch] = table
        for stencil in STENCILS:
            target = measurements[(arch, "StenWeave", stencil, "SASS-ILP")]
            absolute.append(
                {
                    "arch": arch,
                    "stencil": stencil,
                    "gstencils_per_s": target["gstencils_per_s"],
                }
            )
    return matrices, absolute


def export(result, arches, output):
    """Export all plotted values in one table, including the throughput panel."""
    matrices, absolute = result
    rows = []
    for arch in arches:
        for row in matrices[arch]:
            rows.append(
                {
                    "arch": arch,
                    "quantity": "speedup",
                    "unit": "x",
                    "series": row["baseline"],
                    **{
                        name: value for name, value in row.items() if name != "baseline"
                    },
                }
            )
        rates = {
            row["stencil"]: row["gstencils_per_s"]
            for row in absolute
            if row["arch"] == arch
        }
        rows.append(
            {
                "arch": arch,
                "quantity": "throughput",
                "unit": "GStencil/s",
                "series": "SW-SASS-ILP",
                **rates,
                "valid_count": len(rates),
            }
        )
    write_csv(
        output / "fig09.csv",
        rows,
        ["arch", "quantity", "unit", "series", *STENCILS, "geomean", "valid_count"],
    )


def plot(result, arches, output):
    from .plotting import draw_fig09

    matrices, absolute = result
    draw_fig09(matrices, absolute, arches, output)
