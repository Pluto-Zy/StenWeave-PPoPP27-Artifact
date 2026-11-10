"""Figure 14: incremental CUDA, SASS and ILP speedups."""

import statistics

from .catalog import BASELINES, STENCILS
from .data import write_csv

COMPARISONS = (
    ("cuda_over_baseline", "CUDA"),
    ("sass_over_cuda", "+SASS"),
    ("ilp_over_sass", "+ILP"),
)


def analyze(measurements, arches):
    matrices = {}
    for arch in arches:
        table = {name: {"comparison": name} for name, _ in COMPARISONS}
        for stencil in STENCILS:
            references = [
                measurements[(arch, name, stencil, "native")]
                for name in BASELINES
                if (arch, name, stencil, "native") in measurements
            ]
            best = max(references, key=lambda row: row["gstencils_per_s"])
            cuda, sass, ilp = [
                measurements[(arch, "StenWeave", stencil, variant)]
                for variant in ("CUDA", "SASS", "SASS-ILP")
            ]
            for (comparison, _), numerator, denominator in zip(
                COMPARISONS, (cuda, sass, ilp), (best, cuda, sass), strict=True
            ):
                table[comparison][stencil] = (
                    numerator["gstencils_per_s"] / denominator["gstencils_per_s"]
                )
        rows = []
        for name, _ in COMPARISONS:
            row = table[name]
            ratios = [row[stencil] for stencil in STENCILS]
            row.update(
                geomean=statistics.geometric_mean(ratios), valid_count=len(ratios)
            )
            rows.append(row)
        matrices[arch] = rows
    return matrices


def export(result, arches, output):
    """Export both architectures and geometric means in one numerical table."""
    rows = [{"arch": arch, **row} for arch in arches for row in result[arch]]
    write_csv(
        output / "fig14.csv",
        rows,
        ["arch", "comparison", *STENCILS, "geomean", "valid_count"],
    )


def plot(result, arches, output):
    from .plotting import draw_fig14

    draw_fig14(result, arches, output)
