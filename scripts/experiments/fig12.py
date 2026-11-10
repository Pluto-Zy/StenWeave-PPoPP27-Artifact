"""Figure 12: scalar SASS roofline efficiency by temporal blocking depth."""

from .catalog import GPU_NAMES
from .data import write_csv

SUITE = "time-block-sweep"
STENCILS = ("star2d1r", "box2d1r", "star3d1r", "box3d1r")


def analyze(measurements, arches):
    return sorted(
        measurements.values(),
        key=lambda row: (
            STENCILS.index(row["stencil"]),
            arches.index(row["arch"]),
            row["time_block_size"],
        ),
    )


def export(result, arches, output):
    write_csv(output / "fig12.csv", result, list(result[0]))


def plot(result, arches, output):
    from .plotting import draw_fig12

    data = {}
    for row in result:
        key = row["stencil"], GPU_NAMES[row["arch"]]
        data.setdefault(key, {})[row["time_block_size"]] = row["efficiency_percent"]
    draw_fig12(data, arches, output)
