"""Figure 11: regular instruction/shared-memory roofline efficiency."""

from .catalog import GPU_NAMES, STENCILS
from .data import write_csv

SUITE = "roofline"


def analyze(measurements, arches):
    return sorted(
        measurements.values(),
        key=lambda row: (
            arches.index(row["arch"]),
            STENCILS.index(row["stencil"]),
            row["implementation"],
            row["variant"],
        ),
    )


def export(result, arches, output):
    write_csv(output / "fig11.csv", result, list(result[0]))


def plot(result, arches, output):
    from .plotting import draw_fig11

    achieved, intensity = {}, {}
    for row in result:
        name = row["implementation"]
        if name == "StenWeave":
            name += "-" + row["variant"]
        identity = name, GPU_NAMES[row["arch"]]
        achieved.setdefault(identity, {})[row["stencil"]] = row["efficiency_percent"]
        intensity.setdefault(identity, {})[row["stencil"]] = row["arithmetic_intensity"]
    draw_fig11(achieved, intensity, arches, output)
