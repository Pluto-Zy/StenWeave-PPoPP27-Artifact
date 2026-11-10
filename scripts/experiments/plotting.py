"""Adapt validated analysis results to the figure renderers."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .catalog import BASELINES, STENCILS
from .fig14 import COMPARISONS


def draw_fig09(tables, throughput, arches, output):
    from .plot_fig09 import render

    speedups, absolute = [], []
    rates = {
        (row["arch"], row["stencil"]): row["gstencils_per_s"] for row in throughput
    }
    for arch in arches:
        rows = {row["baseline"]: row for row in tables[arch]}
        speedups.append(
            (
                [
                    [rows[name].get(stencil) for stencil in STENCILS]
                    for name in BASELINES
                ],
                [rows[name]["geomean"] for name in BASELINES],
            )
        )
        absolute.append([rates[arch, stencil] for stencil in STENCILS])
    with plt.rc_context():
        render(speedups, absolute, STENCILS, arches, output / "fig09.pdf")


def draw_fig14(tables, arches, output):
    from .plot_fig14 import render

    speedups = []
    for arch in arches:
        rows = {row["comparison"]: row for row in tables[arch]}
        speedups.append(
            {
                name: [rows[name].get(stencil) for stencil in STENCILS]
                + [rows[name]["geomean"]]
                for name, _ in COMPARISONS
            }
        )
    with plt.rc_context():
        render(speedups, STENCILS, arches, output / "fig14.pdf")


def draw_fig10(rows, arches, output):
    from .plot_fig10 import render
    from .size_sweep import STENCILS as SIZE_STENCILS
    from .size_sweep import sizes

    data = {
        (
            row["implementation"],
            row["variant"],
            row["stencil"],
            row["domain_width"],
        ): row["gstencils_per_s"]
        for row in rows
    }
    with plt.rc_context():
        render(
            data,
            {stencil: sizes(stencil) for stencil in SIZE_STENCILS},
            output / "fig10.pdf",
        )


def draw_fig11(achieved, intensity, arches, output):
    from .catalog import GPU_NAMES
    from .plot_fig11 import render

    with plt.rc_context():
        render(
            achieved,
            intensity,
            STENCILS,
            tuple(GPU_NAMES[a] for a in arches),
            output / "fig11.pdf",
        )


def draw_fig12(data, arches, output):
    from .catalog import GPU_NAMES
    from .plot_fig12 import TIME_BLOCKS, render

    values = {
        key: [points.get(depth) for depth in TIME_BLOCKS]
        for key, points in data.items()
    }
    with plt.rc_context():
        render(values, tuple(GPU_NAMES[a] for a in arches), output / "fig12.pdf")


def draw_fig13(data, arches, output):
    from .catalog import GPU_NAMES
    from .plot_fig13 import render

    values = {
        (stencil, GPU_NAMES[arch]): [
            data[stencil, arch, stage] for stage in range(1, 5)
        ]
        for stencil, arch, stage in data
        if stage == 1
    }
    with plt.rc_context():
        render(values, tuple(GPU_NAMES[a] for a in arches), output / "fig13.pdf")


def draw_fig15(data, output):
    from .plot_fig15 import render

    with plt.rc_context():
        render(data, output / "fig15.pdf")
