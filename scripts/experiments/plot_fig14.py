"""Render Figure 14 incremental CUDA, SASS and ILP speedup heatmaps."""

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

from .catalog import GPU_NAMES
from .fig14 import COMPARISONS as ROWS

FIGURE_WIDTH = 3.3368
FONT_SIZE = 8
TICK_SIZE = 7
CASE_RE = re.compile(r"^(star2d|star3d|box2d|box3d)([123])r$")
SPEEDUP_TICKS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
CMAP_NODE_SPEEDUPS = (1.0, 2.0, 4.0, 8.0, 16.0, 22.0)


def make_colormap() -> LinearSegmentedColormap:
    colors = ["#F7FCFD", "#DDF3F6", "#A8DFE7", "#30B0C7", "#007F91", "#004B6B"]
    positions = [
        math.log(value) / math.log(CMAP_NODE_SPEEDUPS[-1])
        for value in CMAP_NODE_SPEEDUPS
    ]
    cmap = LinearSegmentedColormap.from_list(
        "speedup_log", list(zip(positions, colors))
    )
    cmap.set_bad("#ECECEC")
    cmap.set_under(colors[0])
    return cmap


def render(speedups_by_arch, cases, arches, output_pdf: Path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1.80 * len(arches) / 2))
    grid = fig.add_gridspec(
        len(arches),
        2,
        width_ratios=(1, 0.030),
        height_ratios=(1,) * len(arches),
        wspace=0.04,
        hspace=0.50,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(len(arches))]
    colorbar_axis = fig.add_subplot(grid[:, 1])
    cmap = make_colormap()
    norm = LogNorm(vmin=SPEEDUP_TICKS[0], vmax=SPEEDUP_TICKS[-1])
    group_ranges = []
    current_group = None
    for index, case in enumerate(cases):
        group = CASE_RE.fullmatch(case).group(1)
        if group != current_group:
            group_ranges.append([group, index, index])
            current_group = group
        else:
            group_ranges[-1][2] = index

    image = None
    for panel_index, (ax, speedups) in enumerate(zip(axes, speedups_by_arch)):
        data = [
            [float("nan") if value is None else value for value in speedups[row_name]]
            for row_name, _ in ROWS
        ]
        image = ax.imshow(
            data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
        )
        ax.set_title(
            f"({chr(ord('a') + panel_index)}) {GPU_NAMES[arches[panel_index]]}",
            loc="left",
            pad=3,
            fontsize=FONT_SIZE,
        )
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([label for _, label in ROWS], fontsize=6.2)
        ax.set_xticks(range(len(cases) + 1))
        ax.set_xticklabels(
            [f"{CASE_RE.fullmatch(case).group(2)}r" for case in cases] + ["GMean"],
            fontsize=TICK_SIZE,
        )
        if panel_index < len(arches) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([x - 0.5 for x in range(len(cases) + 2)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(ROWS) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(len(cases) - 0.5, color="#555555", linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
        for row_index, row in enumerate(data):
            for column_index, value in enumerate(row):
                if math.isnan(value):
                    label, text_color = "—", "#777777"
                else:
                    label = f"{value:.2f}"
                    text_color = "white" if value >= 1.55 else "#222222"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=TICK_SIZE,
                    fontweight="bold" if column_index == len(cases) else "normal",
                )
        if panel_index == len(arches) - 1:
            for group, first, last in group_ranges:
                ax.text(
                    (first + last) / 2.0,
                    -0.38,
                    group,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=TICK_SIZE,
                )

    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=SPEEDUP_TICKS)
    colorbar.ax.set_yticklabels([f"{tick:.1f}×" for tick in SPEEDUP_TICKS])
    colorbar.ax.tick_params(labelsize=TICK_SIZE, length=2)
    colorbar.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.18, top=0.91, hspace=0.50)
    save_options = {}
    if len(arches) == 1:
        # Include the title, tick labels and family labels on the shorter canvas.
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        save_options = {
            "bbox_inches": Bbox.from_extents(
                0.0, tight.y0 - 0.02, fig.get_figwidth(), tight.y1 + 0.02
            ),
            "pad_inches": 0,
        }
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None},
        **save_options,
    )
    plt.close(fig)
