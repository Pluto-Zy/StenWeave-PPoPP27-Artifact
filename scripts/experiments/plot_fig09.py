"""Render Figure 9 speedup heatmaps with a logarithmic 1–10× color scale."""

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import NullFormatter, NullLocator

from .catalog import BASELINES as BASELINE_ORDER
from .catalog import GPU_NAMES

FIGURE_WIDTH = 3.3368
ORIGINAL_FIGURE_HEIGHT = 3.55
PANEL_HEIGHT_SCALE = 0.55
ORIGINAL_TOP = 0.95
ORIGINAL_BOTTOM = 0.11
BOTTOM_MARGIN_INCHES = 0.32
SECONDARY_LABEL_Y = -0.25
ORIGINAL_HSPACE = 0.16
FONT_SIZE = 8
TICK_SIZE = 7
BASELINE_DISPLAY_NAMES = {
    "DRStencil": "DR",
    "ConvStencil": "Conv",
    "LoRAStencil": "LoRA",
    "FlashFFTStencil": "FFS",
}
CASE_RE = re.compile(r"^(star2d|star3d|box2d|box3d)([123])r$")
SPEEDUP_TICKS = (1, 2, 4, 8, 10)


def make_colormap() -> LinearSegmentedColormap:
    colors = ["#F7FCFD", "#DDF3F6", "#A8DFE7", "#30B0C7", "#007F91"]
    positions = [
        math.log(value / SPEEDUP_TICKS[0])
        / math.log(SPEEDUP_TICKS[-1] / SPEEDUP_TICKS[0])
        for value in SPEEDUP_TICKS
    ]
    cmap = LinearSegmentedColormap.from_list(
        "speedup_log", list(zip(positions, colors))
    )
    cmap.set_bad("#ECECEC")
    cmap.set_under(colors[0])
    return cmap


def render(tables, absolute_tables, cases, arches, output_pdf: Path):
    cmap = make_colormap()
    norm = LogNorm(vmin=SPEEDUP_TICKS[0], vmax=SPEEDUP_TICKS[-1])
    masked_tables = []
    for values, geomeans in tables:
        masked_tables.append(
            [
                [float("nan") if value is None else value for value in row]
                + [float("nan") if geomean is None else geomean]
                for row, geomean in zip(values, geomeans)
            ]
        )

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

    # Compute panel heights and gaps in inches, with fixed top and bottom margins.
    top_margin = ORIGINAL_FIGURE_HEIGHT * (1.0 - ORIGINAL_TOP)
    bottom_margin = BOTTOM_MARGIN_INCHES
    original_panel_height = (
        ORIGINAL_FIGURE_HEIGHT
        * (ORIGINAL_TOP - ORIGINAL_BOTTOM)
        / (2.0 + ORIGINAL_HSPACE)
    )
    inter_panel_gap = 1.25 * ORIGINAL_HSPACE * original_panel_height
    panel_height = PANEL_HEIGHT_SCALE * original_panel_height
    absolute_panel_height = panel_height * len(arches) / 6.0
    figure_height = (
        top_margin
        + len(arches) * panel_height
        + len(arches) * inter_panel_gap
        + absolute_panel_height
        + bottom_margin
    )
    top = 1.0 - top_margin / figure_height
    bottom = bottom_margin / figure_height
    hspace = inter_panel_gap / panel_height

    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    grid = fig.add_gridspec(
        len(arches) + 1,
        2,
        width_ratios=(1, 0.030),
        height_ratios=(len(arches), *([6] * len(arches))),
        wspace=0.05,
        hspace=hspace,
    )
    absolute_axis = fig.add_subplot(grid[0, 0])
    axes = [fig.add_subplot(grid[1, 0])]
    for panel_index in range(1, len(arches)):
        axes.append(fig.add_subplot(grid[panel_index + 1, 0], sharex=axes[0]))
    bottom_axis = axes[-1]
    colorbar_axis = fig.add_subplot(grid[1 : len(arches) + 1, 1])
    image = None
    group_ranges = []
    current_group = None
    for index, case in enumerate(cases):
        group = CASE_RE.fullmatch(case).group(1)
        if group != current_group:
            group_ranges.append([group, index, index])
            current_group = group
        else:
            group_ranges[-1][2] = index

    for panel_index, (ax, data) in enumerate(zip(axes, masked_tables)):
        image = ax.imshow(
            data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
        )
        ax.set_title(
            f"({chr(ord('b') + panel_index)}) Speedup on {GPU_NAMES[arches[panel_index]]}",
            loc="left",
            pad=3,
            fontsize=FONT_SIZE,
        )
        ax.set_yticks(range(len(BASELINE_ORDER)))
        ax.set_yticklabels(
            [BASELINE_DISPLAY_NAMES.get(name, name) for name in BASELINE_ORDER]
        )
        ax.set_xticks(range(len(cases) + 1))
        ax.set_xticklabels(
            [f"{CASE_RE.fullmatch(case).group(2)}r" for case in cases] + [""],
            rotation=0,
            ha="center",
        )
        ax.tick_params(axis="x", labelbottom=panel_index == len(arches) - 1)
        ax.tick_params(axis="x", length=2)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([x - 0.5 for x in range(len(cases) + 2)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(BASELINE_ORDER) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, zorder=4)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(len(cases) - 0.5, color="#555555", linewidth=0.8)
        ax.set_ylim(len(BASELINE_ORDER) - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        column_minima = []
        for column_index in range(len(cases) + 1):
            valid_values = [
                row[column_index] for row in data if not math.isnan(row[column_index])
            ]
            column_minima.append(min(valid_values) if valid_values else None)
        for row_index, row in enumerate(data):
            for column_index, value in enumerate(row):
                if math.isnan(value):
                    label = "—"
                    text_color = "#777777"
                else:
                    label = f"{value:.1f}"
                    text_color = "white" if value >= 16.0 else "#222222"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight=(
                        "bold"
                        if column_index == len(cases)
                        or (
                            not math.isnan(value)
                            and column_minima[column_index] is not None
                            and math.isclose(
                                value, column_minima[column_index], rel_tol=1e-9
                            )
                        )
                        else "normal"
                    ),
                    fontsize=TICK_SIZE,
                )

    # The absolute-performance panel is a categorical, white-background table-like heatmap
    # for absolute SW-SASS-ILP performance on the two architectures.
    absolute_axis.imshow(
        [[0.0] * len(cases) for _ in arches],
        aspect="auto",
        cmap=LinearSegmentedColormap.from_list(
            "absolute_performance_background", ["#F8F8F8", "#F8F8F8"]
        ),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    absolute_axis.set_title(
        r"(a) $P_{\mathrm{eff}}$ of SW-SASS-ILP (GStencils/s)",
        loc="left",
        pad=3,
        fontsize=FONT_SIZE,
    )
    absolute_axis.set_yticks(range(len(arches)))
    absolute_axis.set_yticklabels([GPU_NAMES[arch] for arch in arches])
    absolute_axis.set_xticks(range(len(cases)))
    absolute_axis.set_xticklabels(
        [f"{CASE_RE.fullmatch(case).group(2)}r" for case in cases]
    )
    absolute_axis.tick_params(axis="x", labelbottom=False)
    absolute_axis.tick_params(axis="x", length=2)
    absolute_axis.xaxis.set_minor_locator(NullLocator())
    absolute_axis.yaxis.set_minor_locator(NullLocator())
    absolute_axis.set_xticks([x - 0.5 for x in range(len(cases) + 1)], minor=True)
    absolute_axis.set_yticks([y - 0.5 for y in range(len(arches) + 1)], minor=True)
    absolute_axis.grid(which="minor", color="#D0D0D0", linewidth=0.55, zorder=4)
    absolute_axis.tick_params(which="minor", bottom=False, left=False)
    absolute_axis.set_ylim(len(arches) - 0.5, -0.5)
    for spine in absolute_axis.spines.values():
        spine.set_linewidth(0.5)
    for row_index, values in enumerate(absolute_tables):
        for column_index, value in enumerate(values[: len(cases)]):
            label = "—" if value is None else f"{value:.0f}"
            absolute_axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="#222222" if value is not None else "#777777",
                fontweight="normal",
                fontsize=TICK_SIZE - 0.5,
            )
    for group, first, last in group_ranges:
        bottom_axis.text(
            (first + last) / 2.0,
            SECONDARY_LABEL_Y,
            group,
            transform=bottom_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=TICK_SIZE,
        )
    bottom_axis.text(
        len(cases) + 0.10,
        -0.095,
        "GMean",
        transform=bottom_axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=TICK_SIZE,
    )
    colorbar = fig.colorbar(image, cax=colorbar_axis, ticks=SPEEDUP_TICKS)
    colorbar.ax.set_yticklabels([f"{tick:g}×" for tick in SPEEDUP_TICKS])
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    colorbar.minorticks_off()
    colorbar.ax.tick_params(labelsize=TICK_SIZE, length=2)
    colorbar.outline.set_linewidth(0.6)

    fig.subplots_adjust(left=0.12, right=0.92, bottom=bottom, top=top, hspace=hspace)
    # The absolute-performance panel has no GMean column.  Narrow it by one column while
    # keeping its left edge aligned with the two speedup heatmaps, so the
    # stencil columns remain vertically aligned.
    absolute_position = absolute_axis.get_position()
    absolute_axis.set_position(
        [
            absolute_position.x0,
            absolute_position.y0,
            absolute_position.width * len(cases) / (len(cases) + 1),
            absolute_position.height,
        ]
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    plt.close(fig)
