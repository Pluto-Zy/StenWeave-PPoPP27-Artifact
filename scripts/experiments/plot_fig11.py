"""Render Figure 11 using the paper typography, geometry and bar styling."""

import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import Bbox

FIGURE_WIDTH = 3.3368
FONT_SIZE = 9
AXIS_LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 6

BAR_IMPLEMENTATIONS = ("AN5D", "EBISU", "StenWeave-SASS", "StenWeave-SASS-ILP")
# Implementation colors used in the paper.
IMPLEMENTATION_COLORS = {
    "AN5D": "#E2D5F1",
    "EBISU": "#A5D5FB",
    "StenWeave-SASS": "#B2E0E0",
    "StenWeave-SASS-ILP": "#FFD8B1",
}
IMPLEMENTATION_HATCHES = {
    "AN5D": "",
    "EBISU": "\\\\",
    "StenWeave-SASS": "//",
    "StenWeave-SASS-ILP": "xx",
}
IMPLEMENTATION_DISPLAY_NAMES = {
    "StenWeave-SASS": "SW-SASS",
    "StenWeave-SASS-ILP": "SW-SASS-ILP",
    "AN5D": "AN5D",
    "EBISU": "EBISU",
}
CASE_RE = re.compile(r"^(star2d|box2d|star3d|box3d)([123])r$")


def render(achieved, intensity, cases, arches, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
            "hatch.color": "#444444",
            "hatch.linewidth": 0.4,
        }
    )

    height = 2.40
    if len(arches) == 1:
        # Remove one panel and the gap, retaining the two-panel physical geometry.
        panel_height = 2.40 * (0.858 - 0.12) / (2 + 0.155)
        height -= panel_height * (1 + 0.155)
    fig, axes = plt.subplots(
        len(arches),
        1,
        figsize=(FIGURE_WIDTH, height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    x_positions = list(range(len(cases)))
    bar_width = 0.15
    offsets = {
        implementation: (index - 1.5) * bar_width
        for index, implementation in enumerate(BAR_IMPLEMENTATIONS)
    }

    for ax, arch in zip(axes, arches, strict=True):
        for implementation in BAR_IMPLEMENTATIONS:
            row_key = (implementation, arch)
            if row_key not in achieved or row_key not in intensity:
                continue
            for index, case in enumerate(cases):
                pct = achieved[row_key].get(case)
                ai = intensity[row_key].get(case)
                if pct is None or ai is None:
                    continue
                bar_x = index + offsets[implementation]
                ax.bar(
                    bar_x,
                    pct,
                    width=bar_width,
                    color=IMPLEMENTATION_COLORS[implementation],
                    edgecolor="#444444",
                    linewidth=0.7,
                    linestyle="-",
                    hatch=IMPLEMENTATION_HATCHES[implementation],
                    zorder=3,
                )

        ax.set_ylim(0, 105)
        ax.yaxis.set_major_locator(FixedLocator((0, 50, 90, 100)))
        ax.grid(
            True,
            axis="y",
            color="#b0b0b0",
            linewidth=0.6,
            alpha=0.7,
            linestyle="--",
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.set_xlim(-0.55, len(cases) - 0.45)
        ax.text(
            0.97,
            0.08,
            f"({'a' if arch == 'A100' else 'b'}) {arch}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.20,rounding_size=0.12",
                "facecolor": "white",
                "edgecolor": "#111111",
                "linewidth": 0.65,
            },
            zorder=6,
        )

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        [f"{CASE_RE.fullmatch(case).group(2)}r" for case in cases],
        rotation=0,
        ha="center",
    )
    group_ranges = []
    current_group = None
    for index, case in enumerate(cases):
        group = CASE_RE.fullmatch(case).group(1)
        if group != current_group:
            group_ranges.append([group, index, index])
            current_group = group
        else:
            group_ranges[-1][2] = index
    for group, first, last in group_ranges:
        axes[-1].text(
            (first + last) / 2.0,
            -0.23,
            group,
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=TICK_SIZE,
        )
    fig.supylabel(
        r"Roofline efficiency $\eta_{\mathrm{roof}}$ (%)",
        x=0.012,
        y=0.50,
        fontsize=AXIS_LABEL_SIZE,
    )

    legend_handles = [
        Patch(
            facecolor=IMPLEMENTATION_COLORS[name],
            edgecolor="#444444",
            linewidth=0.7,
            linestyle="-",
            hatch=IMPLEMENTATION_HATCHES[name],
            label=IMPLEMENTATION_DISPLAY_NAMES[name],
        )
        for name in BAR_IMPLEMENTATIONS
    ]
    legend_labels = [IMPLEMENTATION_DISPLAY_NAMES[name] for name in BAR_IMPLEMENTATIONS]
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.565, 0.995 if len(arches) == 2 else 1 - 0.012 / height),
        ncol=4,
        frameon=True,
        fancybox=True,
        edgecolor="#b0b0b0",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.60,
        handletextpad=0.45,
        columnspacing=1.05,
        labelspacing=0.25,
    )

    # Shorten the two panels while keeping the physical bottom margin, the
    # inter-panel gap, and the gap to the legend approximately unchanged.
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.12, top=0.858, hspace=0.155)
    if len(arches) == 1:
        fig.subplots_adjust(
            bottom=2.40 * 0.12 / height,
            top=(2.40 * 0.12 + panel_height) / height,
        )
    for ax in axes:
        position = ax.get_position()
        ax.set_position(
            (
                position.x0,
                position.y0 + (0.02 if len(arches) == 2 else 0.048 / height),
                position.width,
                position.height,
            )
        )
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    vertical_bbox = Bbox.from_extents(
        0.0, tight_bbox.y0 - 0.01, fig.get_figwidth(), tight_bbox.y1 + 0.01
    )
    fig.savefig(output_pdf, format="pdf", bbox_inches=vertical_bbox, pad_inches=0)
    plt.close(fig)
