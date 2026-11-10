"""Render Figure 10 size sweeps as four panels with fixed physical axes dimensions."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from .size_sweep import SERIES, STENCILS, series_for

FIGURE_WIDTH = 3.3368
FONT_SIZE = 8
TICK_SIZE = 7
DISPLAY_NAMES = ("AN5D", "EBISU", "LoRA", "SW-SASS", "SW-SASS-ILP")
COLORS = ("#248A3D", "#C98224", "#3978C2", "#30B0C7", "#AF52DE")
MARKERS = ("o", "^", "s", "D", "x")
# Each panel occupies a fixed width and height in inches.
PANEL_WIDTH = FIGURE_WIDTH * (1.0 - 0.13) / (2.0 + 0.26)
PANEL_HEIGHT = 2.15 * (0.72 - 0.27)
ROW_GAP = 0.25  # inches between the two rows of panels


def tick_sizes(values):
    """Place up to four ticks, including both ends of the selected size range."""
    if len(values) <= 4:
        return values
    return [values[round(i * (len(values) - 1) / 3)] for i in range(4)]


def format_size(exponent):
    return lambda value, _: rf"${int(value)}^{{{exponent}}}$"


def make_figure(data, sizes_by_stencil):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "pdf.fonttype": 42,
        }
    )
    # Place both rows at fixed physical panel dimensions.
    added_height = PANEL_HEIGHT + ROW_GAP
    figure_height = 2.15 + added_height
    fig = plt.figure(figsize=(FIGURE_WIDTH, figure_height))
    for index, (benchmark, panel_label) in enumerate(
        zip(STENCILS, ("(a)", "(b)", "(c)", "(d)"), strict=True)
    ):
        column, row = index % 2, index // 2
        ax = fig.add_axes(
            (
                (FIGURE_WIDTH * 0.13 + column * PANEL_WIDTH * 1.26) / FIGURE_WIDTH,
                (2.15 * (0.27 + 0.13) + (1 - row) * added_height) / figure_height,
                PANEL_WIDTH / FIGURE_WIDTH,
                PANEL_HEIGHT / figure_height,
            )
        )
        x_values = sorted(sizes_by_stencil[benchmark])
        maximum = max(
            data[implementation, variant, benchmark, width]
            for implementation, variant in series_for(benchmark)
            for width in x_values
            if (implementation, variant, benchmark, width) in data
        )
        for series, label, color, marker in zip(
            SERIES, DISPLAY_NAMES, COLORS, MARKERS, strict=True
        ):
            if series not in series_for(benchmark):
                continue
            points = [
                (width, data[(*series, benchmark, width)])
                for width in x_values
                if (*series, benchmark, width) in data
            ]
            valid_x, valid_y = zip(*points) if points else ([], [])
            ax.plot(
                valid_x,
                valid_y,
                color=color,
                marker=marker,
                markersize=3.2 if marker != "+" else 4.2,
                linewidth=1.0,
                markeredgewidth=0.75,
                label=label,
            )
        ax.text(
            0.97,
            0.04,
            f"{panel_label} {benchmark}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.12",
                "facecolor": "white",
                "edgecolor": "#111111",
                "alpha": 0.72,
                "linewidth": 0.6,
            },
            zorder=6,
        )
        half_step = 512 if "2d" in benchmark else 32
        ax.set_xlim(x_values[0] - half_step, x_values[-1] + half_step)
        ax.set_ylim(0, maximum * 1.08)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(FixedLocator(tick_sizes(x_values)))
        ax.grid(
            True,
            axis="y",
            color="#b0b0b0",
            linewidth=0.55,
            linestyle="--",
            alpha=0.7,
        )
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.tick_params(axis="both", which="major", width=0.4)
        exponent = 2 if "2d" in benchmark else 3
        ax.xaxis.set_major_formatter(FuncFormatter(format_size(exponent)))

    # Shared labels avoid repeating the same title on every panel. Their
    # positions preserve the outer margins while the reduced row gap shortens
    # the overall figure.
    xlabel = fig.text(
        0.5,
        0.56 / figure_height,
        "Domain Size",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
    )
    ylabel = fig.text(
        0.07 / FIGURE_WIDTH,
        0.5,
        r"$P_{\mathrm{eff}}$ (GStencils/s)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=FONT_SIZE,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            markersize=3.2 if marker != "+" else 4.2,
            markeredgewidth=0.75,
            linewidth=1.0,
            label=label,
        )
        for label, color, marker in zip(DISPLAY_NAMES, COLORS, MARKERS, strict=True)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, (2.15 * 1.005 + added_height) / figure_height),
        ncol=5,
        frameon=True,
        fancybox=True,
        edgecolor="#b0b0b0",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.45,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    # Center shared titles on the exported page, including the legend bounds.
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    xlabel.set_x((bounds.x0 + bounds.width / 2) / FIGURE_WIDTH)
    ylabel.set_y((bounds.y0 + bounds.height / 2) / figure_height)
    return fig


def render(data, sizes_by_stencil, path):
    fig = make_figure(data, sizes_by_stencil)
    try:
        fig.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.015,
            metadata={"CreationDate": None, "ModDate": None},
        )
    finally:
        plt.close(fig)
