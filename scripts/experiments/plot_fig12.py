"""Render the single-column temporal-depth roofline panels."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

FIGURE_WIDTH = 3.3368
FONT_SIZE = 8
TICK_SIZE = 7
TIME_BLOCKS = tuple(range(1, 21))
CASES = {
    "2D": ("star2d1r", "box2d1r"),
    "3D": ("star3d1r", "box3d1r"),
}
SERIES = (
    ("star-A100", "star2d1r", "A100", "#AF52DE", "o"),
    ("star-H100", "star2d1r", "H100", "#30B0C7", "^"),
    ("box-A100", "box2d1r", "A100", "#007AFF", "s"),
    ("box-H100", "box2d1r", "H100", "#FF9500", "D"),
)


def render(data, arches, output_pdf):
    series = tuple(row for row in SERIES if row[2] in arches)
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

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 1.95))
    shared_maximum = max(
        value for values in data.values() for value in values if value is not None
    )
    for panel_index, (ax, (dimension, _cases)) in enumerate(zip(axes, CASES.items())):
        for label, stencil_2d, arch, color, marker in series:
            stencil = stencil_2d.replace("2d", dimension.lower())
            values = data.get((stencil, arch), [])
            valid = [
                (block, value)
                for block, value in zip(TIME_BLOCKS, values)
                if value is not None
            ]
            if not valid:
                continue
            x_all, y_all = zip(*valid)
            ax.plot(x_all, y_all, color=color, linewidth=1.0, zorder=2)

            ax.plot(
                x_all,
                y_all,
                linestyle="None",
                color=color,
                marker=marker,
                markersize=3.6,
                markeredgewidth=0.65,
                zorder=3,
            )

        ax.text(
            0.97,
            0.04,
            f"({'a' if panel_index == 0 else 'b'}) {dimension}",
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
        max_block = max(
            (
                block
                for _, stencil_2d, arch, _, _ in series
                for block, value in zip(
                    TIME_BLOCKS,
                    data.get((stencil_2d.replace("2d", dimension.lower()), arch), []),
                )
                if value is not None
            ),
            default=20,
        )
        ax.set_xlim(
            (0.3 if dimension == "2D" else 0.5),
            20.7 if dimension == "2D" else max_block + 0.5,
        )
        ax.set_ylim(0, shared_maximum * 1.08)
        ax.xaxis.set_major_locator(
            FixedLocator((1, 5, 10, 15, 20) if dimension == "2D" else (1, 3, 5, 7))
        )
        ax.yaxis.set_major_locator(FixedLocator((0, 25, 50, 75, 95)))
        ax.grid(
            True, axis="y", color="#b0b0b0", linewidth=0.5, linestyle="--", alpha=0.7
        )
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        ax.tick_params(axis="both", which="major", width=0.4)
        if panel_index == 1:
            ax.tick_params(axis="y", labelleft=False)
        for tick in ax.yaxis.get_major_ticks():
            if tick.get_loc() == 95:
                tick.label1.set_verticalalignment("bottom")

    axes[0].set_ylabel("Roofline efficiency (%)", fontsize=FONT_SIZE)
    axes[0].yaxis.label.set_y(0.36)
    fig.supxlabel(r"Temporal-blocking depth $b_T$", fontsize=FONT_SIZE, x=0.565, y=0.20)

    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="-",
            markersize=3.8,
            markeredgewidth=0.65,
            linewidth=1.0,
            label=label,
        )
        for label, _, _, color, marker in series
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.02, 0.15),
        ncol=1,
        frameon=True,
        fancybox=True,
        edgecolor="#b0b0b0",
        facecolor="white",
        framealpha=0.6,
        borderpad=0.45,
        handletextpad=0.25,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.13, right=1.0, bottom=0.24, top=0.72, wspace=0.15)
    for ax in axes:
        position = ax.get_position()
        ax.set_position(
            (position.x0, position.y0 + 0.13, position.width, position.height)
        )
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
