"""Render the execution-model breakdown bar chart."""

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

FIGURE_WIDTH = 3.3368
FIGURE_HEIGHT = 1.80
FONT_SIZE = 8
TICK_SIZE = 7
STAGES = ("Stage 1", "Stage 2", "Stage 3", "Stage 4")
STAGE_LABELS = ("I", "II", "III", "IV")
PAIRS = (
    ("star2d1r", "A100"),
    ("box2d1r", "A100"),
    ("star2d1r", "H100"),
    ("box2d1r", "H100"),
)
COLORS = ("#E2D5F1", "#A5D5FB", "#B2E0E0", "#FFD8B1")


def percentage_labels(values):
    labels = []
    for index, value in enumerate(values):
        if value is None:
            labels.append("N/A")
        elif index == 0 or values[index - 1] in (None, 0):
            labels.append("100%")
        else:
            gain = (value / values[index - 1] - 1.0) * 100.0
            labels.append(f"+{gain:.1f}%" if gain > 1.0 else "~0%")
    return labels


def render(data, arches, output_pdf):
    pairs = [pair for pair in PAIRS if pair[1] in arches]
    all_values = [value for values in data.values() for value in values]
    if not all_values:
        raise ValueError("No numeric breakdown performance values")
    y_max = max(all_values) * 1.18

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    group_centers = [index * 1.45 for index in range(len(pairs))]
    offsets = (-0.41, -0.14, 0.14, 0.41)
    bar_width = 0.24

    for pair_index, pair in enumerate(pairs):
        values = data[pair]
        labels = percentage_labels(values)
        for stage_index, (value, color) in enumerate(zip(values, COLORS, strict=True)):
            if value is None:
                continue
            x = group_centers[pair_index] + offsets[stage_index]
            ax.bar(
                x,
                value,
                width=bar_width,
                color=color,
                edgecolor="#555555",
                linewidth=0.5,
                zorder=3,
            )
            if stage_index > 0:
                if value >= y_max * 0.5:
                    label_y = value - y_max * 0.018
                    label_va = "top"
                else:
                    label_y = value + y_max * 0.012
                    label_va = "bottom"
                ax.text(
                    x,
                    label_y,
                    labels[stage_index],
                    rotation=90,
                    ha="center",
                    va=label_va,
                    fontsize=TICK_SIZE,
                    color="#555555",
                    zorder=4,
                )

    tick_positions = [center + offset for center in group_centers for offset in offsets]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(STAGE_LABELS * len(pairs))
    ax.set_xlim(group_centers[0] - 0.7, group_centers[-1] + 0.7)
    ax.set_ylim(0, 900)
    ax.set_ylabel(r"$P_{\mathrm{eff}}$ (GStencils/s)", fontsize=FONT_SIZE, labelpad=2)
    ax.set_yticks((0, 250, 500, 750))
    ax.grid(
        True,
        axis="y",
        color="#b0b0b0",
        linewidth=0.5,
        linestyle="--",
        alpha=0.7,
    )
    ax.set_axisbelow(True)
    for pair_index, (stencil, arch) in enumerate(pairs):
        ax.text(
            group_centers[pair_index],
            -0.23,
            f"{stencil}\n{arch}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=TICK_SIZE,
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", width=0.5)

    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.25, top=0.73)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    vertical_bbox = Bbox.from_extents(
        0.0, tight_bbox.y0 - 0.01, fig.get_figwidth(), tight_bbox.y1 + 0.01
    )
    fig.savefig(output_pdf, format="pdf", bbox_inches=vertical_bbox, pad_inches=0)
    plt.close(fig)
