"""Render instruction, stall and utilization panels in the paper layout."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, ScaledTranslation

FIGURE_WIDTH = 3.3368
FIGURE_HEIGHT = 2.05
FONT_SIZE = 8
TICK_SIZE = 7
CASES = ("star2d1r", "box2d1r")
IMPLEMENTATIONS = (("AN5D", "A"), ("EBISU", "E"), ("StenWeave-SASS", "SW"))
STAGE_COLORS = ("#A5D5FB", "#FFD8B1", "#B2E0E0", "#E2D5F1")
STALL_COLORS = (
    "#E2D5F1",
    "#F2E6A7",
    "#A5D5FB",
    "#F2C3D2",
    "#B2E0E0",
    "#FFD8B1",
    "#C8E3B8",
)
UTILIZATION_COLORS = ("#A5D5FB", "#B2E0E0", "#E2D5F1")
BAR_OFFSETS = (-0.32, 0.0, 0.32)
CASE_SPACING = 1.35
STACK_HATCHES = ("", "////", r"\\\\", "xxxx", "....", "++++", "oooo")

INSTRUCTION_FIELDS = (
    ("FP/MEM", ("FP64 Instr", "SMEM Instr", "GMEM Instr")),
    ("Addr", ("INT Instr",)),
    ("Sync", ("Sync Instr",)),
    ("Other", ("Other Instr",)),
)

STALL_FIELDS = (
    ("Long SB", ("Long Scoreboard Stall",)),
    ("Short SB", ("Short Scoreboard Stall",)),
    ("MIO Thr.", ("MIO Throttle Stall",)),
    ("Math Thr.", ("Math Throttle Stall",)),
    ("Sync", ("Barrier Stall",)),
    ("Wait", ("Wait Stall", "Other Stall")),
)


def row_sum(row: dict[str, float | None], fields: tuple[str, ...]) -> float:
    return sum(row.get(field) or 0.0 for field in fields)


def render(data, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "hatch.linewidth": 0.35,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharey=True)
    group_centers = (0.0, CASE_SPACING)
    group_titles = ("(a) Inst. mix", "(b) Stall breakdown", "(c) Utilization")
    for panel_index, ax in enumerate(axes):
        for case_index, case in enumerate(CASES):
            center = group_centers[case_index]
            for implementation_index, (implementation, short_name) in enumerate(
                IMPLEMENTATIONS
            ):
                row = data[(implementation, case)]
                x_position = center + BAR_OFFSETS[implementation_index]
                if panel_index == 0:
                    categories = [fields for _, fields in INSTRUCTION_FIELDS]
                    heights = [row_sum(row, fields) for fields in categories]
                    colors = STAGE_COLORS
                    bottom = 0.0
                    for category_index, (height, color) in enumerate(
                        zip(heights, colors)
                    ):
                        ax.bar(
                            x_position,
                            height,
                            bottom=bottom,
                            width=0.28,
                            color=color,
                            hatch=STACK_HATCHES[category_index],
                            edgecolor="#555555",
                            linewidth=0.5,
                            zorder=3,
                        )
                        bottom += height
                elif panel_index == 1:
                    heights = [row_sum(row, fields) for _, fields in STALL_FIELDS]
                    bottom = 0.0
                    for category_index, (height, color) in enumerate(
                        zip(heights, STALL_COLORS)
                    ):
                        ax.bar(
                            x_position,
                            height,
                            bottom=bottom,
                            width=0.28,
                            color=color,
                            hatch=STACK_HATCHES[category_index],
                            edgecolor="#555555",
                            linewidth=0.5,
                            zorder=3,
                        )
                        bottom += height
                else:
                    field = "L1/TEX" if case == "star2d1r" else "FP64 Pipe"
                    value = row.get(field)
                    if value is not None:
                        ax.bar(
                            x_position,
                            value,
                            width=0.28,
                            color=UTILIZATION_COLORS[implementation_index],
                            edgecolor="#555555",
                            linewidth=0.5,
                            zorder=3,
                        )

            ax.text(
                center,
                -0.25,
                case,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=TICK_SIZE,
            )
            if panel_index == 2:
                resource = "L1/TEX" if case == "star2d1r" else "FP64"
                ax.text(
                    center,
                    5.0,
                    resource,
                    transform=ax.transData,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.16,rounding_size=0.10",
                        "facecolor": "white",
                        "edgecolor": "#111111",
                        "alpha": 0.72,
                        "linewidth": 0.5,
                    },
                )

        ax.set_title(group_titles[panel_index], fontsize=FONT_SIZE, pad=4)
        ax.set_xlim(-0.55, CASE_SPACING + 0.55)
        ax.set_ylim(0, 100)
        tick_positions = [
            center + offset for center in group_centers for offset in BAR_OFFSETS
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [short for _ in group_centers for _, short in IMPLEMENTATIONS]
        )
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.grid(
            True, axis="y", color="#b0b0b0", linewidth=0.5, linestyle="--", alpha=0.7
        )
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        ax.tick_params(axis="both", which="major", width=0.5)
        if panel_index > 0:
            ax.tick_params(axis="y", labelleft=False)
        for label in ax.get_xticklabels():
            if label.get_text() == "SW":
                label.set_transform(
                    label.get_transform()
                    + ScaledTranslation(1.5 / 72, 0, fig.dpi_scale_trans)
                )
    axes[0].set_ylabel("Percentage (%)", fontsize=FONT_SIZE, labelpad=3)
    instruction_handles = [
        Patch(
            facecolor=color,
            edgecolor="#555555",
            linewidth=0.5,
            hatch=STACK_HATCHES[index],
            label=label,
        )
        for index, ((label, _), color) in enumerate(
            zip(INSTRUCTION_FIELDS, STAGE_COLORS)
        )
    ]
    stall_handles = [
        Patch(
            facecolor=color,
            edgecolor="#555555",
            linewidth=0.5,
            hatch=STACK_HATCHES[index],
            label=label,
        )
        for index, ((label, _), color) in enumerate(zip(STALL_FIELDS, STALL_COLORS))
    ]
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.60, wspace=0.12)
    for ax in axes:
        position = ax.get_position()
        ax.set_position(
            (position.x0, position.y0 + 0.15, position.width, position.height)
        )
    legend_kwargs = {
        "frameon": True,
        "fancybox": True,
        "edgecolor": "#999999",
        "facecolor": "white",
        "framealpha": 0.92,
        "borderpad": 0.35,
        "handletextpad": 0.25,
        "columnspacing": 0.55,
        "labelspacing": 0.2,
        "fontsize": 6.5,
    }
    fig.legend(
        handles=instruction_handles,
        loc="upper center",
        bbox_to_anchor=(0.27, 0.995),
        ncol=2,
        **legend_kwargs,
    )
    fig.legend(
        handles=stall_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.995),
        ncol=3,
        **legend_kwargs,
    )
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    vertical_bbox = Bbox.from_extents(
        0.0, tight_bbox.y0 - 0.01, fig.get_figwidth(), tight_bbox.y1 + 0.01
    )
    fig.savefig(output_pdf, format="pdf", bbox_inches=vertical_bbox, pad_inches=0)
    plt.close(fig)
