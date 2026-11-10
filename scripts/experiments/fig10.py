"""Figure 10: effective throughput across square and cube domains."""

from .data import write_csv
from .size_sweep import ARCHES as SIZE_ARCHES
from .size_sweep import STENCILS, series_for, sizes

ARCHES = SIZE_ARCHES
SUITE = "size-sweep"


def analyze(measurements, arches):
    return [
        measurements[(arch, implementation, stencil, variant, width)]
        for arch in arches
        for stencil in STENCILS
        for implementation, variant in series_for(stencil)
        for width in sizes(stencil)
    ]


def exclusions():
    from run.suites.size_sweep import exclusions as excluded_sizes

    return excluded_sizes()


def export(result, arches, output):
    write_csv(
        output / "fig10.csv",
        result,
        [
            "experiment",
            "arch",
            "implementation",
            "stencil",
            "variant",
            "domain_width",
            "rank",
            "gstencils_per_s",
            "statistic",
            "config_id",
            "config",
            "source_file",
            "source_row",
        ],
    )


def plot(result, arches, output):
    from .plotting import draw_fig10

    draw_fig10(result, arches, output)
