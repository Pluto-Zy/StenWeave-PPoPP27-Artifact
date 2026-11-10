"""Figure ordering and expected benchmark coverage from frozen configurations."""

from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
ARCHES = ("sm80", "sm90")
GPU_NAMES = {"sm80": "A100", "sm90": "H100"}
# Match the stencil-family order used in the paper figures.
STENCILS = (
    "star2d1r",
    "star2d2r",
    "star2d3r",
    "star3d1r",
    "star3d2r",
    "star3d3r",
    "box2d1r",
    "box2d2r",
    "box2d3r",
    "box3d1r",
)
VARIANTS = ("CUDA", "SASS", "SASS-ILP")
BASELINES = (
    "DRStencil",
    "AN5D",
    "EBISU",
    "ConvStencil",
    "LoRAStencil",
    "FlashFFTStencil",
)
RESULT_NAMES = ("an5d", "ebisu", "convstencil", "lorastencil", "flashfft", "stenweave")


def key(row):
    identity = row["arch"], row["implementation"], row["stencil"], row["variant"]
    if row.get("experiment") == "size-sweep":
        return (*identity, row["domain_width"])
    return identity


def expected(arches, figures, read):
    """Read only source configs; analysis requires no CUDA build or raw logs."""
    result = {}
    variants = VARIANTS if "14" in figures else ("SASS-ILP",)
    for arch in arches:
        path = ROOT / f"configs/stenweave/runner/{arch}/benchmark.toml"
        config = tomllib.loads(read(path).decode())
        for case in config["case"]:
            stencil, suffix = case["name"].split(f"-{arch}-")
            variant = {"cuda": "CUDA", "sass": "SASS", "sass-ilp": "SASS-ILP"}[suffix]
            if variant in variants:
                result[(arch, "StenWeave", stencil, variant)] = {
                    "config_id": case["kernel"]["id"],
                    "statistic": "mean_sample_throughput",
                }
        for implementation in BASELINES:
            if implementation == "DRStencil":
                for stencil in STENCILS:
                    result[(arch, implementation, stencil, "native")] = {}
                continue
            path = ROOT / f"configs/baselines/{implementation}/{arch}.toml"
            config = tomllib.loads(read(path).decode())
            for stencil in config["stencils"]:
                result[(arch, implementation, stencil, "native")] = {
                    "statistic": (
                        "median_time_throughput"
                        if implementation == "FlashFFTStencil"
                        else config["statistic"]
                    ),
                }
    return result
