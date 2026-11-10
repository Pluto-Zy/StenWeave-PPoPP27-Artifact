"""Fixed DRStencil reference throughput in GStencil/s for each architecture."""

SOURCES = {
    arch: {
        "file": "scripts/experiments/drstencil.py",
        "arch": arch,
        "implementation": "DRStencil",
        "unit": "GStencil/s",
    }
    for arch in ("sm80", "sm90")
}
THROUGHPUT = {
    "sm80": {
        "star2d1r": 113.213,
        "star2d2r": 91.492,
        "star2d3r": 67.618,
        "box2d1r": 92.98,
        "box2d2r": 45.59,
        "box2d3r": 27.172,
        "star3d1r": 122.794,
        "star3d2r": 26.057,
        "star3d3r": 25.218,
        "box3d1r": 21.232,
    },
    "sm90": {
        "star2d1r": 139.232,
        "star2d2r": 95.005,
        "star2d3r": 88.744,
        "box2d1r": 124.262,
        "box2d2r": 105.689,
        "box2d3r": 64.895,
        "star3d1r": 122.794,
        "star3d2r": 28.764,
        "star3d3r": 25.218,
        "box3d1r": 37.104,
    },
}


def measurements(arches):
    return [
        {
            "arch": arch,
            "implementation": "DRStencil",
            "stencil": stencil,
            "variant": "native",
            "gstencils_per_s": rate,
            "statistic": "fixed_paper_value",
            "source_file": "scripts/experiments/drstencil.py",
            "source_row": stencil,
            "reference": SOURCES[arch],
        }
        for arch in arches
        for stencil, rate in THROUGHPUT[arch].items()
    ]
