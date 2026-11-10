"""Collect the A100 domain-size sweep for Figure 10.

Use --impl baselines to collect baseline points independently of the native
StenWeave inventory. EBISU participates only in the 2D size sweeps.
"""

from .common import baseline, baseline_tasks, stenweave_tasks

IMPLEMENTATION_NAMES = ("an5d", "ebisu", "lorastencil", "stenweave")
ARCHES = ("sm80",)
PROFILE = False
STENCILS = ("star2d1r", "box2d1r", "star3d1r", "box3d1r")
# Edit this list to select plot points; preserve the full native TOML inventory.
SIZES_2D = list(range(1024, 20481, 1024))
SIZES_3D = list(range(64, 1025, 64))


def participates(implementation, stencil):
    return implementation.lower() != "ebisu" or "2d" in stencil


def exclusions():
    return [
        {
            "implementation": "EBISU",
            "stencil": stencil,
            "reason": "EBISU is not evaluated for 3D size sweeps.",
        }
        for stencil in STENCILS
        if not participates("ebisu", stencil)
    ]


def plan(arch, implementations=None):
    implementations = implementations or IMPLEMENTATION_NAMES
    sizes = {2: SIZES_2D, 3: SIZES_3D}
    if any(
        not values
        or len(set(values)) != len(values)
        or any(type(v) is not int or v <= 0 for v in values)
        for values in sizes.values()
    ):
        raise ValueError("Size lists must contain distinct positive integers")
    tasks = []
    for implementation in IMPLEMENTATION_NAMES[:-1]:
        if implementation not in implementations:
            continue
        config, _ = baseline(implementation, arch)
        for entry in baseline_tasks(implementation, arch, STENCILS):
            if not participates(implementation, entry["stencil"]):
                continue
            dim = config["stencils"][entry["stencil"]]["dim"]
            for width in sizes[dim]:
                tasks.append(
                    {
                        **entry,
                        "id": f"{entry['id']}-{width}",
                        "width": width,
                        "domain_width": width,
                        "rank": dim,
                    }
                )
    if "stenweave" not in implementations:
        return tasks
    native = stenweave_tasks(arch, "size-sweep", widths=sizes)
    expected = {
        (stencil, variant, width)
        for stencil in STENCILS
        for variant in ("sass", "sass-ilp")
        for width in sizes[2 if "2d" in stencil else 3]
    }
    if {(t["stencil"], t["variant"], t["domain_width"]) for t in native} != expected:
        raise ValueError(
            "Selected sizes do not match the prepared StenWeave configurations"
        )
    return tasks + native
