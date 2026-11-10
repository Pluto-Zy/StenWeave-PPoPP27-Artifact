"""Collect two independent representative-launch profiles for Figure 15."""

from .common import baseline_tasks, stenweave_tasks

IMPLEMENTATION_NAMES = ("an5d", "ebisu", "stenweave")
ARCHES = ("sm80",)
PROFILE = True
STENCILS = ("star2d1r", "box2d1r")


def plan(arch):
    tasks = (
        baseline_tasks("an5d", arch, STENCILS)
        + baseline_tasks("ebisu", arch, STENCILS)
        + [
            t
            for t in stenweave_tasks(arch, "benchmark", variants=("sass",))
            if t["stencil"] in STENCILS
        ]
    )
    return [
        {**task, "id": f"{task['id']}-collection{index}", "collection": index}
        for task in tasks
        for index in (1, 2)
    ]
