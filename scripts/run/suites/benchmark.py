"""Collect the shared benchmark inputs for Figures 9 and 14."""

from .common import IMPLEMENTATIONS, baseline_tasks, stenweave_tasks

IMPLEMENTATION_NAMES = tuple(IMPLEMENTATIONS)
ARCHES = ("sm80", "sm90")
PROFILE = False


def plan(arch):
    tasks = []
    for implementation in IMPLEMENTATION_NAMES:
        tasks.extend(
            stenweave_tasks(arch, "benchmark")
            if implementation == "stenweave"
            else baseline_tasks(implementation, arch)
        )
    return tasks
