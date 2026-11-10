"""Collect one representative kernel per roofline point for Figure 11."""

from .common import baseline_tasks, stenweave_tasks

IMPLEMENTATION_NAMES = ("an5d", "ebisu", "stenweave")
ARCHES = ("sm80", "sm90")
PROFILE = True


def plan(arch):
    return (
        baseline_tasks("an5d", arch)
        + baseline_tasks("ebisu", arch)
        + stenweave_tasks(arch, "roofline")
    )
