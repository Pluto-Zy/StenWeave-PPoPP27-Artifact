"""Collect the four prepared cumulative CUDA stages for Figure 13."""

from .common import stenweave_tasks

IMPLEMENTATION_NAMES = ("stenweave",)
ARCHES = ("sm80", "sm90")
PROFILE = False


def plan(arch):
    return stenweave_tasks(arch, "breakdown")
