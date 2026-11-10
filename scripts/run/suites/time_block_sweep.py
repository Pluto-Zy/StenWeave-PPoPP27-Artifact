"""Collect the prepared temporal-depth roofline points for Figure 12."""

from .common import stenweave_tasks

IMPLEMENTATION_NAMES = ("stenweave",)
ARCHES = ("sm80", "sm90")
PROFILE = True


def plan(arch):
    return stenweave_tasks(arch, "time-block-sweep")
