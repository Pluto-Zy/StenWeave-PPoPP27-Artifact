"""SM90 latency policy and prologue entry points."""

from codegen.sass.arch import prologue
from codegen.sass.arch.base import SM90

POLICY = SM90
FP64_LATENCY = POLICY.fp64_latency


def d2_prologue(code):
    prologue.d2_prologue(code, SM90)


def star3d_prologue(code):
    prologue.star3d_prologue(code, SM90)


def box3d_prologue(code):
    prologue.box3d_prologue(code, SM90)
