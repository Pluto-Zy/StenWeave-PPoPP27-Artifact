from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Phase(Enum):
    Head = auto()
    Pipeline = auto()
    Tail = auto()


@dataclass(frozen=True)
class CodeGenStatus:
    phase: Phase

    # The location of the current plane relative to `phase`.
    #
    # This is a relative position. For `Phase.Head`, this is the offset relative to the first row,
    # so this is also the actual row where the plane is located. For `Phase.Pipeline`, this is the
    # compile-time index after performing `elememts_per_time_step` loop unrolling relative to the
    # first row on the basis of `Phase.Head`, which is different from the runtime position of the
    # plane. For `Phase.Tail`, this is the offset relative to the last row, so it decreases from -1.
    plane: int

    # Represents the runtime offset of `plane`.
    #
    # Since the input may not be exactly unrolled in the pipeline phase, we will leave some rows
    # that are not enough to fill one iteration. The number of remaining rows is a runtime value
    # because it depends on the height of the input.
    #
    # When `phase` is `Phase.Tail`, `plane` is the offset relative to the last row, so it cannot
    # represent the row offset brought by this runtime value. We use this value to adjust the
    # register allocation in `Phase.Tail`.
    runtime_offset: int = 0
