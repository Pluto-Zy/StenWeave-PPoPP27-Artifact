"""Backend-neutral Star/Box topology, plane lifecycle and coefficient ordering."""

from dataclasses import dataclass
from enum import Enum, auto

from codegen.state import CodeGenStatus, Phase


class PipelineTopology:
    radius: int
    time_block_size: int

    @property
    def stream_halo(self) -> int:
        return self.radius * self.time_block_size

    @property
    def head_rows(self) -> int:
        return self.head_end

    @property
    def tail_rows(self) -> int:
        return -self.tail_start

    @property
    def halo_head_extension(self) -> int:
        unroll = self.planes_per_iteration
        return (self.stream_halo + unroll - 1) // unroll * unroll

    def pipeline_contract(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in (
                "head_rows",
                "tail_rows",
                "planes_per_iteration",
                "stream_halo",
                "halo_head_extension",
            )
        }


@dataclass(frozen=True)
class StarTopology(PipelineTopology):
    radius: int
    time_block_size: int

    @property
    def temporal_stride(self) -> int:
        return self.radius

    @property
    def elements_per_time_step(self) -> int:
        return 2 * self.radius + 1

    @property
    def head_end(self) -> int:
        return self.temporal_stride * self.time_block_size + 1

    @property
    def planes_per_iteration(self) -> int:
        return 2 * self.elements_per_time_step

    @property
    def tail_start(self) -> int:
        return -self.radius


@dataclass(frozen=True)
class BoxTopology(PipelineTopology):
    radius: int
    time_block_size: int

    @property
    def temporal_stride(self) -> int:
        return self.radius + 1

    @property
    def elements_per_time_step(self) -> int:
        return 2 * self.radius + 2

    @property
    def head_end(self) -> int:
        return self.temporal_stride * self.time_block_size + 1

    @property
    def planes_per_iteration(self) -> int:
        return self.elements_per_time_step

    @property
    def tail_start(self) -> int:
        return -max(self.radius - 1, 1)


class StencilComputationSource(Enum):
    """
    This enum represents the position of the source operand that contributes to the current `Plane`.
    """

    # The plane contributes its own value to itself, i.e., self-update computation.
    Self = auto()

    # The other plane that is in the same column as the current plane contributes to the current
    # plane.
    SameColumn = auto()

    # The other plane that is in a different column from the current plane contributes to the
    # current plane.
    OtherColumn = auto()


@dataclass(frozen=True)
class PlaneState:
    index: int
    center_offset: int
    target_offset: int
    in_range: bool
    meaningful: bool
    # Raw completion boundary; storage handoff may need this even without current work.
    finished: bool


@dataclass(frozen=True)
class StarPlane(PlaneState):
    source_in_range: bool
    computation_sources: frozenset[StencilComputationSource]

    @property
    def finishes_current_iteration(self) -> bool:
        return self.finished and bool(self.computation_sources)


def plane_in_range(index: int, phase: Phase) -> bool:
    if phase == Phase.Head:
        return index >= 0

    if phase == Phase.Tail:
        return index < 0

    return True


def star_planes(radius: int, status: CodeGenStatus, time_step: int) -> list[StarPlane]:
    """Plane lifecycle before backend storage, addresses or scheduling exist."""

    target = status.plane - time_step * radius
    result = []

    for offset in range(-2 * radius, 1):
        center = offset + radius
        index = target + offset
        in_range = plane_in_range(index, status.phase)
        meaningful = plane_in_range(index + radius, status.phase)

        sources = set()
        if meaningful:
            if offset == -1 and in_range:
                sources.add(StencilComputationSource.Self)

            if offset != 0 and plane_in_range(target, status.phase):
                sources.add(StencilComputationSource.SameColumn)

            if center == -1:
                sources.add(StencilComputationSource.OtherColumn)

        result.append(
            StarPlane(
                index,
                center,
                offset,
                in_range,
                meaningful,
                center == -radius
                or (status.phase == Phase.Tail and center <= -1 and target + 1 >= 0),
                plane_in_range(target, status.phase),
                frozenset(sources),
            )
        )

    return result


@dataclass(frozen=True)
class BoxPlane(PlaneState):
    needs_processing: bool

    @property
    def finishes_current_iteration(self) -> bool:
        return self.finished and self.needs_processing


def box_planes(radius: int, status: CodeGenStatus, time_step: int) -> list[BoxPlane]:
    target = status.plane - time_step * (radius + 1)
    source = target - 1
    source_in_range = plane_in_range(source, status.phase)
    result = []

    for offset in range(-2 * radius - 1, 1):
        center = offset + radius + 1
        index = target + offset
        meaningful = plane_in_range(index + radius, status.phase)

        result.append(
            BoxPlane(
                index,
                center,
                offset,
                plane_in_range(index, status.phase),
                meaningful,
                center == -radius or (status.phase == Phase.Tail and center <= 0 and target >= 0),
                meaningful and offset != 0 and source_in_range,
            )
        )

    return result


def star_coefficient_index_2d(x: int, y: int) -> int | None:
    """Center is 0; each distance orders directions y-, y+, x-, x+ for (x, y).

    Non-axis-aligned coordinates return None.
    """
    if x and y:
        return None

    if x == y == 0:
        return 0

    distance = abs(x) + abs(y)
    direction = (0 if y < 0 else 1) if y else (2 if x < 0 else 3)

    return 1 + 4 * (distance - 1) + direction


def star_coefficient_index_3d(z: int, y: int, x: int) -> int:
    """Center is 0; each distance orders directions z-, z+, x-, x+, y-, y+.

    At most one coordinate may be nonzero.
    """
    assert sum(v != 0 for v in (z, y, x)) <= 1

    if z == y == x == 0:
        return 0

    distance = abs(z) + abs(y) + abs(x)
    direction = (0 if z < 0 else 1) if z else ((2 if x < 0 else 3) if x else (4 if y < 0 else 5))

    return 1 + 6 * (distance - 1) + direction


def box_coefficient_index_2d(radius: int, x: int, y: int) -> int:
    """Order width-(2r+1) x-columns 0, -1, +1, -2, +2, ...; y runs -r to +r."""
    return (2 * abs(x) - int(x < 0)) * (2 * radius + 1) + y + radius


def box_coefficient_index_3d(radius: int, z: int, y: int, x: int) -> int:
    """Use ((y+r)*width + x+r)*width + z+r: z changes fastest, then x, then y."""
    width = 2 * radius + 1
    return ((y + radius) * width + x + radius) * width + z + radius
