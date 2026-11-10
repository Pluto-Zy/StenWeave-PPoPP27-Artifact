from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from types import NotImplementedType
from typing import NamedTuple

from codegen.spec import CrossSectionSize


class Coord2D(NamedTuple):
    """
    Type for a 2D array index or shape, represented in NumPy axis order `(y, x)`. The first
    component is the row/y coordinate, and the second component is the column/x coordinate.
    """

    y: int
    x: int

    def total_size(self) -> int:
        return self.y * self.x

    @staticmethod
    def _components(value: object) -> tuple[int, int] | None:
        """
        Coord2D(y, x) -> (y, x)
        int           -> (int, int)
        """

        if isinstance(value, Coord2D):
            return value.y, value.x

        if isinstance(value, int):
            return value, value

        return None

    def _binary(
        self,
        other: object,
        op: Callable[[int, int], int],
        *,
        reflected: bool = False,
    ) -> Coord2D | NotImplementedType:
        components = self._components(other)

        if components is None:
            return NotImplemented

        other_y, other_x = components

        if reflected:
            return type(self)(op(other_y, self.y), op(other_x, self.x))
        else:
            return type(self)(op(self.y, other_y), op(self.x, other_x))

    def _unary(self, op: Callable[[int], int]) -> Coord2D:
        return type(self)(op(self.y), op(self.x))

    def _divmod(
        self,
        other: object,
        *,
        reflected: bool = False,
    ) -> tuple[Coord2D, Coord2D] | NotImplementedType:
        components = self._components(other)

        if components is None:
            return NotImplemented

        other_y, other_x = components

        if reflected:
            y_quotient, y_remainder = divmod(other_y, self.y)
            x_quotient, x_remainder = divmod(other_x, self.x)
        else:
            y_quotient, y_remainder = divmod(self.y, other_y)
            x_quotient, x_remainder = divmod(self.x, other_x)

        return (type(self)(y_quotient, x_quotient), type(self)(y_remainder, x_remainder))

    def __add__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.add)

    def __sub__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.sub)

    def __mul__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.mul)

    def __floordiv__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.floordiv)

    def __mod__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.mod)

    def __divmod__(self, other: object) -> tuple[Coord2D, Coord2D] | NotImplementedType:
        return self._divmod(other)

    def __radd__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.add, reflected=True)

    def __rsub__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.sub, reflected=True)

    def __rmul__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.mul, reflected=True)

    def __rfloordiv__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.floordiv, reflected=True)

    def __rmod__(self, other: object) -> Coord2D | NotImplementedType:
        return self._binary(other, operator.mod, reflected=True)

    def __rdivmod__(self, other: object) -> tuple[Coord2D, Coord2D] | NotImplementedType:
        return self._divmod(other, reflected=True)

    def __pos__(self) -> Coord2D:
        return self._unary(operator.pos)

    def __neg__(self) -> Coord2D:
        return self._unary(operator.neg)

    def __abs__(self) -> Coord2D:
        return self._unary(operator.abs)

    def __invert__(self) -> Coord2D:
        return self._unary(operator.invert)


class Coord3D(NamedTuple):
    """
    Type for a 3D array index or shape, represented in NumPy axis order `(z, y, x)`. The first
    component is the depth/z coordinate, the second component is the row/y coordinate, and the third
    component is the column/x coordinate.
    """

    z: int
    y: int
    x: int


@dataclass(frozen=True)
class CrossSectionGeometry:
    """Rank-independent publication geometry for a thread's local cross section."""

    size: CrossSectionSize
    radius: int

    @property
    def border_size(self) -> int:
        sizes = self.size.named().values()
        return math.prod(sizes) - math.prod(max(0, v - 2 * self.radius) for v in sizes)

    def chunk_size(self, time_block_size: int) -> int:
        size = self.border_size * time_block_size
        return size + (size % 2 == 0)
