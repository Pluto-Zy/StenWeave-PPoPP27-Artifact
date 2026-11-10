"""Rank-2 SASS addressing shared by Star and Box strategies."""

from __future__ import annotations

from typing import Protocol

from codegen.sass.ir import (
    GlobalMemoryLoad,
    GlobalMemoryStore,
    RegisterPlane,
)
from codegen.sass.pipeline import Rank2DPipeline


class Plane(Protocol):
    index: int

    def to_local(self, index: int) -> RegisterPlane: ...


class Addressing2D(Rank2DPipeline):
    def instr_group_count(self) -> int:
        """
        This function returns the number of instruction groups needed to load all the neighbor
        elements from shared memory. Each instruction group consists of a number of leading LDS
        instructions that loads `self.leading_load_count` elements, followed by several computation
        instructions that use the loaded elements.
        """

        # Number of neighbor elements that should be loaded from shared memory.
        neighbor_count = 2 * self.radius
        # In one instruction group, the leading LDS instruction will load `self.leading_load_count`
        # elements. This value indicates how many such instruction groups we need to load all the
        # neighbor elements.
        return (neighbor_count + self.leading_load_count - 1) // self.leading_load_count

    def smem_slot_of(self, local_index: int) -> int | None:
        """
        Returns the shared-memory slot index of a column within a thread, or `None` if the column is
        an interior column that is never published to shared memory.

        Only the edge columns of each thread (the leftmost `radius` and the rightmost `radius`) are
        published to shared memory, and they occupy the slots `0, 1, ..., smem_slot_size - 1` in
        their left-to-right order. Both `gen_smem_comp_operations()` and `SharedMemoryStore` must
        agree on this mapping so that a value written by one thread can be read back by a
        neighboring thread.
        """

        # The left interval of elements to be published: [0, left_end).
        left_end = min(self.radius, self.group_size)
        # The right interval of elements to be published: [right_start, group_size).
        right_start = max(0, self.group_size - self.radius)

        if right_start <= left_end:
            # The left and right intervals overlap, which means that all elements are published to
            # shared memory.
            return local_index

        if local_index < left_end:
            # The element is in the left interval.
            return local_index

        if local_index >= right_start:
            # The element is in the right interval.
            return left_end + (local_index - right_start)

        # The element is in the interior interval, which is never published to shared memory.
        return None

    def gen_gmem_load_operations(self, ptr_reg: str, plane: Plane) -> list[GlobalMemoryLoad]:
        """
        This function generates several instructions to load all thread-local elements of `plane`
        from global memory pointed to by `ptr_reg`.
        """

        # If we are loading an even number of `double` elements, use `LDG.E.128`, otherwise use
        # `LDG.E.64`.
        size = 128 if self.group_size % 2 == 0 else 64
        return [
            GlobalMemoryLoad(
                dst_planes=[plane.to_local(i)],
                desc_reg=self.global_mem_desc,
                ptr_reg=ptr_reg,
                ptr_offset=i,
                size=size,
                write_scoreboard=self.gmem_load_write_scoreboard(plane.index),
                read_scoreboard=self.gmem_load_read_scoreboard(plane.index),
                comment=f"Load the value of local index {i} from global memory.",
            )
            for i in range(0, self.group_size, size // 64)
        ]

    def gen_gmem_store_operations(self, ptr_reg: str, plane: Plane) -> list[GlobalMemoryStore]:
        """
        This function generates several instructions to store all thread-local elements of `plane`
        to global memory pointed to by `ptr_reg`.
        """

        # If we are storing an even number of `double` elements, use `STG.E.128`, otherwise use
        # `STG.E.64`.
        size = 128 if self.group_size % 2 == 0 else 64
        return [
            GlobalMemoryStore(
                src_planes=[plane.to_local(i)],
                desc_reg=self.global_mem_desc,
                ptr_reg=ptr_reg,
                ptr_offset=i,
                size=size,
                pred=self.output_predicate(plane),
                read_scoreboard=self.gmem_store_read_scoreboard(),
                comment=f"Store the value of local index {i} to global memory.",
            )
            for i in range(0, self.group_size, size // 64)
        ]
