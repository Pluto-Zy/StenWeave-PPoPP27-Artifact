"""Rank-3 SASS addressing shared by Star and Box strategies."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from codegen.geometry import Coord2D
from codegen.sass.dependency import DependencyManager
from codegen.sass.ir import (
    GlobalMemoryLoad,
    GlobalMemoryStore,
    InstrGroup,
    RegisterPlane,
    SharedMemoryStore,
    UpdatePtr,
)
from codegen.sass.pipeline import Rank3DPipeline


class Plane(Protocol):
    index: int
    time_step: int
    regs: NDArray[np.str_]

    def to_local(self, index: Coord2D) -> RegisterPlane: ...


class Addressing3D(Rank3DPipeline):
    @property
    def shared_mem_read(self):
        return self.shared_mem_reads

    @shared_mem_read.setter
    def shared_mem_read(self, value):
        self.shared_mem_reads = value

    @property
    def shared_mem_write(self):
        return self.shared_mem_writes

    @shared_mem_write.setter
    def shared_mem_write(self, value):
        self.shared_mem_writes = value

    def smem_slot_of(self, local_index: Coord2D) -> int | None:
        """
        Returns the shared-memory slot of a published local element, or `None` if the element is not
        published.

        The published elements form a border of thickness `radius` around the local matrix. They are
        packed contiguously in the row-major order of the original matrix.
        """

        height, width = self.group_sizes
        y, x = local_index

        assert 0 <= y < height
        assert 0 <= x < width

        inner_height = max(0, height - 2 * self.radius)
        inner_width = max(0, width - 2 * self.radius)

        # If either dimension has no interior interval, every element is published and the packed
        # order is identical to the row-major order of the original matrix.
        if inner_height == 0 or inner_width == 0:
            return y * width + x

        mid_row_begin = self.radius
        bottom_row_begin = height - self.radius
        mid_col_begin = self.radius
        right_col_begin = width - self.radius

        if y < mid_row_begin:
            # The top border contains complete rows.
            return y * width + x

        top_slot_count = width * mid_row_begin
        mid_row_slot_count = 2 * self.radius
        if y < bottom_row_begin:
            # Middle rows publish only their leftmost and rightmost `radius` elements.
            mid_row = y - self.radius
            row_begin = top_slot_count + mid_row * mid_row_slot_count

            if x < mid_col_begin:
                return row_begin + x
            elif x < right_col_begin:
                return None
            else:
                return row_begin + mid_col_begin + (x - right_col_begin)

        # The bottom border contains complete rows.
        mid_slot_count = inner_height * mid_row_slot_count
        bottom_row = y - bottom_row_begin

        return top_slot_count + mid_slot_count + bottom_row * width + x

    def gen_and_distribute_smem_stores(
        self, plane: Plane, groups: list[InstrGroup]
    ) -> list[InstrGroup]:
        """
        This function generates instructions to store all local elements of a `Plane` into shared
        memory and evenly distributes these instructions across each instruction group in `groups`.
        It returns a new list of instruction groups that includes the original groups along with the
        newly generated shared memory store instructions.
        """

        smem_stores = [
            SharedMemoryStore(
                src_planes=[plane.to_local(local_index)],
                addr_base=self.shared_mem_writes[0],
                addr_offset=plane.time_step * self.smem_slot_size + self.smem_slot_of(local_index),
                comment=f"Store the value of local index {local_index} at time step {plane.time_step} to shared memory.",
            )
            for local_index in self.thread_local_indices
            if self.smem_slot_of(local_index) is not None
        ]

        # We want to evenly interleave `smem_stores` into the instruction sequence, ensuring that
        # there are even gaps between shared memory read and write instructions.
        avg_insert_count = len(smem_stores) // len(groups)
        remainder = len(smem_stores) % len(groups)

        for group in groups:
            # We only focus on the computation instructions within the group in this loop.
            group = group.comp_instrs
            insert_count = avg_insert_count + (1 if remainder > 0 else 0)
            avg_gap = len(group) // (insert_count + 1)

            for index, smem_store in enumerate(smem_stores[:insert_count]):
                # We want to insert `smem_store` into the middle of the gap between two adjacent
                # computation instructions, so we add 1 to `avg_gap` to account for the instruction
                # itself.
                insert_pos = (index + 1) * avg_gap + index
                group.insert(insert_pos, smem_store)

            smem_stores = smem_stores[insert_count:]
            remainder -= 1

        assert len(smem_stores) == 0, (
            "All shared-memory store instructions should have been inserted."
        )

        return groups

    def gen_gmem_load_operations(self, ptr_regs: list[str], plane: Plane) -> list[GlobalMemoryLoad]:
        """
        Generates instructions that load all thread-local elements of `plane`.

        Each register in `ptr_regs` points to the first element of one global-memory row.
        """

        assert len(ptr_regs) == self.group_sizes.y

        size = 128 if self.group_sizes.x % 2 == 0 else 64
        element_count = size // 64

        operations: list[GlobalMemoryLoad] = []

        for y in range(self.group_sizes.y):
            for x in range(0, self.group_sizes.x, element_count):
                local_indices = [Coord2D(y=y, x=x + offset) for offset in range(element_count)]
                operations.append(
                    GlobalMemoryLoad(
                        dst_planes=[plane.to_local(local_index) for local_index in local_indices],
                        desc_reg=self.global_mem_desc,
                        ptr_reg=ptr_regs[y],
                        ptr_offset=x,
                        size=size,
                        write_scoreboard=self.gmem_load_write_scoreboard(plane.index),
                        read_scoreboard=self.gmem_load_read_scoreboard(plane.index),
                        comment=(f"Load local indices {local_indices} from global memory."),
                    )
                )

        return operations

    def gen_gmem_store_operations(
        self, ptr_regs: list[str], plane: Plane
    ) -> list[GlobalMemoryStore]:
        """
        Generates instructions that store all thread-local elements of `plane`.

        Each register in `ptr_regs` points to the first element of one global-memory row.
        """

        assert len(ptr_regs) == self.group_sizes.y

        size = 128 if self.group_sizes.x % 2 == 0 else 64
        element_count = size // 64

        operations: list[GlobalMemoryStore] = []

        for y in range(self.group_sizes.y):
            for x in range(0, self.group_sizes.x, element_count):
                local_indices = [Coord2D(y=y, x=x + offset) for offset in range(element_count)]
                operations.append(
                    GlobalMemoryStore(
                        src_planes=[plane.to_local(local_index) for local_index in local_indices],
                        desc_reg=self.global_mem_desc,
                        ptr_reg=ptr_regs[y],
                        ptr_offset=x,
                        size=size,
                        pred=self.output_predicate(plane),
                        read_scoreboard=self.gmem_store_read_scoreboard(),
                        comment=(f"Store local indices {local_indices} to global memory."),
                    )
                )

        return operations

    def gen_update_ptr_operations(
        self,
        src_ptrs: list[str],
        dst_ptrs: list[str],
        offset: int,
        wait_scoreboard: set[int],
        ptr_name: str,
        gmem_operations: list[GlobalMemoryLoad | GlobalMemoryStore],
        dep_manager: DependencyManager,
    ) -> list[UpdatePtr]:
        """
        Generates a series of `UpdatePtr` instructions that advance each pointer in `src_ptrs` by
        `offset` times `global_mem_stride`, storing the results in the corresponding `dst_ptrs`. It
        also establishes dependencies between the generated `UpdatePtr` instructions and the given
        `gmem_operations` using the `dep_manager`.
        """

        update_ptrs = [
            UpdatePtr(
                src_reg=src_ptr,
                dst_reg=dst_ptr,
                stride_reg=self.global_mem_stride,
                offset=offset,
                wait_scoreboard=wait_scoreboard,
                comment=f"Update {ptr_name}[{index}] by {offset} * global_mem_stride.",
            )
            for index, (src_ptr, dst_ptr) in enumerate(zip(src_ptrs, dst_ptrs, strict=False))
        ]

        assert len(gmem_operations) % len(update_ptrs) == 0
        for producer, consumer in zip(
            update_ptrs, gmem_operations[:: len(gmem_operations) // len(update_ptrs)], strict=False
        ):
            dep_manager.add_dependency(producer, consumer, stall=5)

        return update_ptrs

    def gen_and_schedule_gmem_load_operations(
        self, new_plane: Plane, dep_manager: DependencyManager, groups: list[InstrGroup]
    ):
        """
        Generates global memory load and pointer updating instructions for `new_plane` and schedules
        them with the existing `groups`. It also establishes dependencies between the generated load
        instructions and the existing operations using the `dep_manager`.
        """

        slot = self.precomputed_addr_slot(new_plane.index)

        # If `slot` is not `None`, place the newly loaded value into `slot`. This avoids overwriting
        # the value in `planes[0][0].regs`.
        if slot is not None:
            new_plane.regs = slot

        # Generate global memory load instructions for `new_plane`.
        gmem_loads = self.gen_gmem_load_operations(
            ptr_regs=self.inputs if slot is None else slot[:, -1], plane=new_plane
        )

        if slot is not None:
            # For the head phase, we need to insert the LDG instructions into the start of the first
            # group.
            groups[0].load_instrs[:0] = gmem_loads
            return

        # We need to update the global memory pointer before loading the next plane if the there is
        # no precomputed address slot for it.
        offset = new_plane.index if new_plane.index == self.precomputed_addr_slot_size else 1
        wait_scoreboard = (
            set()
            if new_plane.index == self.precomputed_addr_slot_size
            else {self.gmem_load_read_scoreboard(new_plane.index - 1)}
        )
        update_ptrs = self.gen_update_ptr_operations(
            src_ptrs=self.inputs,
            dst_ptrs=self.inputs,
            offset=offset,
            wait_scoreboard=wait_scoreboard,
            ptr_name="self.inputs",
            gmem_operations=gmem_loads,
            dep_manager=dep_manager,
        )

        # Insert the `UpdatePtr` instruction before the first computation of the time step.
        groups[0].comp_instrs[:0] = update_ptrs

        # Insert the global memory load instructions.
        #
        # We insert the LDG instructions after the `UpdatePtr` operation that generates its address
        # and the last computation instruction that uses its destination operand.
        for gmem_load in reversed(gmem_loads):
            dst_regs = {dst_plane.reg() for dst_plane in gmem_load.dst_planes}

            for group in reversed(groups):
                for op_index in range(len(group.comp_instrs), 0, -1):
                    operation = group.comp_instrs[op_index - 1]

                    # Check if the current instruction is the `UpdatePtr` instruction that generates
                    # the address for this LDG instruction.
                    produces_load_address = (
                        isinstance(operation, UpdatePtr) and operation.dst_reg == gmem_load.ptr_reg
                    )

                    # Check if the current instruction is the last computation that uses the
                    # destination operand of this LDG instruction.
                    reads_load_destination = any(
                        # We must use register-level comparison instead of plane-level comparison
                        # here, because the destination plane of `gmem_load` is `new_plane`, which
                        # is different from the planes used by other computation instructions.
                        src_plane.reg() in dst_regs
                        for src_plane in operation.src_planes
                    )

                    if produces_load_address or reads_load_destination:
                        # Insert the `gmem_load` instruction after `operation`.
                        group.comp_instrs.insert(op_index, gmem_load)
                        break
                else:
                    continue
                break
