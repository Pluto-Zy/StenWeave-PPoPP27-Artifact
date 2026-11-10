from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TypeAlias

import numpy as np

from codegen.geometry import Coord2D, Coord3D, CrossSectionGeometry
from codegen.logical import StarTopology, StencilComputationSource, star_coefficient_index_3d
from codegen.sass.addressing.d3 import Addressing3D
from codegen.sass.arch import sm80
from codegen.sass.computation.star import StarComputation
from codegen.sass.dependency import DependencyManager
from codegen.sass.emitter import SASSEmitter
from codegen.sass.ir import (
    GlobalMemoryStore,
    InstrGroup,
    SASSOperation,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
    UpdatePtr,
)
from codegen.sass.pipeline import Pipeline
from codegen.spec import CrossSectionSize
from codegen.state import CodeGenStatus, Phase

# Type alias for a 2D array of strings, representing a register list.
RegList2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.str_]]


@dataclass(eq=False)
class Plane:
    # The time step that the plane belongs to.
    time_step: int = 0
    # The index of the plane that is relative to the current phase.
    index: int = 0
    # The offset of the plane relative to the center plane.
    center_offset: int = 0
    # The offset of the plane relative to the target plane.
    target_offset: int = 0

    # Represents whether the plane itself is within the valid range.
    in_range: bool = False
    # Canonical logical fact: this plane produces its final result in this iteration.
    finishes_current_iteration: bool = False
    # Records the set of all computation sources that contribute to the current `Plane`.
    computation_sources: set[StencilComputationSource] = field(default_factory=set)

    # The registers of this plane, stored as a row-major 2D array.
    regs: RegList2D = field(default_factory=lambda: np.empty((0, 0), dtype=np.str_))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def has_intra_thread_computation(self) -> bool:
        return (
            StencilComputationSource.Self in self.computation_sources
            or StencilComputationSource.SameColumn in self.computation_sources
        )

    def has_inter_thread_computation(self) -> bool:
        return StencilComputationSource.OtherColumn in self.computation_sources

    def to_local(self, index: Coord2D) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index)

    def to_remote(self, index: Coord2D, remote_reg: str) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index, remote_reg=remote_reg)


@dataclass(frozen=True)
class LocalPlane:
    """
    Represents a local element in a `Plane`.

    If a group has multiple elements, each thread may hold multiple consecutive elements of a
    `Plane`. This class represents a local element of a `Plane` in that thread. It consists of a
    `Plane` object and the 2D coordinates of the local element relative to the thread.

    If the index is within `plane.regs`, `LocalPlane` refers to a local element owned by the current
    thread, and the register that stores this element can be accessed through `reg()`. Otherwise,
    `LocalPlane` refers to an element owned by another thread. In that case, `remote_reg` cannot be
    `None`; it is the register used by the current thread to store that element.
    """

    plane: Plane
    local_index: Coord2D
    remote_reg: str | None = None

    def is_thread_local(self) -> bool:
        return all(
            0 <= local_index < group_size
            for local_index, group_size in zip(
                self.local_index, self.plane.regs.shape, strict=False
            )
        )

    def is_remote(self) -> bool:
        return not self.is_thread_local()

    def reg(self) -> str:
        if self.is_thread_local():
            return self.plane.regs[self.local_index]
        else:
            assert self.remote_reg is not None
            return self.remote_reg

    def element_position(self) -> tuple[Coord2D, Coord2D]:
        """
        Returns the position of the element in the thread that owns it, which is a tuple of two
        `Coord2D`. The first coordinate is the thread offset relative to the current thread, and the
        second coordinate is the local index of the element within that thread.

        Note that this method assumes that the group shape is equal to the register shape in the
        plane.

        Note that if the `LocalPlane` is thread-local, the first coordinate will be (0, 0), and the
        second coordinate will be equal to `local_index`.
        """

        return divmod(self.local_index, Coord2D(*self.plane.regs.shape))


class CodeGen(StarComputation, Pipeline, Addressing3D, SASSEmitter):
    plane_type = Plane

    def __init__(
        self,
        radius: int,
        time_block_size: int,
        group_sizes: Coord2D,
        leading_load_count: int = 1,
        min_comp_stall: int = 1,
        arch=sm80,
    ):
        assert 0 < radius < 4
        # We only have 2 free scoreboards that can be allocated to shared memory loads, so
        # `leading_load_count` cannot exceed 2.
        assert 0 < leading_load_count <= 2
        assert min_comp_stall > 0

        self.topology = StarTopology(radius, time_block_size)
        self.arch = arch
        self.codes = []

        self.radius = radius
        # Number of time steps performed in one iteration.
        self.time_block_size = time_block_size
        # Shape of the local matrix held by one thread.
        self.group_sizes = group_sizes

        # Number of elements loaded by the LDS instructions that lead an instruction group.
        self.leading_load_count = leading_load_count
        # The minimum stall cycles between two instructions.
        self.min_comp_stall = min_comp_stall

        # Number of elements that should be saved in registers for each time step.
        self.elements_per_time_step = self.topology.elements_per_time_step

        # Each element is a double-precision floating-point number, which occupies 2 registers.
        # Number of elements published to shared memory by each thread for one plane and one time
        # step.
        #
        # A thread publishes the border of its local matrix with a thickness of `radius`. The
        # interior elements are used only by the owning thread and therefore do not need to be
        # stored in shared memory. If the local matrix is too small to contain an interior region,
        # all elements are published.
        self.geometry = CrossSectionGeometry(
            CrossSectionSize(x=self.group_sizes.x, y=self.group_sizes.y), self.radius
        )
        self.smem_slot_size = self.geometry.border_size
        # Size of the shared-memory chunk assigned to each thread, measured in FP64 elements.
        #
        # For each time step, the published border elements are packed into a contiguous 1D region
        # in the row-major order of the local matrix. The regions for all time steps are then placed
        # consecutively in the thread's shared-memory chunk.
        self.smem_chunk_size = self.geometry.chunk_size(self.time_block_size)

        # Maximum thread offset that may be required to access the stencil halo along each
        # dimension.
        #
        # Since each thread owns `group_sizes` elements, the required thread-space radius is
        # `ceil(radius / group_sizes)` for each dimension.
        #
        # For example, with `radius == 2`:
        # - `group_sizes == Coord2D(y=2, x=2)` requires at most one neighboring thread.
        # - `group_sizes == Coord2D(y=1, x=1)` requires neighboring threads up to two positions away.
        self.thread_halo_radius = (self.radius + self.group_sizes - 1) // self.group_sizes

        # Number of planes processed in each iteration in the pipeline phase.
        self.planes_per_iteration = self.topology.planes_per_iteration

        # Number of slots (registers) that store the precomputed addresses of the planes in the head
        # phase.
        #
        # We must ensure that there are at least 2 slots, because we will pre-load 2 planes in the
        # prologue.
        self.precomputed_addr_slot_size = max(2, self.topology.stream_halo)
        # Only when the following conditions are met, the `precomputed_addr_slot()` function may go
        # out of bounds in the second branch.
        if self.time_block_size == 2 and self.radius >= 2:
            self.precomputed_addr_slot_size -= 1

        # This dictionary records the minimum stall cycles between two instruction types.
        self.stall_dict = {
            (StencilComputation, StencilComputation): arch.FP64_LATENCY,
            (SharedMemoryLoad, StencilComputation): 2,
            (SharedMemoryStore, StencilComputation): 2,
            (StencilComputation, SharedMemoryStore): arch.FP64_LATENCY,
            (StencilComputation, GlobalMemoryStore): arch.FP64_LATENCY,
        }

        # Replacements for registers in some instructions. It works like a register copy at compile
        # time.
        self.replacements = {}
        self.candidate_replacements = {}

        # Number of registers used by each thread.
        self.reg_count = 0

        self.global_mem_stride = self.alloc_reg()

        # This register stores the value of the predicate register for jumping to each label. This
        # value is calculated based on `remain_pipeline_rows`.
        self.bra_pred_set = self.alloc_reg()

        # Each thread needs a pointer to the starting address of each row of elements it holds.
        self.inputs = [self.alloc_reg(2) for _ in range(self.group_sizes.y)]
        self.outputs = [self.alloc_reg(2) for _ in range(self.group_sizes.y)]

        # Elements belonging to different thread rows are not contiguous, so a separate base pointer
        # is required for each reachable y offset.
        #
        # The pointer at offset zero addresses the current thread row. The remaining pointers
        # address thread rows in the range:
        #
        #     [-thread_halo_radius.y, thread_halo_radius.y]
        self.shared_mem_reads = [self.alloc_reg() for _ in range(2 * self.thread_halo_radius.y + 1)]
        self.shared_mem_writes = [
            self.alloc_reg() for _ in range(2 * self.thread_halo_radius.y + 1)
        ]
        self.neighbors = [self.alloc_reg(2) for _ in range(self.neighbor_elem_count())]

        # This predicate register indicates whether the current thread needs to write the result to
        # global memory.
        self.write_result = "P5"
        self.pipeline_phase_loop_pred = "P4"
        self.bra_preds = tuple(f"P{i}" for i in range(4))

        # UR1:UR0 are used.
        self.global_mem_desc = "UR0"

        # The number of iterations in the pipeline phase.
        #
        # Note that although this value can be stored in a 32-bit register, UR3 is also needed
        # because 64-bit intermediate values will be generated during the calculation.
        self.pipeline_phase_iterations = "UR2"

        self.regs = tuple(
            tuple(
                # Generate a register array of shape `group_sizes`, allocating registers with
                # `alloc_reg(2)` in row-major order.
                np.array([self.alloc_reg(2) for _ in range(self.group_sizes.total_size())]).reshape(
                    self.group_sizes
                )
                for _ in range(self.elements_per_time_step)
            )
            for _ in range(self.time_block_size)
        )

        # Local indices of all elements owned by the current thread, in row-major order.
        self.thread_local_indices = tuple(
            Coord2D(y=y, x=x) for y in range(self.group_sizes.y) for x in range(self.group_sizes.x)
        )

        # Generate the coordinates of all neighboring elements located in other threads for the
        # current thread. Note that these coordinates are local coordinates relative to the current
        # thread.
        surrounding_coords = chain.from_iterable(
            chain(
                (Coord2D(y=-offset, x=x) for x in range(self.group_sizes.x)),
                (
                    Coord2D(y=self.group_sizes.y - 1 + offset, x=x)
                    for x in range(self.group_sizes.x)
                ),
                (Coord2D(y=y, x=-offset) for y in range(self.group_sizes.y)),
                (
                    Coord2D(y=y, x=self.group_sizes.x - 1 + offset)
                    for y in range(self.group_sizes.y)
                ),
            )
            for offset in range(1, self.radius + 1)
        )

        # This dictionary maps the local index of an element located in another thread to the target
        # register and scoreboard number when loading that element into the current thread.
        self.neighbor_map = {
            elem_pos: (
                self.neighbors[index % len(self.neighbors)],
                1 + index % self.leading_load_count,
            )
            for index, elem_pos in enumerate(surrounding_coords)
        }

        # Each thread should contribute 2 registers as the uniform registers.
        self.alloc_reg(2)

    def instr_group_count(self) -> int:
        """
        This function returns the number of instruction groups needed to load all the neighbor
        elements from shared memory. Each instruction group consists of a number of leading LDS
        instructions that loads `self.leading_load_count` elements, followed by several computation
        instructions that use the loaded elements.
        """

        # Number of neighbor elements that should be loaded from shared memory.
        neighbor_count = 2 * self.radius * sum(self.group_sizes)
        # In one instruction group, the leading LDS instruction will load `self.leading_load_count`
        # elements. This value indicates how many such instruction groups we need to load all the
        # neighbor elements.
        return (neighbor_count + self.leading_load_count - 1) // self.leading_load_count

    def coeff_addr(self, offset: Coord3D) -> int:
        return star_coefficient_index_3d(*offset) * 8

    def precomputed_addr_slot(self, plane: int) -> RegList2D | None:
        if 0 <= plane < self.precomputed_addr_slot_size:
            if plane <= self.radius:
                return self.regs[0][plane]
            elif plane <= 2 * self.radius:
                plane += 3
                return self.regs[plane // (self.radius + 1)][plane % (self.radius + 1)]

            return self.regs[plane // self.radius][plane % self.radius + 1]
        else:
            return None

    def gen_prologue(self):
        self.arch.star3d_prologue(self)

    def gen_inter_thread_computation(
        self, src_plane: LocalPlane, dst_plane: Plane
    ) -> list[StencilComputation]:
        """
        This function constructs a sequence of `StencilComputation` for inter-thread computations.
        It generates all inter-thread computations that use `src_plane` as the source operand.
        `dst_plane` is the destination operand plane for inter-thread computations. For star
        stencils, inter-thread computations only occur on planes with a center offset of -1.
        """

        assert dst_plane.has_inter_thread_computation()

        results: list[StencilComputation] = []

        # We generate computation instructions in the order of y-, x-, x+, y+. For each direction,
        # we start from the farthest element and gradually move towards the center element.
        offsets = chain(
            # y- direction
            (Coord2D(y=offset, x=0) for offset in range(-self.radius, 0)),
            # x- direction
            (Coord2D(y=0, x=offset) for offset in range(-self.radius, 0)),
            # x+ direction
            (Coord2D(y=0, x=offset) for offset in range(self.radius, 0, -1)),
            # y+ direction
            (Coord2D(y=offset, x=0) for offset in range(self.radius, 0, -1)),
        )

        for neighbor_offset in offsets:
            neighbor_id = src_plane.local_index + neighbor_offset
            dst_local_plane = dst_plane.to_local(neighbor_id)
            # Note that if the offset relative to `src_plane` is negative, then the coefficient
            # offset relative to the stencil computation centered on the neighbor itself is
            # positive, and vice versa.
            coeff_coord = Coord3D(z=0, y=-neighbor_offset.y, x=-neighbor_offset.x)

            # Check if the neighbor element is owned by the current thread.
            if not dst_local_plane.is_thread_local():
                continue

            computation = StencilComputation(
                src_planes=[src_plane, dst_local_plane],
                dst_planes=[dst_local_plane],
                coeff_addr=self.coeff_addr(coeff_coord),
                comment=f"Inter computation: center offset {neighbor_id}, neighbor offset {coeff_coord}",
            )

            results.append(computation)

        return results

    def gen_intra_thread_computation(
        self, src_plane: Plane, dst_plane: Plane
    ) -> list[StencilComputation]:
        """
        This function generates all intra-thread computation instructions that contribute the values
        of `src_plane` to `dst_plane`. It returns a list of `StencilComputation` objects.
        """

        assert dst_plane.has_intra_thread_computation()

        # The coefficient coordinate of the neighbor element relative to the center element.
        #
        # All intra-thread computations occur between elements in the z direction, so the y and x
        # components of the neighbor coefficient coordinate are always 0.
        coeff_coord = Coord3D(z=src_plane.index - dst_plane.index - self.radius, y=0, x=0)

        return [
            StencilComputation(
                # The number of elements in `src_planes` determines whether we use DMUL or DFMA
                # instructions. We only use DMUL for the following cases:
                #
                # - All self-update computations.
                # - The first iteration of the head phase, because at this point all registers have
                #   no initial values. At this point, the index of `src_plane` is 0.
                #
                # Note that in the first iteration of the head phase, there will be no inter-thread
                # computations, because there are no values in shared memory yet. So we only need to
                # consider intra-thread computations here.
                src_planes=[src_plane.to_local(i)]
                + (
                    [dst_plane.to_local(i)]
                    if src_plane.index != 0 and src_plane is not dst_plane
                    else []
                ),
                dst_planes=[dst_plane.to_local(i)],
                coeff_addr=self.coeff_addr(coeff_coord),
                comment=f"Intra computation: center offset {i}, neighbor offset {coeff_coord}",
            )
            for i in self.thread_local_indices
        ]

    def gen_smem_dep_inter_comps(self, src_plane: Plane, dst_plane: Plane) -> list[InstrGroup]:
        """
        This function generates the corresponding LDS instructions and computation instructions that
        depend on those instructions for all neighbor elements that need to be loaded from shared
        memory. It returns a list of `InstrGroup` objects, each containing at most
        `self.leading_load_count` LDS instructions and all computation instructions that depend on
        them. The length of the returned list is `self.instr_group_count()`.

        `src_plane` is the plane where the neighbor element is located, and `dst_plane` is the
        target plane for inter-thread computations. For star stencils, inter-thread computations
        only occur on planes with a center offset of -1.
        """

        assert src_plane.target_offset == -1
        assert dst_plane.center_offset == -1
        assert dst_plane.has_inter_thread_computation()

        results = [InstrGroup() for _ in range(self.instr_group_count())]

        for index, (local_index, (target_reg, scoreboard)) in enumerate(self.neighbor_map.items()):
            src_local_plane = src_plane.to_remote(index=local_index, remote_reg=target_reg)

            # Determine the position of the remote element.
            thread_offset, thread_local_index = src_local_plane.element_position()
            # The slot index of the neighbor element within shared memory.
            smem_slot = self.smem_slot_of(thread_local_index)
            assert smem_slot is not None

            # Generate `LDS` instruction.
            load_instr = SharedMemoryLoad(
                # Load the remote element into the register of `src_local_plane`.
                dst_planes=[src_local_plane],
                addr_base=self.shared_mem_reads[thread_offset.y],
                addr_offset=(
                    thread_offset.x * self.smem_chunk_size
                    + dst_plane.time_step * self.smem_slot_size
                    + smem_slot
                ),
                write_scoreboard=scoreboard,
                comment=f"Load neighbor element of local index {src_local_plane.local_index} from shared memory.",
            )

            # Generate the inter-thread computation instructions that depend on the loaded value.
            comp_instrs = self.gen_inter_thread_computation(src_local_plane, dst_plane)

            target_group = results[index // self.leading_load_count]
            target_group.load_instrs.append(load_instr)
            target_group.comp_instrs.extend(comp_instrs)

        return results

    def gen_smem_indep_inter_comp_self_update(
        self, src_plane: Plane, dst_plane: Plane
    ) -> list[StencilComputation]:
        """
        This function generates inter-thread computation instructions that do not depend on shared
        memory. It returns a list of `StencilComputation` objects representing these instructions.

        All elements of `src_plane` will participate in several computations as source operands. If
        a source operand no longer participates in any computation, this function will also emit a
        self-update computation for that operand.
        """

        assert dst_plane.center_offset == -1, (
            "Inter-thread computations only occur on planes with center offset -1."
        )
        assert dst_plane.has_inter_thread_computation()

        results: list[StencilComputation] = []

        src_plane_predefined = src_plane
        if src_plane.time_step == 0:
            slot = self.precomputed_addr_slot(src_plane.index)
            # Since we use the precomputed address slot mechanism, we need to insert a patch here.
            # The actual value of `src_plane` may be stored in the predefined slot rather than in
            # `src_plane.regs`. Therefore, when loading the value of `src_plane`, we need to replace
            # it with the register in the predefined slot.
            #
            # In fact, we considered the case where `planes[-1]` is stored in the predefined slot in
            # `construct_planes()`, because `planes[-1]` is only read in one iteration, so we can
            # safely replace `planes[-1].regs` directly with the predefined slot. For `planes[-2]`,
            # we cannot do this, because when initiating self-updates, we need to write the result
            # of the self-update into the normal register of `planes[-2]` rather than the predefined
            # slot, to release the dependency on the predefined slot register. Therefore, at this
            # point, `src_plane.regs` holds the normal register.
            if slot is not None:
                src_plane_predefined = replace(src_plane, regs=slot)

        # Each source register in `src_plane_predefined` contributes its value to a destination
        # register in `dst_plane`.
        for center_id in self.thread_local_indices:
            src_local_plane = src_plane_predefined.to_local(center_id)
            results.extend(self.gen_inter_thread_computation(src_local_plane, dst_plane))

            # If the source operand has already participated in all computations, generate the
            # self-update computation for it.
            #
            # Note that we need to check whether to emit `self_update` here, because the conditions
            # for generating "inter-thread computations" and "self-update instructions" are
            # different. We can prove the following two propositions:
            #
            # 1. If we need to generate self-update instructions, then we must generate inter-thread
            #    computation instructions.
            #
            # We can try to prove the contrapositive of this proposition: If we do not need to
            # generate inter-thread computation instructions, then we do not need to generate
            # self-update instructions.
            #
            # If we do not need to generate inter-thread computation instructions, then this means
            # that `dst_plane` is meaningless, which means that `dst_plane.index + radius` is not
            # within the valid range. Note that `dst_plane.index + radius` is exactly
            # `src_plane.index`, which means that `src_plane` is meaningless, and its value is
            # always 0, so no self-update is needed.
            #
            # 2. If we need to generate inter-thread computation instructions, then we do not
            #    necessarily need to generate self-update instructions.
            #
            # This is because "generating inter-thread computation instructions" only means that
            # `src_plane` itself is within the valid range, but does not mean that the computation
            # result of `src_plane` is meaningful. The latter requires that
            # `src_plane.index + radius` is also within the valid range.
            #
            # In summary, we have already checked that we need to generate inter-thread computations
            # at the beginning of this method, so we only need to further check whether to generate
            # self-update instructions.
            if StencilComputationSource.Self in src_plane.computation_sources:
                coeff_coord = Coord3D(z=-self.radius, y=0, x=0)
                self_update = StencilComputation(
                    src_planes=[src_local_plane],
                    # Write the result of the self-update to the normal register of `src_plane`,
                    # rather than the predefined slot, to release the dependency on the predefined
                    # slot register.
                    dst_planes=[src_plane.to_local(center_id)],
                    coeff_addr=self.coeff_addr(coeff_coord),
                    comment=f"Self-update: center offset {center_id}, neighbor offset {coeff_coord}",
                )
                results.append(self_update)

        return results

    def distribute_independent_instructions(
        self, target_groups: list[InstrGroup], indep_ops: list[SASSOperation]
    ) -> list[InstrGroup]:
        """
        This function evenly distributes all independent instructions in `indep_ops` to the end of
        each group in `target_groups`.
        """

        total_instrs = len(indep_ops) + sum(len(group.comp_instrs) for group in target_groups)
        avg_insert_count = total_instrs // len(target_groups)
        remainder = total_instrs % len(target_groups)

        for group_id, group in enumerate(target_groups):
            # We only focus on the computation instructions within the group in this loop.
            group = group.comp_instrs
            insert_count = avg_insert_count + (1 if remainder > 0 else 0) - len(group)

            if group_id != len(target_groups) - 1:
                group.extend(indep_ops[:insert_count])
            else:
                # The last few instructions of the last group usually have scoreboard dependencies
                # on the shared memory load instructions of that group. It is better to insert
                # independent instructions before such instructions.
                CodeGen.insert_before(
                    group,
                    indep_ops[:insert_count],
                    lambda op: isinstance(op, StencilComputation) and len(op.wait_scoreboard) > 0,
                )

            indep_ops = indep_ops[insert_count:]
            remainder -= 1

        assert len(indep_ops) == 0, "Not all independent instructions have been distributed."

        return target_groups

    def gen_and_schedule_gmem_store_operations(
        self,
        status: CodeGenStatus,
        planes: list[Plane],
        dep_manager: DependencyManager,
        groups: list[InstrGroup],
    ):
        """
        Generates global memory store and pointer updating instructions for the finished planes in
        `planes` and schedules them with the existing `groups`. It also establishes dependencies
        between the generated store instructions and the existing operations using `dep_manager`.
        """

        # Find all planes in the last time step that are marked as finished and involved in
        # computations.
        finished_planes = [plane for plane in planes if plane.finishes_current_iteration]
        assert len(finished_planes) - 1 <= self.radius - 1

        if not finished_planes:
            return

        # List of `UpdatePtr` instructions for each finished plane.
        update_ptrs: list[list[UpdatePtr]] = []
        global_stores = []

        for index, plane in enumerate(finished_planes):
            addr_regs = (
                self.outputs
                if index == len(finished_planes) - 1
                else planes[-(index + 2)].regs[:, 0]
            )
            global_store = self.gen_gmem_store_operations(ptr_regs=addr_regs, plane=plane)
            global_stores.extend(global_store)

            if status.phase != Phase.Head or plane.index != -self.radius:
                wait_scoreboard = (
                    {self.gmem_store_read_scoreboard()} if addr_regs is self.outputs else set()
                )
                update_ptrs.append(
                    self.gen_update_ptr_operations(
                        src_ptrs=self.outputs,
                        dst_ptrs=addr_regs,
                        offset=index + 1,
                        wait_scoreboard=wait_scoreboard,
                        ptr_name="self.outputs",
                        gmem_operations=global_store,
                        dep_manager=dep_manager,
                    )
                )

        if update_ptrs:
            # A pointer temporary may borrow a data register from another finished plane. Insert
            # each borrowed update after the final physical-register read of its destination.
            # Iterate in reverse so front insertions keep their original order.
            borrowed_update_ptrs = list(chain.from_iterable(update_ptrs[:-1]))
            for update_ptr in reversed(borrowed_update_ptrs):
                for group in reversed(groups):
                    for op_index in range(len(group.comp_instrs) - 1, -1, -1):
                        operation = group.comp_instrs[op_index]
                        if any(src.reg() == update_ptr.dst_reg for src in operation.src_planes):
                            group.comp_instrs.insert(op_index + 1, update_ptr)
                            break
                    else:
                        continue
                    break
                else:
                    groups[-1].comp_instrs.insert(0, update_ptr)

            groups[-1].comp_instrs.extend(update_ptrs[-1])

        groups[-1].comp_instrs.extend(global_stores)
