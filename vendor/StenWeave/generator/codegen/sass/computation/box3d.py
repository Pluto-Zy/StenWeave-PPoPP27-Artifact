from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import chain
from typing import TypeAlias

import numpy as np

from codegen.geometry import Coord2D, Coord3D, CrossSectionGeometry
from codegen.logical import BoxTopology, box_coefficient_index_3d
from codegen.sass.addressing.d3 import Addressing3D
from codegen.sass.arch import sm80
from codegen.sass.computation.box import BoxComputation, ContributionPlanPreview, batched
from codegen.sass.dependency import DependencyManager
from codegen.sass.emitter import SASSEmitter
from codegen.sass.ir import (
    GlobalMemoryStore,
    InstrGroup,
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

    # Represents whether the plane itself is within the valid range.
    in_range: bool = False
    need_processing: bool = False
    # Canonical logical fact: this plane produces its final result in this iteration.
    finishes_current_iteration: bool = False

    # The registers of this plane, stored as a row-major 2D array.
    regs: RegList2D = field(default_factory=lambda: np.empty((0, 0), dtype=np.str_))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def to_local(self, index: Coord2D) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index)

    def to_remote(self, index: Coord2D, remote_reg: str) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index, inner_reg=remote_reg)


@dataclass(frozen=True)
class LocalPlane:
    """
    Represents a local element in a `Plane`.

    If a group has multiple elements, each thread may hold multiple consecutive elements of a
    `Plane`. This class represents a local element of a `Plane` in that thread. It consists of a
    `Plane` object and the 2D coordinates of the local element relative to the thread.

    If `plane` is `None`, then this object represents a free register `inner_reg`. In this case,
    `local_index` is meaningless.

    Otherwise, if the index is within `plane.regs`, `LocalPlane` refers to a local element owned by
    the current thread, and the register that stores this element can be accessed through `reg()`.
    Otherwise, `LocalPlane` refers to an element owned by another thread. In that case, `inner_reg`
    cannot be `None`; it is the register used by the current thread to store that element.
    """

    plane: Plane | None
    local_index: Coord2D
    inner_reg: str | None = None

    def free_reg(reg: str) -> LocalPlane:
        return LocalPlane(plane=None, local_index=Coord2D(0, 0), inner_reg=reg)

    def is_free_reg(self) -> bool:
        return self.plane is None

    def is_thread_local(self) -> bool:
        return not self.is_free_reg() and all(
            0 <= local_index < group_size
            for local_index, group_size in zip(
                self.local_index, self.plane.regs.shape, strict=False
            )
        )

    def is_remote(self) -> bool:
        return not self.is_free_reg() and all(
            not (0 <= local_index < group_size)
            for local_index, group_size in zip(
                self.local_index, self.plane.regs.shape, strict=False
            )
        )

    def reg(self) -> str:
        if self.is_thread_local():
            return self.plane.regs[self.local_index]
        else:
            assert self.inner_reg is not None
            return self.inner_reg

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
        assert not self.is_free_reg()

        return divmod(self.local_index, Coord2D(*self.plane.regs.shape))


class CodeGen(BoxComputation, Pipeline, Addressing3D, SASSEmitter):
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

        self.topology = BoxTopology(radius, time_block_size)
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
        self.precomputed_addr_slot_size = max(2, (self.radius + 1) * self.time_block_size - 1)

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

        # These extra registers are used to store the computation results of the self-updating
        # planes (`planes[-2]`), preventing overwriting the values of those planes. This is because
        # these planes contribute not only to themselves but also to neighboring columns.
        #
        # = NOTE ===================================================================================
        # Note that the number of registers used in `extra_regs` is not theoretically minimal. When
        # `group_sizes` is not 1, we directly allocate temporary registers for all elements. This
        # simplifies the implementation of the code. If registers become a bottleneck in the future,
        # we can consider allocating registers only for the elements that need them.
        # ==========================================================================================
        self.extra_regs = (
            np.array([self.alloc_reg(2) for _ in range(self.group_sizes.total_size())]).reshape(
                self.group_sizes
            )
            if self.group_sizes != Coord2D(1, 1)
            else []
        )

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
                (
                    Coord2D(y=-offset, x=x)
                    for x in range(-self.radius, self.group_sizes.x + self.radius)
                ),
                (
                    Coord2D(y=self.group_sizes.y - 1 + offset, x=x)
                    for x in range(-self.radius, self.group_sizes.x + self.radius)
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
        neighbor_count = (
            self.group_sizes + 2 * self.radius
        ).total_size() - self.group_sizes.total_size()
        # In one instruction group, the leading LDS instruction will load `self.leading_load_count`
        # elements. This value indicates how many such instruction groups we need to load all the
        # neighbor elements.
        return (neighbor_count + self.leading_load_count - 1) // self.leading_load_count

    def coeff_addr(self, offset: Coord3D) -> int:
        return box_coefficient_index_3d(self.radius, *offset) * 8

    def precomputed_addr_slot(self, plane: int) -> RegList2D | None:
        if 0 <= plane < self.precomputed_addr_slot_size:
            if plane <= self.radius + 1:
                return self.regs[0][plane]
            else:
                plane += 1
                return self.regs[plane // (self.radius + 1)][plane % (self.radius + 1) + 1]
        else:
            return None

    def gen_prologue(self):
        self.arch.box3d_prologue(self)

    def is_valid_neighbor(self, src_col: Coord2D, dst_col: Coord2D) -> bool:
        """
        Returns whether the two columns with coordinates `dst_col` and `src_col` are within a valid
        radius range.
        """

        return all(
            -self.radius <= dst - src <= self.radius
            for src, dst in zip(src_col, dst_col, strict=False)
        )

    def gen_contribution_plan_preview(self) -> list[ContributionPlanPreview]:
        """
        Generates a contribution plan preview suitable for the current code generator configuration,
        which is the contribution order between columns within a single time step. The length of the
        returned list is `self.instr_group_count()`.
        """

        flat_results = [
            (src_col, dst_col)
            for src_col in self.thread_local_indices
            for dst_col in self.thread_local_indices
            if self.is_valid_neighbor(src_col, dst_col)
        ]

        # Compute the total number of contributions, where each column contributes to the
        # surrounding (2r + 1) ** 2 columns.
        total_contributions = self.group_sizes.total_size() * ((2 * self.radius + 1) ** 2)
        # The number of instruction groups.
        group_count = self.instr_group_count()
        # Average contributions per instruction group.
        avg_contributions = total_contributions // self.instr_group_count()

        final_results = [ContributionPlanPreview() for _ in range(group_count)]
        for group_id, smem_dep_columns in enumerate(
            # Each column in `self.neighbor_map` corresponds to an LDS instruction, so we group the
            # columns into `self.leading_load_count` columns per group.
            batched(self.neighbor_map.keys(), self.leading_load_count)
        ):
            current_group = final_results[group_id]
            next_group = final_results[min(group_id + 1, group_count - 1)]

            current_group.load_columns.extend(smem_dep_columns)

            insert_count = (
                avg_contributions - len(current_group.contributions)
                if group_id + 1 != group_count
                else len(flat_results)
            )
            current_group.contributions.extend(flat_results[:insert_count])
            flat_results = flat_results[insert_count:]

            next_group.contributions.extend(
                (src_col, dst_col)
                for src_col in smem_dep_columns
                for dst_col in self.thread_local_indices
                if self.is_valid_neighbor(src_col, dst_col)
            )

        return final_results

    def preview_to_computations(
        self,
        preview: tuple[Coord2D, Coord2D],
        producers: dict[Coord2D, StencilComputation],
        planes: list[Plane],
    ) -> list[StencilComputation]:
        """
        This function generates the corresponding computation instructions based on the contribution
        preview. The contribution plan is a tuple containing the local indices of the source and
        target columns. This function converts the contribution plan into a sequence of computation
        instructions, where each instruction represents a computation operation from the source
        column to the target column. The `producers` list keeps track of the last computation that
        writes to `planes[-2]` for each local index.
        """

        results: list[StencilComputation] = []

        src_col, dst_col = preview
        # In box3d stencil, `planes[-2]` is always the source plane for all contributions.
        src_plane = planes[-2]

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
            #
            # Note that in most cases we still use `src_plane`, and `src_plane_predefined` is only
            # used to temporarily replace the source operand of this computation.
            if slot is not None:
                src_plane_predefined = replace(src_plane, regs=slot)

        if src_col in self.thread_local_indices:
            src_local_plane = src_plane_predefined.to_local(src_col)
        else:
            # If the source column comes from another thread, the register holding the source value
            # should be obtained from `neighbor_map`.
            src_local_plane = src_plane_predefined.to_remote(
                index=src_col, remote_reg=self.neighbor_map[src_col][0]
            )

        # Represents whether we need to accumulate the existing result at the target column, which
        # determines whether we use MUL or FMA instructions. If there is already a computation that
        # writes to the target column, we need to accumulate the existing result, hence we use FMA
        # instructions.
        use_acc = dst_col in producers

        # Note that in box3d stencil, `planes[-1]` does not participate in any computation.
        for dst_plane in planes[:-1]:
            if not dst_plane.need_processing:
                continue

            if dst_plane is src_plane and len(self.extra_regs) != 0:
                # For contributions to `planes[-2]`, we write the result to an extra register to
                # avoid overwriting the original value of `planes[-2]`.
                dst_local_plane = LocalPlane.free_reg(self.extra_regs[dst_col])
            else:
                dst_local_plane = dst_plane.to_local(dst_col)

            # Determine whether to use MUL or FMA instructions for the current computation. We have
            # the following rules:
            #
            # - If there are already computations that contribute to the current column, we always
            #   use FMA instructions.
            # - Otherwise, we use MUL instructions for contributions to the following planes:
            #   - Contributions to `planes[-2]`.
            #   - Or, when `planes[-2].index` is 0, contributions to all planes.
            if not use_acc and (dst_plane is src_plane or src_plane.index == 0):
                # Use MUL instructions for contributions to the current column.
                src_planes = [src_local_plane]
            else:
                # Use FMA instructions for contributions to the current column.
                src_planes = [src_local_plane, dst_local_plane]

            neighbor_offset = Coord3D(
                z=src_plane.index - dst_plane.index - self.radius,
                y=src_col.y - dst_col.y,
                x=src_col.x - dst_col.x,
            )
            computation = StencilComputation(
                src_planes=src_planes,
                dst_planes=[dst_local_plane],
                coeff_addr=self.coeff_addr(neighbor_offset),
                comment=f"Contribution from neighbor {neighbor_offset} to local index {dst_col}",
            )

            if dst_plane is src_plane:
                # If the current computation contributes to `planes[-2]`, we update the producer for
                # the current column.
                producers[dst_col] = computation

            results.append(computation)

        return results

    def gen_smem_load(self, target_plane: LocalPlane, scoreboard: int) -> SharedMemoryLoad:
        # Determine the position of the remote element.
        thread_offset, thread_local_index = target_plane.element_position()
        # The slot index of the neighbor element within shared memory.
        smem_slot = self.smem_slot_of(thread_local_index)
        assert smem_slot is not None

        return SharedMemoryLoad(
            dst_planes=[target_plane],
            addr_base=self.shared_mem_reads[thread_offset.y],
            addr_offset=(
                thread_offset.x * self.smem_chunk_size
                + target_plane.plane.time_step * self.smem_slot_size
                + smem_slot
            ),
            write_scoreboard=scoreboard,
            comment=f"Load neighbor element of local index {target_plane.local_index} at time step {target_plane.plane.time_step} from shared memory.",
        )

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
        assert len(finished_planes) * 2 - 1 <= self.elements_per_time_step

        if not finished_planes:
            return

        # List of `UpdatePtr` instructions for each finished plane.
        update_ptrs: list[list[UpdatePtr]] = []
        global_stores = []

        for index, plane in enumerate(finished_planes):
            addr_regs = (
                self.outputs
                if index == len(finished_planes) - 1
                else planes[len(finished_planes) + index].regs[:, 0]
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
