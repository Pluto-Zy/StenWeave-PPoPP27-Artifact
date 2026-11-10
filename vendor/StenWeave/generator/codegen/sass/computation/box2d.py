from __future__ import annotations

from dataclasses import dataclass, field, replace

from codegen.geometry import CrossSectionGeometry
from codegen.logical import BoxTopology, box_coefficient_index_2d
from codegen.sass.addressing.d2 import Addressing2D
from codegen.sass.arch import sm80
from codegen.sass.computation.box import BoxComputation, ContributionPlanPreview, batched
from codegen.sass.emitter import SASSEmitter
from codegen.sass.ir import (
    GlobalMemoryStore,
    InstrGroup,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
)
from codegen.sass.pipeline import OutputAddressLayout, Pipeline
from codegen.spec import CrossSectionSize


@dataclass
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

    # The registers of this plane. The value is a list whose length is equal to `group_size`.
    regs: list[str] = field(default_factory=list)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def to_local(self, index: int) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index)

    def to_remote(self, index: int, remote_reg: str) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index, inner_reg=remote_reg)


@dataclass(frozen=True)
class LocalPlane:
    """
    Represents a local element in a `Plane`.

    When `group_size` is greater than 1, each thread may hold multiple consecutive elements of a
    `Plane`. This class represents a local element of a `Plane` in that thread. It consists of a
    `Plane` object and the index of that local element relative to that thread.

    If `plane` is `None`, then this object represents a free register `inner_reg`. In this case,
    `local_index` is meaningless.

    Otherwise, if `local_index` is in `[0, group_size)`, then `LocalPlane` refers to a local element
    owned by the current thread, and the register that stores that element can be accessed through
    `reg()`. Otherwise, `LocalPlane` refers to an element owned by another thread. In this case,
    `inner_reg` cannot be `None`, and it will represent the register used in the current thread to
    store that element, and the `reg()` method will return `inner_reg`.
    """

    plane: Plane | None
    local_index: int
    inner_reg: str | None = None

    def free_reg(reg: str) -> LocalPlane:
        return LocalPlane(plane=None, local_index=0, inner_reg=reg)

    def is_free_reg(self) -> bool:
        return self.plane is None

    def is_thread_local(self) -> bool:
        return not self.is_free_reg() and 0 <= self.local_index < len(self.plane.regs)

    def is_remote(self) -> bool:
        return not self.is_free_reg() and (
            self.local_index < 0 or self.local_index >= len(self.plane.regs)
        )

    def reg(self) -> str:
        if self.is_thread_local():
            return self.plane.regs[self.local_index]
        else:
            assert self.inner_reg is not None
            return self.inner_reg

    def element_position(self) -> tuple[int, int]:
        """
        Returns the position of the element in the thread that owns it, which is a tuple of two
        integers. The first integer is the thread offset relative to the current thread, and the
        second integer is the local index of the element within that thread.

        Note that this method assumes that the group size is equal to the number of registers in the
        plane.

        Note that if the `LocalPlane` is thread-local, the first integer will be 0, and the second
        integer will be equal to `local_index`.
        """

        assert not self.is_free_reg()

        group_size = len(self.plane.regs)

        thread_offset = self.local_index // group_size
        local_index = self.local_index % group_size

        return thread_offset, local_index


class CodeGen(BoxComputation, Addressing2D, Pipeline, SASSEmitter):
    output_address_layout = OutputAddressLayout.AfterOutputs
    plane_type = Plane

    def __init__(
        self,
        radius: int,
        time_block_size: int,
        group_size: int,
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
        #
        # Number of continuous elements in a single plane held by one thread.
        self.group_size = group_size

        # Number of elements loaded by the LDS instructions that lead an instruction group.
        self.leading_load_count = leading_load_count
        # The minimum stall cycles between two instructions.
        self.min_comp_stall = min_comp_stall

        # Number of elements that should be saved in registers for each time step.
        self.elements_per_time_step = self.topology.elements_per_time_step

        # Each element is a double-precision floating-point number, which occupies 2 registers.
        # Number of elements that should be published to the shared memory for each plane in each
        # time step.
        self.geometry = CrossSectionGeometry(CrossSectionSize(x=self.group_size), self.radius)
        self.smem_slot_size = self.geometry.border_size
        # The size of each chunk in shared memory, in number of double-precision floating-point
        # numbers. Each thread occupies `self.smem_chunk_size` elements in shared memory.
        self.smem_chunk_size = self.geometry.chunk_size(self.time_block_size)

        # Maximum thread offset that may be required to access the stencil halo.
        #
        # Since each thread owns `group_size` elements, the required thread-space radius is
        # `ceil(radius / group_size)`.
        #
        # For example, with `radius == 2`:
        # - `group_size == 2` requires at most one neighboring thread.
        # - `group_size == 1` requires neighboring threads up to two positions away.
        self.thread_halo_radius = (self.radius + self.group_size - 1) // self.group_size

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

        self.input = self.alloc_reg(2)
        self.output = self.alloc_reg(2)

        self.shared_mem_read = self.alloc_reg()
        self.shared_mem_write = self.alloc_reg()
        self.neighbors = [self.alloc_reg(2) for _ in range(self.neighbor_elem_count())]

        # These extra registers are used to store the computation results of the self-updating
        # planes (`planes[-2]`), preventing overwriting the values of those planes. This is because
        # these planes contribute not only to themselves but also to neighboring columns.
        #
        # = NOTE ===================================================================================
        # Note that the number of registers used in `extra_regs` is not theoretically minimal. When
        # `group_size` is not 1, we directly allocate temporary registers for all elements. This
        # simplifies the implementation of the code. If registers become a bottleneck in the future,
        # we can consider allocating registers only for the elements that need them.
        # ==========================================================================================
        self.extra_regs = [
            self.alloc_reg(2) for _ in range(self.group_size if self.group_size > 1 else 0)
        ]

        # This dictionary maps the local index of an element located in another thread to the target
        # register and scoreboard number when loading that element into the current thread.
        self.neighbor_map = {
            elem_pos: (
                self.neighbors[index % len(self.neighbors)],
                1 + index % self.leading_load_count,
            )
            for index, elem_pos in enumerate(
                x for i in range(self.radius) for x in (self.group_size + i, -i - 1)
            )
        }

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
                tuple(self.alloc_reg(2) for _ in range(self.group_size))
                for _ in range(self.elements_per_time_step)
            )
            for _ in range(self.time_block_size)
        )

        # Each thread should contribute 2 registers as the uniform registers.
        self.alloc_reg(2)

    def coeff_addr(self, offset: tuple[int, int]) -> int:
        return box_coefficient_index_2d(self.radius, *offset) * 8

    def precomputed_addr_slot(self, plane: int) -> str | None:
        if 0 <= plane < self.precomputed_addr_slot_size:
            if plane <= self.radius + 1:
                return self.regs[0][plane]
            else:
                plane += 1
                return self.regs[plane // (self.radius + 1)][plane % (self.radius + 1) + 1]
        else:
            return None

    def gen_prologue(self):
        self.arch.d2_prologue(self)

    def gen_contribution_plan_preview(self) -> list[ContributionPlanPreview]:
        """
        Generates a contribution plan preview suitable for the current code generator configuration,
        which is the contribution order between columns within a single time step. The length of the
        returned list is `self.instr_group_count()`.
        """

        flat_results = [
            (src, dst)
            for src in range(self.group_size)
            for dst in range(self.group_size)
            if -self.radius <= dst - src <= self.radius
        ]

        # Compute the total number of contributions, where each column contributes to the
        # surrounding (2r + 1) columns.
        total_contributions = self.group_size * (2 * self.radius + 1)
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
                (src, dst)
                for src in smem_dep_columns
                for dst in range(self.group_size)
                if -self.radius <= dst - src <= self.radius
            )

        return final_results

    def preview_to_computations(
        self,
        preview: tuple[int, int],
        producers: dict[int, StencilComputation],
        planes: list[Plane],
    ) -> list[StencilComputation]:
        """
        This function generates the corresponding computation instructions based on the contribution
        preview. The contribution plan is a tuple containing the local indices of the source and
        target columns. This function converts the contribution plan into a sequence of computation
        instructions, where each instruction represents a computation operation from the source
        column to the target column. The `producers` list keeps track of the last computation that
        writes to `planes[-2]` for each local index (or `None` if there is no such instruction).
        """

        results: list[StencilComputation] = []

        src_col, dst_col = preview
        # In box2d stencil, `planes[-2]` is always the source plane for all contributions.
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

        if 0 <= src_col < self.group_size:
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

        # Note that in box2d stencil, `planes[-1]` does not participate in any computation.
        for dst_plane in planes[:-1]:
            if not dst_plane.need_processing:
                continue

            if dst_plane is src_plane and self.extra_regs:
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

            neighbor_offset = (src_col - dst_col, src_plane.index - dst_plane.index - self.radius)
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
            addr_base=self.shared_mem_read,
            addr_offset=(
                thread_offset * self.smem_chunk_size
                + target_plane.plane.time_step * self.smem_slot_size
                + smem_slot
            ),
            write_scoreboard=scoreboard,
            comment=f"Load neighbor element of local index {target_plane.local_index} from shared memory.",
        )

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
                src_planes=[plane.to_local(i)],
                addr_base=self.shared_mem_write,
                addr_offset=plane.time_step * self.smem_slot_size + self.smem_slot_of(i),
                comment=f"Store the value of local index {i} to shared memory.",
            )
            for i in range(self.group_size)
            if self.smem_slot_of(i) is not None
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
