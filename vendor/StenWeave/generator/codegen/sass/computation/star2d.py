from __future__ import annotations

from dataclasses import dataclass, field, replace

from codegen.geometry import CrossSectionGeometry
from codegen.logical import StarTopology, StencilComputationSource, star_coefficient_index_2d
from codegen.sass.addressing.d2 import Addressing2D
from codegen.sass.arch import sm80
from codegen.sass.computation.star import StarComputation
from codegen.sass.emitter import SASSEmitter
from codegen.sass.ir import (
    GlobalMemoryStore,
    InstrGroup,
    SASSOperation,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
)
from codegen.sass.pipeline import OutputAddressLayout, Pipeline
from codegen.spec import CrossSectionSize


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

    # The registers of this plane. The value is a list whose length is equal to `group_size`.
    regs: list[str] = field(default_factory=list)

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

    def to_local(self, index: int) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index)

    def to_remote(self, index: int, remote_reg: str) -> LocalPlane:
        return LocalPlane(plane=self, local_index=index, remote_reg=remote_reg)


@dataclass(frozen=True)
class LocalPlane:
    """
    Represents a local element in a `Plane`.

    When `group_size` is greater than 1, each thread may hold multiple consecutive elements of a
    `Plane`. This class represents a local element of a `Plane` in that thread. It consists of a
    `Plane` object and the index of that local element relative to that thread.

    If `local_index` is in `[0, group_size)`, then `LocalPlane` refers to a local element owned by
    the current thread, and the register that stores that element can be accessed through `reg()`.
    Otherwise, `LocalPlane` refers to an element owned by another thread. In this case, `remote_reg`
    cannot be `None`, and it will represent the register used in the current thread to store that
    element, and the `reg()` method will return `remote_reg`.
    """

    plane: Plane
    local_index: int
    remote_reg: str | None = None

    def is_thread_local(self) -> bool:
        return 0 <= self.local_index < len(self.plane.regs)

    def is_remote(self) -> bool:
        return not self.is_thread_local()

    def reg(self) -> str:
        if self.is_thread_local():
            return self.plane.regs[self.local_index]
        else:
            assert self.remote_reg is not None
            return self.remote_reg

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

        group_size = len(self.plane.regs)

        thread_offset = self.local_index // group_size
        local_index = self.local_index % group_size

        return thread_offset, local_index


class CodeGen(StarComputation, Addressing2D, Pipeline, SASSEmitter):
    output_address_layout = OutputAddressLayout.ReverseTail
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

        self.topology = StarTopology(radius, time_block_size)
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
        index = star_coefficient_index_2d(*offset)
        assert index is not None

        return index * 8

    def precomputed_addr_slot(self, plane: int) -> str | None:
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
        self.arch.d2_prologue(self)

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

        # We generate computation instructions in the following order: first to the left
        # (contributing to elements with negative offsets), then to the right (contributing to
        # elements with positive offsets), and from farthest to nearest relative to the center
        # element.
        for offset in (*range(-self.radius, 0), *range(self.radius, 0, -1)):
            neighbor_id = src_plane.local_index + offset

            # Check if the neighbor element is owned by the current thread.
            if neighbor_id < 0 or neighbor_id >= self.group_size:
                continue

            computation = StencilComputation(
                src_planes=[src_plane, dst_plane.to_local(neighbor_id)],
                dst_planes=[dst_plane.to_local(neighbor_id)],
                # Note that if the offset relative to `src_plane` is negative, then the coefficient
                # offset relative to the stencil computation centered on the neighbor itself is
                # positive, and vice versa.
                coeff_addr=self.coeff_addr((-offset, 0)),
                comment=f"Inter computation: center offset {neighbor_id}, neighbor offset {(-offset, 0)}",
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
                coeff_addr=self.coeff_addr((0, src_plane.index - dst_plane.index - self.radius)),
                comment=f"Intra computation: center offset {i}, neighbor offset {(0, src_plane.index - dst_plane.index - self.radius)}",
            )
            for i in range(self.group_size)
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
                addr_base=self.shared_mem_read,
                addr_offset=(
                    thread_offset * self.smem_chunk_size
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
        for center_id in range(self.group_size):
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
                self_update = StencilComputation(
                    src_planes=[src_local_plane],
                    # Write the result of the self-update to the normal register of `src_plane`,
                    # rather than the predefined slot, to release the dependency on the predefined
                    # slot register.
                    dst_planes=[src_plane.to_local(center_id)],
                    coeff_addr=self.coeff_addr((0, -self.radius)),
                    comment=f"Self-update: center offset {center_id}, neighbor offset {(0, -self.radius)}",
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

        avg_insert_count = len(indep_ops) // len(target_groups)
        remainder = len(indep_ops) % len(target_groups)

        for group_id, group in enumerate(target_groups):
            insert_count = avg_insert_count + (1 if remainder > 0 else 0)
            # We only focus on the computation instructions within the group in this loop.
            group = group.comp_instrs

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

    def gen_and_distribute_smem_stores(
        self, plane: Plane, grouped_operations: list[InstrGroup]
    ) -> list[InstrGroup]:
        """
        This function generates instructions to store all local elements of a `Plane` into shared
        memory and evenly distributes these instructions across each instruction group in
        `grouped_operations`. It returns a new list of instruction groups that includes the original
        groups along with the newly generated shared memory store instructions.
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
        avg_insert_count = len(smem_stores) // len(grouped_operations)
        remainder = len(smem_stores) % len(grouped_operations)

        for group in grouped_operations:
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

        return grouped_operations
