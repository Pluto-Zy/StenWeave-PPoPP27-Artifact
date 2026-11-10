from __future__ import annotations

from dataclasses import dataclass, field

from codegen.cuda.addressing.d3 import VolumeAddressing
from codegen.cuda.operations import (
    CUDAOperation,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
)
from codegen.logical import (
    StarTopology,
    StencilComputationSource,
    plane_in_range,
    star_coefficient_index_3d,
    star_planes,
)
from codegen.state import CodeGenStatus


@dataclass(eq=False)
class Plane:
    index: int
    center_offset: int
    target_offset: int
    finishes_current_iteration: bool
    need_independent_computation: bool
    need_neighbor_computation: bool
    src_reg: str
    dst_reg: str

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class StencilGroup:
    """
    Represents a group of stencil computations.

    In general, a `StencilGroup` has the following pattern:
    - 2 `LDS` instructions to load elements into registers.
    - 1 `STS` instruction to write the target element to shared memory.
    - Several computation instructions.
    """

    operations: list[CUDAOperation] = field(default_factory=list)


class CodeGen(VolumeAddressing):
    global_load_positions = (None, 0)
    global_store_pointer_stride = None
    topology_type = StarTopology
    address_slot_adjustment = 0
    neighbor_count = 4
    full_cross_section = False

    def precomputed_addr_slot(self, plane: int) -> str | None:
        if 0 <= plane < self.precomputed_addr_slot_size:
            if plane <= self.radius:
                return self.regs[0][plane]
            else:
                return self.regs[plane // self.radius][plane % self.radius + 1]
        else:
            return None

    def construct_planes(self, status: CodeGenStatus, time_step: int) -> list[Plane]:
        logical_planes = star_planes(self.radius, status, time_step)

        def plane_constructor(list_index: int) -> Plane:
            logical = logical_planes[list_index]
            index = logical.index
            center_offset = logical.center_offset
            target_offset = logical.target_offset
            finished = logical.finished
            need_independent_computation = logical.meaningful and logical.source_in_range
            need_neighbor_computation = (
                StencilComputationSource.OtherColumn in logical.computation_sources
            )

            src_reg = self.regs[time_step][
                CodeGen.wrap_add(index, status.runtime_offset, self.elements_per_time_step)
            ]

            if finished and time_step != self.time_block_size - 1:
                dst_reg = self.regs[time_step + 1][
                    CodeGen.wrap(
                        index + self.radius + status.runtime_offset, self.elements_per_time_step
                    )
                ]
                if center_offset == -self.radius:
                    pass
                else:
                    key = CodeGenStatus(
                        status.phase, status.plane + list_index, status.runtime_offset
                    )
                    if key not in self.candidate_replacements:
                        self.candidate_replacements[key] = {}

                    self.candidate_replacements[key][dst_reg] = src_reg
                    dst_reg = src_reg
            else:
                dst_reg = src_reg

            if time_step == 0 and target_offset == 0:
                slot = self.precomputed_addr_slot(index)
                if slot is not None:
                    src_reg = slot

            src_reg = self.replacements.get(src_reg, src_reg)
            dst_reg = self.replacements.get(dst_reg, dst_reg)

            return Plane(
                index,
                center_offset,
                target_offset,
                logical.finishes_current_iteration,
                need_independent_computation,
                need_neighbor_computation,
                src_reg,
                dst_reg,
            )

        return [plane_constructor(i) for i in range(self.elements_per_time_step)]

    def gen_memory_independent_computations(
        self,
        planes: list[Plane],
        groups: list[StencilGroup],
        plane_op_map: dict[Plane, StencilComputation],
    ):
        """
        This function generates computation instructions that do not depend on shared-memory load
        results. It operates on the set of planes for a single cell within a single time step.

        For a star3d stencil there are (2 * radius + 1) such computations for each cell/time-step.
        To map these computations onto the 2 * radius stencil groups, the first group receives two
        computation instructions while each of the remaining groups receives one.

        `plane_op_map` saves a mapping from `Plane` to the corresponding `StencilComputation`
        operation that generates the result of the plane.
        """
        assert len(groups) == 2 * self.radius
        assert all(len(group.operations) == 0 for group in groups)

        def construct_stencil_computation(plane: Plane) -> StencilComputation:
            assert plane.need_independent_computation

            src_regs = [planes[-1].src_reg]
            if plane.target_offset != 0 and planes[-1].index != 0:
                src_regs.append(plane.src_reg)

            coeff_index = star_coefficient_index_3d(-plane.center_offset, 0, 0)

            result = StencilComputation(
                src_regs=src_regs,
                dst_reg=plane.dst_reg,
                coeff_index=coeff_index,
                comment=f"Vertical plane {plane.center_offset}",
            )

            plane_op_map[plane] = result

            return result

        if planes[self.radius - 1].need_independent_computation:
            assert planes[self.radius - 1].center_offset == -1
            groups[0].operations.append(construct_stencil_computation(planes[self.radius - 1]))

        for id, plane in enumerate(p for p in planes if p.center_offset != -1):
            if not plane.need_independent_computation:
                continue

            groups[id].operations.append(construct_stencil_computation(plane))

    def gen_smem_load_and_memory_dependent_computations(
        self,
        time_step: int,
        memory_plane: Plane,
        memory_plane_producer: StencilComputation | None,
        groups: list[StencilGroup],
        plane_op_map: dict[Plane, StencilComputation],
    ):
        if not memory_plane.need_neighbor_computation:
            return

        def gen_operations(
            elem_pos: tuple[int, int],
        ) -> tuple[SharedMemoryLoad, SharedMemoryLoad, StencilComputation, StencilComputation]:
            """
            This function generates two `LDS` instructions and two computation instructions. The
            `LDS` instructions load neighbor elements from shared memory, and the corresponding
            computation instructions use these loaded elements for computation.

            `elem_pos` specifies the position of the neighbor elements as a 2D coordinate. The
            function generates `LDS` instructions to load two symmetric elements. For example, if
            `elem_pos` is `(2, 0)`, the generated instructions will load elements at positions
            `(2, 0)` and `(-2, 0)`. If `elem_pos` is `(0, 3)`, the instructions will load elements
            at `(0, 3)` and `(0, -3)`.

            """
            assert elem_pos[0] == 0 or elem_pos[1] == 0
            assert elem_pos != (0, 0)
            nonlocal memory_plane_producer

            smem_load1 = SharedMemoryLoad(
                dst_reg=self.neighbors[0 if elem_pos[1] == 0 else 2],
                addr_base=self.shared_mem_reads[-elem_pos[1]],
                addr_offset=time_step - self.smem_chunk_size * elem_pos[0],
                comment="Load neighbor at ({}, {}) of time step {} from shared memory.".format(
                    -elem_pos[0], -elem_pos[1], time_step
                ),
            )
            smem_load2 = SharedMemoryLoad(
                dst_reg=self.neighbors[1 if elem_pos[1] == 0 else 3],
                addr_base=self.shared_mem_reads[elem_pos[1]],
                addr_offset=time_step + self.smem_chunk_size * elem_pos[0],
                comment="Load neighbor at {} of time step {} from shared memory.".format(
                    elem_pos, time_step
                ),
            )

            coeff_index1 = star_coefficient_index_3d(0, -elem_pos[1], -elem_pos[0])
            coeff_index2 = star_coefficient_index_3d(0, elem_pos[1], elem_pos[0])

            comp1 = StencilComputation(
                src_regs=[
                    smem_load1.dst_reg,
                    memory_plane.src_reg
                    if memory_plane_producer is None
                    else memory_plane_producer.dst_reg,
                ],
                dst_reg=memory_plane.dst_reg,
                coeff_index=coeff_index1,
                comment="Compute with neighbor at ({}, {}).".format(-elem_pos[0], -elem_pos[1]),
            )
            comp2 = StencilComputation(
                src_regs=[smem_load2.dst_reg, memory_plane.dst_reg],
                dst_reg=memory_plane.dst_reg,
                coeff_index=coeff_index2,
                comment="Compute with neighbor at {}.".format(elem_pos),
            )

            memory_plane_producer = comp2

            return (smem_load1, smem_load2, comp1, comp2)

        for offset in range(self.radius):
            (smem_load1, smem_load2, comp1, comp2) = gen_operations((offset + 1, 0))
            groups[offset * 2].operations[:0] = [smem_load1, smem_load2]
            CodeGen.insert_before_computation(
                groups[offset * 2 + 1].operations,
                [comp1, comp2],
            )

            (smem_load1, smem_load2, comp1, comp2) = gen_operations((0, offset + 1))
            groups[offset * 2 + 1].operations[:0] = [smem_load1, smem_load2]
            if offset == self.radius - 1:
                groups[offset * 2 + 1].operations.extend([comp1, comp2])
            else:
                CodeGen.insert_before_computation(
                    groups[offset * 2 + 2].operations,
                    [comp1, comp2],
                )

        plane_op_map[memory_plane] = memory_plane_producer

    def computation_groups(self, planes, status: CodeGenStatus):
        stencil_groups: list[list[StencilGroup]] = []
        plane_op_map: dict[Plane, StencilComputation] = {}

        for time_step_planes in planes:
            time_step_groups: list[StencilGroup] = [StencilGroup() for _ in range(2 * self.radius)]
            self.gen_memory_independent_computations(
                time_step_planes, time_step_groups, plane_op_map
            )
            stencil_groups.append(time_step_groups)

        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=False)
        ):
            if not plane_in_range(time_step_planes[-1].index, status.phase):
                continue

            operation = SharedMemoryStore(
                src_reg=time_step_planes[-1].src_reg,
                addr_base=self.shared_mem_writes[0],
                addr_offset=time_step,
                comment=f"Store the result of time step {time_step} to shared memory.",
            )
            time_step_groups[-1].operations.insert(0, operation)

        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=False)
        ):
            self.gen_smem_load_and_memory_dependent_computations(
                time_step=time_step,
                memory_plane=time_step_planes[self.radius - 1],
                memory_plane_producer=plane_op_map.get(time_step_planes[self.radius - 1], None),
                groups=time_step_groups,
                plane_op_map=plane_op_map,
            )

        return stencil_groups

    def finished_outputs(self, planes):
        result = [p for p in planes if p.finishes_current_iteration]
        assert len(result) <= self.radius

        return result

    def borrow_output_address(self, planes, index: int, count: int) -> str:
        return planes[-(index + 2)].dst_reg
