from __future__ import annotations

from dataclasses import dataclass, field

from codegen.cuda.addressing.d3 import VolumeAddressing
from codegen.cuda.operations import (
    CUDAOperation,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
)
from codegen.logical import BoxTopology, box_coefficient_index_3d, box_planes, plane_in_range
from codegen.state import CodeGenStatus


@dataclass(eq=False)
class Plane:
    index: int
    center_offset: int
    target_offset: int
    finishes_current_iteration: bool
    need_processing: bool
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
    global_load_positions = (2, 1)
    global_store_pointer_stride = 2
    topology_type = BoxTopology
    address_slot_adjustment = 1
    neighbor_count = 3
    full_cross_section = True

    def precomputed_addr_slot(self, plane: int) -> str | None:
        if 0 <= plane < self.precomputed_addr_slot_size:
            if plane <= self.radius + 1:
                return self.regs[0][plane]
            else:
                plane += 1
                return self.regs[plane // (self.radius + 1)][plane % (self.radius + 1) + 1]
        else:
            return None

    def construct_planes(self, status: CodeGenStatus, time_step: int) -> list[Plane]:
        logical_planes = box_planes(self.radius, status, time_step)

        def plane_constructor(list_index: int) -> Plane:
            logical = logical_planes[list_index]
            index = logical.index
            center_offset = logical.center_offset
            target_offset = logical.target_offset
            finished = logical.finished
            need_processing = logical.needs_processing

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

            if time_step == 0 and (target_offset == 0 or target_offset == -1):
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
                need_processing,
                src_reg,
                dst_reg,
            )

        return [plane_constructor(i) for i in range(self.elements_per_time_step)]

    def coeff_index(self, pos: tuple[int, int], target_offset: int) -> int:
        return box_coefficient_index_3d(
            self.radius, abs(target_offset) - 1 - self.radius, pos[1], pos[0]
        )

    def gen_memory_independent_computations(
        self,
        planes: list[Plane],
        groups: list[StencilGroup],
        plane_op_map: dict[Plane, StencilComputation],
    ):
        """
        This function generates computation instructions that do not depend on shared-memory load
        results. It operates on the set of planes for a single cell within a single time step.

        `plane_op_map` saves a mapping from `Plane` to the corresponding `StencilComputation`
        operation that generates the result of the plane.
        """
        assert len(groups) == (2 * self.radius + 1) ** 2
        assert all(len(group.operations) == 0 for group in groups)

        source_plane = planes[-2]

        def construct_stencil_computation(plane: Plane) -> StencilComputation:
            assert plane.need_processing

            src_regs = [source_plane.src_reg]
            if plane.target_offset != -1 and source_plane.index != 0:
                src_regs.append(plane.src_reg)

            result = StencilComputation(
                src_regs=src_regs,
                dst_reg=plane.dst_reg,
                coeff_index=self.coeff_index((0, 0), plane.target_offset),
                comment=f"Vertical plane {plane.center_offset}",
            )

            plane_op_map[plane] = result

            return result

        groups[0].operations = [
            construct_stencil_computation(plane) for plane in planes if plane.need_processing
        ]

    def gen_smem_load_and_memory_dependent_computations(
        self,
        planes: list[list[Plane]],
        groups: list[list[StencilGroup]],
        plane_op_map: dict[Plane, StencilComputation],
    ):
        neighbor_index = 0

        def gen_operations(
            pos1: tuple[int, int], pos2: tuple[int, int], time_step: int, planes: list[Plane]
        ) -> tuple[
            SharedMemoryLoad, SharedMemoryLoad, list[StencilComputation], list[StencilComputation]
        ]:
            """
            The function generates two `LDS` instructions and the corresponding computation
            instructions that use the loaded neighbor elements for computation. `pos1` and `pos2`
            specify the positions of the neighbor elements as 2D coordinates.
            """
            assert any(plane.need_processing for plane in planes)
            nonlocal neighbor_index

            (smem_load1, smem_load2) = tuple(
                SharedMemoryLoad(
                    dst_reg=self.neighbors[(neighbor_index + i) % len(self.neighbors)],
                    addr_base=self.shared_mem_reads[pos_y],
                    addr_offset=time_step + self.smem_chunk_size * pos_x,
                    comment="Load neighbor at ({}, {}) of time step {} from shared memory.".format(
                        pos_x, pos_y, time_step
                    ),
                )
                for (i, (pos_x, pos_y)) in enumerate((pos1, pos2))
            )

            result1: list[StencilComputation] = []
            result2: list[StencilComputation] = []

            for plane in planes:
                if not plane.need_processing:
                    continue

                producer = plane_op_map.get(plane)
                comp1 = StencilComputation(
                    src_regs=[
                        smem_load1.dst_reg,
                        plane.src_reg if producer is None else producer.dst_reg,
                    ],
                    dst_reg=plane.dst_reg,
                    coeff_index=self.coeff_index(pos1, plane.target_offset),
                    comment="Compute from neighbor {} for plane {}".format(
                        pos1, plane.center_offset
                    ),
                )
                comp2 = StencilComputation(
                    src_regs=[smem_load2.dst_reg, plane.dst_reg],
                    dst_reg=plane.dst_reg,
                    coeff_index=self.coeff_index(pos2, plane.target_offset),
                    comment="Compute from neighbor {} for plane {}".format(
                        pos2, plane.center_offset
                    ),
                )

                plane_op_map[plane] = comp2
                result1.append(comp1)
                result2.append(comp2)

            neighbor_index += 2

            return (smem_load1, smem_load2, result1, result2)

        first_smem_read = True

        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, groups, strict=False)
        ):
            if not any(plane.need_processing for plane in time_step_planes):
                continue

            assert len(time_step_groups) == (2 * self.radius + 1) ** 2
            assert len(time_step_groups) // 2 == 2 * self.radius * (self.radius + 1)

            for n in range(2 * self.radius * (self.radius + 1)):
                pos = (n % (2 * self.radius + 1), n // (2 * self.radius + 1))
                pos1 = (pos[0] - self.radius, pos[1] - self.radius)
                pos2 = (-pos1[0], -pos1[1])

                (smem_load1, smem_load2, comps1, comps2) = gen_operations(
                    pos1, pos2, time_step, time_step_planes
                )

                target_group = time_step_groups[2 * n]
                target_group.operations.insert(0, smem_load1)
                target_group.operations.insert(1 if first_smem_read and n < 2 else 2, smem_load2)
                time_step_groups[2 * n + 1].operations.extend(comps1)
                time_step_groups[2 * n + 2].operations.extend(comps2)

            first_smem_read = False

    def computation_groups(self, planes, status: CodeGenStatus):
        plane_op_map: dict[Plane, StencilComputation] = {}
        stencil_groups: list[list[StencilGroup]] = [
            [StencilGroup() for _ in range((2 * self.radius + 1) ** 2)]
            for _ in range(self.time_block_size)
        ]

        for time_step_planes, time_step_groups in zip(planes, stencil_groups, strict=False):
            self.gen_memory_independent_computations(
                time_step_planes, time_step_groups, plane_op_map
            )

        self.gen_smem_load_and_memory_dependent_computations(
            planes=planes, groups=stencil_groups, plane_op_map=plane_op_map
        )

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
            assert not time_step_planes[-1].need_processing

            if time_step_groups[0].operations and isinstance(
                time_step_groups[0].operations[0], SharedMemoryLoad
            ):
                time_step_groups[1].operations.insert(0, operation)
            else:
                time_step_groups[0].operations.append(operation)

        return stencil_groups

    def finished_outputs(self, planes):
        result = [p for p in planes if p.finishes_current_iteration]
        assert len(result) * 2 - 1 <= self.elements_per_time_step

        return result

    def borrow_output_address(self, planes, index: int, count: int) -> str:
        return planes[count + index].src_reg
