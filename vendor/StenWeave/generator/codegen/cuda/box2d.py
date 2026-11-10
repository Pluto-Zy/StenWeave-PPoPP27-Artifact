from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from codegen.cuda.operations import (
    CUDAOperation,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
)
from codegen.cuda.pipeline import SourcePipeline
from codegen.logical import BoxTopology, box_coefficient_index_2d, box_planes, plane_in_range
from codegen.state import CodeGenStatus


@dataclass
class Plane:
    index: int
    center_offset: int
    target_offset: int
    need_processing: bool
    finishes_current_iteration: bool
    src_reg: str
    dst_reg: str


class Direction(Enum):
    Vertical = auto()
    Horizontal = auto()

    def is_vertical(self) -> bool:
        return self == Direction.Vertical

    def is_horizontal(self) -> bool:
        return self == Direction.Horizontal


@dataclass
class StencilGroup:
    """
    Represents a group of stencil computations in the same direction.
    """

    time_step: int
    direction: Direction
    center_offset: int
    operations: list[CUDAOperation] = field(default_factory=list)


class CodeGen(SourcePipeline):
    global_load_positions = (2, 1)
    global_store_pointer_stride = 2

    def __init__(self, radius: int, time_block_size: int, overlap_size: int):
        assert 0 < radius < 4
        assert overlap_size % 2 == 0

        self.topology = BoxTopology(radius, time_block_size)
        self.codes = []
        self.radius = radius
        self.time_block_size = time_block_size
        self.overlap_size = overlap_size
        self.elements_per_time_step = self.topology.elements_per_time_step
        self.planes_per_iteration = self.topology.planes_per_iteration
        self.smem_chunk_size = self.time_block_size + (1 if self.time_block_size % 2 == 0 else 0)
        self.precomputed_addr_slot_size = (self.radius + 1) * self.time_block_size - 1

        self.replacements = {}
        self.candidate_replacements = {}

        self.global_mem_stride = "global_mem_stride"
        self.input = "input"
        self.output = "output"

        self.shared_mem_read = "shared_mem_read"
        self.shared_mem_write = "shared_mem_write"

        self.left_neighbor = "left_neighbor"
        self.right_neighbor1 = "right_neighbor1"
        self.right_neighbor2 = "right_neighbor2"
        self.write_result = "write_result"
        self.pipeline_phase_iterations = "pipeline_phase_iterations"
        self.remain_rows = "remain_rows"

        self.regs = tuple(
            tuple(f"reg_{time_step_start}_{i}" for i in range(self.elements_per_time_step))
            for time_step_start in range(self.time_block_size)
        )
        self.code_indent = 4

    def generate_reg_declarations(self) -> str:
        name_type_pairs = [
            (self.global_mem_stride, "unsigned"),
            (self.shared_mem_read, "double *"),
            (self.shared_mem_write, "double *"),
            (self.left_neighbor, "double"),
            (self.right_neighbor1, "double"),
            (self.right_neighbor2, "double"),
            (self.write_result, "bool"),
            (self.pipeline_phase_iterations, "unsigned"),
            (self.remain_rows, "unsigned"),
            *((reg, "double") for time_step in self.regs for reg in time_step),
        ]
        declarations = (f"{type_} {name};" for (name, type_) in name_type_pairs)
        return "\n    ".join(declarations)

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
        self.add_section_title("Prologue")
        codes = []
        codes.extend(
            f"\n            unsigned const ptr_offset = (blockDim.x - {self.overlap_size}) * blockIdx.x + threadIdx.x;\n            {self.input} = {self.input} + ptr_offset;\n            {self.output} = {self.output} + ptr_offset;\n            {self.global_mem_stride} = gridDim.x * blockDim.x - {self.overlap_size} * (gridDim.x - 1);\n            ".splitlines()
        )

        codes.append(f"{self.regs[0][0]} = *{self.input};")
        codes.extend(
            f"{self.precomputed_addr_slot(row)} = bit_cast<double>({self.input} + {self.global_mem_stride} * {row});"
            for row in range(1, self.precomputed_addr_slot_size)
        )
        codes.append(f"{self.regs[0][1]} = *bit_cast<double *>({self.precomputed_addr_slot(1)});")
        codes.extend(
            f"\n            unsigned const pipeline_rows = height - {(self.radius + 1) * self.time_block_size + 1 + max(self.radius - 1, 1)};\n            {self.pipeline_phase_iterations} = pipeline_rows / {self.elements_per_time_step};\n            {self.remain_rows} = pipeline_rows % {self.elements_per_time_step};\n            ".splitlines()
        )
        codes.extend(
            f"\n            unsigned const smem_read_offset = (threadIdx.x + {self.radius}) * {self.smem_chunk_size};\n            unsigned const smem_write_offset = (blockDim.x + {2 * self.radius} + threadIdx.x) * {self.smem_chunk_size};\n            {self.shared_mem_read} = shared_mem + smem_read_offset;\n            {self.shared_mem_write} = shared_mem + smem_write_offset;\n            ".splitlines()
        )
        codes.append(
            f"{self.write_result} = (threadIdx.x >= {self.overlap_size // 2} || blockIdx.x == 0) && (threadIdx.x < blockDim.x - {self.overlap_size // 2} || blockIdx.x == gridDim.x - 1)"
        )

        for code in codes:
            self.add_code(code)

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
                need_processing,
                logical.finishes_current_iteration,
                src_reg,
                dst_reg,
            )

        return [plane_constructor(i) for i in range(self.elements_per_time_step)]

    def right_neighbor(self, time_step: int, offset: int) -> str:
        index = time_step * self.radius + offset - 1
        return self.right_neighbor1 if index % 2 == 0 else self.right_neighbor2

    def construct_stencil_group(self, planes: list[Plane], group: StencilGroup) -> StencilGroup:
        common_src_reg = (
            planes[-2].src_reg
            if group.direction.is_vertical()
            else self.left_neighbor
            if group.center_offset < 0
            else self.right_neighbor(group.time_step, group.center_offset)
        )

        def construct_stencil_computation(plane: Plane) -> StencilComputation:
            assert plane.need_processing

            src_regs = [common_src_reg]
            if (
                plane.target_offset != -1 and planes[-2].index != 0
            ) or group.direction.is_horizontal():
                src_regs.append(plane.src_reg if group.direction.is_vertical() else plane.dst_reg)

            coeff_index = box_coefficient_index_2d(
                self.radius, group.center_offset, abs(plane.target_offset) - 1 - self.radius
            )

            return StencilComputation(
                src_regs=src_regs,
                dst_reg=plane.dst_reg,
                coeff_index=coeff_index,
                comment=f"{group.direction} {plane.center_offset}",
            )

        group.operations = [
            construct_stencil_computation(plane) for plane in planes[:-1] if plane.need_processing
        ]

        return group

    def computation_groups(self, planes, status: CodeGenStatus):
        stencil_groups: list[list[StencilGroup]] = []

        for time_step, time_step_planes in enumerate(planes):
            if all(not plane.need_processing for plane in time_step_planes):
                stencil_groups.append(
                    [
                        StencilGroup(
                            direction=Direction.Vertical, center_offset=0, time_step=time_step
                        )
                    ]
                )
                continue

            time_step_groups: list[StencilGroup] = []
            vertical_stencil_group = StencilGroup(
                direction=Direction.Vertical, center_offset=0, time_step=time_step
            )
            time_step_groups.append(
                self.construct_stencil_group(time_step_planes, vertical_stencil_group)
            )

            for offset in range(self.radius):
                offset += 1
                left_stencil_group = StencilGroup(
                    direction=Direction.Horizontal, center_offset=-offset, time_step=time_step
                )
                time_step_groups.append(
                    self.construct_stencil_group(time_step_planes, left_stencil_group)
                )

                right_stencil_group = StencilGroup(
                    direction=Direction.Horizontal, center_offset=offset, time_step=time_step
                )
                time_step_groups.append(
                    self.construct_stencil_group(time_step_planes, right_stencil_group)
                )

            stencil_groups.append(time_step_groups)

        first_smem_read = True
        last_smem_read = None

        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=False)
        ):
            assert len(time_step_groups) == self.radius * 2 + 1 or (
                all(not p.need_processing for p in time_step_planes) and len(time_step_groups) == 1
            )

            if not plane_in_range(time_step_planes[-2].index, status.phase):
                continue

            assert len(time_step_groups) == self.radius * 2 + 1

            if first_smem_read:
                time_step_groups[0].operations.insert(
                    0,
                    SharedMemoryLoad(
                        dst_reg=self.left_neighbor,
                        addr_base=self.shared_mem_read,
                        addr_offset=time_step - self.smem_chunk_size,
                        comment=f"Load left neighbor at offset -1 of time step {time_step}",
                    ),
                )
                time_step_groups[0].operations.insert(
                    1,
                    SharedMemoryLoad(
                        dst_reg=self.right_neighbor(time_step=time_step, offset=1),
                        addr_base=self.shared_mem_read,
                        addr_offset=time_step + self.smem_chunk_size,
                        comment=f"Load right neighbor at offset 1 of time step {time_step}",
                    ),
                )

            right_neighbor_groups = time_step_groups[2::2]
            for group_id, right_neighbor_group in enumerate(right_neighbor_groups):
                offset = 1 if group_id == len(right_neighbor_groups) - 1 else group_id + 2
                time_step = (
                    time_step + 1 if group_id == len(right_neighbor_groups) - 1 else time_step
                )

                right_neighbor_group.operations.insert(
                    0,
                    SharedMemoryLoad(
                        dst_reg=self.left_neighbor,
                        addr_base=self.shared_mem_read,
                        addr_offset=time_step - self.smem_chunk_size * offset,
                        comment=f"Load left neighbor at offset -{offset} of time step {time_step}",
                    ),
                )
                right_neighbor_group.operations.insert(
                    1 if first_smem_read and group_id == 0 else 2,
                    SharedMemoryLoad(
                        dst_reg=self.right_neighbor(time_step=time_step, offset=offset),
                        addr_base=self.shared_mem_read,
                        addr_offset=time_step + self.smem_chunk_size * offset,
                        comment=f"Load right neighbor at offset {offset} of time step {time_step}",
                    ),
                )

            first_smem_read = False
            last_smem_read = time_step_groups[-1]

        if last_smem_read is not None:
            last_smem_read.operations = [
                op for op in last_smem_read.operations if not isinstance(op, SharedMemoryLoad)
            ]

        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=False)
        ):
            if not plane_in_range(time_step_planes[-1].index, status.phase):
                continue

            operation = SharedMemoryStore(
                src_reg=time_step_planes[-1].src_reg,
                addr_base=self.shared_mem_write,
                addr_offset=time_step,
                comment=f"Store the result of time step {time_step} to shared memory.",
            )

            if len(time_step_groups) == 1:
                assert len(time_step_groups[0].operations) == 0
                time_step_groups[0].operations.append(operation)
            elif isinstance(time_step_groups[0].operations[0], SharedMemoryLoad):
                time_step_groups[1].operations.insert(0, operation)
            else:
                time_step_groups[0].operations.append(operation)

        return stencil_groups

    @property
    def publication_buffers(self):
        return self.shared_mem_read, self.shared_mem_write

    @publication_buffers.setter
    def publication_buffers(self, buffers):
        self.shared_mem_read, self.shared_mem_write = buffers

    def finished_outputs(self, planes):
        result = [p for p in planes if p.finishes_current_iteration]
        assert len(result) * 2 - 1 <= self.elements_per_time_step

        return result

    def borrow_output_address(self, planes, index: int, count: int) -> str:
        return planes[count + index].src_reg
