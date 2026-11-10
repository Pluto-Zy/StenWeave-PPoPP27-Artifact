"""Historical box2d staged generator.

The operation ordering is intentionally retained for breakdown experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from codegen.cuda.breakdown.operations import (
    CUDAOperation,
    Dependency,
    GlobalMemoryLoad,
    GlobalMemoryStore,
    SharedMemoryLoad,
    SharedMemoryStore,
    StencilComputation,
    SyncThreads,
    UpdatePtr,
)
from codegen.cuda.breakdown.pipeline import BreakdownPipeline, CodeGenConfig
from codegen.logical import BoxTopology
from codegen.state import Phase


class Action(Enum):
    UpdatePipelinePhaseLoopVariable = auto()


@dataclass
class CodeGenStatus:
    phase: Phase

    # The location of the current plane relative to `phase`.
    #
    # This is a relative position. For `Phase.Head`, this is the offset relative to the first row,
    # so this is also the actual row where the plane is located. For `Phase.Pipeline`, this is the
    # compile-time index after performing `elememts_per_time_step` loop unrolling relative to the
    # first row on the basis of `Phase.Head`, which is different from the runtime position of the
    # plane. For `Phase.Tail`, this is the offset relative to the last row, so it decreases from -1.
    plane: int

    # Represents the runtime offset of `plane`.
    #
    # Since the input may not be exactly unrolled in the pipeline phase, we will leave some rows
    # that are not enough to fill one iteration. The number of remaining rows is a runtime value
    # because it depends on the height of the input.
    #
    # When `phase` is `Phase.Tail`, `plane` is the offset relative to the last row, so it cannot
    # represent the row offset brought by this runtime value. We use this value to adjust the
    # register allocation in `Phase.Tail`.
    runtime_offset: int = 0

    def __hash__(self):
        return hash((self.phase, self.plane, self.runtime_offset))


@dataclass
class Plane:
    # The index of the plane that is relative to the current phase.
    index: int
    # The offset of the plane relative to the center plane.
    center_offset: int
    # The offset of the plane relative to the target plane.
    target_offset: int

    # Whether this plane needs to be processed in the current iteration.
    need_processing: bool
    # Represents whether the computation of this plane will be finished in the current iteration.
    plane_finished: bool

    # The source register of this plane.
    src_reg: str
    # The destination register of this plane.
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
    # Represents the offset of the source plane in this group relative to the center plane. Only
    # valid when `direction` is `Direction.Horizontal`.
    center_offset: int
    operations: list[CUDAOperation] = field(default_factory=list)


@dataclass
class GatherContributions(CUDAOperation):
    src_regs: tuple[str, str, str] = ("", "", "")
    dst_reg: str = ""

    def to_cuda(self) -> str:
        older, middle, recent = self.src_regs
        return f"{self.dst_reg} = {older} + {middle}; {self.dst_reg} = {self.dst_reg} + {recent};"


class CodeGen(BreakdownPipeline):
    def __init__(
        self,
        radius: int,
        time_block_size: int,
        overlap_size: int,
        codegen_config: CodeGenConfig,
    ):
        assert 0 < radius < 4
        assert overlap_size % 2 == 0

        self.codes = []
        self.radius = radius
        self.codegen_config = codegen_config
        # Number of time steps performed in one iteration.
        self.time_block_size = time_block_size
        self.topology = BoxTopology(radius, time_block_size)
        # Number of elements that are overlapped between two blocks.
        self.overlap_size = overlap_size

        # Number of elements that should be saved in registers for each time step.
        self.elements_per_time_step = self.topology.elements_per_time_step
        # Each element is a double-precision floating-point number, which occupies 2 registers.
        self.regs_per_time_step = self.elements_per_time_step * 2

        # The size of each chunk in shared memory, in number of double-precision floating-point
        # numbers. Each thread occupies `self.smem_chunk_size` elements in shared memory.
        self.smem_chunk_size = self.time_block_size + (1 if self.time_block_size % 2 == 0 else 0)

        self.precomputed_addr_slot_size = (self.radius + 1) * self.time_block_size - 1

        # Replacements for registers in some instructions. It works like a register copy at compile
        # time.
        self.replacements = {}
        self.candidate_replacements = {}

        self.global_mem_stride = "global_mem_stride"
        self.input = "input"
        self.output = "output"

        self.shared_mem_read = "shared_mem_read"
        self.shared_mem_write = "shared_mem_write"

        self.left_neighbor = "left_neighbor"
        # R11:R10 are used.
        self.right_neighbor1 = "right_neighbor1"
        # R13:R12 are used.
        self.right_neighbor2 = "right_neighbor2"

        # This predicate register indicates whether the current thread needs to write the result to
        # global memory.
        self.write_result = "write_result"

        self.global_mem_desc = "global_mem_desc"

        # The number of iterations in the pipeline phase.
        #
        # Note that although this value can be stored in a 32-bit register, UR3 is also needed
        # because 64-bit intermediate values will be generated during the calculation.
        self.pipeline_phase_iterations = "pipeline_phase_iterations"
        self.remain_rows = "remain_rows"

        self.regs = tuple(
            tuple(f"reg_{time_step_start}_{i}" for i in range(self.elements_per_time_step))
            for time_step_start in range(self.time_block_size)
        )

        self.code_indent = 4

    @property
    def reuse_shared_rows(self) -> bool:
        return (
            self.radius == 1
            and self.codegen_config.gather_row_contributions
            and self.codegen_config.comm_centric_smem_scheduling
            and not self.codegen_config.layered_point_update
        )

    def row_contribution_history(self, time_step: int, slot: int) -> str:
        return f"row_contribution_history_{time_step}_{slot}"

    def generate_reg_declarations(self) -> str:
        name_type_pairs = [
            (self.global_mem_stride, "unsigned"),
            (self.shared_mem_read, "double *"),
            (self.shared_mem_write, "double *"),
            (self.left_neighbor, "double"),
            (self.right_neighbor1, "double"),
            *(((self.right_neighbor2, "double"),) if not self.reuse_shared_rows else ()),
            (self.write_result, "bool"),
            (self.pipeline_phase_iterations, "unsigned"),
            (self.remain_rows, "unsigned"),
            *((reg, "double") for time_step in self.regs for reg in time_step),
            *(
                (f"temp_{i}", "double")
                for i in range(
                    3 if self.reuse_shared_rows else 2 * self.radius * (2 * self.radius + 1)
                )
            ),
        ]

        if self.reuse_shared_rows:
            name_type_pairs.extend(
                (self.row_contribution_history(time_step, slot), "double")
                for time_step in range(self.time_block_size)
                for slot in range(3)
            )

        declarations = (f"{type_} {name} {{}};" for name, type_ in name_type_pairs)
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

    def ldg_scoreboard(self, plane: int) -> int:
        """
        Returns the write scoreboard of the `LDG` instruction that loads the specified plane.
        """
        if plane < 0 or plane > self.precomputed_addr_slot_size:
            plane = self.precomputed_addr_slot_size

        return 0 if plane % 2 == 0 else 3

    def gen_prologue(self):
        self.add_section_title("Prologue")

        codes = []

        # We first compute the value of `self.input` so that we can issue the first `LDG`
        # instruction as soon as possible.
        #
        # `self.global_mem_stride` is also computed here because it is used later.
        codes.extend(
            f"""
            unsigned const ptr_offset = (blockDim.x - {self.overlap_size}) * blockIdx.x + threadIdx.x;
            {self.input} = {self.input} + ptr_offset;
            {self.output} = {self.output} + ptr_offset;
            {self.global_mem_stride} = gridDim.x * blockDim.x - {self.overlap_size} * (gridDim.x - 1);
            """.splitlines()
        )

        # Load the first row from the global memory.
        codes.append(f"{self.regs[0][0]} = *{self.input};")

        # Compute the addresses of other rows in the head phase.
        codes.extend(
            f"{self.precomputed_addr_slot(row)} = bit_cast<double>({self.input} + {self.global_mem_stride} * {row});"
            for row in range(1, self.precomputed_addr_slot_size)
        )

        # Load the second row from the global memory.
        if self.codegen_config.comm_centric_smem_scheduling:
            codes.append(
                f"{self.regs[0][1]} = *bit_cast<double *>({self.precomputed_addr_slot(1)});"
            )

        # While loading the second row, we compute other variables.
        #
        # 1. Determine `self.remain_pipeline_rows`.
        #
        # We need to handle `a = (radius + 1) * time_block_size + 1` rows in the head phase, and
        # `b = max(radius - 1, 1)` rows in the tail phase. Since we handle `elements_per_time_step`
        # rows in each iteration in the pipeline phase, the number of rows remaining to be handled
        # in the pipeline phase is `(height - a - b) % elements_per_time_step`. Note that all values
        # except `height` are constants.
        #
        # - Used registers:
        #   (O1) height
        #   (O2) remain_pipeline_rows
        #   (O3) self.bra_pred_set
        #   (O4) pipeline_rows: the number of rows handled in the pipeline phase.
        #   (O5) self.pipeline_iterations: the number of iterations in the pipeline phase.
        #
        # # Load `height` argument.
        # (1) ULDC (O1), c[0x0][0x170];
        # # pipeline_rows = height - a - b
        # (2) UIADD3 (O4), (O1), constant, URZ;
        # # pipeline_iterations = pipeline_rows // elements_per_time_step
        # #
        # # We use the method from [libdivide](https://github.com/ridiculousfish/libdivide/blob/eea3b2436925396287abe64298e0e8a1c5674f68/libdivide.h#L1001)
        # # to compute a // b for constant b.
        # if magic == 0:
        #     (6) USHF.R.U32.HI (O5), URZ, more, (O4);
        # else:
        #     (3) UIMAD.WIDE.U32 (O5), (O4), magic, URZ;
        #     if more & LIBDIVIDE_ADD_MARKER != 0:
        #         (4) UIADD3 (O5), (O4), -(Next Reg of O5), URZ;
        #         (5) ULEA.HI (O5), (O5), (Next Reg of O5), URZ, 0x1F;
        #         (6) USHF.R.U32.HI (O5), URZ, more & LIBDIVIDE_32_SHIFT_MASK, (O5);
        #     else:
        #         (6) USHF.R.U32.HI (O5), URZ, more, (Next Reg of O5);
        # # remain_pipeline_rows = pipeline_rows - pipeline_iterations * elements_per_time_step
        # (7) UIMAD (O2), (O5), -constant, (O4);
        # # Constant 1.
        # (8) IMAD.MOV.U32 (O3), RZ, RZ, 0x1;
        # # bra_pred_set = 1 << remain_pipeline_rows
        # (9) SHF.L.U32 (O3), (O3), (O2), RZ;
        #
        # - Dependencies:
        #   - (1) ->2 (2) ->4 (3) ->4 (4) ->4 (5) ->4 (6) ->4 (7)
        #   - (7) ->12 (9)
        #   - (8) ->5 (9)
        #
        # = NOTE ===================================================================================
        # For box2d stencil, `elements_per_time_step` is always even, so some cases of the modulo
        # calculation can be further simplified. For example, when the radius is 1,
        # `elements_per_time_step` is 4.
        # ==========================================================================================
        codes.extend(
            f"""
            unsigned const pipeline_rows = height - {(self.radius + 1) * self.time_block_size + 1 + max(self.radius - 1, 1)};
            {self.pipeline_phase_iterations} = pipeline_rows / {self.elements_per_time_step};
            {self.remain_rows} = pipeline_rows % {self.elements_per_time_step};
            """.splitlines()
        )

        # 2. Compute `shared_mem_read` and `shared_mem_write`.
        #
        # shared_mem_read = (threadIdx.x + radius) * smem_chunk_size * 8
        # shared_mem_write = (blockDim.x + 2 * radius + threadIdx.x) * smem_chunk_size * 8
        #
        # - Used registers:
        #   - (O1) constant_chunk_size: value of `smem_chunk_size * 8`
        #   - (O2) self.shared_mem_read
        #   - (O3) self.shared_mem_write
        #
        # - Input:
        #   - (I1) thread_id_x
        #
        # # Constant `smem_chunk_size * 8`.
        # (10) HFMA2.MMA (O1), -RZ, RZ, 0, smem_chunk_size * 8;
        # # shared_mem_read = threadIdx.x * smem_chunk_size * 8 + radius * smem_chunk_size * 8
        # (11) IMAD (O2), (I1), (O1), radius * smem_chunk_size * 8;
        # # a = threadIdx.x + blockDim.x
        # (12) IADD3 (O3), (I1), c[0x0][0x0], RZ;
        # # shared_mem_write = (a + 2 * radius) * smem_chunk_size * 8
        # (13) IMAD (O3), (O3), (O1), 2 * radius * smem_chunk_size * 8;
        #
        # - Dependencies:
        #   - (10) ->10 (11)
        #   - (10) ->10 (13)
        #   - (12) ->6 (13)
        if self.codegen_config.segmented_smem:
            if self.codegen_config.layered_point_update:
                codes.extend(
                    f"""
                    unsigned const smem_read_offset = (threadIdx.x + {self.radius}) * {self.smem_chunk_size};
                    unsigned const smem_write_offset = (blockDim.x + {2 * self.radius} + threadIdx.x) * {self.smem_chunk_size};
                    {self.shared_mem_read} = shared_mem + smem_read_offset;
                    {self.shared_mem_write} = shared_mem + smem_write_offset;
                    """.splitlines()
                )
            else:
                codes.extend(
                    f"""
                    unsigned const smem_read_offset = (threadIdx.x + {self.radius * (2 * self.radius + 1)}) * {self.smem_chunk_size};
                    unsigned const smem_write_offset = (blockDim.x + {self.radius + self.radius * (2 * self.radius + 1)} + threadIdx.x) * {self.smem_chunk_size};
                    {self.shared_mem_read} = shared_mem + smem_read_offset;
                    {self.shared_mem_write} = shared_mem + smem_write_offset;
                    """.splitlines()
                )
        else:
            if self.codegen_config.layered_point_update:
                codes.extend(
                    f"""
                    unsigned const smem_read_offset = {self.radius};
                    unsigned const smem_write_offset = (blockDim.x + {self.radius}) * {self.time_block_size} + {self.radius};
                    {self.shared_mem_read} = shared_mem + smem_read_offset;
                    {self.shared_mem_write} = shared_mem + smem_write_offset;
                    """.splitlines()
                )
            else:
                codes.extend(
                    f"""
                    unsigned const smem_read_offset = {self.radius * (2 * self.radius + 1)};
                    unsigned const smem_write_offset = (blockDim.x + {self.radius}) * {self.time_block_size} + {self.radius * (2 * self.radius + 1)};
                    {self.shared_mem_read} = shared_mem + smem_read_offset;
                    {self.shared_mem_write} = shared_mem + smem_write_offset;
                    """.splitlines()
                )

        # 3. Compute the `write_result_` predicate.
        #
        # - Used registers:
        #   (O1) self.write_result
        #   (O3) grid_last: gridDim.x - 1
        #   (O4) pred_block_last: true if threadIdx.x < block_last
        #   (O5) pred_grid_last: true if blockIdx.x == grid_last
        #   (O6) grid_dim_x
        #
        # - Input:
        #   (I1) block_id_x
        #   (I2) thread_id_x
        #   (I3) block_dim_x
        #
        # # Load `gridDim.x`.
        # (14) IMAD.MOV.U32 (O6), RZ, RZ, c[0x0][0xc];
        # (15) IADD3 (O2), (I3), -constant RZ;
        # # b = gridDim.x - 1
        # (16) IADD3 (O3), (O6), -0x1, RZ;
        # # c = c || blockIdx.x == 0
        # (18) ISETP.EQ.OR (O1), PT, RZ, (I1), (O1);
        # # d = threadIdx.x < a
        # (19) ISETP.LT.U32.AND (O4), PT, (I2), (O2), PT;
        # # e = blockIdx.x == b
        # (20) ISETP.EQ.AND (O5), PT, (O3), (I1), PT;
        # # result = c && (d || e)
        # (21) PLOP3.LUT (O1), PT, (O1), (O4), (O5), 0xE0, 0x0;
        #
        # - Dependencies:
        #   - (15) ->6 (19)
        #   - (14) ->5 (16) ->6 (20)
        #   - (17) ->5 (18)
        #   - (18) ->5 (21)
        #   - (19) ->5 (21)
        #   - (20) ->5 (21)
        codes.append(
            f"{self.write_result} = (threadIdx.x >= {self.overlap_size // 2} || blockIdx.x == 0) && (threadIdx.x < blockDim.x - {self.overlap_size // 2} || blockIdx.x == gridDim.x - 1)"
        )

        for code in codes:
            self.add_code(code)

    def construct_planes(self, status: CodeGenStatus, time_step: int) -> list[Plane]:
        # The index of the target plane, which is the last bottom plane of the current time step.
        target_plane_index = status.plane - time_step * (self.radius + 1)
        # The source plane is the previous plane of the target plane for box stencil.
        source_plane_index = target_plane_index - 1

        def plane_meaningful(plane: int) -> bool:
            """
            This function determines whether a certain plane of the given phase is meaningful, i.e.,
            its result is within the valid range.
            """

            return CodeGen.plane_in_range(plane + self.radius, status.phase)

        def plane_constructor(list_index: int) -> Plane:
            target_offset = list_index - self.elements_per_time_step + 1
            center_offset = target_offset + self.radius + 1
            index = target_plane_index + target_offset

            need_processing = plane_meaningful(index) and CodeGen.plane_in_range(
                source_plane_index, status.phase
            )
            plane_finished = center_offset == -self.radius or (
                status.phase == Phase.Tail and center_offset <= 0 and target_plane_index >= 0
            )

            # The register of this plane.
            src_reg = self.regs[time_step][
                CodeGen.wrap_add(index, status.runtime_offset, self.elements_per_time_step)
            ]

            if plane_finished and time_step != self.time_block_size - 1:
                # The computation of the current plane is complete, so we need to write its result
                # to the next time step. However, for the last time step, we don't modify the
                # destination register.
                dst_reg = self.regs[time_step + 1][
                    CodeGen.wrap(
                        index + self.radius + status.runtime_offset,
                        self.elements_per_time_step,
                    )
                ]

                if center_offset == -self.radius:
                    # For the first top plane, its result can be directly written to the next time
                    # step, because the value in the corresponding register of the next time step is
                    # already useless.
                    pass
                else:
                    # Otherwise, we cannot write the result to the next time step, because the value
                    # in the destination register is still needed. We record this replacement
                    # relationship in `self.candidate_replacements` and keep `dst_reg` unchanged.
                    # When we can write the result to the correct register, we will replace all uses
                    # of the destination register with `dst_reg`.
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
                # Use the precomputed address register if possible.
                if slot is not None:
                    src_reg = slot

            # Replace `src_reg` and `dst_reg` according to `self.replacements`.
            src_reg = self.replacements.get(src_reg, src_reg)
            dst_reg = self.replacements.get(dst_reg, dst_reg)

            return Plane(
                index,
                center_offset,
                target_offset,
                need_processing,
                plane_finished,
                src_reg,
                dst_reg,
            )

        return [plane_constructor(i) for i in range(self.elements_per_time_step)]

    def right_neighbor(self, time_step: int, offset: int) -> str:
        index = time_step * self.radius + offset - 1
        return self.right_neighbor1 if index % 2 == 0 else self.right_neighbor2

    def construction_noscatter_stencil_group(
        self, planes: list[Plane], group: StencilGroup
    ) -> StencilGroup:
        if self.reuse_shared_rows:
            source = (
                planes[-2].src_reg
                if group.direction.is_vertical()
                else self.left_neighbor
                if group.center_offset < 0
                else self.right_neighbor1
            )
            base = 0 if group.direction.is_vertical() else 3 if group.center_offset < 0 else 6
            group.operations = [
                StencilComputation(
                    src_regs=[source]
                    if group.direction.is_vertical()
                    else [source, f"temp_{offset}"],
                    dst_reg=f"temp_{offset}",
                    coeff_addr=(base + offset) * 8,
                    comment=f"Gather row {offset}, step {group.time_step}, {group.direction.name}",
                )
                for offset in range(3)
            ]

            return group

        def construct_stencil_computation(plane: Plane) -> StencilComputation:
            assert plane.need_processing

            if group.direction.is_vertical():
                src_regs = [plane.src_reg]
                if plane is not planes[0]:
                    src_regs.append(planes[0].dst_reg)
            else:
                idx = (
                    2 * abs(group.center_offset) - 2
                    if group.center_offset < 0
                    else 2 * group.center_offset - 1
                )
                idx = idx * (2 * self.radius + 1) + abs(plane.target_offset) - 1
                src_regs = [f"temp_{idx}", planes[0].dst_reg]

            # Determine the coefficients for the current computation.
            if group.direction.is_vertical():
                base = 0
            else:
                base = (abs(group.center_offset) - 1) * 2
                base += 1 if group.center_offset < 0 else 2

            coeff_addr = (base * (2 * self.radius + 1) + abs(plane.target_offset) - 1) * 8

            return StencilComputation(
                src_regs=src_regs,
                dst_reg=planes[0].dst_reg,
                coeff_addr=coeff_addr,
                comment=f"{group.direction} {plane.center_offset}",
            )

        group.operations = [
            construct_stencil_computation(plane) for plane in planes[:-1] if plane.need_processing
        ]

        return group

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

            # Determine the coefficients for the current computation.
            if group.direction.is_vertical():
                base = 0
            else:
                base = (abs(group.center_offset) - 1) * 2
                base += 1 if group.center_offset < 0 else 2

            coeff_addr = (base * (2 * self.radius + 1) + abs(plane.target_offset) - 1) * 8

            return StencilComputation(
                src_regs=src_regs,
                dst_reg=plane.dst_reg,
                coeff_addr=coeff_addr,
                comment=f"{group.direction} {plane.center_offset}",
            )

        group.operations = [
            construct_stencil_computation(plane) for plane in planes[:-1] if plane.need_processing
        ]

        return group

    def compute_min_stalls(
        operations: list[CUDAOperation],
        deps: list[Dependency],
        per_op_min_stall: int = 1,
    ):
        graph = [[] for _ in range(len(operations) + 1)]
        for i in range(len(operations)):
            graph[i].append((i + 1, per_op_min_stall))

        for dep in deps:
            # Find the index of the producer and consumer operations.
            producer_index = operations.index(dep.producer)
            consumer_index = operations.index(dep.consumer)
            assert producer_index < consumer_index

            graph[producer_index].append((consumer_index, dep.required_stall))

        y = [0] * (len(operations) + 1)
        for u in range(len(operations) + 1):
            for v, w in graph[u]:
                y[v] = max(y[v], y[u] + w)

        for id, op in enumerate(operations):
            op.stall = y[id + 1] - y[id]

    def stencil_pipeline(self, status: CodeGenStatus, extra_actions: list[Action] | None = None):
        if extra_actions is None:
            extra_actions = []

        # Merge the replacements that can be applied in the current phase.
        self.replacements.update(self.candidate_replacements.get(status, {}))

        # Construct planes for each time step.
        planes: list[list[Plane]] = [
            self.construct_planes(status, time_step) for time_step in range(self.time_block_size)
        ]

        dependencies: list[Dependency] = []
        stencil_groups: list[list[StencilGroup]] = []

        for time_step, time_step_planes in enumerate(planes):
            if all(not plane.need_processing for plane in time_step_planes):
                # No plane needs to be processed in the current time step. We add an empty stencil
                # group to keep the time step structure consistent.
                stencil_groups.append(
                    [
                        StencilGroup(
                            direction=Direction.Vertical,
                            center_offset=0,
                            time_step=time_step,
                        )
                    ]
                )
                continue

            # The stencil groups in the current time step.
            time_step_groups: list[StencilGroup] = []

            construct_stencil_group = (
                self.construct_stencil_group
                if self.codegen_config.layered_point_update
                else self.construction_noscatter_stencil_group
            )

            # Vertical.
            vertical_stencil_group = StencilGroup(
                direction=Direction.Vertical, center_offset=0, time_step=time_step
            )
            time_step_groups.append(
                construct_stencil_group(time_step_planes, vertical_stencil_group)
            )
            producer_group = vertical_stencil_group

            # Horizontal.
            for offset in range(self.radius):
                offset += 1

                # Left.
                left_stencil_group = StencilGroup(
                    direction=Direction.Horizontal,
                    center_offset=-offset,
                    time_step=time_step,
                )
                time_step_groups.append(
                    construct_stencil_group(time_step_planes, left_stencil_group)
                )

                # Left dependency.
                for producer, consumer in zip(
                    producer_group.operations,
                    left_stencil_group.operations,
                    strict=True,
                ):
                    dependencies.append(
                        Dependency(producer=producer, consumer=consumer, required_stall=6)
                    )

                # Right.
                right_stencil_group = StencilGroup(
                    direction=Direction.Horizontal,
                    center_offset=offset,
                    time_step=time_step,
                )
                time_step_groups.append(
                    construct_stencil_group(time_step_planes, right_stencil_group)
                )

                # Right dependency.
                for producer, consumer in zip(
                    left_stencil_group.operations, right_stencil_group.operations, strict=True
                ):
                    dependencies.append(
                        Dependency(producer=producer, consumer=consumer, required_stall=6)
                    )

                producer_group = right_stencil_group

            stencil_groups.append(time_step_groups)

        if self.reuse_shared_rows:
            for time_step, time_step_groups in enumerate(stencil_groups):
                if len(time_step_groups) != 3:
                    continue

                time_step_groups[-1].operations.append(
                    GatherContributions(
                        src_regs=(
                            self.row_contribution_history(time_step, 0),
                            self.row_contribution_history(time_step, 2),
                            "temp_2",
                        ),
                        dst_reg=planes[time_step][0].dst_reg,
                    )
                )

        # Records whether the current time step is the first time step that should read data from
        # shared memory.
        first_smem_read = True
        # Records the last time step that has read data from shared memory.
        last_smem_read = None

        # Insert operations for shared memory load.
        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=True)
        ):
            assert len(time_step_groups) == self.radius * 2 + 1 or (
                all(not p.need_processing for p in time_step_planes) and len(time_step_groups) == 1
            )

            # We only need to read data from shared memory if the previous plane of the target plane
            # is in range.
            if not CodeGen.plane_in_range(time_step_planes[-2].index, status.phase):
                continue

            assert len(time_step_groups) == self.radius * 2 + 1

            if self.codegen_config.layered_point_update:
                if first_smem_read:
                    time_step_groups[0].operations.insert(
                        0,
                        SharedMemoryLoad(
                            dst_reg=self.left_neighbor,
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, -1),
                            write_scoreboard=1,
                            comment=f"Load left neighbor at offset -1 of time step {time_step}",
                        ),
                    )
                    time_step_groups[0].operations.insert(
                        1,
                        SharedMemoryLoad(
                            dst_reg=self.right_neighbor(time_step=time_step, offset=1),
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, 1),
                            write_scoreboard=2,
                            comment=f"Load right neighbor at offset 1 of time step {time_step}",
                        ),
                    )

                for left_neighbor_group in time_step_groups[1::2]:
                    left_neighbor_group.operations[0].wait_scoreboard.append(1)

                right_neighbor_groups = time_step_groups[2::2]
                for group_id, right_neighbor_group in enumerate(right_neighbor_groups):
                    # The last group should issue the memory access instruction to load the neighbor
                    # elements of the first group of the next time step.
                    offset = 1 if group_id == len(right_neighbor_groups) - 1 else group_id + 2
                    time_step = (
                        time_step + 1 if group_id == len(right_neighbor_groups) - 1 else time_step
                    )

                    right_neighbor_group.operations.insert(
                        0,
                        SharedMemoryLoad(
                            dst_reg=self.left_neighbor,
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, -offset),
                            write_scoreboard=1,
                            comment=f"Load left neighbor at offset -{offset} of time step {time_step}",
                        ),
                    )
                    right_neighbor_group.operations.insert(
                        1 if first_smem_read and group_id == 0 else 2,
                        SharedMemoryLoad(
                            dst_reg=self.right_neighbor(time_step=time_step, offset=offset),
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, offset),
                            write_scoreboard=2,
                            comment=f"Load right neighbor at offset {offset} of time step {time_step}",
                        ),
                    )

                    right_neighbor_group.operations[1].wait_scoreboard.append(2)
            elif self.reuse_shared_rows:
                # Form the new row's contributions once, then gather three rows
                # for one output instead of scattering into output accumulators.
                for direction, offset, destination in (
                    (0, -1, self.left_neighbor),
                    (1, 1, self.right_neighbor1),
                ):
                    time_step_groups[0].operations.insert(
                        direction,
                        SharedMemoryLoad(
                            dst_reg=destination,
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, offset),
                        ),
                    )
            else:
                for i in range(self.radius * (2 * self.radius + 1)):
                    time_step_groups[0].operations.insert(
                        0,
                        SharedMemoryLoad(
                            dst_reg=f"temp_{i}",
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, -(i + 1)),
                        ),
                    )

                for i in range(self.radius * (2 * self.radius + 1)):
                    time_step_groups[0].operations.insert(
                        0,
                        SharedMemoryLoad(
                            dst_reg=f"temp_{i + self.radius * (2 * self.radius + 1)}",
                            addr_base=self.shared_mem_read,
                            addr_offset=self.gen_smem_addr_offset(time_step, i + 1),
                        ),
                    )

            first_smem_read = False
            last_smem_read = time_step_groups[-1]

        if last_smem_read is not None:
            # Remove unnecessary shared memory load operations in the last time step.
            last_smem_read.operations = [
                op for op in last_smem_read.operations if not isinstance(op, SharedMemoryLoad)
            ]
            last_smem_read.operations[0].wait_scoreboard.append(2)

        # Insert operations for shared memory store.
        for time_step, (time_step_planes, time_step_groups) in enumerate(
            zip(planes, stencil_groups, strict=True)
        ):
            if not CodeGen.plane_in_range(time_step_planes[-1].index, status.phase):
                # The last plane of the current time step is not in range, so we don't need to store
                # the result to shared memory.
                continue

            operation = SharedMemoryStore(
                src_reg=time_step_planes[-1].src_reg,
                addr_base=self.shared_mem_write
                if self.codegen_config.comm_centric_smem_scheduling
                else self.shared_mem_read,
                addr_offset=self.gen_smem_addr_offset(time_step, 0),
                comment=f"Store the result of time step {time_step} to shared memory.",
            )

            if time_step == 0:
                scoreboard = self.ldg_scoreboard(time_step_planes[-1].index)
                operation.wait_scoreboard.append(scoreboard)
            else:
                # For other time steps, the result to be written comes from the previous time step.
                # Find the operation that produces this result.
                producer = next(
                    (
                        op
                        for op in stencil_groups[time_step - 1][-1].operations
                        if isinstance(op, StencilComputation)
                    ),
                    None,
                )

                if producer is not None:
                    dependencies.append(
                        Dependency(producer=producer, consumer=operation, required_stall=6)
                    )

            # Insert the store operation.
            if len(time_step_groups) == 1:
                assert len(time_step_groups[0].operations) == 0

                if self.codegen_config.comm_centric_smem_scheduling:
                    time_step_groups[0].operations.append(operation)
                else:
                    time_step_groups[0].operations.insert(0, operation)
                    time_step_groups[0].operations.insert(1, SyncThreads())
            else:
                if self.codegen_config.comm_centric_smem_scheduling:
                    if isinstance(time_step_groups[0].operations[0], SharedMemoryLoad):
                        # If the first group contains memory access operations, insert the store
                        # operation into the second group to avoid stacking too many memory access
                        # instructions.
                        time_step_groups[1].operations.insert(0, operation)
                    else:
                        time_step_groups[0].operations.append(operation)
                else:
                    time_step_groups[0].operations.insert(0, operation)
                    time_step_groups[0].operations.insert(1, SyncThreads())

        # Insert operations for global memory load.
        plane_index = (
            planes[0][-1].index + 1
            if self.codegen_config.comm_centric_smem_scheduling
            else planes[0][-1].index
        )
        if CodeGen.plane_in_range(plane_index, status.phase):
            # The next plane of the target plane of the first time step is in range, so we need to
            # load it from global memory.
            slot = self.precomputed_addr_slot(plane_index)

            if slot is None:
                update_ptr = UpdatePtr(
                    src_reg=self.input,
                    dst_reg=self.input,
                    stride_reg=self.global_mem_stride,
                    offset=plane_index if plane_index == self.precomputed_addr_slot_size else 1,
                    comment="Update the global memory pointer.",
                )
                global_load = GlobalMemoryLoad(
                    dst_reg=planes[0][0].src_reg,
                    desc_reg=self.global_mem_desc,
                    ptr_reg=self.input,
                    write_scoreboard=self.ldg_scoreboard(plane_index),
                    comment="Load a row from global memory.",
                )
                dependencies.append(
                    Dependency(producer=update_ptr, consumer=global_load, required_stall=5)
                )

                assert len(stencil_groups[0]) == self.radius * 2 + 1
                stencil_groups[0][0].operations.insert(2, update_ptr)

                if self.codegen_config.comm_centric_smem_scheduling:
                    stencil_groups[0][1].operations.insert(1, global_load)
                else:
                    stencil_groups[0][0].operations.insert(3, global_load)
            else:
                if status.phase != Phase.Head or plane_index > 1:
                    global_load = GlobalMemoryLoad(
                        dst_reg=slot,
                        desc_reg=self.global_mem_desc,
                        ptr_reg=f"bit_cast<double *>({slot})",
                        write_scoreboard=self.ldg_scoreboard(plane_index),
                        comment="Load a row from global memory.",
                    )
                    stencil_groups[0][0].operations.insert(0, global_load)

        # Find out all planes that are finished in the last time step.
        finished_planes = [
            (id, plane)
            for id, plane in enumerate(planes[-1])
            if plane.plane_finished and plane.need_processing
        ]
        assert len(finished_planes) * 2 - 1 <= self.elements_per_time_step

        # Insert operations for global memory store.
        if finished_planes:
            update_ptrs = []
            global_stores = []

            for id, plane in finished_planes:
                addr_reg = (
                    self.output
                    if id == len(finished_planes) - 1
                    else planes[-1][len(finished_planes) + id].src_reg
                )
                need_bitcast = addr_reg != self.output

                global_store = GlobalMemoryStore(
                    src_reg=plane.dst_reg,
                    desc_reg=self.global_mem_desc,
                    ptr_reg=f"bit_cast<double *>({addr_reg})" if need_bitcast else addr_reg,
                    pred=self.write_result,
                )
                global_stores.append(global_store)
                dependencies.append(
                    Dependency(
                        producer=stencil_groups[-1][-1].operations[id],
                        consumer=global_store,
                        required_stall=6,
                    )
                )

                if status.phase != Phase.Head or plane.index != -self.radius:
                    update_ptr = UpdatePtr(
                        src_reg=self.output,
                        dst_reg=addr_reg,
                        stride_reg=self.global_mem_stride,
                        offset=id + 1,
                        comment="Update the global memory pointer.",
                    )
                    update_ptrs.append(update_ptr)
                    dependencies.append(
                        Dependency(producer=update_ptr, consumer=global_store, required_stall=5)
                    )

            for id, op in enumerate(update_ptrs):
                stencil_groups[-1][-1].operations.insert(id * 2, op)

            stencil_groups[-1][-1].operations.extend(global_stores)

        # Gather all operations in the current pipeline.
        operations: list[CUDAOperation] = [
            op
            for time_step_groups in stencil_groups
            for group in time_step_groups
            for op in group.operations
        ]

        if not operations:
            # No operation needs to be performed in the current pipeline.
            return

        CodeGen.compute_min_stalls(operations, dependencies)

        for time_step, time_step_groups in enumerate(stencil_groups):
            if not time_step_groups[0].operations:
                continue

            self.add_section_title(
                f"Stencil Computation\nPhase: {status.phase}\nTime Step {time_step}"
            )

            for group in time_step_groups:
                for op in group.operations:
                    if op.pred is None:
                        self.add_code(op.to_cuda(), op.comment)
                    else:
                        self.add_code(f"if ({op.pred}) {{ {op.to_cuda()} }}")

            if self.reuse_shared_rows and CodeGen.plane_in_range(
                planes[time_step][-2].index, status.phase
            ):
                older = self.row_contribution_history(time_step, 0)
                middle = self.row_contribution_history(time_step, 1)
                recent = self.row_contribution_history(time_step, 2)
                self.add_code(f"{older} = {middle}")
                self.add_code(f"{middle} = temp_0")
                self.add_code(f"{recent} = temp_1")

        self.add_barrier()

        if not self.codegen_config.segmented_smem:
            self.add_code(
                f"""if (cond) {{
                    // auto const temp = {self.shared_mem_read};
                    // {self.shared_mem_read} = {self.shared_mem_write};
                    // {self.shared_mem_write} = temp;
                    {self.shared_mem_read} = (double *)__shfl_up_sync(0xFFFFFFFF, (unsigned) {self.shared_mem_read}, 1);
                }}"""
            )

    def head_phase(self):
        for plane in range(self.topology.head_end):
            self.add_newline()
            self.add_section_title(f"Head phase plane {plane} begin")
            self.add_newline()

            self.stencil_pipeline(CodeGenStatus(phase=Phase.Head, plane=plane))

            self.shared_mem_read, self.shared_mem_write = (
                self.shared_mem_write,
                self.shared_mem_read,
            )

    def pipeline_phase(self):
        self.add_section_title("Pipeline Phase")

        self.add_code(
            f"while ({self.pipeline_phase_iterations}--) {{",
            "Begin of the pipeline phase loop.",
        )
        self.code_indent += 4

        # The starting plane index of the pipeline phase.
        self.pipeline_phase_plane_start = self.topology.head_end

        for i in range(self.elements_per_time_step):
            first_plane = self.pipeline_phase_plane_start + i

            self.add_newline()
            self.add_section_title(f"Pipeline phase plane {first_plane} (unroll {i}) begin")
            self.add_newline()

            self.stencil_pipeline(
                CodeGenStatus(phase=Phase.Pipeline, plane=first_plane),
                extra_actions=[Action.UpdatePipelinePhaseLoopVariable] if i == 0 else [],
            )

            assert self.elements_per_time_step % 2 == 0
            self.shared_mem_read, self.shared_mem_write = (
                self.shared_mem_write,
                self.shared_mem_read,
            )

        self.code_indent -= 4
        self.add_code("}", "End of the pipeline phase loop.")

    def gen_tail_phase(self):
        # Save the pointers so that we can restore them in each case of the tail phase.
        smem_ptrs = (self.shared_mem_read, self.shared_mem_write)
        self.add_code(f"switch ({self.remain_rows}) {{", "Begin of tail phase switch.")

        for remain_pipeline_rows in range(self.elements_per_time_step, 0, -1):
            remain_pipeline_rows -= 1

            # Clear the candidate replacements and replacements.
            self.candidate_replacements.clear()
            self.replacements.clear()

            self.add_code(f"case {remain_pipeline_rows}: {{")
            self.code_indent += 4

            # Generate codes for the remaining pipeline rows.
            for i in range(remain_pipeline_rows):
                first_plane = self.pipeline_phase_plane_start + i

                self.add_newline()
                self.add_section_title(
                    f"Remaining Pipeline phase plane {first_plane} (unroll {i}) begin"
                )
                self.add_newline()

                self.stencil_pipeline(CodeGenStatus(phase=Phase.Pipeline, plane=first_plane))

                self.shared_mem_read, self.shared_mem_write = (
                    self.shared_mem_write,
                    self.shared_mem_read,
                )

            # Generate codes for the tail phase.
            plane_start = self.topology.tail_start
            for plane in range(plane_start, self.topology.head_end + plane_start):
                self.add_newline()
                self.add_section_title(f"Tail phase plane {plane} begin")
                self.add_newline()

                self.stencil_pipeline(
                    CodeGenStatus(
                        phase=Phase.Tail,
                        plane=plane,
                        # Note that we must sub `plane_start` here since the plane index in the tail
                        # phase starts from `plane_start`.
                        runtime_offset=self.pipeline_phase_plane_start
                        + remain_pipeline_rows
                        - plane_start,
                    )
                )

                self.shared_mem_read, self.shared_mem_write = (
                    self.shared_mem_write,
                    self.shared_mem_read,
                )

            self.add_code("return", "EXIT")
            self.add_newline()

            self.code_indent -= 4
            self.add_code("}", f"End of case {remain_pipeline_rows}.")

            self.shared_mem_read, self.shared_mem_write = smem_ptrs

        self.add_code("}", "End of tail phase switch.")
        self.add_code("__builtin_unreachable()")
