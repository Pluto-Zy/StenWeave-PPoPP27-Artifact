"""Historical star2d staged generator.

The operation ordering is intentionally retained for breakdown experiments.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto

from codegen.cuda.breakdown.pipeline import BreakdownPipeline, CodeGenConfig
from codegen.logical import StarTopology
from codegen.state import Phase


class Action(Enum):
    UpdatePipelinePhaseLoopVariable = auto()
    SaveSMemPtr = auto()
    SwapSMemPtr = auto()


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

    actions: list[Action] = field(default_factory=list)


class SwapSMemPtrStage(Enum):
    NOP = auto()
    Save = auto()
    Swap = auto()


class PlaneDirection(Enum):
    Vertical = auto()
    Horizontal = auto()

    def is_vertical(self) -> bool:
        return self == PlaneDirection.Vertical

    def is_horizontal(self) -> bool:
        return self == PlaneDirection.Horizontal


@dataclass
class PlaneInfo:
    # The index of the plane.
    index: int
    # The direction of the plane.
    direction: PlaneDirection
    # The offset of the plane relative to the center plane.
    center_offset: int
    # The offset of the plane relative to the target plane.
    target_offset: int

    # Whether this plane needs to be processed in the current iteration.
    need_processing: bool = True

    # The source register of this plane.
    src_reg: list[str] = field(default_factory=list)
    # The destination register of this plane.
    dst_reg: str = ""

    # Represents whether the computation of this plane will be finished in the current iteration.
    plane_finished: bool = False
    # The address of the coefficient of this plane in constant memory.
    coefficient: int = 0

    # The number of cycles to wait after computing the value in `dst_reg`.
    stall: int = 0
    # The scoreboard to wait before computing the value in `dst_reg`.
    wait_scoreboard: list[int] = field(default_factory=list)
    # The scoreboard to set after computing the value in `dst_reg`.
    write_scoreboard: int | None = None
    # The scoreboard to set after reading the value in `dst_reg`.
    read_scoreboard: int | None = None


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
        assert overlap_size >= 2 * radius * time_block_size

        self.codes = []
        self.radius = radius
        self.codegen_config = codegen_config
        # Number of time steps performed in one iteration.
        self.time_block_size = time_block_size
        self.overlap_size = overlap_size
        self.topology = StarTopology(self.radius, self.time_block_size)

        # Number of elements that should be saved in registers for each time step.
        self.elements_per_time_step = self.topology.elements_per_time_step
        # Each element is a double-precision floating-point number, which occupies 2 registers.
        self.regs_per_time_step = self.elements_per_time_step * 2

        # The size of each chunk in shared memory, in number of double-precision floating-point
        # numbers. Each thread occupies `self.smem_chunk_size` elements in shared memory.
        self.smem_chunk_size = self.time_block_size + (1 if self.time_block_size % 2 == 0 else 0)
        # Number of planes processed in each iteration in the pipeline phase.
        self.planes_per_iteration = self.topology.planes_per_iteration

        self.replacements = {}
        self.candidate_replacements = {}

        self.global_mem_stride = "global_mem_stride"
        self.input = "input"
        self.output = "output"

        self.shared_mem_read = "shared_mem_read"
        self.shared_mem_write = "shared_mem_write"

        self.left_neighbor = "left_neighbor"
        self.right_neighbor = "right_neighbor"

        # This predicate register indicates whether the current thread needs to write the result to
        # global memory.
        self.write_result = "write_result"

        # UR1:UR0 are used.
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

    def generate_reg_declarations(self) -> str:
        name_type_pairs = [
            (self.global_mem_stride, "unsigned"),
            (self.shared_mem_read, "double *"),
            (self.shared_mem_write, "double *"),
            (self.left_neighbor, "double"),
            (self.right_neighbor, "double"),
            (self.write_result, "bool"),
            (self.pipeline_phase_iterations, "unsigned"),
            (self.remain_rows, "unsigned"),
            *((reg, "double") for time_step in self.regs for reg in time_step),
        ]

        if self.codegen_config.gather_update:
            name_type_pairs.extend(
                (f"gather_{age}_{time_step}", "double")
                for time_step in range(self.time_block_size)
                for age in ("older", "recent")
            )
            name_type_pairs.append(("gather_incoming", "double"))

        declarations = (f"{type_} {name};" for name, type_ in name_type_pairs)
        return "\n    ".join(declarations)

    def precomputed_addr_slot(self, plane: int) -> str | None:
        if 0 < plane < self.topology.stream_halo:
            if plane <= self.radius:
                plane -= 1

            return self.regs[plane // self.radius][plane % self.radius + 1]
        else:
            return None

    def ldg_scoreboard(self, plane: int) -> int:
        """
        Returns the write scoreboard of the `LDG` instruction that loads the specified plane.
        """
        if plane < 0 or plane >= self.topology.stream_halo:
            plane = self.topology.stream_halo

        return 0 if plane % 2 == 0 else 4

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

        # While loading the first row, we compute the address of other rows and issue a second `LDG`
        # instruction.
        #
        # We still need to load `radius * time_block_size + 1` rows for the head phase. However, for
        # the final 2 rows, we compute their addresses on the fly as what we do in the pipeline
        # phase. Therefore, we only need to precompute the addresses of
        # `radius * time_block_size - 1` rows.
        codes.extend(
            f"{self.precomputed_addr_slot(row)} = bit_cast<double>({self.input} + {self.global_mem_stride} * {row});"
            for row in range(1, self.topology.stream_halo)
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
        # We need to handle `radius * time_block_size + 1` rows in the head phase, and `radius` rows
        # in the tail phase. Since we handle `elements_per_time_step` rows in each iteration in the
        # pipeline phase, the number of rows remaining to be handled in the pipeline phase is
        # `(height - radius * time_block_size - radius - 1) % elements_per_time_step`. Note that all
        # values except `height` are constants.
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
        # # pipeline_rows = height - radius * time_block_size - radius - 1
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
        codes.extend(
            f"""
            unsigned const pipeline_rows = height - {self.radius * self.time_block_size + self.radius + 1};
            {self.pipeline_phase_iterations} = pipeline_rows / {self.planes_per_iteration};
            {self.remain_rows} = pipeline_rows % {self.planes_per_iteration};
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
                unsigned const smem_read_offset = {self.radius};
                unsigned const smem_write_offset = (blockDim.x + {self.radius}) * {self.time_block_size} + {self.radius};
                {self.shared_mem_read} = shared_mem + smem_read_offset;
                {self.shared_mem_write} = shared_mem + smem_write_offset;
                """.splitlines()
            )

        # 3. Compute the `write_result_` predicate.
        #
        # - Used registers:
        #   (O1) self.write_result
        #   (O2) block_last: blockDim.x - (overlap_size / 2)
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
        # # a = blockDim.x - (overlap_size / 2)
        # (15) IADD3 (O2), (I3), -(overlap_size / 2), RZ;
        # # b = gridDim.x - 1
        # (16) IADD3 (O3), (O6), -0x1, RZ;
        # # c = threadIdx.x >= (overlap_size / 2)
        # (17) ISETP.GT.U32.AND (O1), PT, (I2), (overlap_size / 2) - 1, PT;
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
        #
        # 4. Compute the addresses for the remaining rows in the head phase.
        #
        # 5. Generate predicates in the first chunk.
        #
        # used_pred_count = min(len(self.bra_preds), self.elements_per_time_step - 1)
        # mask = (1 << used_pred_count) - 1
        #
        # - Used registers:
        #   - (O1) some predicates in `self.bra_preds`
        #
        # - Input:
        #   - (I1) self.bra_pred_set
        #
        # # Generate predicates.
        # (22) R2P PR, (I1), mask;
        # # Shift `bra_pred_set`.
        # (23) SHF.R.U32.HI (I1), RZ, len(self.bra_preds), (I1);
        #
        # - Dependencies:
        #   - (9) ->6 (22) ->1 (23)
        codes.append(
            f"{self.write_result} = (threadIdx.x >= {self.overlap_size // 2} || blockIdx.x == 0) && (threadIdx.x < blockDim.x - {self.overlap_size // 2} || blockIdx.x == gridDim.x - 1)"
        )

        for code in codes:
            self.add_code(code)

    def gen_plane_info(
        self, status: CodeGenStatus, time_step: int
    ) -> tuple[list[list[PlaneInfo]], list[PlaneInfo], list[PlaneInfo]]:
        # The index of the target plane, which is the last bottom plane of the current time step.
        target_plane_index = status.plane - time_step * self.radius
        # The index of the center plane.
        center_plane_index = target_plane_index - self.radius
        # The write scoreboard of the `LDG` instruction issued by the pipeline to which the current
        # computation belongs.
        #
        # Note that we must use `status.plane` rather than `target_plane_index` here.
        cur_ldg_scoreboard = self.ldg_scoreboard(status.plane + 1)

        def construct_plane_info(list_index: int, direction: PlaneDirection) -> PlaneInfo:
            center_offset = (
                list_index
                if list_index <= self.radius
                else list_index - self.elements_per_time_step
            )
            target_offset = center_offset - self.radius

            index = (
                center_plane_index + center_offset
                if direction.is_vertical()
                else center_plane_index - 1
            )
            # The register of this plane.
            plane_reg = self.regs[time_step][
                CodeGen.wrap_add(index, status.runtime_offset, self.elements_per_time_step)
            ]

            result = PlaneInfo(index, direction, center_offset, target_offset)
            # We currently set the destination register to be the plane register itself, and we will
            # modify it later if necessary.
            #
            # This value is invalid for horizontal planes, but we won't use it anyway.
            result.dst_reg = plane_reg

            return result

        # The information of horizontal planes (i.e., the left and right neighboring elements).
        #
        # The first element represents the center element itself. We store the center element in
        # `vertical_planes[0]`, so `horizontal_planes[0]` is `None`.
        horizontal_planes: list[PlaneInfo] = [
            None if i == 0 else construct_plane_info(i, direction=PlaneDirection.Horizontal)
            for i in range(self.elements_per_time_step)
        ]
        # The information of vertical planes (i.e., the top and bottom neighboring elements).
        vertical_planes: list[PlaneInfo] = [
            construct_plane_info(i, direction=PlaneDirection.Vertical)
            for i in range(self.elements_per_time_step)
        ]

        target_plane = vertical_planes[self.radius]

        def plane_meaningful(plane: int) -> bool:
            """
            This function determines whether a certain plane of the given phase is meaningful, i.e.,
            its result is within the valid range.
            """

            return CodeGen.plane_in_range(plane + self.radius, status.phase)

        # Determine whether each plane needs to be processed.
        #
        # If the plane with offset -1 is meaningful, all horizontal planes should be processed.
        horizontal_valid = plane_meaningful(vertical_planes[-1].index)
        for plane in horizontal_planes[1:]:
            plane.need_processing = horizontal_valid

        # Represents whether the target plane is within the valid range. If the target plane is out
        # of range, the value of the plane is always zero, so all vertical planes do not need to be
        # processed.
        target_plane_in_range = CodeGen.plane_in_range(target_plane.index, status.phase)
        for plane in vertical_planes:
            plane.need_processing = plane_meaningful(plane.index) and target_plane_in_range

        # The address slot for the `LDG` instruction issued by the previous pipeline.
        last_ldg_addr_slot = self.precomputed_addr_slot(status.plane)
        if time_step == 0 and last_ldg_addr_slot is not None:
            # Move the value from the slot to the target plane.
            target_plane.src_reg = [last_ldg_addr_slot]
        else:
            target_plane.src_reg = [target_plane.dst_reg]

        # Determines the source and destination registers for each plane.
        for plane in vertical_planes:
            if plane is target_plane:
                continue

            plane.src_reg = [target_plane.src_reg[0]]

            if plane.target_offset != 0 and target_plane.index != 0:
                plane.src_reg.append(plane.dst_reg)

        for plane in horizontal_planes[1:]:
            # Currently, we assume that the computation of neighboring planes reads the result of
            # `vertical_planes[-1]` and writes the result back to that register. However, this is
            # not entirely correct. For example, the last computation should write the result to the
            # correct destination register. We will make corrections later.
            plane.src_reg = [
                self.left_neighbor if plane.center_offset < 0 else self.right_neighbor,
                vertical_planes[-1].dst_reg,
            ]
            plane.dst_reg = vertical_planes[-1].dst_reg

        # Modify the destination registers of some special planes.
        for plane in vertical_planes:
            plane.plane_finished = plane.center_offset == -self.radius or (
                status.phase == Phase.Tail
                and plane.center_offset <= -1
                and target_plane.index + 1 >= 0
            )

            if plane.plane_finished and time_step != self.time_block_size - 1:
                # The computation of the current plane is complete, so we need to write its result
                # to the next time step. However, for the last time step, we don't modify the
                # destination register.
                dst_reg = self.regs[time_step + 1][
                    CodeGen.wrap(
                        plane.index + self.radius + status.runtime_offset,
                        self.elements_per_time_step,
                    )
                ]

                if plane.center_offset == -self.radius:
                    # For `vertical_planes[-radius]`, its result can be directly written to the next
                    # time step, because the value in the corresponding register of the next time
                    # step is already useless.
                    plane.dst_reg = dst_reg
                else:
                    # Otherwise, we cannot write the result to the next time step, because the value
                    # in the destination register is still needed. We record this replacement
                    # relationship in `self.candidate_replacements` and keep `plane.dst_reg`
                    # unchanged. When we can write the result to the correct register, we will
                    # replace all uses of the destination register with `plane.dst_reg`.
                    self.candidate_replacements[dst_reg] = plane.dst_reg

        for plane in horizontal_planes[1:]:
            plane.plane_finished = False

        # Note that the result of the last right neighbor should be written to the destination
        # register of `vertical_planes[-1]`.
        final_right_neighbor = horizontal_planes[1]
        final_right_neighbor.dst_reg, vertical_planes[-1].dst_reg = (
            vertical_planes[-1].dst_reg,
            final_right_neighbor.dst_reg,
        )
        final_right_neighbor.plane_finished, vertical_planes[-1].plane_finished = (
            vertical_planes[-1].plane_finished,
            final_right_neighbor.plane_finished,
        )

        if target_plane.dst_reg in self.candidate_replacements:
            # Only when `target_plane` touches a replacement in `candidate_replacements`, the
            # replacement can be enabled.
            self.replacements[target_plane.dst_reg] = self.candidate_replacements[
                target_plane.dst_reg
            ]

        # Replace all uses of the registers in `self.replacements` with their corresponding values.
        for plane in itertools.chain(vertical_planes, horizontal_planes[1:]):
            plane.src_reg = [
                self.replacements.get(reg, reg) for reg in plane.src_reg if reg is not None
            ]
            plane.dst_reg = self.replacements.get(plane.dst_reg, plane.dst_reg)

        # Determine the scoreboards for each plane.
        for plane in vertical_planes:
            plane.write_scoreboard = (
                3 if plane.center_offset == -self.radius else cur_ldg_scoreboard
            )

        for plane in horizontal_planes[1:]:
            plane.wait_scoreboard = [1 if plane.center_offset < 0 else 2]
            plane.write_scoreboard = 2 if plane.center_offset < 0 else 1

        # Adjust the scoreboards: the result of the last right neighbor should be written to the
        # destination register of `vertical_planes[-1]`, so its write scoreboard should be the same
        # as that of `vertical_planes[-1]`. The result of `vertical_planes[-1]` will be used as the
        # input of the computation of the first left neighbor, so its write scoreboard should be the
        # same as the wait scoreboard of the first left neighbor.
        final_right_neighbor.write_scoreboard = vertical_planes[-1].write_scoreboard
        vertical_planes[-1].write_scoreboard = horizontal_planes[-self.radius].wait_scoreboard[0]

        # Determine the coefficients for each plane.
        vertical_planes[0].coefficient = 0
        for offset in range(1, self.radius + 1):
            # The starting address of the coefficient of the element with offset `offset`.
            addr = (offset - 1) * 4 * 8 + 8
            # Top.
            vertical_planes[offset].coefficient = addr
            # Bottom.
            vertical_planes[-offset].coefficient = addr + 8
            # Left.
            horizontal_planes[-offset].coefficient = addr + 16
            # Right.
            horizontal_planes[offset].coefficient = addr + 24

        # Allocate the computation order.
        result: list[list[PlaneInfo]] = []
        for offset in range(self.radius, 0, -1):
            result.append(
                [
                    vertical_planes[-offset],
                    vertical_planes[offset],
                    horizontal_planes[-offset],
                    horizontal_planes[offset],
                ]
            )

        if self.radius > 1:
            # `target_plane` must be processed last.
            result[0][1], result[-1][1] = result[-1][1], result[0][1]
            # Put `vertical_planes[-1]` first and `vertical_planes[-radius]` second.
            result[0][0], result[0][1], result[-1][0] = result[-1][0], result[0][0], result[0][1]

        result[-1].insert(0, vertical_planes[0])

        # Remove all planes that do not need to be processed.
        result = [[x for x in item if x.need_processing] for item in result]
        # Remove all empty lists.
        result = [item for item in result if item]

        def find_index(planes: list[PlaneInfo], pred) -> int:
            return next(
                (i for i, plane in enumerate(planes) if pred(plane)),
                -1,
            )

        # Set the basic stall cycles for each plane.
        for plane in vertical_planes:
            plane.stall = 1

        for plane in horizontal_planes[1:]:
            # The computation of the left neighbor needs to stall for 6 cycles because its result
            # will be used by the right neighbor. The right neighbor is usually the last computation
            # in a group, so it stalls for 1 cycle to immediately issue the next memory instruction.
            plane.stall = 6 if plane.center_offset < 0 else 1

            if plane.plane_finished:
                # If the result of the current plane is already complete, its value needs to be used
                # in the next cycle. We need to stall for enough cycles to ensure the correctness of
                # the result.
                plane.stall = max(plane.stall, 6)

        if result:
            # The result of `vertical_planes[-1]` is used by the `horizontal_planes` in the same
            # group, so we need to ensure that there is at least a 6-cycle stall between them.
            producer_index = find_index(result[0], lambda x: x is vertical_planes[-1])
            consumer_index = find_index(result[0], lambda x: x.direction.is_horizontal())

            if producer_index != -1 and consumer_index != -1:
                assert producer_index < consumer_index

                # Gather the latency of the instructions between them.
                latency = sum(plane.stall for plane in result[0][producer_index:consumer_index])
                additional_latency = max(0, 6 - latency)
                result[0][consumer_index - 1].stall += additional_latency

            # The result of handling right neighbors will be used by the left neighbor in the next
            # group, so we need to ensure that there is at least a 6-cycle stall between them.
            for group_index, group in enumerate(result[:-1]):
                if group[-1].direction.is_horizontal():
                    producer = group[-1]
                    assert producer.center_offset > 0

                    next_group = result[group_index + 1]
                    # The meaning of `len(next_group)` is the number of instructions between
                    # `producer` and the first horizontal plane in `next_group`. The last 2 planes
                    # in `next_group` are horizontal planes, so we need to subtract them. We also
                    # need to add 2 `LDS` instructions.
                    producer.stall = max(producer.stall, 6 - len(next_group))

        return result, vertical_planes, horizontal_planes

    def gather_computation(
        self,
        status: CodeGenStatus,
        time_step: int,
        target_plane: PlaneInfo,
        finished_planes: list[PlaneInfo],
    ) -> None:
        assert self.radius == 1

        incoming_valid = CodeGen.plane_in_range(target_plane.index, status.phase)
        if incoming_valid:
            self.add_code(f"gather_incoming = {target_plane.src_reg[0]}")

        if finished_planes:
            assert len(finished_planes) == 1
            destination = finished_planes[0].dst_reg
            self.add_code(f"{destination} = gather_recent_{time_step} * neighbor_coeffs[0]")

            if status.phase != Phase.Head or target_plane.index >= 2:
                self.add_code(
                    f"{destination} = gather_older_{time_step} * neighbor_coeffs[1] + {destination}"
                )

            if incoming_valid:
                self.add_code(
                    f"{destination} = gather_incoming * neighbor_coeffs[2] + {destination}"
                )

            self.add_code(
                f"{destination} = {self.left_neighbor} * neighbor_coeffs[3] + {destination}"
            )
            self.add_code(
                f"{destination} = {self.right_neighbor} * neighbor_coeffs[4] + {destination}"
            )

        if incoming_valid:
            if status.phase != Phase.Head or target_plane.index >= 1:
                self.add_code(f"gather_older_{time_step} = gather_recent_{time_step}")

            self.add_code(f"gather_recent_{time_step} = gather_incoming")

    def stencil_pipeline(self, status: CodeGenStatus):
        # The address slot for the `LDG` instruction to be issued in the current pipeline.
        cur_ldg_addr_slot = (
            self.precomputed_addr_slot(status.plane + 1)
            if self.codegen_config.comm_centric_smem_scheduling
            else self.precomputed_addr_slot(status.plane)
        )
        # The address slot for the `LDG` instruction issued by the previous pipeline.
        last_ldg_addr_slot = self.precomputed_addr_slot(status.plane)

        # The write scoreboard of the `LDG` instruction issued by the pipeline to which the current
        # computation belongs.
        cur_ldg_scoreboard = self.ldg_scoreboard(status.plane + 1)
        # The write scoreboard of the `LDG` instruction issued by the previous pipeline.
        last_ldg_scoreboard = self.ldg_scoreboard(status.plane)
        # The scoreboard that is not used by the current pipeline.
        unused_ldg_scoreboard = 0 if cur_ldg_scoreboard == 4 else 4

        # Represents whether we have already waited for the scoreboard of the previous row.
        prev_row_scoreboard = False

        for time_step in range(self.time_block_size):
            plane_info, vertical, _ = self.gen_plane_info(status, time_step)
            target_plane = vertical[self.radius]
            need_global_load = time_step == 0 and CodeGen.plane_in_range(
                status.plane + 1, status.phase
            )
            # Record the register that the `STS` instruction uses. We must ensure that this register
            # is not modified before the `STS` instruction reads it.
            sts_target_reg = ""
            # Represents whether the previous group has issued an `STG` instruction. In this case,
            # if the current group also needs to issue an `STG` instruction, the instruction to
            # update the address should wait for the scoreboard.

            if not plane_info:
                # No plane needs to be processed in this time step.
                continue

            self.add_section_title(
                f"Stencil Computation\nPhase: {status.phase}\nTime Step {time_step}"
            )

            if need_global_load and cur_ldg_addr_slot is not None:  # noqa: SIM102
                # `cur_ldg_addr_slot` being not `None` means that the address of the current plane
                # has been precomputed. This means that we can issue the `LDG` instruction
                # immediately without waiting for the previous `LDG` instruction to complete,
                # because we don't need to wait for a new plane to be available.
                #
                # Note that the `LDG` instruction for plane 1 has already been issued in
                # `gen_prologue`.
                if status.plane != 0:
                    self.add_code(
                        f"{cur_ldg_addr_slot} = *bit_cast<double *>({cur_ldg_addr_slot});",
                        "Load the row {} from global memory to the precomputed slot.".format(
                            status.plane + 1
                        ),
                    )

            if (
                need_global_load
                and not self.codegen_config.comm_centric_smem_scheduling
                and cur_ldg_addr_slot is None
            ):
                # The address of the next plane has not been precomputed, so we need to compute it
                # now.
                offset = status.plane if status.plane == self.topology.stream_halo else 1
                self.add_code(
                    f"{self.input} = {self.input} + {self.global_mem_stride} * {offset};",
                    "Compute the input address of the next plane.",
                )

                # Issue the `LDG` instruction for the next plane.
                dst_reg = self.regs[0][
                    CodeGen.wrap(
                        status.plane + status.runtime_offset,
                        self.elements_per_time_step,
                    )
                ]
                self.add_code(
                    f"{dst_reg} = *{self.input}",
                    "Load the row {} from global memory.".format(status.plane),
                )

            for group_id, planes in enumerate(plane_info):
                # Combine `time_step` and `group_id` to compute the global time step. It is like
                # treating each group as a separate time step and connecting all groups into a
                # one-dimensional sequence.
                global_time_step = group_id + time_step * self.radius

                # Determine whether we should use the saved shared memory pointer.
                should_use_saved_shared_mem_ptr = (
                    Action.SwapSMemPtr in status.actions and global_time_step <= 4
                )
                smem_write = (
                    self.saved_shared_mem_write
                    if should_use_saved_shared_mem_ptr
                    else self.shared_mem_write
                )
                smem_read = (
                    self.saved_shared_mem_read
                    if should_use_saved_shared_mem_ptr
                    else self.shared_mem_read
                )

                if group_id == 0 and not self.codegen_config.comm_centric_smem_scheduling:
                    target_plane_reg = target_plane.src_reg[0]
                    # Remove the `.reuse` flag if it exists.
                    target_plane_reg = target_plane_reg.removesuffix(".reuse")

                    # Store the target plane to shared memory.
                    if status.phase != Phase.Tail or target_plane.index < 0:
                        # The target plane is still in the valid range, so we need to store its
                        # value to shared memory.

                        # If the scoreboard of the previous row has not been waited for, we should
                        # wait for it now. Otherwise, we wait for the scoreboard of the previous
                        # time step.

                        # Record the register that the `STS` instruction uses.
                        sts_target_reg = (
                            last_ldg_addr_slot
                            if time_step == 0 and last_ldg_addr_slot is not None
                            else target_plane_reg
                        )
                        self.add_code(
                            f"{smem_read}[{self.gen_smem_addr_offset(time_step, 0)}] = {sts_target_reg};",
                            "Store the target plane to shared memory.",
                        )
                        self.add_code("__syncthreads();")

                if planes[-1].direction.is_horizontal():
                    # We need to compute horizontal planes in this iterations, so we should issue
                    # some `LDS` instructions to load data from shared memory.
                    assert planes[-2].direction.is_horizontal()

                    left_offset = planes[-2].center_offset
                    right_offset = planes[-1].center_offset

                    left_addr = self.gen_smem_addr_offset(time_step, left_offset)
                    right_addr = self.gen_smem_addr_offset(time_step, right_offset)

                    self.add_code(
                        f"{planes[-2].src_reg[0]} = {smem_read}[{left_addr}];",
                        "Load the left neighbor at offset {}.".format(left_offset),
                    )
                    self.add_code(
                        f"{planes[-1].src_reg[0]} = {smem_read}[{right_addr}];",
                        "Load the right neighbor at offset {}.".format(right_offset),
                    )

                if group_id == 0 and self.codegen_config.comm_centric_smem_scheduling:
                    target_plane_reg = target_plane.src_reg[0]
                    # Remove the `.reuse` flag if it exists.
                    target_plane_reg = target_plane_reg.removesuffix(".reuse")

                    # Store the target plane to shared memory.
                    if status.phase != Phase.Tail or target_plane.index < 0:
                        # The target plane is still in the valid range, so we need to store its
                        # value to shared memory.

                        # If the scoreboard of the previous row has not been waited for, we should
                        # wait for it now. Otherwise, we wait for the scoreboard of the previous
                        # time step.

                        # Record the register that the `STS` instruction uses.
                        sts_target_reg = (
                            last_ldg_addr_slot
                            if time_step == 0 and last_ldg_addr_slot is not None
                            else target_plane_reg
                        )
                        self.add_code(
                            f"{smem_write}[{self.gen_smem_addr_offset(time_step, 0)}] = {sts_target_reg};",
                            "Store the target plane to shared memory.",
                        )

                if group_id == 0:  # noqa: SIM102
                    if (
                        need_global_load
                        and cur_ldg_addr_slot is None
                        and self.codegen_config.comm_centric_smem_scheduling
                    ):
                        # The address of the next plane has not been precomputed, so we need to
                        # compute it now.
                        offset = (
                            status.plane + 1 if status.plane == self.topology.stream_halo - 1 else 1
                        )
                        self.add_code(
                            f"{self.input} = {self.input} + {self.global_mem_stride} * {offset};",
                            "Compute the input address of the next plane.",
                        )

                # Records the planes whose computation has been completed in this loop.
                finished_planes = [plane for plane in planes if plane.plane_finished]

                if time_step == self.time_block_size - 1 and finished_planes:
                    assert len(finished_planes) == 1

                    # Issue instruction to compute the address of output.
                    if finished_planes[0].index != -self.radius:
                        # This is not the first plane we output, so we need to update the value of
                        # `self.output`.
                        self.add_code(
                            f"{self.output} = {self.output} + {self.global_mem_stride};",
                            "Compute the output address of the next plane.",
                        )

                # Issue the computation instructions.
                if self.codegen_config.gather_update:
                    self.gather_computation(status, time_step, target_plane, finished_planes)

                for plane in () if self.codegen_config.gather_update else planes:
                    assert len(plane.src_reg) in (1, 2)

                    if plane.dst_reg == sts_target_reg:
                        # We must ensure that the `STS` instruction has read the correct value.
                        plane.wait_scoreboard.append(unused_ldg_scoreboard)

                    if not prev_row_scoreboard:
                        plane.wait_scoreboard.append(last_ldg_scoreboard)
                        prev_row_scoreboard = True

                    if len(plane.src_reg) == 1:
                        self.add_code(
                            f"{plane.dst_reg} = {plane.src_reg[0]} * neighbor_coeffs[{plane.coefficient // 8}];",
                            "Plane with center offset {o}, {d}.".format(
                                o=plane.center_offset, d=plane.direction
                            ),
                        )
                    else:
                        self.add_code(
                            f"{plane.dst_reg} = {plane.src_reg[0]} * neighbor_coeffs[{plane.coefficient // 8}] + {plane.src_reg[1]};",
                            "Plane with center offset {o}, {d}.".format(
                                o=plane.center_offset, d=plane.direction
                            ),
                        )

                if group_id == 0:  # noqa: SIM102
                    if (
                        need_global_load
                        and cur_ldg_addr_slot is None
                        and self.codegen_config.comm_centric_smem_scheduling
                    ):
                        # Issue the `LDG` instruction for the next plane.
                        dst_reg = self.regs[0][
                            CodeGen.wrap(
                                status.plane + 1 + status.runtime_offset,
                                self.elements_per_time_step,
                            )
                        ]
                        self.add_code(
                            f"{dst_reg} = *{self.input}",
                            "Load the row {} from global memory.".format(status.plane + 1),
                        )

                if time_step == self.time_block_size - 1:
                    for plane in finished_planes:
                        self.add_code(
                            f"if ({self.write_result}) {{ *{self.output} = {plane.dst_reg}; }}"
                        )

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
        for plane in range(self.time_block_size * self.radius + 1):
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
            f"while ({self.pipeline_phase_iterations}--) {{", "Begin of the pipeline phase loop."
        )
        self.code_indent += 4

        # The starting plane index of the pipeline phase.
        self.pipeline_phase_plane_start = self.time_block_size * self.radius + 1

        for i in range(self.planes_per_iteration):
            first_plane = self.pipeline_phase_plane_start + i

            self.add_newline()
            self.add_section_title(f"Pipeline phase plane {first_plane} (unroll {i}) begin")
            self.add_newline()

            actions = []
            self.stencil_pipeline(
                CodeGenStatus(phase=Phase.Pipeline, plane=first_plane, actions=actions)
            )

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

        for remain_pipeline_rows in range(self.planes_per_iteration, 0, -1):
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
            for plane in range(
                self.topology.tail_start, self.topology.head_end + self.topology.tail_start
            ):
                self.add_newline()
                self.add_section_title(f"Tail phase plane {plane} begin")
                self.add_newline()

                self.stencil_pipeline(
                    CodeGenStatus(
                        phase=Phase.Tail,
                        plane=plane,
                        # Note that we must add `self.radius` here since the plane index in the tail
                        # phase starts from `-self.radius`.
                        runtime_offset=self.pipeline_phase_plane_start
                        + remain_pipeline_rows
                        + self.radius,
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
