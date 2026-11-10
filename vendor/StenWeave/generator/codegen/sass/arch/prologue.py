"""Shared prologue builders with explicit architecture instruction branches.

The host resolves geometry. Device code derives thread/CTA positions using borrowed
pipeline registers; architecture branches retain cluster setup and shared-memory geometry.
"""

import re


def next_reg(reg: str) -> str:
    index = re.search(r"\d", reg)
    assert index is not None
    index = index.start()

    return f"{reg[:index]}{int(reg[index:]) + 1}"


def d2_prologue(self, policy):
    self.add_section_title("Prologue")

    initial_reg_count = self.reg_count
    usable_regs = [
        reg
        for time_step_regs in self.regs
        for group in time_step_regs
        for reg in group
        if all(
            reg != self.precomputed_addr_slot(p)[-1] for p in range(self.precomputed_addr_slot_size)
        )
        and all(
            reg != target_reg for row in (0, 1) for target_reg in self.precomputed_addr_slot(row)
        )
    ]

    # Borrow both 32-bit halves of each available FP64 register pair.
    usable_regs = [half for reg in usable_regs for half in (reg, next_reg(reg))]

    def alloc_usable_reg(count: int) -> list[str]:
        if len(usable_regs) < count:
            assert self.time_block_size == 1, "Extra prologue registers require TB=1"

            while len(usable_regs) < count:
                reg = self.alloc_reg(2)
                usable_regs.extend((reg, next_reg(reg)))

        regs = usable_regs[:count]
        del usable_regs[:count]

        return regs

    thread_id_x, block_dim_x = alloc_usable_reg(2)
    grid_dim_x, ptr_offset = alloc_usable_reg(2)
    constant_chunk_size, constant_group_size = alloc_usable_reg(2)

    if policy.is_sm90:
        cga_cta_id = constant_group_size
        smem_read_base, smem_write_base = alloc_usable_reg(2)

    _, grid_last = alloc_usable_reg(2)

    if self.reg_count != initial_reg_count:
        # Uniform registers require two registers beyond the highest thread register used.
        self.alloc_reg(2)

    remain_pipeline_rows = "UR6"
    block_id_x = "UR4"
    pred_block_last = "P0"
    pred_grid_last = "P1"

    # TABLE_TRUE: FXU->IMAD 6, IMAD->IMAD 6, IMAD->MIO 8;
    # predicate math 5, predicate branch/store 13 on both target tables.
    def parameter(name):
        return policy.uniform_parameter(name)

    if not policy.is_sm90:
        self.add_code("[B------:R-:W-:-:S01]", "MOV R1, c[0x0][0x28]")

    self.add_code("[B------:R-:W0:-:S01]", f"S2R {thread_id_x}, SR_TID.X")
    self.add_code("[B------:R-:W1:-:S01]", f"S2UR {block_id_x}, SR_CTAID.X")
    self.add_code("[B------:R-:W-:-:S01]", f"MOV {block_dim_x}, c[0x0][0x0]")
    self.add_code("[B------:R-:W-:-:S01]", f"MOV {grid_dim_x}, c[0x0][0xc]")

    self.add_code("[B------:R-:W-:-:S01]", f"MOV {ptr_offset}, {parameter('cta_stride_x')}")
    self.add_code("[B------:R-:W-:-:S01]", f"MOV {constant_group_size}, {8 * self.group_size:#x}")
    self.add_code(
        "[B------:R-:W-:-:S01]", f"MOV {self.global_mem_stride}, {parameter('row_pitch')}"
    )
    self.add_code(
        "[B------:R-:W-:-:S03]",
        f"ULDC.64 {self.global_mem_desc}, c[0x0][{policy.global_mem_desc_offset:#x}]",
    )

    self.add_code(
        "[B01----:R-:W-:-:S06]", f"IMAD {ptr_offset}, {ptr_offset}, {block_id_x}, {thread_id_x}"
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"IMAD.WIDE.U32 {self.input}, {ptr_offset}.reuse, {constant_group_size}.reuse, "
        f"c[0x0][{policy.input_arg_offset:#x}]",
    )
    self.add_code(
        "[B------:R-:W-:-:S07]",
        f"IMAD.WIDE.U32 {self.output}, {ptr_offset}, {constant_group_size}, "
        f"c[0x0][{policy.output_arg_offset:#x}]",
    )

    for row in range(2):
        if row:
            self.add_code(
                "[B------:R-:W-:-:S08]",
                f"IMAD.WIDE.U32 {self.precomputed_addr_slot(row)[-1]}, "
                f"{self.global_mem_stride}, {8 * self.group_size:#x}, {self.input}",
            )

        plane = self.plane_type(index=row, regs=self.regs[0][row])
        pointer = self.input if row == 0 else self.precomputed_addr_slot(row)[-1]
        for operation in self.gen_gmem_load_operations(ptr_reg=pointer, plane=plane):
            operation.stall = 1
            self.emit_operation(operation)

    # Ownership comparisons use constant operands; independent address setup hides predicates.
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ISETP.GE.U32.AND {self.write_result}, PT, {thread_id_x}, {parameter('write_begin_x')}, PT",
    )
    self.add_code("[B------:R-:W-:-:S01]", f"IADD3 {grid_last}, {grid_dim_x}, -0x1, RZ")
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ISETP.LT.U32.AND {pred_block_last}, PT, {thread_id_x}, {parameter('write_end_x')}, PT",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]", f"IADD3 {self.shared_mem_write}, {thread_id_x}, {block_dim_x}, RZ"
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ULDC {self.pipeline_phase_iterations}, {parameter('iterations_arg')}",
    )
    self.add_code(
        "[B------:R-:W-:-:S02]", f"ULDC {remain_pipeline_rows}, {parameter('remainder_arg')}"
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ISETP.EQ.OR {self.write_result}, PT, RZ, {block_id_x}, {self.write_result}",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]", f"ISETP.EQ.AND {pred_grid_last}, PT, {grid_last}, {block_id_x}, PT"
    )

    if policy.is_sm90:
        # Keep two cycles before the immediately following scoreboard-1 wait on Hopper.
        # With one, LEA can consume the borrowed register's pre-S2R value as a DSMEM rank.
        self.add_code("[B------:R-:W1:-:S02]", f"S2R {cga_cta_id}, SR_CgaCtaId")
        self.add_code(
            "[B-1----:R-:W-:-:S01]",
            f"LEA {smem_read_base}, {cga_cta_id}, "
            f"{1024 + self.thread_halo_radius * self.smem_chunk_size * 8:#x}, 0x18",
        )
        self.add_code(
            "[B------:R-:W-:-:S05]",
            f"LEA {smem_write_base}, {cga_cta_id}, "
            f"{1024 + 2 * self.thread_halo_radius * self.smem_chunk_size * 8:#x}, 0x18",
        )

        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"IMAD {self.shared_mem_read}, {thread_id_x}, {self.smem_chunk_size * 8:#x}, {smem_read_base}",
        )
        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"IMAD {self.shared_mem_write}, {self.shared_mem_write}, {self.smem_chunk_size * 8:#x}, {smem_write_base}",
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S04]", f"MOV {constant_chunk_size}, {self.smem_chunk_size * 8:#x}"
        )
        self.add_code(
            "[B------:R-:W-:-:S03]",
            f"PLOP3.LUT {self.write_result}, PT, {self.write_result}, {pred_block_last}, {pred_grid_last}, 0xE0, 0x0",
        )
        self.add_code("[B------:R-:W-:-:S04]", f"MOV {self.bra_pred_set}, 0x1")

        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"IMAD {self.shared_mem_read}, {thread_id_x}.reuse, {constant_chunk_size}.reuse, "
            f"{self.thread_halo_radius * self.smem_chunk_size * 8:#x}",
        )
        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"IMAD {self.shared_mem_write}, {self.shared_mem_write}, {constant_chunk_size}, "
            f"{2 * self.thread_halo_radius * self.smem_chunk_size * 8:#x}",
        )

    if policy.is_sm90:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"PLOP3.LUT {self.write_result}, PT, {self.write_result}, {pred_block_last}, {pred_grid_last}, 0xE0, 0x0",
        )
        self.add_code("[B------:R-:W-:-:S06]", f"MOV {self.bra_pred_set}, 0x1")

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"SHF.L.U32 {self.bra_pred_set}, {self.bra_pred_set}, {remain_pipeline_rows}, RZ",
    )

    for row in range(2, self.precomputed_addr_slot_size):
        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"IMAD.WIDE.U32 {self.precomputed_addr_slot(row)[-1]}, "
            f"{self.global_mem_stride}, {row * 8 * self.group_size:#x}, {self.input}",
        )

    self.add_code(
        "[B------:R-:W-:-:S05]",
        f"ISETP.NE.AND P6, PT, RZ, {parameter('store_head')}, {self.write_result}",
    )

    pred_mask = (1 << min(len(self.bra_preds), self.planes_per_iteration - 1)) - 1
    self.add_code("[B------:R-:W-:-:S01]", f"R2P PR, {self.bra_pred_set}, {pred_mask:#x}")
    self.add_code(
        "[B------:R-:W-:-:S01]", f"ISETP.NE.AND P4, PT, RZ, {parameter('store_head')}, PT"
    )
    self.add_code(
        "[B------:R-:W-:-:S06]",
        f"SHF.R.U32.HI {self.bra_pred_set}, RZ, {len(self.bra_preds):#x}, {self.bra_pred_set}",
    )


def d3_prologue(self, policy, is_box):
    self.add_section_title("Prologue")

    initial_reg_count = self.reg_count
    if not is_box and (not policy.is_sm90):
        self.add_code("[B------:R-:W-:-:S01]", "MOV R1, c[0x0][0x28]")

    assert self.precomputed_addr_slot_size >= 2

    precomputed_pointer_regs = {
        reg
        for plane in range(self.precomputed_addr_slot_size)
        for reg in self.precomputed_addr_slot(plane)[:, -1].ravel()
    }

    loaded_plane_regs = {
        reg
        for plane in range(min(2, self.precomputed_addr_slot_size))
        for reg in self.precomputed_addr_slot(plane).ravel()
    }

    unavailable_regs = precomputed_pointer_regs | loaded_plane_regs
    usable_regs = [
        reg
        for time_step_regs in self.regs
        for group in time_step_regs
        for reg in group.ravel()
        if reg not in unavailable_regs
    ]
    usable_regs = [half for reg in usable_regs for half in (reg, next_reg(reg))]

    def alloc_usable_reg(count: int) -> list[str]:
        if len(usable_regs) < count:
            assert self.time_block_size == 1, "Extra prologue registers require TB=1"

            while len(usable_regs) < count:
                reg = self.alloc_reg(2)
                usable_regs.extend((reg, next_reg(reg)))

        regs = usable_regs[:count]
        del usable_regs[:count]

        return regs

    block_stride_x = next_reg(self.outputs[-1])
    block_stride_y = next_reg(self.inputs[-1])
    plane_x_length = self.outputs[-1]

    thread_id_x, thread_id_y = alloc_usable_reg(2)
    ptr_offset = next_reg(self.inputs[-1])
    constant8 = self.inputs[-1]
    row_id = next_reg(self.inputs[-1])
    col_id = next_reg(self.outputs[-1])
    thread_off_x = self.inputs[-1]
    thread_off_y = self.inputs[-1]

    assert len(self.shared_mem_writes) >= 3

    if is_box:
        smem_halo_size_x = self.shared_mem_writes[-3]
    else:
        constant_chunk_size = self.shared_mem_writes[-3]

    block_dim_x = self.shared_mem_reads[0]
    row_offset = self.shared_mem_writes[-2]
    max_smem_read_size = self.shared_mem_writes[-1]

    block_x_last, grid_x_last = alloc_usable_reg(2)
    block_y_last, overlap_size_half_x = alloc_usable_reg(2)
    grid_y_last = self.shared_mem_reads[1]
    overlap_size_half_y, constant_one = alloc_usable_reg(2)

    if self.reg_count != initial_reg_count:
        # Keep the uniform-register reservation beyond newly allocated thread registers.
        self.alloc_reg(2)

    if policy.is_sm90:
        cga_cta_id = overlap_size_half_x
        smem_base = constant_one

    remain_pipeline_rows = "UR7"
    block_id_x = "UR4"
    block_id_y = "UR5"

    pred_block_x_last = "P0"
    pred_grid_x_last = "P1"
    pred_block_y_last = "P2"
    pred_grid_y_last = "P3"
    pred_temp1 = "P4"
    pred_temp2 = "P5"

    imad_ldg_gap = 8
    self.add_code(
        "[B------:R-:W0:-:S01]", "S2R {}, SR_TID.X".format(thread_id_x), "(201) Load `threadIdx.x`."
    )
    self.add_code(
        "[B------:R-:W2:-:S01]",
        "S2UR {}, SR_CTAID.X".format(block_id_x),
        "(203) Load `blockIdx.x`.",
    )
    self.add_code(
        "[B------:R-:W1:-:S01]", "S2R {}, SR_TID.Y".format(thread_id_y), "(202) Load `threadIdx.y`."
    )
    self.add_code(
        "[B------:R-:W3:-:S01]",
        "S2UR {}, SR_CTAID.Y".format(block_id_y),
        "(204) Load `blockIdx.y`.",
    )

    self.add_code(
        "[B0-----:R-:W-:Y:S01]",
        "IMAD {d}, {a}, {b}, RZ".format(
            d=thread_off_x, a=thread_id_x, b=f"{self.group_sizes.x:#x}"
        ),
        "(207) thread_off_x = threadIdx.x * group_size_x.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {block_stride_x}, {policy.uniform_parameter('cta_stride_x')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:-:S05]",
        f"MOV {plane_x_length}, {policy.uniform_parameter('row_pitch')}",
        "Host-resolved launch uniform.",
    )

    self.add_code(
        "[B--2---:R-:W-:-:S01]",
        "IMAD {d}, {a}, {b}, {c}".format(d=col_id, a=block_stride_x, b=block_id_x, c=thread_off_x),
        "(209) col_id = block_stride_x * blockIdx.x + thread_off_x.",
    )

    self.add_code(
        "[B-1----:R-:W-:-:S01]",
        "IMAD {d}, {a}, {b}, RZ".format(
            d=thread_off_y, a=thread_id_y, b=f"{self.group_sizes.y:#x}"
        ),
        "(206) thread_off_y = threadIdx.y * group_size_y.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {block_stride_y}, {policy.uniform_parameter('cta_stride_y')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:-:S05]",
        f"MOV {self.global_mem_stride}, {policy.uniform_parameter('plane_stride')}",
        "Host-resolved launch uniform.",
    )

    self.add_code(
        "[B---3--:R-:W-:-:S01]",
        "IMAD {d}, {a}, {b}, {c}".format(d=row_id, a=block_stride_y, b=block_id_y, c=thread_off_y),
        "(208) row_id = block_stride_y * blockIdx.y + thread_off_y.",
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "IMAD.MOV.U32 {d}, RZ, RZ, 0x8".format(d=constant8),
        "(205) Constant 8.",
    )
    self.add_code(
        "[B------:R-:W-:Y:S04]",
        "ULDC.64 {}, c[0x0][{global_mem_desc_offset:#x}]".format(
            self.global_mem_desc, global_mem_desc_offset=policy.global_mem_desc_offset
        ),
        "(211) Load global memory descriptor.",
    )

    self.add_code(
        "[B------:R-:W-:Y:S06]",
        "IMAD {d}, {a}, {b}, {c}".format(d=ptr_offset, a=plane_x_length, b=row_id, c=col_id),
        "(210) ptr_offset = plane_x_length * row_id + col_id.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "IMAD.WIDE.U32 {d}, {a}, {b}, c[0x0][{output_arg_offset:#x}]".format(
            d=self.outputs[0], a=ptr_offset, b=constant8, output_arg_offset=policy.output_arg_offset
        ),
        "(213) outputs[0] += ptr_offset * 8.",
    )
    self.add_code(
        "[B------:R-:W-:Y:S{:02}]".format(imad_ldg_gap if len(self.inputs) == 1 else 6),
        "IMAD.WIDE.U32 {d}, {a}, {b}, c[0x0][{input_arg_offset:#x}]".format(
            d=self.inputs[0], a=ptr_offset, b=constant8, input_arg_offset=policy.input_arg_offset
        ),
        "(212) inputs[0] += ptr_offset * 8.",
    )

    for index, (input, output) in enumerate(
        zip(self.inputs[1:], self.outputs[1:], strict=False), start=1
    ):
        self.add_code(
            "[B------:R-:W-:-:S{:02}]".format(
                1 if index != len(self.outputs) - 1 else imad_ldg_gap
            ),
            "IMAD.WIDE.U32 {d}, {a}, {b:#x}, {c}".format(
                d=input, a=plane_x_length, b=index * 8, c=self.inputs[0]
            ),
            "(214) inputs[{}] = inputs[0] + {} * plane_x_length * 8.".format(index, index),
        )
        self.add_code(
            "[B------:R-:W-:-:S08]",
            "IMAD.WIDE.U32 {d}, {a}, {b:#x}, {c}".format(
                d=output, a=plane_x_length, b=index * 8, c=self.outputs[0]
            ),
            "(215) outputs[{}] = outputs[0] + {} * plane_x_length * 8.".format(index, index),
        )

    first_plane = self.plane_type(index=0, regs=self.regs[0][0])
    gmem_loads = self.gen_gmem_load_operations(ptr_regs=self.inputs, plane=first_plane)

    for op in gmem_loads:
        op.stall = 1
        self.add_code(op.to_control_codes(), op.to_sass(), op.comment)

    for index, input in enumerate(self.inputs):
        self.add_code(
            "[B------:R-:W-:-:S{:02}]".format(1 if index != len(self.inputs) - 1 else imad_ldg_gap),
            "IMAD.WIDE.U32 {d}, {a}, 0x8, {c}".format(
                d=self.precomputed_addr_slot(1)[index, -1], a=self.global_mem_stride, c=input
            ),
            "Precompute the address of row 1 for inputs[{}].".format(index),
        )

    second_plane = self.plane_type(index=1, regs=self.regs[0][1])
    gmem_loads = self.gen_gmem_load_operations(
        ptr_regs=self.precomputed_addr_slot(1)[:, -1], plane=second_plane
    )

    for op in gmem_loads:
        op.stall = 1
        self.add_code(op.to_control_codes(), op.to_sass(), op.comment)

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ULDC {self.pipeline_phase_iterations}, {policy.uniform_parameter('iterations_arg')}",
        "Host-resolved launch uniform.",
    )

    if is_box:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "MOV {d}, {value:#x}".format(
                d=smem_halo_size_x, value=self.radius * self.smem_chunk_size * 8
            ),
            "(201) Constant `radius * smem_chunk_size * 8` = {}.".format(
                self.radius * self.smem_chunk_size * 8
            ),
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "MOV {d}, {value:#x}".format(d=constant_chunk_size, value=self.smem_chunk_size * 8),
            "(201) Constant `smem_chunk_size * 8` = {}.".format(self.smem_chunk_size * 8),
        )

    self.add_code(
        "[B------:R-:W-:-:S02]",
        "IMAD.MOV.U32 {}, RZ, RZ, c[0x0][0x0]".format(block_dim_x),
        "(202) Load `blockDim.x`.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {overlap_size_half_x}, {policy.uniform_parameter('write_begin_x')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "IMAD.MOV.U32 {}, RZ, RZ, 0x1".format(constant_one),
        "(301) Constant 1.",
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {overlap_size_half_y}, {policy.uniform_parameter('write_begin_y')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:Y:S08]",
        "IMAD.MOV.U32 {}, RZ, RZ, 0x1".format(self.bra_pred_set),
        "(108) Constant 1.",
    )

    if is_box:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, {b:#x}, {c}".format(
                d=row_offset, a=block_dim_x, b=self.smem_chunk_size * 8, c=smem_halo_size_x
            ),
            "(203) row_offset = (blockDim.x + self.thread_halo_radius.x) * smem_chunk_size * 8",
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, {b}, {c:#x}".format(
                d=row_offset,
                a=block_dim_x,
                b=constant_chunk_size,
                c=self.thread_halo_radius.x * self.smem_chunk_size * 8,
            ),
            "(203) row_offset = (blockDim.x + self.thread_halo_radius.x) * smem_chunk_size * 8",
        )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {block_x_last}, {policy.uniform_parameter('write_end_x')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:-:S06]",
        "ISETP.GE.U32.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_temp1, a=thread_id_x, b=overlap_size_half_x
        ),
        "(308) {} = threadIdx.x >= overlap_size_x // group_size_x // 2".format(pred_temp1),
    )

    if policy.is_sm90:
        self.add_code(
            "[B------:R-:W1:-:S06]",
            "S2R {}, SR_CgaCtaId".format(cga_cta_id),
            "(210) Load `SR_CgaCtaId` after overlap_size_half_x is dead.",
        )

    if is_box:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, {b:#x}, {c}".format(
                d=self.shared_mem_reads[0],
                a=thread_id_x,
                b=self.smem_chunk_size * 8,
                c=smem_halo_size_x,
            ),
            "(204) temp = (tid.x + self.thread_halo_radius.x) * smem_chunk_size * 8",
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, {b}, {c:#x}".format(
                d=self.shared_mem_reads[0],
                a=thread_id_x,
                b=constant_chunk_size,
                c=self.thread_halo_radius.x * self.smem_chunk_size * 8,
            ),
            "(204) temp = (tid.x + self.thread_halo_radius.x) * smem_chunk_size * 8",
        )

    self.add_code(
        "[B------:R-:W-:-:S02]",
        "IADD3 {d}, -{a}, c[0x0][0x10], RZ".format(d=grid_y_last, a=constant_one),
        "(307) grid_y_last = gridDim.y - 1",
    )

    if is_box:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, c[0x0][0x4], {c}".format(
                d=max_smem_read_size, a=row_offset, c=smem_halo_size_x
            ),
            "(206) max_smem_read_size = row_offset * blockDim.y + radius * smem_chunk_size * 8",
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, c[0x0][0x4], RZ".format(d=max_smem_read_size, a=row_offset),
            "(206) max_smem_read_size = row_offset * blockDim.y",
        )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"MOV {block_y_last}, {policy.uniform_parameter('write_end_y')}",
        "Host-resolved launch uniform.",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "IADD3 {d}, -{a}, c[0x0][0xc], RZ".format(d=grid_x_last, a=constant_one),
        "(306) grid_x_last = gridDim.x - 1",
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ULDC {remain_pipeline_rows}, {policy.uniform_parameter('remainder_arg')}",
        "Host-resolved launch uniform.",
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "IMAD {d}, {a}, {b}, {c}".format(
            d=self.shared_mem_reads[0], a=thread_id_y, b=row_offset, c=self.shared_mem_reads[0]
        ),
        "(205) shared_mem_reads[0] += tid.y * row_offset",
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.GE.U32.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_temp2, a=thread_id_y, b=overlap_size_half_y
        ),
        "(313) {} = threadIdx.y >= overlap_size_y // group_size_y // 2".format(pred_temp2),
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.EQ.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_grid_y_last, a=grid_y_last, b=block_id_y
        ),
        "(316) {} = blockIdx.y == grid_y_last".format(pred_grid_y_last),
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.EQ.OR {d}, PT, RZ, {b}, {c}".format(d=pred_temp1, b=block_id_x, c=pred_temp1),
        "(309) {} = {} || blockIdx.x == 0".format(pred_temp1, pred_temp1),
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.LT.U32.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_block_x_last, a=thread_id_x, b=block_x_last
        ),
        "(310) {} = threadIdx.x < block_x_last".format(pred_block_x_last),
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.EQ.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_grid_x_last, a=grid_x_last, b=block_id_x
        ),
        "(311) {} = blockIdx.x == grid_x_last".format(pred_grid_x_last),
    )

    shared_mem_read_offsets = [
        *range(1, self.thread_halo_radius.y + 1),
        *range(-1, -self.thread_halo_radius.y - 1, -1),
    ]
    assert len(shared_mem_read_offsets) == len(self.shared_mem_reads) - 1

    for offset in shared_mem_read_offsets:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMAD {d}, {a}, {b:#x}, {c}".format(
                d=self.shared_mem_reads[offset], a=row_offset, b=offset, c=self.shared_mem_reads[0]
            ),
            "(207) shared_mem_reads[{}] = shared_mem_reads[0] + {} * row_offset".format(
                offset, offset
            ),
        )

    if policy.is_sm90:
        self.add_code(
            "[B-1----:R-:W-:-:S04]",
            "LEA {d}, {a}, 0x400, 0x18".format(d=smem_base, a=cga_cta_id),
            "(211) smem_base = (SR_CgaCtaId << 0x18) + 0x400",
        )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.EQ.OR {d}, PT, RZ, {b}, {c}".format(d=pred_temp2, b=block_id_y, c=pred_temp2),
        "(314) {} = {} || blockIdx.y == 0".format(pred_temp2, pred_temp2),
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "ISETP.LT.U32.AND {d}, PT, {a}, {b}, PT".format(
            d=pred_block_y_last, a=thread_id_y, b=block_y_last
        ),
        "(315) {} = threadIdx.y < block_y_last".format(pred_block_y_last),
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "SHF.L.U32 {d}, {a}, {b}, RZ".format(
            d=self.bra_pred_set, a=self.bra_pred_set, b=remain_pipeline_rows
        ),
        "(109) bra_pred_set = 1 << remain_pipeline_rows",
    )

    for row in range(2, self.precomputed_addr_slot_size):
        for input_index, input_reg in enumerate(self.inputs):
            self.add_code(
                "[B------:R-:W-:-:S01]",
                "IMAD.WIDE.U32 {d}, {a}, {b:#x}, {c}".format(
                    d=self.precomputed_addr_slot(row)[input_index, -1],
                    a=self.global_mem_stride,
                    b=row * 8,
                    c=input_reg,
                ),
                "Precompute the address of row {} for inputs[{}].".format(row, input_index),
            )

    for physical_index in range(1, len(self.shared_mem_reads)):
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IMNMX.U32 {d}, {a}, {b}, PT".format(
                d=self.shared_mem_reads[physical_index],
                a=self.shared_mem_reads[physical_index],
                b=max_smem_read_size,
            ),
            "(208) shared_mem_reads[{}] = min(shared_mem_reads[{}], max_smem_read_size)".format(
                physical_index, physical_index
            ),
        )

    if policy.is_sm90:
        for physical_index in range(len(self.shared_mem_reads)):
            self.add_code(
                "[B------:R-:W-:-:S06]",
                "IADD3 {d}, {a}, {b}, RZ".format(
                    d=self.shared_mem_reads[physical_index],
                    a=self.shared_mem_reads[physical_index],
                    b=smem_base,
                ),
                "(212) shared_mem_reads[{}] += smem_base".format(physical_index),
            )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "PLOP3.LUT {d}, PT, {a}, {b}, {c}, 0xE0, 0x0".format(
            d=pred_temp1, a=pred_temp1, b=pred_block_x_last, c=pred_grid_x_last
        ),
        "(312) {} = {} && ({} || {})".format(
            pred_temp1, pred_temp1, pred_block_x_last, pred_grid_x_last
        ),
    )
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "PLOP3.LUT {d}, PT, {a}, {b}, {c}, 0xE0, 0x0".format(
            d=pred_temp2, a=pred_temp2, b=pred_block_y_last, c=pred_grid_y_last
        ),
        "(317) {} = {} && ({} || {})".format(
            pred_temp2, pred_temp2, pred_block_y_last, pred_grid_y_last
        ),
    )

    if is_box:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IADD3 {d}, {a}, {b:#x}, {c}".format(
                d=self.shared_mem_writes[0],
                a=self.shared_mem_reads[0],
                b=self.smem_chunk_size * 8,
                c=max_smem_read_size,
            ),
            "(209) shared_mem_writes[0] = shared_mem_reads[0] + max_smem_read_size + smem_chunk_size * 8",
        )
    else:
        self.add_code(
            "[B------:R-:W-:-:S01]",
            "IADD3 {d}, {a}, {b}, RZ".format(
                d=self.shared_mem_writes[0], a=self.shared_mem_reads[0], b=max_smem_read_size
            ),
            "(209) shared_mem_writes[0] = shared_mem_reads[0] + max_smem_read_size",
        )

    used_pred_count = min(len(self.bra_preds), self.planes_per_iteration - 1)
    pred_mask = (1 << used_pred_count) - 1
    self.add_code(
        "[B------:R-:W-:-:S01]",
        "R2P PR, {a}, {b:#x}".format(a=self.bra_pred_set, b=pred_mask),
        "(401) Generate predicates.",
    )

    for physical_index in range(1, len(self.shared_mem_writes)):
        if is_box:
            self.add_code(
                "[B------:R-:W-:-:S01]",
                "IADD3 {d}, {a}, {b:#x}, {c}".format(
                    d=self.shared_mem_writes[physical_index],
                    a=self.shared_mem_reads[physical_index],
                    b=self.smem_chunk_size * 8,
                    c=max_smem_read_size,
                ),
                "(209) shared_mem_writes[{}] = shared_mem_reads[{}] + max_smem_read_size + smem_chunk_size * 8".format(
                    physical_index, physical_index
                ),
            )
        else:
            self.add_code(
                "[B------:R-:W-:-:S01]",
                "IADD3 {d}, {a}, {b}, RZ".format(
                    d=self.shared_mem_writes[physical_index],
                    a=self.shared_mem_reads[physical_index],
                    b=max_smem_read_size,
                ),
                "(209) shared_mem_writes[{}] = shared_mem_reads[{}] + max_smem_read_size".format(
                    physical_index, physical_index
                ),
            )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        "PLOP3.LUT {d}, PT, {a}, {b}, PT, 0x80, 0x0".format(
            d=self.write_result, a=pred_temp1, b=pred_temp2
        ),
        "(318) write_result = {} && {}".format(pred_temp1, pred_temp2),
    )

    self.add_code(
        "[B------:R-:W-:-:S04]",
        "SHF.R.U32.HI {d}, RZ, {b:#x}, {c}".format(
            d=self.bra_pred_set, b=len(self.bra_preds), c=self.bra_pred_set
        ),
        "(402) Shift `bra_pred_set`.",
    )

    self.add_code(
        "[B------:R-:W-:-:S01]",
        f"ISETP.NE.AND P6, PT, RZ, {policy.uniform_parameter('store_head')}, {self.write_result}",
        "Head ownership; consumers are separated by the static head computation.",
    )
    self.add_code(
        "[B------:R-:W-:-:S08]",
        f"ISETP.NE.AND P4, PT, RZ, {policy.uniform_parameter('store_head')}, PT",
        "Halo-head branch predicate; P4 is dead after the ownership conjunction.",
    )


def star3d_prologue(code, policy):
    d3_prologue(code, policy, is_box=False)


def box3d_prologue(code, policy):
    d3_prologue(code, policy, is_box=True)
