from __future__ import annotations

from dataclasses import replace
from enum import Enum, auto
from itertools import pairwise

from codegen.logical import plane_in_range
from codegen.sass.dependency import DependencyManager
from codegen.sass.ir import (
    InstrGroup,
    SASSOperation,
    StencilComputation,
    UpdatePipelinePhaseLoopVariable,
    UpdatePtr,
)
from codegen.state import CodeGenStatus, Phase


class Action(Enum):
    UpdatePipelinePhaseLoopVariable = auto()


class OutputAddressLayout(Enum):
    """SASS plane registers available for borrowing by output pointers."""

    ReverseTail = auto()
    AfterOutputs = auto()


class Pipeline:
    def head_phase(self):
        self.in_head = True

        for plane in range(self.topology.head_end):
            self.add_newline()
            self.add_section_title(f"Head phase plane {plane} begin")
            self.add_newline()

            self.stencil_pipeline(CodeGenStatus(phase=Phase.Head, plane=plane))

            self.shared_mem_read, self.shared_mem_write = (
                self.shared_mem_write,
                self.shared_mem_read,
            )

        extra = self.topology.halo_head_extension
        self.add_code("[B------:R-:W-:-:S05]", "@P4 BRA `(.L_AFTER_HALO_HEAD)")

        for plane in range(self.topology.head_end, self.topology.head_end + extra):
            self.stencil_pipeline(CodeGenStatus(phase=Phase.Pipeline, plane=plane))
            self.shared_mem_read, self.shared_mem_write = (
                self.shared_mem_write,
                self.shared_mem_read,
            )

        self.add_label(".L_AFTER_HALO_HEAD")
        self.in_head = False

    def pipeline_phase(self):
        self.add_section_title("Pipeline Phase")

        label = ".L_PIPELINE_PHASE_BEGIN"
        # The halo-head branch also uses P4. Resolve loop entry from the loaded counter
        # and allow the uniform-to-thread predicate dependency to settle before branching.
        self.add_code(
            "[B------:R-:W-:-:S01]",
            f"ISETP.EQ.U32.AND P4, PT, RZ, {self.pipeline_phase_iterations}, PT",
        )
        self.add_code(
            "[B------:R-:W-:Y:S12]",
            f"ISETP.NE.AND P6, PT, RZ, {self.arch.POLICY.uniform_parameter('store_tail')}, {self.write_result}",
        )
        self.add_code("[B------:R-:W-:-:S05]", "@P4 BRA `(.L_AFTER_PIPELINE)")
        self.add_label(label)

        # The starting plane index of the pipeline phase.
        self.pipeline_phase_plane_start = self.topology.head_end + self.topology.halo_head_extension

        for i in range(self.planes_per_iteration):
            first_plane = self.pipeline_phase_plane_start + i

            self.add_newline()
            self.add_section_title(f"Pipeline phase plane {first_plane} (unroll {i}) begin")
            self.add_newline()

            self.stencil_pipeline(
                CodeGenStatus(phase=Phase.Pipeline, plane=first_plane),
                extra_actions=[Action.UpdatePipelinePhaseLoopVariable] if i == 0 else [],
            )

            assert self.planes_per_iteration % 2 == 0
            self.shared_mem_read, self.shared_mem_write = (
                self.shared_mem_write,
                self.shared_mem_read,
            )

        self.add_code(
            "[B------:R-:W-:-:S05]",
            "@{p} BRA `({label})".format(p=self.pipeline_phase_loop_pred, label=label),
            "Go to the head of the pipeline phase loop",
        )

        self.add_label(".L_AFTER_PIPELINE")

    def gen_tail_phase(self):
        # Save the pointers so that we can restore them in each case of the tail phase.
        smem_ptrs = (self.shared_mem_read, self.shared_mem_write)

        for remain_pipeline_rows in range(self.planes_per_iteration, 0, -1):
            remain_pipeline_rows -= 1

            # Clear the candidate replacements and replacements.
            self.candidate_replacements.clear()
            self.replacements.clear()

            self.add_label(f".L_TAIL_PHASE_CASE_{remain_pipeline_rows}")
            self.add_section_title(f"Tail Phase Case {remain_pipeline_rows} begin")

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
                        # Note that we must add `self.radius` here since the plane index in the tail
                        # phase starts from `-self.radius`.
                        runtime_offset=self.pipeline_phase_plane_start
                        + remain_pipeline_rows
                        - plane_start,
                    )
                )

                self.shared_mem_read, self.shared_mem_write = (
                    self.shared_mem_write,
                    self.shared_mem_read,
                )

            self.add_code("[B------:R-:W-:-:S01]", "EXIT")
            self.add_newline()

            self.shared_mem_read, self.shared_mem_write = smem_ptrs

    def gen_tail_phase_branch(self):
        required_preds = self.planes_per_iteration - 1

        for i in range(min(required_preds, len(self.bra_preds))):
            self.add_code(
                "[B------:R-:W-:-:S05]",
                "@P{p} BRA `(.L_TAIL_PHASE_CASE_{i})".format(p=i, i=i),
                "Remain pipeline rows = {}".format(i),
            )

        for start in range(len(self.bra_preds), required_preds, len(self.bra_preds)):
            pred_mask = (1 << len(self.bra_preds)) - 1
            self.add_code(
                "[B------:R-:W-:-:S01]",
                "R2P PR, {a}, {b:#x}".format(a=self.bra_pred_set, b=pred_mask),
                "Generate predicates.",
            )
            self.add_code(
                "[B------:R-:W-:Y:S12]",
                "SHF.R.U32.HI {d}, RZ, {b:#x}, {c}".format(
                    d=self.bra_pred_set, b=len(self.bra_preds), c=self.bra_pred_set
                ),
                "Shift `bra_pred_set`.",
            )

            count = min(required_preds - start, len(self.bra_preds))
            for i in range(count):
                self.add_code(
                    "[B------:R-:W-:-:S05]",
                    "@P{p} BRA `(.L_TAIL_PHASE_CASE_{i})".format(p=i, i=i + start),
                    "Remain pipeline rows = {}".format(i + start),
                )

    def generate(self):
        self.gen_prologue()
        self.add_newline()

        self.head_phase()
        self.add_newline()

        self.pipeline_phase()
        self.add_newline()

        self.gen_tail_phase_branch()
        self.add_newline()

        self.gen_tail_phase()


class StencilPipeline:
    def output_predicate(self, plane):
        halo = self.topology.stream_halo
        if self.current_status.phase == Phase.Tail and plane.index + self.radius >= -halo:
            return "P6"

        if getattr(self, "in_head", False) and plane.index + self.radius < halo:
            return "P6"

        return self.write_result

    def stencil_pipeline(self, status: CodeGenStatus, extra_actions: list[Action] | None = None):
        self.current_status = status

        if extra_actions is None:
            extra_actions = []

        # Merge the replacements that can be applied in the current phase.
        self.replacements.update(self.candidate_replacements.get(status, {}))

        # Dependency management.
        dep_manager = DependencyManager()

        # Construct planes for each time step.
        planes = [
            self.construct_planes(status, time_step) for time_step in range(self.time_block_size)
        ]

        # Generate the shared-memory load/store and computation instructions for each time step.
        operations: list[list[InstrGroup]] = [
            self.gen_smem_comp_operations(time_step_planes, dep_manager)
            for time_step_planes in planes
        ]

        # Handle the planes that are completed in the current iteration.
        #
        # In this step, we need to establish dependencies between time steps. Specifically, we need
        # to do the following two things:
        #
        # - If `planes[0]` of a certain time step has completed its computation, its result must be
        #   written to the next time step (except for the last time step). Therefore, we need to
        #   find the last computation that writes to this plane and modify its destination plane to
        #   the plane in the next time step.
        # - If other planes of a certain time step have completed their computation, the results of
        #   these planes cannot be written to the next time step yet, because the values in the
        #   destination register are still useful. We need to add them to `candidate_replacements`.
        #   When the destination registers are no longer used, we will replace all uses of the
        #   destination registers with the registers of the plane.
        for time_step, time_step_planes in enumerate(planes[:-1]):
            for index, plane in enumerate(time_step_planes):
                if not plane.finishes_current_iteration:
                    continue

                dst_plane = planes[time_step + 1][
                    (index + (self.elements_per_time_step - 1)) % len(planes[time_step + 1])
                ]

                if index == 0:
                    # The result of `planes[0]` should be output to the next time step. We first
                    # find out the last computation that writes to this plane.
                    #
                    # Note that we must enumerate each local element of the plane.
                    for local_index in self.local_indices():
                        local_plane = plane.to_local(local_index)
                        for computation in (
                            op
                            for group in reversed(operations[time_step])
                            for op in reversed(group.comp_instrs)
                        ):
                            if local_plane in computation.dst_planes:
                                assert len(computation.dst_planes) == 1
                                # Modify the destination plane to the corresponding plane in the next
                                # time step.
                                computation.dst_planes[0] = dst_plane.to_local(local_index)

                                computation.wait_scoreboard.add(self.gmem_store_read_scoreboard())
                                break

                else:
                    # The result of other planes should not be output to the next time step yet, so
                    # we add them to `candidate_replacements`.
                    #
                    # The meaning of this record is: from the moment when `status` and `key` are
                    # consistent, all `dst_plane.regs` will be replaced with `plane.regs`.
                    key = CodeGenStatus(status.phase, status.plane + index, status.runtime_offset)
                    self.candidate_replacements.setdefault(key, {})[
                        self.replacement_key(dst_plane.regs)
                    ] = plane.regs

        # `planes[-1]` of time step 0 comes from global memory, so we need to add a scoreboard
        # dependency for the first instruction that uses `planes[-1]`.
        gmem_plane = planes[0][-1]
        for computation in (op for group in operations[0] for op in group.comp_instrs):
            if any(local_plane.plane is gmem_plane for local_plane in computation.src_planes):
                computation.wait_scoreboard.add(self.gmem_load_write_scoreboard(gmem_plane.index))
                break

        self.schedule_global_memory(status, planes, dep_manager, operations)

        if Action.UpdatePipelinePhaseLoopVariable in extra_actions:
            assert status.phase == Phase.Pipeline

            # Number of groups in each time step.
            num_groups_per_time_step = len(operations[0])
            total_groups = sum(len(time_step_groups) for time_step_groups in operations)

            extra_operations: list[UpdatePipelinePhaseLoopVariable] = [
                UpdatePipelinePhaseLoopVariable(
                    target=self.pipeline_phase_iterations,
                    stage=0,
                    comment="Decrement the pipeline phase loop variable.",
                ),
                UpdatePipelinePhaseLoopVariable(
                    target=self.pipeline_phase_iterations,
                    stage=1,
                    comment="Check whether the pipeline phase loop variable is not zero.",
                ),
                UpdatePipelinePhaseLoopVariable(
                    target=self.pipeline_phase_loop_pred,
                    stage=2,
                    comment="Generate the pipeline phase loop predicate.",
                ),
            ]

            for i, op in enumerate(extra_operations):
                if i >= total_groups:
                    # If the number of extra operations exceeds the total number of groups, we will
                    # insert the remaining operations into the last group of the last time step.
                    operations[-1][-1].comp_instrs.append(op)
                    continue

                time_step = i // num_groups_per_time_step
                group = i % num_groups_per_time_step

                if group == num_groups_per_time_step - 1:
                    operations[time_step][group].comp_instrs.insert(0, op)
                else:
                    operations[time_step][group].comp_instrs.append(op)

            dep_manager.add_dependency(
                producer=extra_operations[0], consumer=extra_operations[1], stall=4
            )
            dep_manager.add_dependency(
                producer=extra_operations[1], consumer=extra_operations[2], stall=6
            )

        # Gather all operations in the current pipeline.
        flatten_operations: list[SASSOperation] = [
            op
            for time_step_groups in operations
            for group in time_step_groups
            for op in group.load_instrs + group.comp_instrs
        ]

        if not flatten_operations:
            # No operation needs to be performed in the current pipeline.
            return

        # Build dependencies for all operations in the current pipeline.
        dep_manager.add_dependencies(flatten_operations, self.stall_dict)

        # Set the registers of `StencilComputation` based on `src_planes` and `dst_planes`.
        for op in flatten_operations:
            if isinstance(op, StencilComputation):
                op.set_regs()

        # Add `.reuse` flag to stencil computations.
        computations = [op for op in flatten_operations if isinstance(op, StencilComputation)]
        for op1, op2 in pairwise(computations):
            if op1.src_regs[0] == op2.src_regs[0] and op1.dst_reg != op1.src_regs[0]:
                op1.src_regs[0] += ".reuse"

            if (
                len(op1.src_regs) > 1
                and len(op2.src_regs) > 1
                and op1.src_regs[1] == op2.src_regs[1]
                and op1.dst_reg != op1.src_regs[1]
            ):
                op1.src_regs[1] += ".reuse"

        dep_manager.compute_min_stalls(flatten_operations, per_op_min_stall=self.min_comp_stall)

        for time_step, time_step_groups in enumerate(operations):
            if all(not (group.load_instrs or group.comp_instrs) for group in time_step_groups):
                # No operation needs to be performed in the current time step.
                continue

            self.add_section_title(
                f"Stencil Computation\nPhase: {status.phase}\nTime Step {time_step}"
            )

            for group in time_step_groups:
                for op in group.load_instrs + group.comp_instrs:
                    self.emit_operation(op)

        self.add_barrier()


class Rank2DPipeline(StencilPipeline):
    def output_address_registers(self, planes, count: int) -> list[str]:
        # Borrow only registers whose values no longer contribute to this iteration.
        if self.output_address_layout == OutputAddressLayout.ReverseTail:
            assert count <= self.radius
            slots = [-(index + 2) for index in range(count - 1)]
        else:
            assert count * 2 - 1 <= self.elements_per_time_step
            slots = list(range(count, 2 * count - 1))

        return [planes[slot].regs[0] for slot in slots] + ([self.output] if count else [])

    def local_indices(self):
        return range(self.group_size)

    @staticmethod
    def replacement_key(regs):
        return regs

    def schedule_global_memory(self, status, planes, dep_manager, operations):
        # Construct and insert global memory load instructions into time step 0.
        #
        # We need to construct a new `Plane` object that represents the new row to be loaded from
        # global memory. This row will be loaded into the register of `planes[0][0]`, so we
        # construct this object from `planes[0][0]` and modify its `index` attribute.
        #
        # We need to skip loading planes 0 and 1 in the head phase, as their load instructions have
        # already been issued in the prologue.
        gmem_plane = planes[0][-1]
        new_plane = replace(planes[0][0], index=gmem_plane.index + 1)
        slot = self.precomputed_addr_slot(new_plane.index)

        # If `slot` is not `None`, place the newly loaded value into `slot` This avoids overwriting
        # the value in `planes[0][0].regs`.
        if slot is not None:
            new_plane.regs = slot

        if plane_in_range(new_plane.index, status.phase) and (
            status.phase != Phase.Head or new_plane.index > 1
        ):
            # The next plane is in range, so we need to load it from global memory.
            gmem_loads = self.gen_gmem_load_operations(
                ptr_reg=self.input if slot is None else slot[-1], plane=new_plane
            )

            if slot is None:
                # We need to update the global memory pointer before loading the next plane.
                update_ptr = UpdatePtr(
                    src_reg=self.input,
                    dst_reg=self.input,
                    stride_reg=self.global_mem_stride,
                    offset=(
                        new_plane.index if new_plane.index == self.precomputed_addr_slot_size else 1
                    )
                    * self.group_size,
                    wait_scoreboard=set()
                    if new_plane.index == self.precomputed_addr_slot_size
                    else {self.gmem_load_read_scoreboard(new_plane.index - 1)},
                    comment="Update the global memory pointer.",
                )
                dep_manager.add_dependency(producer=update_ptr, consumer=gmem_loads[0], stall=5)

                # Insert the `UpdatePtr` instruction before the first computation of time step 0.
                operations[0][0].comp_instrs.insert(0, update_ptr)

            if slot is None:
                # Insert all LDG instructions to the end of the first time step.
                operations[0][-1].comp_instrs.extend(gmem_loads)
            else:
                # For the head phase, we need to insert the LDG instructions into the first group of
                # time step 0.
                operations[0][0].load_instrs[:0] = gmem_loads

        # Find all planes in the last time step that are marked as finished and involved in
        # computations.
        finished_planes = [plane for plane in planes[-1] if plane.finishes_current_iteration]
        output_addresses = self.output_address_registers(planes[-1], len(finished_planes))

        if finished_planes:
            update_ptrs = []
            global_stores = []

            for index, plane in enumerate(finished_planes):
                addr_reg = output_addresses[index]
                global_store = self.gen_gmem_store_operations(ptr_reg=addr_reg, plane=plane)

                halo = self.topology.stream_halo
                if (status.phase == Phase.Tail and plane.index + self.radius >= -halo) or (
                    getattr(self, "in_head", False) and plane.index + self.radius < halo
                ):
                    for store in global_store:
                        store.pred = "P6"

                global_stores.extend(global_store)

                if status.phase != Phase.Head or plane.index != -self.radius:
                    wait_scoreboard = (
                        {self.gmem_store_read_scoreboard()} if addr_reg == self.output else set()
                    )
                    update_ptr = UpdatePtr(
                        src_reg=self.output,
                        dst_reg=addr_reg,
                        stride_reg=self.global_mem_stride,
                        offset=(index + 1) * self.group_size,
                        wait_scoreboard=wait_scoreboard,
                        comment="Update the global memory pointer.",
                    )
                    update_ptrs.append(update_ptr)
                    dep_manager.add_dependency(
                        producer=update_ptr, consumer=global_store[0], stall=5
                    )

            if update_ptrs:
                # A pointer temporary may borrow a data register from another finished plane. Insert
                # each borrowed update after the final physical-register read of its destination.
                # Iterate in reverse so front insertions keep their original order.
                for update_ptr in reversed(update_ptrs[:-1]):
                    for group in reversed(operations[-1]):
                        for op_index in range(len(group.comp_instrs) - 1, -1, -1):
                            operation = group.comp_instrs[op_index]
                            if any(src.reg() == update_ptr.dst_reg for src in operation.src_planes):
                                group.comp_instrs.insert(op_index + 1, update_ptr)
                                break
                        else:
                            continue
                        break
                    else:
                        operations[-1][-1].comp_instrs.insert(0, update_ptr)

                # Advance the real output pointer only after all derived addresses exist.
                operations[-1][-1].comp_instrs.append(update_ptrs[-1])

            operations[-1][-1].comp_instrs.extend(global_stores)


class Rank3DPipeline(StencilPipeline):
    def local_indices(self):
        return self.thread_local_indices

    @staticmethod
    def replacement_key(regs):
        return id(regs)

    def schedule_global_memory(self, status, planes, dep_manager, operations):
        new_plane = replace(planes[0][0], index=planes[0][-1].index + 1)
        if plane_in_range(new_plane.index, status.phase) and (
            status.phase != Phase.Head or new_plane.index > 1
        ):
            self.gen_and_schedule_gmem_load_operations(new_plane, dep_manager, operations[0])

        self.gen_and_schedule_gmem_store_operations(status, planes[-1], dep_manager, operations[-1])
