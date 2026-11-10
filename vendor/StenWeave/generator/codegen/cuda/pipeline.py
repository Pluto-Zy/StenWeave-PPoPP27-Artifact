"""Shared CUDA source emission, temporal loops and operation bookkeeping."""

from __future__ import annotations

from codegen.cuda.operations import (
    GlobalMemoryLoad,
    GlobalMemoryStore,
    StencilComputation,
    UpdatePtr,
)
from codegen.logical import plane_in_range
from codegen.state import CodeGenStatus, Phase


class SourcePipeline:
    def add_code(self, code: str, comment=None):
        code = code.strip()
        if len(code) > 0 and code[-1] != ";":
            code += ";"

        if comment is not None:
            result = f"{' ' * self.code_indent}{code:<60}    // {comment}"
        else:
            result = f"{' ' * self.code_indent}{code}"

        self.codes.append(result)

    def add_section_title(self, title):
        self.codes.append(f"    //{'=' * 92}//")

        for line in title.splitlines():
            self.codes.append(f"    // {line:<90} //")

        self.codes.append(f"    //{'=' * 92}//")

    def add_newline(self):
        self.codes.append("")

    def add_barrier(self):
        self.add_code("__syncthreads()")

    def generate(self):
        self.gen_prologue()
        self.add_newline()

        self.head_phase()
        self.add_newline()

        self.pipeline_phase()
        self.add_newline()

        self.gen_tail_phase()

    def wrap(value: int, range: int) -> int:
        value %= range

        if value < 0:
            return value + range
        else:
            return value

    def wrap_add(lhs: int, rhs: int, range: int) -> int:
        assert range > 0, "Range must be positive."

        return SourcePipeline.wrap(lhs + rhs, range)

    def head_phase(self):
        for plane in range(self.topology.head_end):
            self.add_newline()
            self.add_section_title(f"Head phase plane {plane} begin")
            self.add_newline()
            self.stencil_pipeline(CodeGenStatus(phase=Phase.Head, plane=plane))
            self.publication_buffers = self.publication_buffers[::-1]

    def pipeline_phase(self):
        self.add_section_title("Pipeline Phase")
        self.add_code(
            f"while ({self.pipeline_phase_iterations}--) {{", "Begin of the pipeline phase loop."
        )
        self.code_indent += 4
        self.pipeline_phase_plane_start = self.topology.head_end

        for i in range(self.planes_per_iteration):
            first_plane = self.pipeline_phase_plane_start + i
            self.add_newline()
            self.add_section_title(f"Pipeline phase plane {first_plane} (unroll {i}) begin")
            self.add_newline()
            self.stencil_pipeline(
                CodeGenStatus(phase=Phase.Pipeline, plane=first_plane),
            )
            assert self.planes_per_iteration % 2 == 0
            self.publication_buffers = self.publication_buffers[::-1]

        self.code_indent -= 4
        self.add_code("}", "End of the pipeline phase loop.")

    def gen_tail_phase(self):
        smem_ptrs = self.publication_buffers
        self.add_code(f"switch ({self.remain_rows}) {{", "Begin of tail phase switch.")

        for remain_pipeline_rows in range(self.planes_per_iteration, 0, -1):
            remain_pipeline_rows -= 1
            self.candidate_replacements.clear()
            self.replacements.clear()
            self.add_code(f"case {remain_pipeline_rows}: {{")
            self.code_indent += 4

            for i in range(remain_pipeline_rows):
                first_plane = self.pipeline_phase_plane_start + i
                self.add_newline()

                self.add_section_title(
                    f"Remaining Pipeline phase plane {first_plane} (unroll {i}) begin"
                )
                self.add_newline()
                self.stencil_pipeline(CodeGenStatus(phase=Phase.Pipeline, plane=first_plane))
                self.publication_buffers = self.publication_buffers[::-1]

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
                        runtime_offset=self.pipeline_phase_plane_start
                        + remain_pipeline_rows
                        - self.topology.tail_start,
                    )
                )
                self.publication_buffers = self.publication_buffers[::-1]

            self.add_code("return", "EXIT")
            self.add_newline()

            self.code_indent -= 4
            self.add_code("}", f"End of case {remain_pipeline_rows}.")
            self.publication_buffers = smem_ptrs

        self.add_code("}", "End of tail phase switch.")
        self.add_code("__builtin_unreachable()")

    def stencil_pipeline(self, status: CodeGenStatus):
        self.replacements.update(self.candidate_replacements.get(status, {}))

        planes = [self.construct_planes(status, step) for step in range(self.time_block_size)]
        groups = self.computation_groups(planes, status)

        self.load_next_plane(planes, groups, status)
        self.store_finished_planes(planes, groups, status)
        self.render_groups(groups, status)

    def render_groups(self, groups, status: CodeGenStatus):
        if not any(group.operations for step in groups for group in step):
            return

        for time_step, step in enumerate(groups):
            if not any(group.operations for group in step):
                continue

            self.add_section_title(
                f"Stencil Computation\nPhase: {status.phase}\nTime Step {time_step}"
            )
            for group in step:
                for op in group.operations:
                    if op.pred is None:
                        self.add_code(op.to_cuda(), op.comment)
                    else:
                        self.add_code(f"if ({op.pred}) {{ {op.to_cuda()} }}")

        self.add_barrier()

    @staticmethod
    def insert_before_computation(operations, inserted):
        for i, op in enumerate(operations):
            if isinstance(op, StencilComputation):
                operations[i:i] = inserted

                return

        operations.extend(inserted)

    def load_next_plane(self, planes, groups, status: CodeGenStatus):
        index = planes[0][-1].index + 1
        if not plane_in_range(index, status.phase):
            return

        slot = self.precomputed_addr_slot(index)
        if slot is None:
            advance = UpdatePtr(
                src_reg=self.input,
                dst_reg=self.input,
                stride_reg=self.global_mem_stride,
                offset=index if index == self.precomputed_addr_slot_size else 1,
                comment="Update the global memory pointer.",
            )
            load = GlobalMemoryLoad(
                dst_reg=planes[0][0].src_reg,
                ptr_reg=self.input,
                comment="Load a row from global memory.",
            )

            pointer_index, load_index = self.global_load_positions
            if pointer_index is None:
                self.insert_before_computation(groups[0][0].operations, [advance])
            else:
                groups[0][0].operations.insert(pointer_index, advance)

            groups[0][1].operations.insert(load_index, load)
        elif status.phase != Phase.Head or index > 1:
            groups[0][0].operations.insert(
                0,
                GlobalMemoryLoad(
                    dst_reg=slot,
                    ptr_reg=f"bit_cast<double *>({slot})",
                    comment="Load a row from global memory.",
                ),
            )

    def store_finished_planes(self, planes, groups, status: CodeGenStatus):
        finished = self.finished_outputs(planes[-1])
        advances, stores = [], []

        for index, plane in enumerate(finished):
            address = (
                self.output
                if index == len(finished) - 1
                else self.borrow_output_address(planes[-1], index, len(finished))
            )
            stores.append(
                GlobalMemoryStore(
                    src_reg=plane.dst_reg,
                    ptr_reg=f"bit_cast<double *>({address})" if address != self.output else address,
                    pred=self.write_result,
                )
            )

            if status.phase != Phase.Head or plane.index != -self.radius:
                advances.append(
                    UpdatePtr(
                        src_reg=self.output,
                        dst_reg=address,
                        pointer_bits=address != self.output,
                        stride_reg=self.global_mem_stride,
                        offset=index + 1,
                        comment="Update the global memory pointer.",
                    )
                )

        target = groups[-1][-1].operations
        if self.global_store_pointer_stride is None:
            self.insert_before_computation(target, advances[:1])
            target.extend(advances[1:])
        else:
            for index, op in enumerate(advances):
                target.insert(index * self.global_store_pointer_stride, op)

        target.extend(stores)


class VolumePipeline(SourcePipeline):
    def generate_reg_declarations(self) -> str:
        name_type_pairs = [
            (self.global_mem_stride, "unsigned"),
            (self.write_result, "bool"),
            (self.pipeline_phase_iterations, "unsigned"),
            (self.remain_rows, "unsigned"),
            *((reg, "double") for time_step in self.regs for reg in time_step),
            *((reg, "double *") for reg in self.shared_mem_reads),
            *((reg, "double *") for reg in self.shared_mem_writes),
            *((reg, "double") for reg in self.neighbors),
        ]
        declarations = (f"{type_} {name};" for (name, type_) in name_type_pairs)
        return "\n    ".join(declarations)
