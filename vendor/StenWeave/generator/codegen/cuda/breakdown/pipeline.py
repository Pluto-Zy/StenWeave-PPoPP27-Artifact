"""Shared source emission for the historical staged breakdown experiment."""

from __future__ import annotations

from dataclasses import dataclass

from codegen.state import Phase


@dataclass
class CodeGenConfig:
    comm_centric_smem_scheduling: bool = True
    layered_point_update: bool = True
    segmented_smem: bool = True
    gather_row_contributions: bool = False
    gather_update: bool = False


class BreakdownPipeline:
    def add_code(self, code: str, comment=None):
        code = code.strip()

        if len(code) > 0 and code[-1] != ";":
            code += ";"

        if comment is not None:
            result = f"{' ' * self.code_indent}{code:<60}    // {comment}"
        else:
            result = f"{' ' * self.code_indent}{code}"

        self.codes.append(result)

    def add_comment(self, comment):
        self.codes.append(f"    // {comment}")

    def add_section_title(self, title):
        self.codes.append(f"    //{'=' * 92}//")

        for line in title.splitlines():
            self.codes.append(f"    // {line:<90} //")

        self.codes.append(f"    //{'=' * 92}//")

    def add_label(self, label):
        self.codes.append(f"{label}:")

    def add_newline(self):
        self.codes.append("")

    def add_barrier(self):
        # = IMPORTANT ==============================================================================
        # The latency of this instruction will affect the correctness of the code.
        # ==========================================================================================
        self.add_code("__syncthreads()")

    def gen_wait_scoreboard(*scoreboards: int) -> str:
        return "".join(f"{i}" if i in scoreboards else "-" for i in range(6))

    def plane_in_range(plane: int, phase: Phase) -> bool:
        """
        This function determines whether the value of the given plane is within the valid range.
        """

        if phase == Phase.Head:
            return plane >= 0

        if phase == Phase.Tail:
            return plane < 0

        return True

    def gen_smem_addr_offset(self, time_step: int, thread_offset: int):
        if self.codegen_config.segmented_smem:
            return time_step + thread_offset * self.smem_chunk_size
        else:
            return f"({time_step} * int(blockDim.x + {self.radius}) + int(threadIdx.x) + {thread_offset})"

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

        return BreakdownPipeline.wrap(lhs + rhs, range)

    def wrap_sub(lhs: int, rhs: int, range: int) -> int:
        assert range > 0, "Range must be positive."

        return BreakdownPipeline.wrap(lhs - rhs, range)
