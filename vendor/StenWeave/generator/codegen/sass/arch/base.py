"""Small architecture policy values shared by SASS prologue builders.

The builders manually interleave independent setup work. This module
keeps ABI and latency facts explicit without introducing a new scheduler or a
large inheritance hierarchy around the emitters.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitecturePolicy:
    name: str
    fp64_latency: int
    global_mem_desc_offset: int
    input_arg_offset: int
    output_arg_offset: int
    uniform_arg_offset: int

    def uniform_parameter(self, name: str) -> str:
        names = (
            "row_pitch",
            "plane_stride",
            "cta_stride_x",
            "cta_stride_y",
            "write_begin_x",
            "write_end_x",
            "write_begin_y",
            "write_end_y",
            "iterations_arg",
            "remainder_arg",
            "store_head",
            "store_tail",
        )
        return f"c[0x0][{self.uniform_arg_offset + 4 * names.index(name):#x}]"

    @property
    def is_sm90(self) -> bool:
        return self.name == "sm90"


SM80 = ArchitecturePolicy("sm80", 6, 0x118, 0x160, 0x168, 0x170)
SM90 = ArchitecturePolicy("sm90", 8, 0x208, 0x210, 0x218, 0x220)
