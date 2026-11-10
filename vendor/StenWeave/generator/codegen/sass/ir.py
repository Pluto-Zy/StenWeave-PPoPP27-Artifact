from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class RegisterPlane(Protocol):
    def reg(self) -> str: ...


@dataclass
class SASSOperation:
    stall: int = 0
    wait_scoreboard: set[int] = field(default_factory=set)
    write_scoreboard: int | None = None
    read_scoreboard: int | None = None
    pred: str | None = None
    comment: str | None = None

    src_planes: list[RegisterPlane] = field(default_factory=list)
    dst_planes: list[RegisterPlane] = field(default_factory=list)

    def to_control_codes(self) -> str:
        wait = "".join(f"{i}" if i in self.wait_scoreboard else "-" for i in range(6))
        write = f"{self.write_scoreboard}" if self.write_scoreboard is not None else "-"
        read = f"{self.read_scoreboard}" if self.read_scoreboard is not None else "-"
        yield_bit = "Y" if self.stall > 2 else "-"

        return f"[B{wait}:R{read}:W{write}:{yield_bit}:S{self.stall:02}]"


@dataclass
class StencilComputation(SASSOperation):
    src_regs: list[str] = field(default_factory=list)
    dst_reg: str = ""
    coeff_addr: int = 0

    def set_regs(self):
        assert len(self.src_planes) in (1, 2)
        assert len(self.dst_planes) == 1

        self.src_regs = [plane.reg() for plane in self.src_planes]
        self.dst_reg = self.dst_planes[0].reg()

    def to_sass(self):
        assert len(self.src_regs) in (1, 2)

        if len(self.src_regs) == 1:
            return "DMUL {d}, {a}, c[0x3][{b:#x}]".format(
                d=self.dst_reg, a=self.src_regs[0], b=self.coeff_addr
            )
        else:
            return "DFMA {d}, {a}, c[0x3][{b:#x}], {c}".format(
                d=self.dst_reg, a=self.src_regs[0], b=self.coeff_addr, c=self.src_regs[1]
            )


@dataclass
class SharedMemoryLoad(SASSOperation):
    addr_base: str = ""
    addr_offset: int = 0

    def to_sass(self):
        assert len(self.dst_planes) == 1
        dst_reg = self.dst_planes[0].reg()
        return f"LDS.64 {dst_reg}, [{self.addr_base}+{self.addr_offset * 8:#x}]"


@dataclass
class SharedMemoryStore(SASSOperation):
    addr_base: str = ""
    addr_offset: int = 0

    def to_sass(self):
        assert len(self.src_planes) == 1
        src_reg = self.src_planes[0].reg()
        return f"STS.64 [{self.addr_base}+{self.addr_offset * 8:#x}], {src_reg}"


@dataclass
class GlobalMemoryLoad(SASSOperation):
    desc_reg: str = ""
    ptr_reg: str = ""
    ptr_offset: int = 0
    size: int = 64

    def to_sass(self):
        assert len(self.dst_planes) in (1, self.size // 64)
        dst_reg = self.dst_planes[0].reg()
        return f"LDG.E.{self.size} {dst_reg}, desc[{self.desc_reg}][{self.ptr_reg}.64+{self.ptr_offset * 8:#x}]"


@dataclass
class GlobalMemoryStore(SASSOperation):
    desc_reg: str = ""
    ptr_reg: str = ""
    ptr_offset: int = 0
    size: int = 64

    def to_sass(self):
        assert len(self.src_planes) in (1, self.size // 64)
        src_reg = self.src_planes[0].reg()
        return f"STG.E.{self.size} desc[{self.desc_reg}][{self.ptr_reg}.64+{self.ptr_offset * 8:#x}], {src_reg}"


@dataclass
class UpdatePtr(SASSOperation):
    src_reg: str = ""
    dst_reg: str = ""
    stride_reg: str = ""
    offset: int = 1

    def to_sass(self):
        return "IMAD.WIDE.U32 {d}, {a}, {b:#x}, {c}".format(
            d=self.dst_reg, a=self.stride_reg, b=self.offset * 8, c=self.src_reg
        )


@dataclass
class UpdatePipelinePhaseLoopVariable(SASSOperation):
    target: str = ""
    stage: int = 0

    def to_sass(self):
        return {
            0: "UIADD3 {d}, {a}, -0x1, URZ".format(d=self.target, a=self.target),
            1: "UISETP.NE.U32.AND UP0, UPT, {}, URZ, UPT".format(self.target),
            2: "PLOP3.LUT {}, PT, PT, PT, UP0, 0x80, 0x0".format(self.target),
        }[self.stage]


@dataclass
class InstrGroup:
    """
    Represents a group of instructions, which is led by several shared memory access instructions,
    followed by several other computation instructions.
    """

    load_instrs: list[SharedMemoryLoad] = field(default_factory=list)
    comp_instrs: list[SASSOperation] = field(default_factory=list)
