from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUDAOperation:
    stall: int = 1
    wait_scoreboard: list[int] = field(default_factory=list)
    write_scoreboard: int | None = None
    read_scoreboard: int | None = None
    pred: str | None = None
    comment: str | None = None


@dataclass
class StencilComputation(CUDAOperation):
    src_regs: list[str] = field(default_factory=list)
    dst_reg: str = ""
    coeff_addr: int = 0

    def to_cuda(self):
        assert len(self.src_regs) in (1, 2)

        if len(self.src_regs) == 1:
            return f"{self.dst_reg} = {self.src_regs[0]} * neighbor_coeffs[{self.coeff_addr // 8}];"
        else:
            return f"{self.dst_reg} = {self.src_regs[0]} * neighbor_coeffs[{self.coeff_addr // 8}] + {self.src_regs[1]};"


@dataclass
class SharedMemoryLoad(CUDAOperation):
    dst_reg: str = ""
    addr_base: str = ""
    addr_offset: int = 0

    def to_cuda(self):
        return f"{self.dst_reg} = {self.addr_base}[{self.addr_offset}];"


@dataclass
class SharedMemoryStore(CUDAOperation):
    src_reg: str = ""
    addr_base: str = ""
    addr_offset: int = 0

    def to_cuda(self):
        return f"{self.addr_base}[{self.addr_offset}] = {self.src_reg};"


@dataclass
class GlobalMemoryLoad(CUDAOperation):
    dst_reg: str = ""
    desc_reg: str = ""
    ptr_reg: str = ""

    def to_cuda(self):
        return f"{self.dst_reg} = *{self.ptr_reg};"


@dataclass
class GlobalMemoryStore(CUDAOperation):
    src_reg: str = ""
    desc_reg: str = ""
    ptr_reg: str = ""

    def to_cuda(self):
        return f"*{self.ptr_reg} = {self.src_reg};"


@dataclass
class UpdatePtr(CUDAOperation):
    src_reg: str = ""
    dst_reg: str = ""
    stride_reg: str = ""
    offset: int = 1

    def to_cuda(self):
        if self.dst_reg.startswith("reg"):
            return f"{self.dst_reg} = bit_cast<double>({self.stride_reg} * {self.offset} + {self.src_reg});"
        else:
            return f"{self.dst_reg} = {self.stride_reg} * {self.offset} + {self.src_reg};"


@dataclass
class UpdatePipelinePhaseLoopVariable(CUDAOperation):
    target: str = ""
    stage: int = 0

    def to_cuda(self):
        return ""


@dataclass
class Dependency:
    producer: CUDAOperation
    consumer: CUDAOperation
    required_stall: int


@dataclass
class SyncThreads(CUDAOperation):
    def to_cuda(self):
        return "__syncthreads();"
