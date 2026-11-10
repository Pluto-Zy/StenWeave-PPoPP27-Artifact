from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUDAOperation:
    pred: str | None = None
    comment: str | None = None


@dataclass
class StencilComputation(CUDAOperation):
    src_regs: list[str] = field(default_factory=list)
    dst_reg: str = ""
    coeff_index: int = 0

    def to_cuda(self):
        assert len(self.src_regs) in (1, 2)

        if len(self.src_regs) == 1:
            return f"{self.dst_reg} = {self.src_regs[0]} * neighbor_coeffs[{self.coeff_index}];"
        else:
            return f"{self.dst_reg} = {self.src_regs[0]} * neighbor_coeffs[{self.coeff_index}] + {self.src_regs[1]};"


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
    ptr_reg: str = ""

    def to_cuda(self):
        return f"{self.dst_reg} = *{self.ptr_reg};"


@dataclass
class GlobalMemoryStore(CUDAOperation):
    src_reg: str = ""
    ptr_reg: str = ""

    def to_cuda(self):
        return f"*{self.ptr_reg} = {self.src_reg};"


@dataclass
class UpdatePtr(CUDAOperation):
    src_reg: str = ""
    dst_reg: str = ""
    stride_reg: str = ""
    offset: int = 1

    pointer_bits: bool = False

    def to_cuda(self):
        if self.pointer_bits:
            return f"{self.dst_reg} = bit_cast<double>({self.stride_reg} * {self.offset} + {self.src_reg});"
        else:
            return f"{self.dst_reg} = {self.stride_reg} * {self.offset} + {self.src_reg};"
