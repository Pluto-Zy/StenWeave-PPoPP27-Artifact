import importlib
import re
import string
from dataclasses import dataclass


@dataclass
class ControlCodes:
    wait_bar: tuple[int, ...]
    read_bar: int
    write_bar: int
    yield_bit: bool
    stall_cycles: int

    def to_args(self, args: dict):
        args["bar"] = self.wait_bar
        args["rb"] = self.read_bar
        args["wb"] = self.write_bar
        args["yb"] = self.yield_bit
        args["stall"] = self.stall_cycles

    @classmethod
    def from_string(cls, code_str: str) -> "ControlCodes":
        wait_bar = ()
        read_bar = None
        write_bar = None
        yield_bit = False
        stall_cycles = 0

        for part in code_str.strip("[]").split(":"):
            part = part.strip()
            assert len(part) > 0

            name = part[0]
            part = part[1:]

            if name == "B":
                wait_bar = tuple(i for i, ch in enumerate(part) if ch != "-")
            elif name == "R":
                read_bar = int(part) if part != "-" else 7
            elif name == "W":
                write_bar = int(part) if part != "-" else 7
            elif name == "Y":
                yield_bit = True
            elif name == "S":
                stall_cycles = int(part)
            elif name == "-":
                yield_bit = False
            else:
                raise ValueError(f"Unknown control code: {name}")

        return cls(
            wait_bar=wait_bar,
            read_bar=read_bar,
            write_bar=write_bar,
            yield_bit=yield_bit,
            stall_cycles=stall_cycles,
        )


@dataclass
class SASSOperand:
    raw_content: str
    negated: bool
    absoluted: bool
    notted: bool
    inverted: bool

    def to_args(self, args: dict, name: str, mod_list: list[str] | None = None):
        if mod_list is None:
            mod_list = []

        for attr_name, mod_name in zip(
            ["negated", "absoluted", "notted", "inverted"],
            ["negate", "absolute", "not", "invert"],
            strict=False,
        ):
            if mod_name in mod_list:
                args[f"{name}_{mod_name}"] = getattr(self, attr_name)


@dataclass
class RegisterOperand(SASSOperand):
    index: str
    reuse: bool

    def reg_name(self) -> str:
        return f"R{self.index}"

    def to_args(self, a, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"R{name}", mod_list)
        args[f"R{name}"] = getattr(a.Register, self.reg_name())

        if mod_list is not None and "reuse" in mod_list:
            args[f"reuse_src_{name}"] = self.reuse


@dataclass
class UniformRegisterOperand(SASSOperand):
    index: str

    def reg_name(self) -> str:
        return f"UR{self.index}"

    def to_args(self, a, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"UR{name}", mod_list)
        args[f"UR{name}"] = getattr(a.UniformRegister, self.reg_name())


@dataclass
class PredicateRegisterOperand(SASSOperand):
    index: str

    def reg_name(self) -> str:
        return f"P{self.index}"

    def to_args(self, a, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"P{name}", mod_list)
        args[f"P{name}"] = getattr(a.Predicate, self.reg_name())


@dataclass
class UniformPredicateRegisterOperand(SASSOperand):
    index: str

    def reg_name(self) -> str:
        return f"UP{self.index}"

    def to_args(self, a, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"UP{name}", mod_list)
        args[f"UP{name}"] = getattr(a.UniformPredicate, self.reg_name())


@dataclass
class ImmmediateOperand(SASSOperand):
    value: int

    def to_args(self, _, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"S{name}", mod_list)
        args[f"S{name}"] = self.value


@dataclass
class ConstantAddressOperand(SASSOperand):
    bank: int
    addr: int

    def to_args(self, _, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"S{name}", mod_list)
        args[f"S{name}_bank"] = self.bank
        args[f"S{name}_addr"] = self.addr


@dataclass
class AddressOperand(SASSOperand):
    base_reg: str
    offset: int

    def to_args(self, a, args: dict, name: str, mod_list: list[str] | None = None):
        super().to_args(args, f"R{name}", mod_list)
        args[f"R{name}"] = getattr(a.Register, self.base_reg)
        args[f"R{name}_offset"] = self.offset


@dataclass
class DescriptorAddressOperand(SASSOperand):
    desc: str
    addr_reg: str
    offset: int = 0

    def to_args(self, a, args: dict, name: str, name2: str, mod_list: list[str] | None = None):
        super().to_args(args, f"R{name}", mod_list)
        args[f"R{name}"] = getattr(a.Register, self.addr_reg)
        args[f"R{name}_UR{name2}"] = getattr(a.UniformRegister, self.desc)
        args[f"R{name}_offset"] = self.offset


@dataclass
class LabelOperand(SASSOperand):
    label: str
    addr: int


@dataclass
class UnknownOperand(SASSOperand):
    pass


@dataclass
class SASSInstruction:
    raw_instr: str
    address: int
    predicate: SASSOperand
    opcode: str
    operands: list[SASSOperand]
    binary: int
    control_codes: ControlCodes
    label: str | None

    def patch_uniform_pred(args: dict):
        if "Pg" in args:
            args["UPg"] = args.pop("Pg")

        if "Pg_not" in args:
            args["UPg_not"] = args.pop("Pg_not")

    def encode_mov(self, a, args: dict) -> int:
        self.operands[0].to_args(a, args, "d")
        self.operands[1].to_args(a, args, "b")

        if isinstance(self.operands[1], ConstantAddressOperand):
            instr_class = a.mov__RC
        elif isinstance(self.operands[1], RegisterOperand):
            instr_class = a.mov__RR
        else:
            assert isinstance(self.operands[1], ImmmediateOperand)
            instr_class = a.mov__RI

        return instr_class(**args).encode()

    def encode_s2r(self, a, args: dict) -> int:
        self.operands[0].to_args(a, args, "d")
        # Replace SR_TID.X with SR_TID_X
        args["SRa"] = getattr(a.SpecialRegister, self.operands[1].raw_content.replace(".", "_"))

        return a.s2r_(**args).encode()

    def encode_s2ur(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)
        self.operands[0].to_args(a, args, "d")
        # Replace SR_TID.X with SR_TID_X
        args["SRa"] = getattr(a.SpecialRegister, self.operands[1].raw_content.replace(".", "_"))

        return a.s2ur_(**args).encode()

    def encode_imad(self, a, args: dict) -> int:
        operand_desc = ""
        for operand in self.operands[1:]:
            if isinstance(operand, RegisterOperand):
                operand_desc += "R"
            elif isinstance(operand, UniformRegisterOperand):
                operand_desc += "U"
            elif isinstance(operand, ConstantAddressOperand):
                operand_desc += "C"
            else:
                operand_desc += "sI"

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = []

            if name == "c" and not isinstance(operand, ImmmediateOperand):
                mod_list.append("negate")

            if name != "d":
                mod_list.append("reuse")

            operand.to_args(a, args, name, mod_list)

        if "U32" in self.opcode:
            args["fmt"] = a.REDUX_SZ.U32
        else:
            args["fmt"] = a.REDUX_SZ.S32

        instr_class = f"imad_{'wide_' if 'WIDE' in self.opcode else ''}_{operand_desc}_{operand_desc.replace('sI', 'I')}"
        return getattr(a, instr_class)(**args).encode()

    def encode_uldc(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        sz = self.opcode.split(".")
        sz = sz[1] if len(sz) > 1 else "32"
        if not sz.startswith((*tuple(string.ascii_letters), "_")):
            sz = "_" + sz

        args["sz"] = getattr(a.SZ_U8_S8_U16_S16_32_64, sz)
        self.operands[0].to_args(a, args, "d")
        self.operands[1].to_args(a, args, "a")

        return a.uldc_const__RCR(**args).encode()

    def encode_iadd3(self, a, args: dict) -> int:
        operand_desc = ""
        for operand in self.operands[1:]:
            if isinstance(operand, RegisterOperand):
                operand_desc += "R"
            elif isinstance(operand, UniformRegisterOperand):
                operand_desc += "U"
            elif isinstance(operand, ConstantAddressOperand):
                operand_desc += "C"
            else:
                operand_desc += "sI"

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = []
            if name != "d":
                mod_list.append("reuse")
                if not isinstance(operand, ImmmediateOperand):
                    mod_list.append("negate")

            operand.to_args(a, args, name, mod_list)

        instr_class = "iadd3_"
        if "sI" in operand_desc:
            instr_class += "imm_"
        else:
            instr_class += "noimm_"

        instr_class += f"_{operand_desc}_{operand_desc.replace('sI', 'I')}"
        return getattr(a, instr_class)(**args).encode()

    def encode_hfma2(self, a, args: dict) -> int:
        assert self.opcode == "HFMA2.MMA"

        for operand, name in zip(self.operands, "dabce", strict=False):
            if name == "e":
                name = "c1"

            mod_list = ["negate", "absolute", "reuse"] if name in "ab" else []
            operand.to_args(a, args, name, mod_list)

        return a.hfma2_mma__RRI(**args).encode()

    def encode_ldg(self, a, args: dict) -> int:
        assert isinstance(self.operands[1], DescriptorAddressOperand), (
            "Second operand of ldg must be a descriptor address."
        )

        size = int(self.opcode.rsplit(".", 1)[-1])
        assert size in (64, 128), "Only 64-bit and 128-bit ldg are supported."

        self.operands[0].to_args(a, args, "d")
        self.operands[1].to_args(a, args, "a", "b")

        args["sz"] = getattr(a.SZ_U8_S8_U16_S16_32_64_128, f"_{size}")
        args["Pnz_not"] = False

        return a.ldg_memdesc__Ra64(**args).encode()

    def encode_uiadd3(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        if isinstance(self.operands[2], ImmmediateOperand):
            instr_class = a.uiadd3__URsIUR_RIR
        else:
            assert isinstance(self.operands[2], UniformRegisterOperand)
            instr_class = a.uiadd3__URURUR_URURUR

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = (
                ["negate"] if isinstance(operand, UniformRegisterOperand) and name != "d" else []
            )
            operand.to_args(a, args, name, mod_list)

        return instr_class(**args).encode()

    def encode_isetp(self, a, args: dict) -> int:
        if isinstance(self.operands[3], RegisterOperand):
            instr_class = a.isetp__RRR_RRR_noEX
        elif isinstance(self.operands[3], UniformRegisterOperand):
            instr_class = a.isetp__RUR_RUR_noEX
        elif isinstance(self.operands[3], ConstantAddressOperand):
            instr_class = a.isetp__RCR_RCR_noEX
        else:
            instr_class = a.isetp__RsIR_RIR_noEX

        for name, member in a.ICmpAll.__members__.items():
            if name in self.opcode:
                args["icmp"] = member
                break

        if "U32" in self.opcode:
            args["fmt"] = a.REDUX_SZ.U32
        else:
            args["fmt"] = a.REDUX_SZ.S32

        for name, member in a.Bop.__members__.items():
            if name in self.opcode:
                args["bop"] = member
                break

        for operand, name in zip(self.operands, "uvabp", strict=False):
            mod_list = ["reuse"]
            if name == "p":
                mod_list.append("not")

            operand.to_args(a, args, name, mod_list)

        return instr_class(**args).encode()

    def encode_uimad(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        operand_desc = ""
        for operand in self.operands[1:]:
            if isinstance(operand, RegisterOperand):
                operand_desc += "R"
            elif isinstance(operand, UniformRegisterOperand):
                operand_desc += "UR"
            elif isinstance(operand, ConstantAddressOperand):
                operand_desc += "C"
            else:
                operand_desc += "sI"

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = []

            if name != "d":
                mod_list.append("reuse")

            if name == "c" and not isinstance(operand, ImmmediateOperand):
                mod_list.append("negate")

            operand.to_args(a, args, name, mod_list)

        if "U32" in self.opcode:
            args["fmt"] = a.REDUX_SZ.U32
        else:
            args["fmt"] = a.REDUX_SZ.S32

        instr_class = "uimad_"
        if "WIDE" in self.opcode:
            instr_class += "wide_"

        instr_class += f"_{operand_desc}_{operand_desc.replace('sI', 'I')}"

        return getattr(a, instr_class)(**args).encode()

    def encode_lea(self, a, args: dict) -> int:
        assert isinstance(self.operands[2], ImmmediateOperand) and isinstance(
            self.operands[3], ImmmediateOperand
        ), "Only immediate offset and scale are supported for lea in this assembler."

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = ["negate", "reuse"] if name == "a" else []

            if name == "c":
                args["scaleU5"] = operand.value
            else:
                operand.to_args(a, args, name, mod_list)

        return a.lea_lo_imm__RuIR_RIR(**args).encode()

    def encode_ushf(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        if "R" in self.opcode:
            args["dir"] = a.SDIR.R
        else:
            args["dir"] = a.SDIR.L

        for name, member in a.FMT.__members__.items():
            if name in self.opcode:
                args["fmt"] = member
                break

        if "HI" in self.opcode:
            args["hilo"] = a.HILO.HI
        else:
            args["hilo"] = a.HILO.LO

        operand_desc = ""
        for operand in self.operands[1:]:
            if isinstance(operand, UniformRegisterOperand):
                operand_desc += "UR"
            else:
                operand_desc += "uI"

        for operand, name in zip(self.operands, "dabc", strict=False):
            operand.to_args(a, args, name, mod_list=[])

        instr_classes = {
            "URURUR": a.ushf__URURUR_URURUR,
            "URURuI": a.ushf__URURuI_URRI,
            "URuIUR": a.ushf__URuIUR_URIR,
        }

        return instr_classes[operand_desc](**args).encode()

    def encode_plop3(self, a, args: dict) -> int:
        if isinstance(self.operands[4], UniformPredicateRegisterOperand):
            instr_class = a.plop3_lut_2out_uniform_
        else:
            instr_class = a.plop3_lut_2out_

        for operand, name in zip(self.operands, "uvpqr56", strict=False):
            mod_list = ["not"] if name in "pqr" else []

            if name == "5":
                args["uimm8"] = operand.value
            elif name == "6":
                args["vimm8"] = operand.value
            else:
                operand.to_args(a, args, name, mod_list)

        return instr_class(**args).encode()

    def encode_shf(self, a, args: dict) -> int:
        if "R" in self.opcode:
            args["dir"] = a.SDIR.R
        else:
            args["dir"] = a.SDIR.L

        for name, member in a.FMT.__members__.items():
            if name in self.opcode:
                args["fmt"] = member
                break

        if "HI" in self.opcode:
            args["hilo"] = a.HILO.HI
        else:
            args["hilo"] = a.HILO.LO

        operand_desc = ""
        for operand in self.operands[1:]:
            if isinstance(operand, RegisterOperand):
                operand_desc += "R"
            elif isinstance(operand, UniformRegisterOperand):
                operand_desc += "U"
            elif isinstance(operand, ConstantAddressOperand):
                operand_desc += "C"
            else:
                operand_desc += "uI"

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = ["reuse"] if name != "d" else []
            operand.to_args(a, args, name, mod_list)

        instr_class = f"shf__{operand_desc}_{operand_desc.replace('uI', 'I')}"
        return getattr(a, instr_class)(**args).encode()

    def encode_r2p(self, a, args: dict) -> int:
        self.operands[1].to_args(a, args, "a", mod_list=["reuse"])
        self.operands[2].to_args(a, args, "b")

        return a.r2p__RIR(**args).encode()

    def encode_sts(self, a, args: dict) -> int:
        self.operands[0].to_args(a, args, "a")
        self.operands[1].to_args(a, args, "b")

        if "64" in self.opcode:
            args["sz"] = a.SZ_U8_S8_U16_S16_32_64_128._64
        else:
            args["sz"] = a.SZ_U8_S8_U16_S16_32_64_128._32

        return a.sts__sImmOffset(**args).encode()

    def encode_dmul(self, a, args: dict) -> int:
        assert isinstance(self.operands[2], ConstantAddressOperand)

        for operand, name in zip(self.operands, "dab", strict=False):
            mod_list = ["negate", "absolute", "reuse"] if name != "d" else []
            operand.to_args(a, args, name, mod_list)

        try:
            return a.dmul__RCR_RC(**args).encode()
        except ValueError as error:
            if "reuse_src_a" not in str(error):
                raise

            args["yb"] = False
            return a.dmul__RCR_RC(**args).encode()

    def encode_bar(self, a, args: dict) -> int:
        assert self.opcode == "BAR.SYNC.DEFER_BLOCKING"

        args["barmode"] = a.BarSync.SYNC
        args["defer_blocking"] = a.DEFER_BLOCKINGONLY.DEFER_BLOCKING

        self.operands[0].to_args(a, args, "b")

        return a.bar__SYNC_dfrBlk_II_optionalCount_II(**args).encode()

    def encode_lds(self, a, args: dict) -> int:
        if "64" in self.opcode:
            args["sz"] = a.LDSSIZE._64
        else:
            args["sz"] = a.LDSSIZE._32

        self.operands[0].to_args(a, args, "d")
        self.operands[1].to_args(a, args, "a")

        return a.lds__sImmOffset(**args).encode()

    def encode_dfma(self, a, args: dict) -> int:
        assert isinstance(self.operands[2], ConstantAddressOperand)

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = ["negate", "absolute", "reuse"] if name != "d" else []
            operand.to_args(a, args, name, mod_list)

        try:
            return a.dfma__RCR_RCR(**args).encode()
        except ValueError as error:
            # Hopper folds reuse flags into the operand-extension scheduling encoding. Its DFMA
            # table has no entry for some reuse + yield combinations emitted by the SM80 scheduler,
            # but the same stall/barrier/reuse combination is encodable without the yield hint.
            # Preserve reuse because these hand-scheduled kernels rely on the operand collector
            # contents; yield only affects warp scheduling.
            if "reuse_src_a, reuse_src_c" not in str(error):
                raise

            args["yb"] = False
            return a.dfma__RCR_RCR(**args).encode()

    def encode_stg(self, a, args: dict) -> int:
        assert isinstance(self.operands[0], DescriptorAddressOperand), (
            "Second operand of stg must be a descriptor address."
        )

        size = int(self.opcode.rsplit(".", 1)[-1])
        assert size in (64, 128), "Only 64-bit and 128-bit stg are supported."

        self.operands[0].to_args(a, args, "a", "c")
        self.operands[1].to_args(a, args, "b")

        args["sz"] = getattr(a.SZ_U8_S8_U16_S16_32_64_128, f"_{size}")

        return a.stg_memdesc__Ra64(**args).encode()

    def encode_uisetp(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        for name, member in a.ICmpAll.__members__.items():
            if name in self.opcode:
                args["icmp"] = member
                break

        if "U32" in self.opcode:
            args["fmt"] = a.REDUX_SZ.U32
        else:
            args["fmt"] = a.REDUX_SZ.S32

        for name, member in a.Bop.__members__.items():
            if name in self.opcode:
                args["bop"] = member
                break

        for operand, name in zip(self.operands, "uvabp", strict=False):
            mod_list = ["not"] if name in "p" else []
            operand.to_args(a, args, name, mod_list)

        return a.uisetp__URURUR_URURUR(**args).encode()

    def encode_bra(self, a, args: dict) -> int:
        value_offset = self.operands[0].addr - self.address - 16
        args["sImm"] = value_offset
        args["Pp_not"] = False

        return a.bra__CONV_DIV(**args).encode()

    def encode_exit(self, a, args: dict) -> int:
        args["Pp_not"] = False

        return a.exit_(**args).encode()

    def encode_nop(self, a, args: dict) -> int:
        return a.nop_(**args).encode()

    def encode_ulea(self, a, args: dict) -> int:
        SASSInstruction.patch_uniform_pred(args)

        assert "HI" in self.opcode, "Only ULEA.HI is supported in this assembler."
        args["hilo"] = a.HIONLY.HI

        assert "X" not in self.opcode

        for operand, name in zip(self.operands, "dabc", strict=False):
            mod_list = ["negate"] if name in "ab" else []
            operand.to_args(a, args, name, mod_list)

        args["scaleU5"] = self.operands[4].value

        return a.ulea_hi_noimm__URURUR_URURUR(**args).encode()

    def encode_imnmx(self, a, args: dict) -> int:
        if isinstance(self.operands[2], ConstantAddressOperand):
            instr_class = a.imnmx__RCR_RCR
        elif isinstance(self.operands[2], ImmmediateOperand):
            instr_class = a.imnmx__RIR_RsIR
        elif isinstance(self.operands[2], RegisterOperand):
            instr_class = a.imnmx__RRR_RRR
        else:
            assert isinstance(self.operands[2], UniformRegisterOperand)
            instr_class = a.imnmx__RUR_RUR

        if "U32" in self.opcode:
            args["fmt"] = a.REDUX_SZ.U32
        else:
            args["fmt"] = a.REDUX_SZ.S32

        for operand, name in zip(self.operands, "dabp", strict=False):
            mod_list = ["reuse"] if name in "ab" else []
            mod_list = ["not"] if name == "p" else mod_list
            operand.to_args(a, args, name, mod_list)

        return instr_class(**args).encode()

    def encode(self, a):
        args = {}

        # Control codes.
        self.control_codes.to_args(args)

        # Predicate.
        self.predicate.to_args(a, args, "g", mod_list=["not"])

        mnemonic = self.opcode.split(".")[0].lower()
        encoder = getattr(self, f"encode_{mnemonic}", None)

        if encoder is not None:
            self.binary = getattr(self, f"encode_{mnemonic}")(a, args)
        else:
            raise NotImplementedError(f"Encoding for opcode {self.opcode} is not implemented.")


class SASSProgram:
    def __init__(self, codes: list[str]):
        # Map from label to instruction address.
        self.label_map: dict[str, SASSInstruction] = {}
        self.instructions: list[SASSInstruction] = []
        self.codes = codes

        # Patterns for parsing operands.
        imm_pattern = r"[+-]?[0-9A-Fa-fxXbBfF]+"
        self.reg_pattern = re.compile(r"R(?P<index>[\dZ]+)")
        self.uniform_reg_pattern = re.compile(r"UR(?P<index>[\dZ]+)")
        self.pred_reg_pattern = re.compile(r"P(?P<index>[\dT]+)")
        self.uniform_pred_reg_pattern = re.compile(r"UP(?P<index>[\dT]+)")

        self.imm_pattern = re.compile(imm_pattern)
        self.constant_addr_pattern = re.compile(
            rf"c\[(?P<bank>{imm_pattern})\]\[(?P<addr>{imm_pattern})\]"
        )
        self.addr_pattern = re.compile(rf"\[(?P<reg>R[\dZ]+)\+(?P<addr>{imm_pattern})\]")
        self.desc_addr_pattern = re.compile(
            rf"desc\[(?P<desc>UR[\dZ]+)\]\[(?P<reg>R[\dZ]+)\.64"
            rf"(?P<addr>{imm_pattern})?\]"
        )
        self.label_pattern = re.compile(r"`\((?P<label>[\w_\.]+)\)")

        self._parse()

    def _parse_operand(self, operand_str: str):
        args = {}
        args["raw_content"] = operand_str

        if operand_str.endswith(".reuse"):
            reuse = True
            operand_str = operand_str[:-6]
        else:
            reuse = False

        negated = operand_str.startswith("-")
        absoluted = operand_str.startswith("|") and operand_str.endswith("|")
        notted = operand_str.startswith("!")
        inverted = operand_str.startswith("~")

        args["negated"] = negated
        args["absoluted"] = absoluted
        args["notted"] = notted
        args["inverted"] = inverted

        if negated or notted or inverted:
            operand_str = operand_str[1:]
        elif absoluted:
            operand_str = operand_str[1:-1]

        m = self.reg_pattern.fullmatch(operand_str)
        if m is not None:
            args["index"] = m.group("index")
            args["reuse"] = reuse
            return RegisterOperand(**args)

        assert not reuse, "Only general-purpose registers can be reused"

        m = self.uniform_reg_pattern.fullmatch(operand_str)
        if m is not None:
            args["index"] = m.group("index")
            return UniformRegisterOperand(**args)

        m = self.pred_reg_pattern.fullmatch(operand_str)
        if m is not None:
            args["index"] = m.group("index")
            return PredicateRegisterOperand(**args)

        m = self.uniform_pred_reg_pattern.fullmatch(operand_str)
        if m is not None:
            args["index"] = m.group("index")
            return UniformPredicateRegisterOperand(**args)

        m = self.imm_pattern.fullmatch(operand_str)
        if m is not None:
            if operand_str.startswith(("0f", "0F")):
                # Float literal in hexadecimal form.
                value = int(operand_str[2:], 16)  # noqa: FURB166 -- SASS uses 0f, not 0x.
            else:
                value = int(operand_str, 0)

            if negated:
                value = -value
                negated = False
                args["negated"] = negated

            args["value"] = value
            return ImmmediateOperand(**args)

        m = self.constant_addr_pattern.fullmatch(operand_str)
        if m is not None:
            args["bank"] = int(m.group("bank"), 0)
            args["addr"] = int(m.group("addr"), 0)
            return ConstantAddressOperand(**args)

        m = self.addr_pattern.fullmatch(operand_str)
        if m is not None:
            args["base_reg"] = m.group("reg")
            args["offset"] = int(m.group("addr"), 0)
            return AddressOperand(**args)

        m = self.desc_addr_pattern.fullmatch(operand_str)
        if m is not None:
            args["desc"] = m.group("desc")
            args["addr_reg"] = m.group("reg")
            args["offset"] = int(m.group("addr") or 0, 0)
            return DescriptorAddressOperand(**args)

        m = self.label_pattern.fullmatch(operand_str)
        if m is not None:
            args["label"] = m.group("label")
            args["addr"] = 0  # Placeholder, will be filled in later.
            return LabelOperand(**args)

        return UnknownOperand(**args)

    def _parse(self):
        pattern = re.compile(
            r"(?P<ctrl_code>\[[BRWYS:\-\d]+\])\s*(@(?P<pred>!?U?P\d))?\s*(?P<instr>.*)\s*;\s*(?P<comment>\/\/.*)?"
        )
        label = None
        addr = 0

        for line in self.codes:
            line = line.strip()

            # Check if the line is a label.
            if line.startswith(".") and line.endswith(":"):
                label = line[:-1]
                continue

            line_match = pattern.fullmatch(line)
            if line_match is None:
                continue

            pred = line_match.group("pred")
            pred = self._parse_operand("PT" if pred is None else pred)

            instr_parts = line_match.group("instr").split()
            operands = [self._parse_operand(part.rstrip(",").strip()) for part in instr_parts[1:]]

            self.instructions.append(
                SASSInstruction(
                    raw_instr=line,
                    address=addr,
                    predicate=pred,
                    opcode=instr_parts[0],
                    operands=operands,
                    binary=0,
                    control_codes=ControlCodes.from_string(line_match.group("ctrl_code")),
                    label=label,
                )
            )

            if label is not None:
                self.label_map[label] = self.instructions[-1]
                label = None

            addr += 16

        # Replace labels in operands with actual addresses.
        for instr in self.instructions:
            for operand in instr.operands:
                if isinstance(operand, LabelOperand):
                    operand.addr = self.label_map[operand.label].address

    def encode(self, version: str) -> list[int]:
        assembler_name = f"codegen.sass.assembler{version}"

        try:
            as_module = importlib.import_module(assembler_name)
        except ImportError as e:
            raise ImportError(
                f"Assembler module for version {version} not found: {assembler_name}"
            ) from e

        for instr in self.instructions:
            try:
                instr.encode(as_module)
            except Exception as error:
                raise ValueError(
                    f"Failed to encode SM{version} instruction at {instr.address:#x}: "
                    f"{instr.raw_instr}"
                ) from error

        return [instr.binary for instr in self.instructions]
