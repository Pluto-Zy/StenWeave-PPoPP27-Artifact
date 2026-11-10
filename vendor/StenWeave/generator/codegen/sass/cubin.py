import io
import os
import subprocess
import tempfile

from elftools.elf.elffile import ELFFile

from codegen.template import replace_placeholders


def generate_cubin(
    template_path: str,
    cubin_path: str,
    replacements: dict,
    cuasm_path: str | None = None,
):
    # CuAssembler is used only by the SM80 template path. SM90 uses the repository's own
    # `SASSProgram` encoder and `generate_cubin_by_patch`, so importing CuAsm here keeps that path
    # independent from CuAssembler (which does not support SM90).
    from CuAsm.CuAsmParser import CuAsmParser

    if cuasm_path is None:
        # Put `cuasm` in the same directory as `cubin`, with the suffix `.cuasm`.
        cuasm_path = os.path.splitext(cubin_path)[0] + ".cuasm"

    # Get the directory of the output cubin file.
    cubin_dir = os.path.dirname(cubin_path)
    if not os.path.exists(cubin_dir):
        os.makedirs(cubin_dir, exist_ok=True)

    # Generate the cuasm file from the template.
    replace_placeholders(template_path, cuasm_path, replacements)
    print(f"Generated cuasm file: {os.path.abspath(cuasm_path)}")

    cap = CuAsmParser()
    cap.parse(cuasm_path)
    cap.saveAsCubin(cubin_path)
    print(f"Generated cubin file: {os.path.abspath(cubin_path)}")


class CubinPatcher:
    def __init__(self, cubin_path: str):
        with open(cubin_path, "rb") as f:
            data = f.read()
            self.elf = ELFFile(io.BytesIO(data))
            self.raw_data = bytearray(data)

        self.function = None

    def find_symbol_index(self, symbol_name: str) -> int:
        symtab = self.elf.get_section_by_name(".symtab")
        if symtab is None:
            raise ValueError("No symbol table found in the cubin file.")

        for i, symbol in enumerate(symtab.iter_symbols()):
            if symbol.name == symbol_name:
                return i

        raise ValueError(f"Symbol '{symbol_name}' not found in the cubin file.")

    def analyze_symbols(self):
        symtab = self.elf.get_section_by_name(".symtab")
        if symtab is None:
            raise ValueError("No symbol table found in the cubin file.")

        for sym in symtab.iter_symbols():
            if not sym.name or sym.name.startswith("."):
                continue

            st_type = sym["st_info"]["type"]

            if st_type == "STT_FUNC":
                if self.function is not None:
                    raise ValueError("Multiple functions found in the cubin file.")

                self.function = sym

    def find_attr_in_section(self, section, attr_id: int, symbol_index: int) -> int:
        data = section.data()
        attr_id = attr_id.to_bytes(2, byteorder="little")

        offset = data.find(attr_id)
        while offset != -1:
            # The format of the attribute entry is as follows:
            # [2 bytes attr_id][2 bytes length][4 bytes symbol_index][value...]
            entry_symbol_index = int.from_bytes(data[offset + 4 : offset + 8], byteorder="little")
            if entry_symbol_index == symbol_index:
                return section["sh_offset"] + offset + 8

            offset = data.find(attr_id, offset + 1)

        raise ValueError(
            f"Attribute with ID {attr_id} and symbol index {symbol_index} not found in the section."
        )

    def patch_reg_count(self, reg_count: int):
        nv_info_section = self.elf.get_section_by_name(".nv.info")
        if nv_info_section is None:
            raise ValueError("No .nv.info section found in the cubin file.")

        symbol_index = self.find_symbol_index(self.function.name)
        attr_offset = self.find_attr_in_section(
            nv_info_section, attr_id=0x2F04, symbol_index=symbol_index
        )

        # The register count is stored as a 4-byte value at the attribute offset.
        self.raw_data[attr_offset : attr_offset + 4] = reg_count.to_bytes(4, byteorder="little")

    def patch_code(self, codes: list[int]):
        sec_index = self.function["st_shndx"]
        text_section = self.elf.get_section(sec_index)
        text_size = text_section["sh_size"]
        text_off = text_section["sh_offset"] + self.function["st_value"]

        assert len(codes) * 16 <= text_size, (
            "The patched code is too large to fit in the original code section."
        )

        new_code = b"".join(code.to_bytes(16, byteorder="little", signed=False) for code in codes)
        self.raw_data[text_off : text_off + len(new_code)] = new_code

    def save(self, output_path: str):
        with open(output_path, "wb") as f:
            f.write(self.raw_data)


def generate_template_cubin(
    output_path: str,
    func_name: str,
    coeff_name: str,
    coeff_size: int,
    args: list[str],
    code_count: int,
    arch_version: str,
):
    code_count = max(code_count - 2, 0)
    code = rf"""
extern "C" __constant__ alignas(8) char {coeff_name}[{coeff_size}];

extern "C" __global__ void {func_name}({", ".join(args)}) {{
"""

    while code_count > 0:
        code += rf"""
    #pragma unroll
    for (unsigned i = 0; i != {min(code_count, 10000)}; ++i) {{
        __syncwarp();
    }}
"""
        code_count -= 10000

    code += "}\n"

    # Compile the code to a cubin file using nvcc.
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False) as f:
        f.write(code.encode("utf-8"))
        temp_cu_path = f.name

    try:
        cmd = ["nvcc", "-cubin", f"-arch=sm_{arch_version}", temp_cu_path, "-o", output_path]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        assert result.returncode == 0, f"nvcc failed with error: {result.stderr}"
    finally:
        if os.path.exists(temp_cu_path):
            os.remove(temp_cu_path)


def generate_cubin_by_patch(
    output_path: str,
    func_name: str,
    coeff_name: str,
    coeff_size: int,
    args: list[str],
    reg_count: int,
    codes: list[int],
    arch_version: str,
):
    # Generate a cubin file for patching.
    generate_template_cubin(
        output_path=output_path,
        func_name=func_name,
        coeff_name=coeff_name,
        coeff_size=coeff_size,
        args=args,
        code_count=len(codes),
        arch_version=arch_version,
    )

    patcher = CubinPatcher(output_path)
    patcher.analyze_symbols()
    patcher.patch_reg_count(reg_count)
    patcher.patch_code(codes)
    patcher.save(output_path)
