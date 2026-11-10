"""Lower a resolved kernel spec through the preserved SASS planners."""

import importlib
from pathlib import Path

from codegen.geometry import Coord2D
from codegen.sass.cubin import generate_cubin, generate_cubin_by_patch
from codegen.spec import KernelSpec


def generate(spec: KernelSpec):
    module = importlib.import_module(f"codegen.sass.computation.{spec.shape}{spec.dimensions}d")
    arch = importlib.import_module(f"codegen.sass.arch.{spec.arch}")
    args = {
        "radius": spec.radius,
        "time_block_size": spec.time_block_size,
        "leading_load_count": spec.leading_load_count,
        "min_comp_stall": spec.min_comp_stall,
        "arch": arch,
    }

    if spec.dimensions == 2:
        args.update(group_size=spec.group_size.x)
    else:
        args.update(
            group_sizes=Coord2D(**spec.group_size.named()),
        )

    code = module.CodeGen(**args)
    code.generate()

    return code


def build(spec: KernelSpec, output: Path, temporary: Path) -> None:
    code = generate(spec)

    if spec.arch == "sm80":
        template = Path(__file__).parents[1] / "templates/sass/kernel.sm80.cuasm.template"
        generate_cubin(
            str(template),
            str(output),
            {
                "code": "\n".join(code.codes),
                "reg_count": code.reg_count,
                "reg_count_hex": f"0x{code.reg_count:08x}",
                "coeff_size": spec.coefficient_count * 8,
            },
            cuasm_path=str(temporary / "kernel.cuasm"),
        )
    else:
        from codegen.sass.assembler import SASSProgram

        generate_cubin_by_patch(
            str(output),
            "stencil_kernel",
            "neighbor_coeffs",
            spec.coefficient_count * 8,
            [
                "double* input",
                "double* output",
                *[
                    f"unsigned {name}"
                    for name in (
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
                ],
            ],
            code.reg_count,
            SASSProgram(code.codes).encode("90"),
            "90",
        )
