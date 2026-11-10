"""CUDA source emission and offline nvcc compilation."""

import importlib
import subprocess
from pathlib import Path

from codegen.spec import KernelSpec
from codegen.template import replace_placeholders


def source(spec: KernelSpec, output: Path) -> None:
    if spec.launch_bounds is None:
        raise ValueError("CUDA source requires launch_bounds")

    if spec.stage is not None:
        from codegen.cuda.breakdown import generate

        code = generate(spec)
        body = "\n".join(code.codes)
        declarations = code.generate_reg_declarations()
    else:
        name = (
            "star2d"
            if spec.shape == "star" and spec.dimensions == 2
            else f"{spec.shape}{spec.dimensions}d"
        )
        module = importlib.import_module(f"codegen.cuda.{name}")
        code = module.CodeGen(spec.radius, spec.time_block_size, spec.minimum_overlap.x)
        code.generate()
        body = "\n".join(code.codes)
        declarations = code.generate_reg_declarations()

    template = Path(__file__).parents[1] / "templates/cuda/kernel.cu.template"
    replace_placeholders(
        template,
        output,
        {
            "code": body,
            "extra_parameters": ", bool cond" if spec.stage is not None else "",
            "reg_declrations": declarations,
            "coeff_count": spec.coefficient_count,
            "launch_annotation": f"__launch_bounds__({spec.launch_bounds})",
        },
    )


def build(spec: KernelSpec, output: Path, temporary: Path) -> None:
    cu = temporary / "kernel.cu"
    source(spec, cu)

    subprocess.run(
        [
            "nvcc",
            "--cubin",
            f"-arch=sm_{spec.arch[2:]}",
            "--ptxas-options=-v",
            str(cu),
            "-o",
            str(output),
        ],
        check=True,
    )
