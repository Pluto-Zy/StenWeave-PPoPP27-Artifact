"""Historical staged CUDA breakdown generators."""

from codegen.profiles.breakdown import features
from codegen.spec import KernelSpec


def generate(spec: KernelSpec):
    from codegen.cuda.breakdown import box, star

    if spec.shape == "star":
        code = star.CodeGen(
            spec.radius,
            spec.time_block_size,
            spec.minimum_overlap.x,
            star.CodeGenConfig(**features(spec.stage, spec.shape, spec.radius)),
        )
    else:
        code = box.CodeGen(
            spec.radius,
            spec.time_block_size,
            spec.minimum_overlap.x,
            box.CodeGenConfig(**features(spec.stage, spec.shape, spec.radius)),
        )

    code.generate()

    return code
