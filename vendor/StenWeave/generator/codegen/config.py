"""Parse and validate the human-maintained artifact matrix before invoking tools."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from codegen.spec import CrossSectionSize, KernelSpec


def size(value: object, dimensions: int, field: str) -> CrossSectionSize:
    if dimensions == 2 and type(value) is int:
        result = CrossSectionSize(value)
    elif dimensions == 3 and isinstance(value, dict) and set(value) == {"x", "y"}:
        result = CrossSectionSize(**value)
    else:
        raise ValueError(f"{field}: use a scalar for 2D or {{x, y}} for 3D")

    if any(type(v) is not int or v <= 0 for v in result.named().values()):
        raise ValueError(f"{field}: dimensions must be positive integers")

    return result


def load_config(path: Path) -> tuple[str, list[KernelSpec]]:
    config = tomllib.loads(path.read_text())
    if config.get("schema_version") != 2:
        raise ValueError("unsupported config schema_version")

    if "sass" in config:
        raise ValueError(
            "top-level sass configuration is no longer supported; "
            "put leading_load_count and min_comp_stall in each SASS variant"
        )

    specs = []
    for kernel in config["kernels"]:
        if set(kernel) - {"backend", "shape", "dimensions", "arch", "profile", "variants"}:
            raise ValueError("unknown kernel field")

        backend, shape, dims = (kernel[k] for k in ("backend", "shape", "dimensions"))
        if backend not in ("cuda", "sass") or shape not in ("star", "box") or dims not in (2, 3):
            raise ValueError("unsupported backend, shape, or dimensions")

        profile = kernel.get("profile", "default")
        if profile not in ("default", "breakdown"):
            raise ValueError("unknown profile")

        for arch in kernel["arch"]:
            if arch not in ("sm80", "sm90"):
                raise ValueError("arch must be sm80 or sm90")

            if profile == "breakdown" and (backend != "cuda" or dims != 2):
                raise ValueError("breakdown requires 2D CUDA")

            for variant in kernel["variants"]:
                if set(variant) - {
                    "radius",
                    "time_block_size",
                    "minimum_overlap",
                    "group_size",
                    "stages",
                    "leading_load_count",
                    "min_comp_stall",
                    "launch_bounds",
                }:
                    raise ValueError("unknown variant field")

                if backend != "sass" and (
                    "leading_load_count" in variant or "min_comp_stall" in variant
                ):
                    raise ValueError("leading_load_count and min_comp_stall require backend='sass'")

                leading_load_count = variant.get("leading_load_count", 1)
                min_comp_stall = variant.get("min_comp_stall", 1)
                if type(leading_load_count) is not int or leading_load_count not in (1, 2):
                    raise ValueError("variant leading_load_count must be an integer in 1..2")

                if type(min_comp_stall) is not int or min_comp_stall < 1:
                    raise ValueError("variant min_comp_stall must be a positive integer")

                launch_bounds = variant.get("launch_bounds")
                if backend == "sass" and "launch_bounds" in variant:
                    raise ValueError("launch_bounds requires backend='cuda'")

                if backend == "cuda" and (
                    type(launch_bounds) is not int or not 1 <= launch_bounds <= 1024
                ):
                    raise ValueError("CUDA variant launch_bounds must be an integer in 1..1024")

                radius, steps = (variant[k] for k in ("radius", "time_block_size"))
                if (
                    type(radius) is not int
                    or radius not in (1, 2, 3)
                    or type(steps) is not int
                    or steps <= 0
                ):
                    raise ValueError("radius must be 1..3 and time_block_size must be positive")

                if shape == "box" and dims == 3 and radius != 1:
                    raise ValueError("box3d supports radius 1 only")

                overlap = size(variant["minimum_overlap"], dims, "minimum_overlap")
                if backend == "cuda" and dims == 3 and overlap.x != overlap.y:
                    raise ValueError("CUDA 3D lowering requires equal x/y minimum_overlap")

                group = (
                    size(variant["group_size"], dims, "group_size") if backend == "sass" else None
                )
                if backend == "cuda" and "group_size" in variant:
                    raise ValueError("CUDA does not accept group_size")

                if any(v < 2 * radius * steps for v in overlap.named().values()):
                    raise ValueError("minimum_overlap must cover both temporal halos")

                if group and any(
                    overlap.named()[axis] % (2 * v) for axis, v in group.named().items()
                ):
                    raise ValueError("minimum_overlap must be divisible by twice group_size")

                if profile == "default" and "stages" in variant:
                    raise ValueError("stages requires breakdown profile")

                for stage in variant.get("stages", [None]):
                    if profile == "breakdown" and stage not in (1, 2, 3, 4):
                        raise ValueError("breakdown requires stages 1..4")

                    spec = KernelSpec(
                        shape,
                        dims,
                        backend,
                        arch,
                        radius,
                        steps,
                        overlap,
                        group,
                        leading_load_count,
                        min_comp_stall,
                        stage,
                        launch_bounds,
                    )
                    specs.append(spec)

    if len({s.id for s in specs}) != len(specs):
        raise ValueError("duplicate kernel specifications")

    return config["artifact"]["name"], specs
