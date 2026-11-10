"""Fully resolved FP64 kernel descriptions shared by emitters and artifact packaging."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CrossSectionSize:
    x: int
    y: int | None = None

    def named(self) -> dict[str, int]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class KernelSpec:
    shape: str
    dimensions: int
    backend: str
    arch: str
    radius: int
    time_block_size: int
    minimum_overlap: CrossSectionSize
    group_size: CrossSectionSize | None = None
    leading_load_count: int = 1
    min_comp_stall: int = 1
    stage: int | None = None
    launch_bounds: int | None = None

    @property
    def coefficient_count(self) -> int:
        if self.shape == "star":
            return 1 + 2 * self.dimensions * self.radius

        return (2 * self.radius + 1) ** self.dimensions

    def metadata(self) -> dict:
        from codegen.profiles.breakdown import features

        record = {
            k: getattr(self, k)
            for k in ("shape", "dimensions", "backend", "arch", "radius", "time_block_size")
        }
        record.update(dtype="f64", minimum_overlap=self.minimum_overlap.named())

        if self.group_size is not None:
            record["group_size"] = self.group_size.named()

        if self.backend == "sass":
            record["sass"] = {
                "leading_load_count": self.leading_load_count,
                "min_comp_stall": self.min_comp_stall,
            }

        if self.backend == "cuda":
            record["cuda"] = {"launch_bounds": self.launch_bounds}

        record["profile"] = (
            {"name": "default"}
            if self.stage is None
            else {
                "name": "breakdown",
                "stage": self.stage,
                "features": features(self.stage, self.shape, self.radius),
            }
        )

        return record

    @property
    def id(self) -> str:
        return hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()[:24]
