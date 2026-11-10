"""Deterministic artifact metadata; paths are always relative to the artifact root."""

import hashlib
import json
from pathlib import Path

from codegen.logical import BoxTopology, StarTopology
from codegen.spec import KernelSpec


def record(spec: KernelSpec, cubin: Path) -> dict:
    data = spec.metadata()
    data.update(
        id=spec.id,
        cubin={
            "path": f"cubins/{spec.id}.cubin",
            "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
            "size": cubin.stat().st_size,
        },
        entry={"symbol": "stencil_kernel", "abi_version": 1 if spec.backend == "cuda" else 2},
        pipeline=(StarTopology if spec.shape == "star" else BoxTopology)(
            spec.radius, spec.time_block_size
        ).pipeline_contract(),
        coefficients={"symbol": "neighbor_coeffs", "count": spec.coefficient_count, "dtype": "f64"},
    )

    return data


def write_manifest(root: Path, name: str, records: list[dict], provenance: dict) -> None:
    data = {
        "schema_version": 2,
        "artifact": {"name": name},
        "provenance": provenance,
        "kernels": records,
    }

    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "manifest.json")
