"""Offline artifact production independent of the runner and its build tree."""

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from codegen.manifest import record, write_manifest
from codegen.spec import KernelSpec

logger = logging.getLogger(__name__)


def build_one(task: tuple[KernelSpec, Path]) -> dict:
    spec, root = task
    output = root / "cubins" / f"{spec.id}.cubin"
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stencil-codegen-", dir=root.parent) as directory:
        temporary = Path(directory)

        if spec.backend == "sass":
            from codegen.sass.backend import build
        else:
            from codegen.cuda.emitter import build

        # Publish a cubin only after the backend completes successfully.
        build(spec, temporary / "kernel.cubin", temporary)
        (temporary / "kernel.cubin").replace(output)

    return record(spec, output)


def build_artifact(name: str, specs: list[KernelSpec], root: Path, config: Path, jobs: int) -> None:
    root = root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))

    previous = None
    try:
        (staging / "cubins").mkdir()
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            records = list(executor.map(build_one, ((spec, staging) for spec in specs)))

        version = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            check=False,
            capture_output=True,
            text=True,
        )

        write_manifest(
            staging,
            name,
            records,
            {
                "python": platform.python_version(),
                "nvcc": version.stdout.strip(),
                "generator_revision": revision.stdout.strip()
                if revision.returncode == 0
                else "unknown",
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            },
        )

        # A completed staging tree is the only thing that may replace a published
        # artifact.  Keep the old tree until the new manifest and every cubin are
        # present, so filtered or failed builds cannot leave a mixed artifact.
        if root.exists():
            previous = Path(tempfile.mkdtemp(prefix=f".{root.name}.previous-", dir=root.parent))
            previous.rmdir()
            os.replace(root, previous)

        try:
            os.replace(staging, root)
        except Exception:
            if previous is not None and not root.exists():
                os.replace(previous, root)

            raise
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)

        raise

    # Publication committed when staging replaced root. Backup cleanup is outside the transaction.
    if previous is not None:
        try:
            shutil.rmtree(previous)
        except OSError as error:
            logger.warning(
                "Published artifact at %s, but could not remove backup %s: %s",
                root,
                previous,
                error,
            )
