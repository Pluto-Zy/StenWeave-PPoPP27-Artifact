"""Validate complete build products before reusing them in a fresh quickcheck."""

import contextlib
import fcntl
import json

from run.suites.common import IMPLEMENTATIONS, ROOT, baseline, sha


def inputs(name, arch, toolchain):
    label = IMPLEMENTATIONS[name]
    patchdir = ROOT / "provenance/patches" / label
    paths = [
        ROOT / f"scripts/build/{name}.py",
        ROOT / f"provenance/{label}.sha256",
        patchdir / "series",
    ]
    paths += [
        patchdir / line.strip()
        for line in (patchdir / "series").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if name == "stenweave":
        paths += [
            ROOT / "configs/quickcheck/generator.toml",
            ROOT / "provenance/CuAssembler.sha256",
        ]
    else:
        paths += [ROOT / f"configs/baselines/{label}/{arch}.toml"]
    if name in ("convstencil", "lorastencil"):
        paths += [ROOT / "scripts/build/tensor_stencils.py"]
    if name == "flashfft":
        paths += [ROOT / "scripts/adapters/flashfft_driver.cu"]
    return {
        "files": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "toolchain": {
            key: info["version"]
            for key, info in (toolchain or {}).items()
            if key != "NCU"
        },
    }


def reusable(name, arch, build_root, compiler):
    """Also support older batches that predate build-input fingerprints."""
    label = IMPLEMENTATIONS[name]
    bindir = build_root / "bin" / label
    if name != "stenweave":
        bindir /= arch
    try:
        manifest = json.loads((bindir / "build.json").read_text())
        if manifest["compiler"].strip() != compiler.strip() or manifest["arch"] != arch:
            return False, "compiler or architecture changed"
        patchdir = ROOT / "provenance/patches" / label
        patches = {
            line.strip(): sha(patchdir / line.strip())
            for line in (patchdir / "series").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        if manifest["patches"] != patches:
            return False, "build patches changed"
        for line in (ROOT / f"provenance/{label}.sha256").read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            if sha(ROOT / path) != digest:
                return False, "source snapshot changed"
        if name == "stenweave":
            if arch == "sm80":
                for line in (
                    (ROOT / "provenance/CuAssembler.sha256").read_text().splitlines()
                ):
                    digest, path = line.split(maxsplit=1)
                    if sha(ROOT / path) != digest:
                        return False, "CuAssembler snapshot changed"
            if sha(bindir / "stencil") != manifest["binary_sha256"]:
                return False, "runner binary changed"
            artifact = build_root / f"artifacts/StenWeave/{arch}/benchmark"
            generated = json.loads((artifact / "build.json").read_text())
            if (
                generated["patches"] != patches
                or generated["compiler"].strip() != compiler.strip()
            ):
                return False, "generator build changed"
            if generated["config_sha256"] != sha(
                ROOT / "configs/quickcheck/generator.toml"
            ):
                return False, "generator configuration changed"
            if sha(artifact / "manifest.json") != generated["manifest_sha256"]:
                return False, "kernel manifest changed"
            for kernel in json.loads((artifact / "manifest.json").read_text())[
                "kernels"
            ]:
                if sha(artifact / kernel["cubin"]["path"]) != kernel["cubin"]["sha256"]:
                    return False, "kernel binary changed"
        elif name == "flashfft":
            if sha(bindir / "flashfft2d") != manifest["binaries"]["flashfft2d"]:
                return False, "binary changed"
            if (
                sha(ROOT / "scripts/adapters/flashfft_driver.cu")
                != manifest["driver_sha256"]
            ):
                return False, "driver source changed"
        else:
            config, path = baseline(name, arch)
            case = config["stencils"]["box2d1r"]
            if name == "an5d":
                cap = case.get("maxrregcount", 0)
                key = case["variant"] + (f"-rreg{cap}" if cap else "")
                if sha(bindir / key) != manifest["binaries"][key]["sha256"]:
                    return False, "binary changed"
            else:
                if (
                    manifest["config_sha256"] != sha(path)
                    or manifest["binaries"]["box2d1r"]["config"] != case
                ):
                    return False, "frozen configuration changed"
                records = manifest["binaries"]["box2d1r"]
                records = (
                    [records]
                    if name == "ebisu"
                    else [records[mode] for mode in ("performance", "check")]
                )
                for record in records:
                    if sha(ROOT / record["path"]) != record["sha256"]:
                        return False, "binary changed"
            source = build_root / "src" / label / arch
            for relative, digest in manifest["source_files"].items():
                if sha(source / relative) != digest:
                    return False, "prepared source changed"
        return True, "verified build"
    except (OSError, ValueError, KeyError, TypeError):
        return False, "build record or product missing/incomplete"


@contextlib.contextmanager
def build_lock(build_root):
    build_root.mkdir(parents=True, exist_ok=True)
    with (build_root / ".quickcheck.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"Another quickcheck is using this build tree: {build_root}"
            ) from error
        yield
