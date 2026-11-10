"""Resolve collection cases from frozen implementation configurations."""

import hashlib
import json
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATIONS = {
    "an5d": "AN5D",
    "ebisu": "EBISU",
    "convstencil": "ConvStencil",
    "lorastencil": "LoRAStencil",
    "flashfft": "FlashFFTStencil",
    "stenweave": "StenWeave",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def baseline(implementation, arch):
    path = ROOT / f"configs/baselines/{IMPLEMENTATIONS[implementation]}/{arch}.toml"
    config = tomllib.loads(path.read_text())
    if config["arch"] != arch:
        raise ValueError(f"Configuration architecture mismatch: {path}")
    return config, path


def task(implementation, stencil, config, case=None, **kwargs):
    identifier = case or stencil
    if "width" in kwargs:
        identifier += f"-{kwargs['width']}"
    return {
        "id": f"{implementation}-{identifier}",
        "implementation": implementation,
        "stencil": stencil,
        "config": str(config.relative_to(ROOT)),
        "config_sha256": sha(config),
        **({"case": case} if case else {}),
        **kwargs,
    }


def baseline_tasks(implementation, arch, stencils=None):
    config, path = baseline(implementation, arch)
    return [
        task(implementation, stencil, path)
        for stencil in config["stencils"]
        if stencils is None or stencil in stencils
    ]


def native_cases(arch, suite):
    path = ROOT / f"configs/stenweave/runner/{arch}/{suite}.toml"
    batch = tomllib.loads(path.read_text())
    return batch, path


def stenweave_tasks(arch, suite, variants=None, widths=None):
    batch, path = native_cases(arch, suite)
    tasks = []
    for case in batch["case"]:
        name = case["name"]
        stencil = name.split("-")[0]
        suffix = name.split(f"-{arch}-", 1)[1]
        variant = "sass-ilp" if suffix.startswith("sass-ilp") else suffix.split("-")[0]
        if variants and variant not in variants:
            continue
        fields = {}
        if suite == "size-sweep":
            width = case["run"]["domain"]["x"]
            if widths and width not in widths[2 if "2d" in stencil else 3]:
                continue
            fields["domain_width"] = width
        tasks.append(
            task(
                "stenweave",
                stencil,
                path,
                name,
                native_suite=suite,
                variant=variant,
                kernel_id=case["kernel"]["id"],
                **fields,
            )
        )
    return tasks


def selected_case_text(path, name, profile=False):
    """Keep exact native syntax, changing only explicit profile sampling fields."""
    text = Path(path).read_text()
    blocks = text.split("[[case]]")
    matches = [
        block
        for block in blocks[1:]
        if tomllib.loads("[[case]]" + block)["case"][0]["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one native case: {name}")
    if not profile:
        return blocks[0] + "[[case]]" + matches[0]
    # Sampling overrides belong to case.run, overriding all suite defaults.
    block = matches[0]
    import re

    for key in (
        "warmup_runs",
        "repeat_runs",
        "measurement_runs",
        "measurement_statistic",
        "check",
        "cooldown_ms",
    ):
        block = re.sub(rf"^{key}\s*=.*\n", "", block, flags=re.MULTILINE)
    block = block.replace(
        "[case.run]\n",
        "[case.run]\n" + "warmup_runs = 0\nrepeat_runs = 1\nmeasurement_runs = 1\n"
        'measurement_statistic = "mean"\ncheck = false\ncooldown_ms = 500\n',
    )
    return blocks[0] + "[[case]]" + block
