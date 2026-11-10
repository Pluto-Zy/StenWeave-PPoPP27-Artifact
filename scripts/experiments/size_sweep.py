"""Figure 10 identities and coverage, independent of builds and GPU access."""

import json

from run.suites.size_sweep import (
    SIZES_2D,
    SIZES_3D,
    STENCILS,
    participates,
)

from .catalog import ROOT, tomllib

ARCHES = ("sm80",)
SERIES = (
    ("AN5D", "native"),
    ("EBISU", "native"),
    ("LoRAStencil", "native"),
    ("StenWeave", "SASS"),
    ("StenWeave", "SASS-ILP"),
)


def sizes(stencil):
    return SIZES_2D if "2d" in stencil else SIZES_3D


def series_for(stencil):
    return tuple(series for series in SERIES if participates(series[0], stencil))


def integer(value):
    if isinstance(value, bool) or not str(value).isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid size-sweep dimension: {value!r}")
    return int(value)


def normalize(result, raw):
    """Require native geometry; compare every available source of the side length."""
    if result["arch"] not in ARCHES or result["stencil"] not in STENCILS:
        raise ValueError("Figure 10 supports only A100 and the four radius-1 stencils")
    if (result["implementation"], result["variant"]) not in series_for(
        result["stencil"]
    ):
        raise ValueError(
            f"Unsupported Figure 10 implementation or variant for {result['stencil']}"
        )
    rank = 2 if "2d" in result["stencil"] else 3
    widths = []
    for source in (raw, raw.get("collection_case", {})):
        for field in ("domain_width", "width"):
            if field in source:
                widths.append(integer(source[field]))
        if "rank" in source and integer(source["rank"]) != rank:
            raise ValueError("Size-sweep rank disagrees with stencil")
    if "dimensions" in raw:
        dimensions = raw["dimensions"]
        if isinstance(dimensions, str):
            dimensions = json.loads(dimensions)
        if not isinstance(dimensions, list) or len(dimensions) != rank:
            raise ValueError("Invalid size-sweep native dimensions")
        widths.extend(integer(value) for value in dimensions)
    if "native" in raw:
        domain = raw["native"]["domain"]
        widths.extend(integer(domain[name]) for name in ("width", "height"))
        depth = integer(domain["depth"])
        if rank == 3:
            widths.append(depth)
        elif depth != 1:
            raise ValueError("2D size-sweep native depth must be one")
        if domain["elements"] != widths[-1] ** rank:
            raise ValueError(
                "Size-sweep native element count disagrees with dimensions"
            )
    if not widths or len(set(widths)) != 1:
        raise ValueError(f"Missing or inconsistent size-sweep side lengths: {widths}")
    result.update(domain_width=widths[0], rank=rank)
    for field in ("steps", "iterations", "repeats", "case", "parameters"):
        if field in raw:
            result[field] = raw[field]
    if "native" in raw:
        result["sampling"] = raw["native"]["run"]
    return result


def expected(arches, read, *, selected):
    if not set(arches).intersection(ARCHES):
        raise ValueError("Figure 10 is an A100-only experiment; select --arch sm80")
    path = ROOT / "configs/stenweave/runner/sm80/size-sweep.toml"
    config = tomllib.loads(read(path).decode())
    result, inventory = {}, {stencil: set() for stencil in STENCILS}
    for case in config["case"]:
        stencil, suffix = case["name"].split("-sm80-")
        if stencil not in inventory:
            raise ValueError(f"Unknown native size-sweep stencil: {stencil}")
        variant, width = suffix.rsplit("-", 1)
        variant = {"sass": "SASS", "sass-ilp": "SASS-ILP"}[variant]
        width = integer(width)
        rank = 2 if "2d" in stencil else 3
        domain = case["run"]["domain"]
        if domain != dict.fromkeys(("x", "y", "z")[:rank], width):
            raise ValueError(
                f"Native size-sweep case is not square/cubic: {case['name']}"
            )
        inventory[stencil].add(width)
        identity = "sm80", "StenWeave", stencil, variant, width
        if identity in result:
            raise ValueError(f"Duplicate native size-sweep case: {identity}")
        result[identity] = {
            "config_id": case["kernel"]["id"],
            "statistic": "mean_sample_throughput",
            "case": case["name"],
            "sampling": {**config["defaults"]["run"], **case["run"]},
        }
    for implementation, _ in SERIES[:3]:
        path = ROOT / f"configs/baselines/{implementation}/sm80.toml"
        baseline = tomllib.loads(read(path).decode())
        for stencil, native_widths in inventory.items():
            if not participates(implementation, stencil):
                continue
            widths = native_widths | set(sizes(stencil))
            c = baseline["stencils"][stencil]
            config_id = stencil
            if implementation == "AN5D":
                cap = c.get("maxrregcount", 0)
                config_id = c["variant"] + (f"-rreg{cap}" if cap else "")
            specification = {
                "statistic": baseline["statistic"],
                "config_id": config_id,
                "repeats": baseline["repeats"],
                "parameters": c,
            }
            if implementation == "AN5D":
                specification["steps"] = baseline["steps"]
            elif implementation == "EBISU":
                specification["steps"] = c["tstep"]
            else:
                specification["iterations"] = c["iterations"]
            for width in widths:
                result[("sm80", implementation, stencil, "native", width)] = (
                    specification
                )
    if selected:
        for stencil, widths in inventory.items():
            values = sizes(stencil)
            if (
                not values
                or len(set(values)) != len(values)
                or any(
                    type(value) is not int or value not in widths for value in values
                )
            ):
                raise ValueError(
                    f"StenWeave configurations do not yet cover selected Figure 10 sizes: {stencil}"
                )
        result = {k: v for k, v in result.items() if k[-1] in sizes(k[2])}
        required = {
            ("sm80", implementation, stencil, variant, width)
            for stencil in STENCILS
            for implementation, variant in series_for(stencil)
            for width in sizes(stencil)
        }
        if result.keys() != required:
            raise ValueError(
                "Native configuration does not cover the selected size sweep"
            )
    return result


def validate(row, specification):
    for field in ("steps", "iterations", "repeats", "case", "parameters"):
        if field not in row:
            continue
        actual = row[field]
        expected = specification.get(field)
        if field in ("steps", "iterations", "repeats"):
            actual = integer(actual)
        if actual != expected:
            raise ValueError(f"Size-sweep {field} differs from frozen selection")
    if "sampling" in row:
        for field in (
            "repeat_runs",
            "measurement_runs",
            "measurement_statistic",
            "cooldown_ms",
            "warmup_runs",
            "time_steps",
            "check",
        ):
            if row["sampling"].get(field) != specification["sampling"].get(field):
                raise ValueError(f"Size-sweep {field} differs from frozen sampling")
