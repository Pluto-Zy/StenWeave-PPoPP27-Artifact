"""Strict readers for complete run batches and explicit measurement exports."""

import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

from .catalog import (
    ARCHES,
    BASELINES,
    RESULT_NAMES,
    ROOT,
    STENCILS,
    VARIANTS,
    expected,
    key,
)
from .drstencil import measurements


class Inputs:
    def __init__(self):
        self.files = {}
        self.replacements = []

    def select(self, rows, identity, result):
        """Later explicit inputs replace earlier successful measurements."""
        if identity in rows:
            previous = rows[identity]
            self.replacements.append(
                {
                    "identity": identity,
                    "previous_source": previous["source_file"],
                    "previous_row": previous["source_row"],
                    "selected_source": result["source_file"],
                    "selected_row": result["source_row"],
                }
            )
            print(
                f"Repeated measurement {identity}: using {result['source_file']}:"
                f"{result['source_row']} (later --input wins)",
                file=sys.stderr,
            )
        rows[identity] = result

    def read(self, path):
        path = Path(path).resolve(strict=True)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        name = str(path)
        if name in self.files and self.files[name]["sha256"] != digest:
            raise ValueError(f"Input changed during analysis: {path}")
        self.files[name] = {"sha256": digest, "bytes": len(content)}
        return content

    def json(self, path):
        return json.loads(self.read(path))


def read_rows(path, inputs):
    text = inputs.read(path).decode("utf-8-sig")
    if path.suffix == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a measurement object: {path}:{number}")
                rows.append((number, row))
        return rows
    if path.suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        return [(number, row) for number, row in enumerate(reader, 2)]
    raise ValueError(f"Expected .jsonl or .csv measurements: {path}")


def batch_file(path, inputs, suites=("benchmark",)):
    """Validate collection completion before accepting its results, including explicit files."""
    directory = path if path.is_dir() else path.parent
    status, plan = directory / "status.json", directory / "plan.json"
    if path.is_dir() or status.exists() or plan.exists():
        state, tasks = inputs.json(status), inputs.json(plan)
        if (
            state.get("status") != "completed"
            or tasks.get("suite") not in suites
            or tasks.get("kind") != "timing"
            or tasks.get("schema_version") != 1
        ):
            raise ValueError(
                f"Not a completed {'/'.join(suites)} collection: {directory}"
            )
        ids = [task["id"] for task in tasks["tasks"]]
        if (
            not ids
            or len(set(ids)) != len(ids)
            or state.get("completed") != len(ids)
            or state.get("task_count") != len(ids)
            or state.get("tasks") != dict.fromkeys(ids, "completed")
        ):
            raise ValueError(f"Incomplete collection task list: {directory}")
        result = directory / "results.jsonl"
        if not path.is_dir() and path != result:
            raise ValueError(f"Use the complete batch results.jsonl: {directory}")
        rows = read_rows(result, inputs)
        found = [row.get("task_id") for _, row in rows]
        if len(found) != len(ids) or set(found) != set(ids):
            raise ValueError(f"Results disagree with collection plan: {directory}")
        by_id = {task["id"]: task for task in tasks["tasks"]}
        implementations = dict(
            zip(RESULT_NAMES, (*BASELINES[1:], "StenWeave"), strict=True)
        )
        for _, row in rows:
            task = by_id[row["task_id"]]
            if (
                row.get("arch") != tasks["arch"]
                or row.get("suite") != tasks["suite"]
                or row.get("experiment") != tasks["suite"]
                or row.get("stencil") != task["stencil"]
                or row.get("implementation") != implementations[task["implementation"]]
                or ("kernel_id" in task and row.get("config_id") != task["kernel_id"])
                or row.get("collection_case") != task
            ):
                raise ValueError(
                    f"Measurement disagrees with planned task: {row['task_id']}"
                )
            config = ROOT / task["config"]
            if hashlib.sha256(inputs.read(config)).hexdigest() != task["config_sha256"]:
                raise ValueError(
                    f"Batch uses a different frozen configuration: {task['id']}"
                )
        return result, rows
    return path, read_rows(path, inputs)


def normalize(row, path, number, suite="benchmark"):
    def fail(message):
        raise ValueError(f"{message}: {path}:{number}")

    if (
        row.get("suite", row.get("experiment")) != suite
        or row.get("experiment") != suite
    ):
        fail(f"Expected a {suite} measurement")
    implementation = row.get("implementation")
    if implementation not in (*BASELINES[1:], "StenWeave"):
        fail("Unknown implementation (DRStencil is supplied by fixed constants)")
    arch, stencil = row.get("arch"), row.get("stencil")
    if arch not in ARCHES or stencil not in STENCILS:
        fail("Unknown architecture or stencil")
    variant = row.get("variant") if implementation == "StenWeave" else "native"
    if implementation == "StenWeave" and variant not in VARIANTS:
        fail("Unknown StenWeave variant")
    raw_rate = row.get("gstencils_per_s")
    if isinstance(raw_rate, bool):
        fail("Throughput must be a positive finite number")
    try:
        rate = float(raw_rate)
    except (TypeError, ValueError):
        fail("Missing or invalid throughput")
    if not math.isfinite(rate) or rate <= 0:
        fail("Throughput must be a positive finite number")
    statistic = row.get("statistic")
    # The FlashFFT native runner records median time before deriving throughput.
    if not statistic and implementation == "FlashFFTStencil" and "median_ms" in row:
        statistic = "median_time_throughput"
    statistic = {"mean": "mean_sample_throughput"}.get(statistic, statistic)
    if statistic not in ("mean_sample_throughput", "median_time_throughput"):
        fail("Unknown primary statistic")
    result = {
        "experiment": suite,
        "arch": arch,
        "implementation": implementation,
        "stencil": stencil,
        "variant": variant,
        "gstencils_per_s": rate,
        "statistic": statistic,
        "config_id": row.get("config_id"),
        "config": row.get("config"),
        "source_file": str(path),
        "source_row": number,
    }
    for field in (
        "run_id",
        "task_id",
        "binary_sha256",
        "log_path",
        "log_paths",
        "measurement_source",
        "native_work_factor",
        "normalization_divisor",
    ):
        if field in row:
            result[field] = row[field]
    return result


def load(paths, arches, figures, inputs):
    from . import fig13
    from .size_sweep import expected as size_expected
    from .size_sweep import normalize as size_normalize

    benchmark = bool({"9", "14"}.intersection(figures))
    size_sweep = "10" in figures
    breakdown = "13" in figures
    profile_suites = tuple(
        suite
        for figure, suite in (
            ("11", "roofline"),
            ("12", "time-block-sweep"),
            ("15", "low-level-metrics"),
        )
        if figure in figures
    )
    profile_rows = {}
    suites = tuple(
        name
        for name, enabled in (
            ("benchmark", benchmark),
            ("size-sweep", size_sweep),
            ("breakdown", breakdown),
        )
        if enabled
    )
    breakdown_expected = fig13.expected(arches, inputs.read) if breakdown else {}
    indexed = {}
    for path in paths:
        path = path.expanduser().resolve(strict=True)
        from . import profiles

        directory = path if path.is_dir() else path.parent
        plan_path = directory / "plan.json"
        is_profile = (
            plan_path.exists() and inputs.json(plan_path).get("kind") == "profile"
        ) or path.name == "profiles.jsonl"
        if not is_profile and path.is_file() and path.suffix == ".jsonl":
            preview = read_rows(path, inputs)
            is_profile = bool(preview and preview[0][1].get("kind") == "profile")
        if is_profile:
            if not profile_suites:
                raise ValueError("Profile input requires --figure 11, 12 or 15")
            source, rows = profiles.read_input(path, inputs, profile_suites)
            for number, row in rows:
                if row.get("arch") in ARCHES and row["arch"] not in arches:
                    continue
                result = profiles.normalize(row, source, number, inputs, profile_suites)
                identity = result["arch"], result["experiment"], result["task_id"]
                inputs.select(profile_rows, identity, result)
            continue
        source, rows = batch_file(path, inputs, suites)
        if not rows:
            raise ValueError(f"Empty measurement source: {source}")
        for number, row in rows:
            if row.get("arch") in ARCHES and row["arch"] not in arches:
                continue
            suite = row.get("experiment")
            if suite not in suites:
                raise ValueError(f"Unexpected experiment {suite!r}: {source}:{number}")
            result = (
                fig13.normalize(row, source, number, breakdown_expected)
                if suite == "breakdown"
                else normalize(row, source, number, suite)
            )
            if suite == "size-sweep":
                result = size_normalize(result, row)
            identity = key(result)
            inputs.select(indexed, identity, result)
    if benchmark:
        for result in measurements(arches):
            indexed[key(result)] = result
    coverage = expected(arches, figures, inputs.read) if benchmark else {}
    if size_sweep:
        coverage.update(size_expected(arches, inputs.read, selected=True))
    if breakdown:
        coverage.update(breakdown_expected)
    missing = coverage.keys() - indexed.keys()
    if missing:
        lines = [" / ".join(map(str, k)) for k in sorted(missing)]
        raise ValueError(
            f"Missing {len(missing)} required measurements:\n  " + "\n  ".join(lines)
        )
    # Unsupported baseline cells have no records; never turn absent data into 0 or 1.
    supported = expected(arches, ("14",), inputs.read) if benchmark else {}
    if size_sweep:
        supported.update(size_expected(arches, inputs.read, selected=False))
    if breakdown:
        supported.update(breakdown_expected)
    if indexed.keys() - supported.keys():
        raise ValueError(
            f"Unsupported measurement combinations: {sorted(indexed.keys() - supported.keys())}"
        )
    for identity, row in indexed.items():
        specification = supported[identity]
        for field in ("statistic", "config_id"):
            if field in specification and row.get(field) != specification[field]:
                raise ValueError(f"{field} does not match frozen selection: {identity}")
        if row.get("experiment") == "size-sweep":
            from .size_sweep import validate

            validate(row, specification)
    for suite in profile_suites:
        from .profiles import expected as profile_expected

        required = profile_expected(arches, inputs.read, suite)
        found = {(arch, task) for arch, kind, task in profile_rows if kind == suite}
        missing_profiles = required.keys() - found
        if missing_profiles:
            raise ValueError(
                f"Missing {len(missing_profiles)} required {suite} profiles: {sorted(missing_profiles)}"
            )
    for row in profile_rows.values():
        identity = (*key(row), row["experiment"])
        if row["experiment"] == "time-block-sweep":
            identity += (row["time_block_size"],)
        elif row["experiment"] == "low-level-metrics":
            identity += (row["collection"],)
        indexed[identity] = row
    return indexed


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
