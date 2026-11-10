"""Analyze experiment results and write paper PDFs directly to figures/.

Examples:
  python3 scripts/experiment.py
  python3 scripts/experiment.py --arch sm80
  python3 scripts/experiment.py --arch sm90 --figure 11
  python3 scripts/experiment.py --input runs/a100 --input runs/h100 --figure 9 14
  python3 scripts/experiment.py --input runs/size-sweep-sm80 --figure 10
  python3 scripts/experiment.py --input runs/tb-a100 --input runs/tb-h100 --figure 12
  python3 scripts/experiment.py --input runs/breakdown-a100 --input runs/breakdown-h100 --figure 13
  python3 scripts/experiment.py --input runs/low-level-a100 --figure 15

By default, read runs/<suite>-<arch> for the selected figures and architectures.
All figures supported by the selected architectures are generated unless --figure
is specified. Use --input to select custom complete batches or CSV/JSONL exports
instead of the default directories.
DRStencil is supplied by fixed reference values in experiments/drstencil.py.
By default, only PDFs are written at the output root. Use --export-data to also
write one numerical CSV and a source manifest per figure in details/.
Only the selected figures are updated when the output directory already exists.
When inputs overlap, later --input values replace earlier measurements of the
same case. Profile batches can be copied with their tasks/ directory for analysis.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import platform
import sys
import tempfile
from pathlib import Path

from experiments import fig09, fig10, fig11, fig12, fig13, fig14, fig15
from experiments.catalog import ARCHES, ROOT
from experiments.data import Inputs, load, write_json
from experiments.drstencil import SOURCES

FIGURES = {
    "9": fig09,
    "10": fig10,
    "11": fig11,
    "12": fig12,
    "13": fig13,
    "14": fig14,
    "15": fig15,
}


def supported_arches(number, arches):
    return tuple(a for a in arches if a in getattr(FIGURES[number], "ARCHES", ARCHES))


def default_inputs(arches, figures):
    paths = dict.fromkeys(
        ROOT / "runs" / f"{getattr(FIGURES[number], 'SUITE', 'benchmark')}-{arch}"
        for number in figures
        for arch in supported_arches(number, arches)
    )
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise ValueError(
            "Missing experiment result directories: "
            + ", ".join(missing)
            + ". Run the corresponding suites or use --input for custom result paths."
        )
    return list(paths)


def generate(measurements, arches, figures, inputs, versions, args, staged):
    details = staged / "details"
    if args.export_data:
        details.mkdir()
    stamp = datetime.datetime.now(
        datetime.timezone.utc  # noqa: UP017
    ).isoformat()
    for number in figures:
        name = f"fig{int(number):02d}"
        module = FIGURES[number]
        figure_arches = supported_arches(number, arches)
        suite = getattr(module, "SUITE", "benchmark")
        selected = {
            identity: row
            for identity, row in measurements.items()
            if row["arch"] in figure_arches
            and row.get("experiment", "benchmark") == suite
        }
        result = module.analyze(selected, figure_arches)
        if not args.tables_only:
            module.plot(result, figure_arches, staged)
        if not args.export_data:
            continue
        module.export(result, figure_arches, details)
        artifacts = {}
        for path in [*staged.glob(f"{name}.*"), details / f"{name}.csv"]:
            content = path.read_bytes()
            artifacts[path.relative_to(staged).as_posix()] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        write_json(
            details / f"{name}-inputs.json",
            {
                "schema_version": 1,
                "status": "completed",
                "created_utc": stamp,
                "figure": number,
                "arches": figure_arches,
                "suite": suite,
                "excluded_combinations": (
                    module.exclusions() if hasattr(module, "exclusions") else []
                ),
                "tables_only": args.tables_only,
                "argv": args.invocation,
                "versions": versions,
                "measurement_count": len(selected),
                "measurements_by_arch": {
                    arch: sum(row["arch"] == arch for row in selected.values())
                    for arch in figure_arches
                },
                "drstencil": (
                    {arch: SOURCES[arch] for arch in figure_arches}
                    if suite == "benchmark"
                    else {}
                ),
                "files": inputs.files,
                "replacements": inputs.replacements,
                "artifacts": artifacts,
                "policy": {
                    "units": "percent; results/byte"
                    if number in ("11", "12")
                    else "percent"
                    if number == "15"
                    else "GStencil/s",
                    "rounding": "display only; tables retain full float precision",
                    "missing_cells": (
                        "all selected size-sweep points required"
                        if number == "10"
                        else "two independent profiles for each implementation and stencil required"
                        if number == "15"
                        else "all frozen cumulative breakdown stages required"
                        if number == "13"
                        else "all frozen temporal-depth profiles required"
                        if number == "12"
                        else "all frozen roofline profiles required; unsupported baseline cells absent"
                        if number == "11"
                        else "unsupported baseline cells excluded from geometric means"
                    ),
                    "formula": (
                        "mean of two independently normalized instruction/stall fractions and active-cycle utilization percentages"
                        if number == "15"
                        else "100 * (DMUL + DFMA) / elapsed_seconds / min(FP64 peak, intensity * shared peak)"
                        if number in ("11", "12")
                        else "SASS-ILP / each baseline"
                        if number == "9"
                        else (
                            "native effective throughput; EBISU is evaluated only in 2D"
                            if number == "10"
                            else "measured effective throughput across cumulative CUDA breakdown stages"
                            if number == "13"
                            else "CUDA / best of six baselines; SASS / CUDA; SASS-ILP / SASS"
                        )
                    ),
                    "source_selection": "complete collection batches or measurement exports",
                    "repeated_measurements": "later --input wins; no averaging or fastest selection",
                },
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--figure",
        nargs="+",
        choices=tuple(FIGURES),
        help="figures to generate (default: all supported by --arch; SM90 excludes 10 and 15)",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=ARCHES,
        default=list(ARCHES),
        help="result architectures to include (default: sm80 sm90)",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="complete batch or CSV/JSONL; repeat to combine; later input wins on overlap (default: runs/<suite>-<arch>)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures",
        help="PDF destination (default: figures/); update only the selected figures",
    )
    parser.add_argument(
        "--export-data",
        action="store_true",
        help="also export one numerical CSV and a source manifest per figure in details/",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="export numerical data without PDFs (implies --export-data)",
    )
    args = parser.parse_args(argv)
    args.export_data = args.export_data or args.tables_only
    args.invocation = list(sys.argv[1:] if argv is None else argv)
    arches = tuple(dict.fromkeys(args.arch))
    figures = (
        tuple(dict.fromkeys(args.figure))
        if args.figure
        else tuple(number for number in FIGURES if supported_arches(number, arches))
    )
    output = args.output_dir.expanduser().resolve()
    try:
        for number in figures:
            if not supported_arches(number, arches):
                supported = ", ".join(getattr(FIGURES[number], "ARCHES", ARCHES))
                raise ValueError(f"Figure {number} requires --arch {supported}")
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output is not a directory: {output}")
        inputs = Inputs()
        measurements = load(
            args.input if args.input is not None else default_inputs(arches, figures),
            arches,
            figures,
            inputs,
        )
        for path in [
            Path(__file__).resolve(),
            *sorted((ROOT / "scripts/experiments").glob("*.py")),
            *([ROOT / "scripts/run/suites/size_sweep.py"] if "10" in figures else []),
            *(
                [
                    ROOT / path
                    for path in (
                        "scripts/run/profile.py",
                        "scripts/run/suites/roofline.py",
                        "scripts/run/suites/time_block_sweep.py",
                        "scripts/run/suites/low_level_metrics.py",
                        "scripts/run/suites/breakdown.py",
                        "scripts/run/suites/common.py",
                        "scripts/run/implementations/profile_targets.py",
                    )
                ]
                if {"11", "12", "15"}.intersection(figures)
                else []
            ),
        ]:
            inputs.read(path)
        versions = {"python": platform.python_version()}
        if not args.tables_only:
            from experiments import plotting  # noqa: F401

            versions.update(
                {
                    name: importlib.metadata.version(name)
                    for name in ("matplotlib", "numpy")
                }
            )
        # Generate and validate everything before publishing to the stable destination.
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".experiment-", dir=output.parent
        ) as temporary:
            staged = Path(temporary)
            generate(measurements, arches, figures, inputs, versions, args, staged)
            destinations = [
                (path, output / path.relative_to(staged))
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            ]
            # Avoid overwriting an explicit input with a generated file.
            for _, target in destinations:
                if str(target.resolve()) in inputs.files:
                    raise ValueError(f"Output would overwrite an input: {target}")
                if target.exists() and not target.is_file():
                    raise ValueError(f"Output file is blocked by a directory: {target}")
            for source, target in destinations:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
        print(
            f"Completed Figures {', '.join(figures)} using {len(measurements)} measurements."
        )
        if not args.tables_only:
            for number in figures:
                print(f"PDF: {output / f'fig{int(number):02d}.pdf'}")
        if args.export_data:
            print(f"Calculation details: {output / 'details'}")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        RuntimeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
