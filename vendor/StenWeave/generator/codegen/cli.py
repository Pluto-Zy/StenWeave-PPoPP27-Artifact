"""Command line interface to the standalone offline generator."""

import argparse
import json
from pathlib import Path

from codegen.artifact import build_artifact
from codegen.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--backend", choices=["cuda", "sass"])
    parser.add_argument("--arch", choices=["sm80", "sm90"])
    parser.add_argument("--profile", choices=["default", "breakdown"])
    parser.add_argument("--list", action="store_true", help="validate and print expanded specs")

    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    name, specs = load_config(args.config)
    specs = [
        s
        for s in specs
        if (args.backend is None or s.backend == args.backend)
        and (args.arch is None or s.arch == args.arch)
    ]
    specs = [
        s
        for s in specs
        if args.profile is None or ((s.stage is None) == (args.profile == "default"))
    ]

    if args.list:
        print(json.dumps([dict(id=s.id, **s.metadata()) for s in specs], indent=2))
        return

    if args.output is None:
        parser.error("--output is required when building")

    build_artifact(name, specs, args.output.resolve(), args.config, args.jobs)
