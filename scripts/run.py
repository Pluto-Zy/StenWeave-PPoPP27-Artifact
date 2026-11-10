"""Collect benchmark or NCU data using frozen implementation configurations."""

import argparse
import datetime
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run import execution, profile, resume
from run.resume import write_json
from run.suites.common import IMPLEMENTATIONS, ROOT, sha

SUITES = (
    "benchmark",
    "size-sweep",
    "roofline",
    "time-block-sweep",
    "breakdown",
    "low-level-metrics",
)


def make_plan(args):
    module = importlib.import_module(f"run.suites.{args.suite.replace('-', '_')}")
    if args.arch not in module.ARCHES:
        raise ValueError(f"{args.suite} supports only {', '.join(module.ARCHES)}")
    selected = []
    for name in args.impl or ["all"]:
        if name == "all":
            selected.extend(module.IMPLEMENTATION_NAMES)
        elif name == "baselines":
            selected.extend(n for n in module.IMPLEMENTATION_NAMES if n != "stenweave")
        else:
            if name not in module.IMPLEMENTATION_NAMES:
                raise ValueError(f"{name} does not participate in {args.suite}")
            selected.append(name)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("No implementations selected")
    planned = (
        module.plan(args.arch, implementations=selected)
        if args.suite == "size-sweep"
        else module.plan(args.arch)
    )
    tasks = [task for task in planned if task["implementation"] in selected]
    for field, values in (("stencil", args.stencil), ("id", args.case)):
        if values:
            missing = set(values) - {task[field] for task in tasks}
            if missing:
                raise ValueError(f"Unknown {field} selection: {sorted(missing)}")
            tasks = [task for task in tasks if task[field] in values]
    if not tasks or len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Task list is empty or contains duplicate IDs")
    return module.PROFILE, tasks


def collect_batch(args, plan, folder, profiling, stamp):
    import time

    tasks = plan["tasks"]
    state = {
        "status": "running",
        "completed": 0,
        "task_count": len(tasks),
        "tasks": {},
        "reused": 0,
    }
    session = {
        "resume": bool(args.resume),
        "utc": stamp,
        "argv": sys.argv,
        "timeout": args.timeout,
        "reused": [],
        "executed": [],
    }
    session_dir = folder / "sessions" / stamp
    current = None
    started = time.monotonic()
    # Verify compatibility before changing any existing status or outputs.
    if not args.resume:
        write_json(folder / "plan.json", plan)
        write_json(folder / "status.json", state)
    try:
        builds = {}
        for task in tasks:
            builds.update(execution.preflight(task, args.arch))
            if profiling:
                profile.commands(
                    task,
                    args.arch,
                    args.device,
                    folder / "tasks" / task["id"],
                    args.ncu_resolved,
                    args.suite,
                    verify=True,
                )
        sys.path.insert(0, str(ROOT / "scripts/adapters"))
        from ebisu import device_info

        environment = {
            "gpu": device_info(str(args.device), args.arch),
            "utc": stamp,
            "python": sys.version,
            "argv": sys.argv,
        }
        if profiling:
            environment["ncu_version"] = subprocess.check_output(
                [args.ncu_resolved, "--version"], text=True, timeout=30
            )
        inputs = resume.snapshot(tasks, builds, environment)
        if args.resume:
            resume.verify_snapshot(resume.read_json(folder / "resume.json"), inputs)
        else:
            write_json(folder / "environment.json", environment)
            write_json(folder / "resume.json", inputs)
    except BaseException as error:
        if not args.resume:
            state.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=str(error),
            )
            write_json(folder / "status.json", state)
        raise
    session_dir.mkdir(parents=True, exist_ok=False)
    session["environment"] = environment
    name = "profiles.jsonl" if profiling else "results.jsonl"
    if (folder / name).exists():
        (folder / name).rename(session_dir / name)
    write_json(folder / "status.json", state)
    rows = []
    try:
        for number, task in enumerate(tasks, 1):
            current = task["id"]
            measured, reason = (
                resume.load_task(folder, task, args.suite, args.arch, profiling)
                if args.resume
                else (None, None)
            )
            if measured is not None:
                print(f"[{number}/{len(tasks)}] REUSE {current}", flush=True)
                state["reused"] += 1
                session["reused"].append(current)
            else:
                if args.resume:
                    detail = (
                        f" ({reason})"
                        if reason and reason != "no completed task checkpoint"
                        else ""
                    )
                    print(f"[{number}/{len(tasks)}] RUN {current}{detail}", flush=True)
                    resume.archive_task(folder, task)
                else:
                    print(f"[{number}/{len(tasks)}] {current}", flush=True)
                state["tasks"][current] = "running"
                write_json(folder / "status.json", state)
                if sha(ROOT / task["config"]) != task["config_sha256"]:
                    raise ValueError(
                        f"Configuration changed during batch: {task['config']}"
                    )
                destination = folder / "tasks" / current
                session["executed"].append(current)
                if profiling:
                    measured = profile.collect(
                        task,
                        args.suite,
                        args.arch,
                        args.device,
                        destination,
                        args.ncu_resolved,
                        args.timeout,
                    )
                else:
                    measured = execution.collect(
                        task,
                        args.suite,
                        args.arch,
                        args.device,
                        destination,
                        args.timeout,
                    )
                resume.save_task(folder, task, measured)
            rows.extend(resume.enrich(measured, task, plan, folder))
            state["tasks"][current] = "completed"
            state["completed"] += 1
            write_json(folder / "status.json", state)
            current = None
        resume.write_text(
            folder / name,
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        )
        state["status"] = "completed"
        write_json(folder / "status.json", state)
    except BaseException as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        if current:
            state["tasks"][current] = state["status"]
        write_json(folder / "status.json", state)
        raise
    finally:
        session.update(
            status=state["status"], elapsed_seconds=time.monotonic() - started
        )
        write_json(session_dir / "session.json", session)
    print(
        f"Completed {len(tasks)} tasks ({state['reused']} reused): {folder}", flush=True
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=SUITES, help="experiment suite; required for a new batch"
    )
    parser.add_argument(
        "--arch",
        choices=("sm80", "sm90"),
        help="GPU architecture; required for a new batch",
    )
    parser.add_argument(
        "--device",
        type=int,
        help="physical CUDA device ordinal; required for a new batch",
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        choices=("all", "baselines", *IMPLEMENTATIONS),
        help="implementations to run (default: all participating implementations)",
    )
    parser.add_argument(
        "--stencil", action="append", help="repeat to filter stencil names"
    )
    parser.add_argument(
        "--case", action="append", help="repeat to select exact task IDs from dry-run"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="fresh batch directory (default: runs/<suite>-<arch>)",
    )
    parser.add_argument(
        "--ncu",
        help="Nsight Compute executable; default NCU, PATH or CUDA installation",
    )
    parser.add_argument(
        "--timeout", type=int, default=7200, help="seconds per task (default: 7200)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show tasks without executing; with --resume, inspect checkpoint reuse candidates",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="continue a batch in place; inherit its suite, architecture, selection and device",
    )
    args = parser.parse_args(argv)
    saved_plan = None
    if args.resume:
        if any(
            getattr(args, key) is not None
            for key in (
                "suite",
                "arch",
                "device",
                "impl",
                "stencil",
                "case",
                "output_dir",
                "ncu",
            )
        ):
            parser.error(
                "--resume inherits suite/arch/device/selection/NCU; only --timeout and --dry-run may be added"
            )
        try:
            args.resume = args.resume.resolve(strict=True)
            saved_plan = resume.read_json(args.resume / "plan.json")
            if saved_plan.get("resume_version") != resume.VERSION:
                raise ValueError(
                    "This batch predates resumable collections; start a fresh batch"
                )
            for key in ("suite", "arch", "device", "ncu"):
                setattr(args, key, saved_plan[key])
            for key in ("impl", "stencil", "case"):
                setattr(args, key, saved_plan["selection"][key])
            args.output_dir = args.resume
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    elif any(getattr(args, key) is None for key in ("suite", "arch", "device")):
        parser.error("--suite, --arch and --device are required for a new batch")
    if args.device < 0 or args.timeout < 1:
        parser.error("--device must be nonnegative and --timeout must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("unset CUDA_VISIBLE_DEVICES; --device selects the physical device")
    try:
        profiling, tasks = make_plan(args)
        if args.ncu and not profiling:
            raise ValueError("--ncu applies only to profiling suites")
        stamp = datetime.datetime.now(
            datetime.timezone.utc  # noqa: UP017
        ).strftime("%Y%m%dT%H%M%S.%fZ")
        folder = (
            args.output_dir or ROOT / "runs" / f"{args.suite}-{args.arch}"
        ).resolve()
        ncu = args.ncu or os.environ.get("NCU") or shutil.which("ncu")
        if profiling:
            if ncu:
                resolved = shutil.which(os.path.expanduser(ncu))
                if not resolved and not args.dry_run:
                    raise ValueError(f"NCU executable not found: {ncu}")
                ncu = str(Path(resolved).resolve()) if resolved else ncu
            else:
                candidate = Path("/usr/local/cuda/bin/ncu")
                ncu = str(candidate) if candidate.is_file() else "ncu"
                if not candidate.is_file() and not args.dry_run:
                    raise ValueError("ncu not found; use --ncu or NCU")
        args.ncu_resolved = ncu
        plan = {
            "schema_version": 1,
            "resume_version": resume.VERSION,
            "selection": {
                key: getattr(args, key) for key in ("impl", "stencil", "case")
            },
            "ncu": ncu if profiling else None,
            "run_id": folder.name,
            "suite": args.suite,
            "arch": args.arch,
            "device": args.device,
            "kind": "profile" if profiling else "timing",
            "tasks": tasks,
        }
        suite_module = importlib.import_module(
            f"run.suites.{args.suite.replace('-', '_')}"
        )
        if hasattr(suite_module, "exclusions"):
            selected = {task["implementation"] for task in tasks}
            stencils = {task["stencil"] for task in tasks}
            plan["excluded_cases"] = [
                row
                for row in suite_module.exclusions()
                if not args.case
                and row["implementation"].lower() in selected
                and row["stencil"] in stencils
            ]
        for task in tasks:
            destination = folder / "tasks" / task["id"]
            if profiling:
                native, command, export, _ = profile.commands(
                    task,
                    args.arch,
                    args.device,
                    destination,
                    ncu,
                    args.suite,
                    verify=False,
                )
                task.update(
                    command=command,
                    export_command=export,
                    launch_selection={
                        key: native[key]
                        for key in ("selection", "launch_skip", "launch_count")
                    },
                )
            else:
                task["command"] = execution.command(
                    task, args.suite, args.arch, args.device, destination
                )
        if saved_plan is not None and saved_plan != plan:
            raise ValueError("Collection plan or commands changed; use a fresh batch")
        if args.dry_run:
            if args.resume:
                plan["recovery"] = {
                    task["id"]: "REUSE"
                    if resume.load_task(folder, task, args.suite, args.arch, profiling)[
                        0
                    ]
                    is not None
                    else "RUN"
                    for task in tasks
                }
                plan["recovery_note"] = (
                    "candidate actions; execution also verifies inputs and GPU identity"
                )
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        if not args.resume:
            try:
                folder.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise ValueError(
                    f"Output path already exists: {folder}. "
                    "If the batch was interrupted, continue it with --resume "
                    f"{folder}; otherwise choose a fresh path with --output-dir."
                ) from error
            (folder / "tasks").mkdir()
        with resume.batch_lock(folder):
            collect_batch(args, plan, folder, profiling, stamp)
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        StopIteration,
        subprocess.SubprocessError,
    ) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Collection interrupted.", file=sys.stderr)
        sys.exit(130)
