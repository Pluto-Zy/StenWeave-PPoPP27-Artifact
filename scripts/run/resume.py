"""Atomic task checkpoints and exclusive ownership of resumable collections."""

import contextlib
import datetime
import fcntl
import json
import os
import platform
import socket
import sys
import tempfile
from pathlib import Path

from . import process
from .suites.common import ROOT, sha

VERSION = 1


def write_text(path, content):
    """Publish a complete file, including across interruption during a write."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def snapshot(tasks, builds, environment):
    paths = [ROOT / "scripts/run.py"]
    paths += sorted((ROOT / "scripts/run").rglob("*.py"))
    paths += sorted((ROOT / "scripts/adapters").glob("*.py"))
    paths += [ROOT / task["config"] for task in tasks]
    return {
        "version": VERSION,
        "root": str(ROOT),
        "host": socket.gethostname(),
        "python": [sys.executable, platform.python_version()],
        "files": {**{str(p.relative_to(ROOT)): sha(p) for p in paths}, **builds},
        # Utilization and allocated memory change during normal use.
        "gpu": [field.strip() for field in environment["gpu"].split(",")[1:5]],
        "ncu_version": environment.get("ncu_version"),
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "LD_LIBRARY_PATH", "CUDA_DEVICE_ORDER")
        },
    }


def verify_snapshot(saved, current):
    if saved != current:
        changed = [key for key in current if saved.get(key) != current[key]]
        if "files" in changed:
            paths = sorted(set(saved.get("files", {})) | set(current["files"]))
            files = [
                path
                for path in paths
                if saved.get("files", {}).get(path) != current["files"].get(path)
            ]
            changed.append("files: " + ", ".join(files[:5]))
        raise ValueError(
            "Cannot resume: collection inputs/environment changed "
            f"({', '.join(changed)}); use a fresh batch directory"
        )


@contextlib.contextmanager
def batch_lock(folder):
    """Keep the lock in collectors too, so killing the parent cannot steal a run."""
    with (folder / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Collection is still active: {folder}") from error
        active = folder / "active-process.json"
        if active.exists():
            owner = read_json(active)
            if owner["host"] != socket.gethostname():
                raise ValueError("Active collector belongs to another host")
            try:
                os.killpg(owner["pgid"], 0)
            except ProcessLookupError:
                active.unlink()
            else:
                raise ValueError(
                    f"Collector process group {owner['pgid']} is still active"
                )
        token = process.BATCH.set((folder, lock.fileno()))
        try:
            yield
        finally:
            process.BATCH.reset(token)


def task_files(folder, task):
    destination = folder / "tasks" / task["id"]
    paths = [
        p for p in destination.rglob("*") if p.is_file() and p.name != "checkpoint.json"
    ]
    outer_log = destination.parent / f"{task['id']}.log"
    if outer_log.is_file():
        paths.append(outer_log)
    return {str(p.relative_to(folder)): sha(p) for p in sorted(paths)}


def save_task(folder, task, rows):
    write_json(
        folder / "tasks" / task["id"] / "checkpoint.json",
        {
            "version": VERSION,
            "task": task,
            "rows": rows,
            "files": task_files(folder, task),
        },
    )


def load_task(folder, task, suite, arch, profiling):
    """Only trust a completed checkpoint with unchanged, reparsable raw files."""
    from . import execution, profile

    try:
        checkpoint = folder / "tasks" / task["id"] / "checkpoint.json"
        if not checkpoint.is_file():
            return None, "no completed task checkpoint"
        saved = read_json(checkpoint)
        if saved["version"] != VERSION or saved["task"] != task:
            raise ValueError("task/checkpoint identity changed")
        if not saved["files"] or saved["files"] != task_files(folder, task):
            raise ValueError("task files missing or changed")
        destination = folder / "tasks" / task["id"]
        rows = (
            profile.read_result(task, suite, arch, destination)
            if profiling
            else execution.read_result(task, arch, destination)
        )
        if rows != saved["rows"]:
            raise ValueError("checkpoint disagrees with native results")
        return rows, None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return None, str(error)


def archive_task(folder, task):
    """Retain every unsuccessful attempt before reusing its canonical task path."""
    destination = folder / "tasks" / task["id"]
    outer_log = destination.parent / f"{task['id']}.log"
    if not destination.exists() and not outer_log.exists():
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = folder / "attempts" / task["id"] / stamp
    archive.mkdir(parents=True, exist_ok=False)
    if destination.exists():
        destination.rename(archive / "task")
    if outer_log.exists():
        outer_log.rename(archive / "wrapper.log")


def enrich(rows, task, plan, folder):
    return [
        {
            **row,
            "suite": plan["suite"],
            "experiment": plan["suite"],
            "task_id": task["id"],
            "run_id": plan["run_id"],
            "collection_case": task,
            "source_results": str(folder / "tasks" / task["id"] / "results.jsonl")
            if plan["kind"] == "timing"
            else None,
        }
        for row in rows
    ]
