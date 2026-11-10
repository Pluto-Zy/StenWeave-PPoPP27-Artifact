"""Render a quickcheck's saved results for a terminal, without GPU access."""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from run.suites.common import IMPLEMENTATIONS


def interactive():
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def paint(text, tone, enabled=True):
    return f"\033[{tone}m{text}\033[0m" if enabled else text


def print_toolchain(toolchain):
    if not toolchain:
        return
    color = interactive() and "NO_COLOR" not in os.environ
    print("\n  " + paint("Toolchain", "1;36", color), flush=True)
    for label, info in toolchain.items():
        print(f"  {label:<12} {info['version']}", flush=True)
        print("               " + paint(info["path"], "2", color), flush=True)
    print(flush=True)


def failure_details(folder, plan, step, error):
    """Keep command dumps in JSON/logs and point the terminal to the real error log."""
    raw = str(error)
    if isinstance(error, subprocess.CalledProcessError):
        reason = f"Command exited with status {error.returncode}."
    elif isinstance(error, subprocess.TimeoutExpired):
        reason = f"Step timed out after {error.timeout}s."
    elif isinstance(error, KeyboardInterrupt):
        reason = "Interrupted by user."
    elif match := re.search(r"returned non-zero exit status (\d+)", raw):
        reason = f"Command exited with status {match[1]}."
    elif raw.startswith("Command "):
        reason = "Command failed; see the error log for details."
    else:
        reason = raw.splitlines()[0][:240] if raw else "Interrupted by user."
    result = {"reason": reason}
    if step.startswith("profile-"):
        target = folder / "profiles" / step.removeprefix("profile-")
        command = getattr(error, "cmd", ())
        filename = "export.log" if "--import" in command else "profile.log"
        log = target / filename
    elif step in {item["id"] for item in plan["steps"]}:
        log = folder / "logs" / f"{step}.log"
    else:
        return result
    result["log"] = str(log)
    if log.is_file():
        # Build wrappers record the exact nested compiler/generator log.
        text = log.read_text(errors="replace")
        nested = re.findall(r"\bfailed(?: \([^)]+\))?; see ([^\r\n]+)", text)
        if nested and (candidate := Path(nested[-1])).is_file():
            result.update(log=str(candidate), wrapper_log=str(log))
    return result


def step_label(name):
    if name == "plot-fig09":
        return "Plot Fig.9 from historical CSV"
    phase, implementation = name.split("-", 1)
    title = {"build": "Build", "timing": "Timing", "checks": "CPU check"}[phase]
    return f"{title} {IMPLEMENTATIONS[implementation]}"


class StepProgress:
    """Keep one live line on terminals; emit ordinary lines in redirected logs."""

    def __init__(self, index, total, label):
        self.index, self.total, self.label = index, total, label
        self.animated = interactive()
        self.color = self.animated and "NO_COLOR" not in os.environ
        self.stop = threading.Event()

    def line(self, status="RUN", frame=0):
        finished = self.index if status == "PASS" else self.index - 1
        filled = int(16 * finished / self.total)
        bar = "=" * filled + "-" * (16 - filled)
        mark = "|/-\\"[frame % 4] if status == "RUN" else " "
        prefix = f"  [{self.index:2}/{self.total}] [{bar}] {mark} {status:<7} "
        elapsed = f"  {duration(time.monotonic() - self.started)}"
        width = shutil.get_terminal_size((100, 24)).columns or 100
        if self.animated and width < 75:
            prefix = f"  [{self.index:2}/{self.total}] {mark} {status:<7} "
        available = max(8, width - len(prefix) - len(elapsed) - 1)
        label = (
            self.label
            if not self.animated or len(self.label) <= available
            else self.label[: available - 3] + "..."
        )
        tone = (
            "32"
            if status == "PASS"
            else "31"
            if status == "FAIL"
            else "36"
            if status == "RUN"
            else "33"
        )
        return paint(prefix, tone, self.color) + label + paint(elapsed, "2", self.color)

    def refresh(self):
        frame = 0
        while not self.stop.wait(0.15):
            frame += 1
            print("\r\033[2K" + self.line(frame=frame), end="", flush=True)

    def __enter__(self):
        self.started = time.monotonic()
        print(self.line(), end="" if self.animated else "\n", flush=True)
        if self.animated:
            self.thread = threading.Thread(target=self.refresh, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        if self.animated:
            self.thread.join()
        status = (
            "PASS"
            if kind is None
            else "STOPPED"
            if issubclass(kind, KeyboardInterrupt)
            else "FAIL"
        )
        print(("\r\033[2K" if self.animated else "") + self.line(status), flush=True)


def duration(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:04.1f}s"


def table(headers, rows, right=()):
    widths = [
        max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))
    ]

    def line(row):
        return (
            "  "
            + "  ".join(
                str(value).rjust(width) if i in right else str(value).ljust(width)
                for i, (value, width) in enumerate(zip(row, widths))
            ).rstrip()
        )

    return [line(headers), line(["-" * width for width in widths]), *map(line, rows)]


def render(folder, plan, state, summary=None, color=False):
    status = state["status"]
    label = {
        "completed": "PASSED",
        "failed": "FAILED",
        "interrupted": "INTERRUPTED",
        "running": "IN PROGRESS",
    }[status]
    heading = f"StenWeave quickcheck  {label}"
    if color:
        tone = "32" if status == "completed" else "31" if status == "failed" else "33"
        heading = f"\033[1;{tone}m{heading}\033[0m"
    elapsed = duration(state["seconds"]) if "seconds" in state else "running"
    lines = [
        "",
        f"  {heading}",
        f"  {'A100' if plan['arch'] == 'sm80' else 'H100'} ({plan['arch']}) | GPU {plan['device']} | Box2D1R | {elapsed}",
        "",
    ]
    completed = {step["id"] for step in state["steps"]}
    steps = [step["id"] for step in plan["steps"]]
    phases = []
    for title, prefix in (
        ("Build", "build-"),
        ("Timing", "timing-"),
        ("Correctness", "checks-"),
    ):
        expected = [name for name in steps if name.startswith(prefix)]
        count = sum(name in completed for name in expected)
        outcome = "PASS" if count == len(expected) else "PENDING"
        if state.get("failed_step", "").startswith(prefix):
            outcome = "FAIL" if status == "failed" else "STOPPED"
        coverage = f"{count}/{len(expected)} implementation runs"
        reused = sum(
            step.get("reused", False)
            for step in state["steps"]
            if step["id"].startswith(prefix)
        )
        if reused:
            coverage += f" ({reused} reused)"
        phases.append([title, outcome, coverage])
    profile_count = len(state["profiles"])
    profile_status = "PASS" if profile_count == len(plan["profiles"]) else "PENDING"
    if state.get("failed_step", "").startswith("profile-"):
        profile_status = "FAIL" if status == "failed" else "STOPPED"
    phases.append(
        [
            "NCU",
            profile_status,
            f"{profile_count}/{len(plan['profiles'])} combined Fig.11 / Fig.15 reports",
        ]
    )
    plot_status = "PASS" if "plot-fig09" in completed else "PENDING"
    if state.get("failed_step") == "plot-fig09":
        plot_status = "FAIL" if status == "failed" else "STOPPED"
    phases.append(["Plotting", plot_status, "Fig.9 from historical CSV"])
    lines += table(["Stage", "Status", "Coverage"], phases)
    if plan["arch"] == "sm90":
        lines += ["", "  SKIP  ConvStencil on H100 (excluded from this quickcheck)."]
    if summary:
        names = {label: key for key, label in IMPLEMENTATIONS.items()}
        rows = []
        for row in summary["timings"]:
            checked = f"checks-{names[row['implementation']]}" in completed
            rows.append(
                [
                    row["implementation"],
                    row["variant"]
                    if row["implementation"] == "StenWeave"
                    else "native",
                    "PASS" if checked else "PENDING",
                    f"{row['gstencils_per_s']:.2f}",
                ]
            )
        lines += [""] + table(
            ["Implementation", "Variant", "CPU check", "GStencil/s"], rows, right=(3,)
        )
        lines += [
            "  Timings use short, implementation-specific workloads; use full experiments for paper comparisons."
        ]
    if state["profiles"]:
        rows = []
        for row in state["profiles"]:
            low = row["low_level"]
            rows.append(
                [
                    row["report"]["implementation"],
                    f"{row['roofline']['efficiency_percent']:.2f}",
                    f"{low['pc_samples']:,.0f}",
                    f"{low['pc_dropped_bytes']:,.0f}",
                ]
            )
        lines += [""] + table(
            ["NCU workload", "Roofline (%)", "PC samples", "Dropped bytes"],
            rows,
            right=(1, 2, 3),
        )
        lines += ["  NCU uses one combined capture per workload; StenWeave uses SASS."]
    failure_lines = []
    if status in ("failed", "interrupted"):
        step = state.get("failed_step", "")
        failure = state.get("failure") or failure_details(
            folder, plan, step, state.get("error", "")
        )
        failure_lines += [
            "",
            f"  Stopped at: {step_label(step) if step in steps else step}",
            f"  Reason: {failure['reason']}",
        ]
        if wrapper := failure.get("wrapper_log"):
            failure_lines += [f"  Wrapper log: {wrapper}"]
        if log := failure.get("log"):
            failure_lines += [
                "",
                "  " + paint("OPEN THIS ERROR LOG:", "1;33", color),
                "    " + paint(log, "1", color),
            ]
    if "plot-fig09" in completed:
        lines += ["", f"  Figure 9: {folder / 'figures/fig09.pdf'}"]
    lines += [f"  Results:  {folder}", f"  Details:  {folder / 'status.json'}", ""]
    lines += failure_lines
    if color:
        for i in range(3, len(lines)):
            lines[i] = re.sub(
                r"\b(PASS|FAIL|PENDING|STOPPED|SKIP)\b",
                lambda match: paint(
                    match[0],
                    "32"
                    if match[0] == "PASS"
                    else "31"
                    if match[0] == "FAIL"
                    else "33",
                ),
                lines[i],
            )
            if (
                lines[i].startswith("  Stage ")
                or lines[i].startswith("  Implementation ")
                or lines[i].startswith("  NCU workload ")
            ):
                lines[i] = paint(lines[i], "1;36")
    return "\n".join(lines)


def print_report(folder, show_toolchain=True):
    plan = json.loads((folder / "plan.json").read_text())
    state = json.loads((folder / "status.json").read_text())
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    environment_path = folder / "environment.json"
    if show_toolchain and environment_path.is_file():
        print_toolchain(json.loads(environment_path.read_text()).get("toolchain"))
    print(
        render(
            folder,
            plan,
            state,
            summary,
            color=interactive() and "NO_COLOR" not in os.environ,
        ),
        flush=True,
    )
    return state["status"]
