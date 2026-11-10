"""Run a collector and terminate its process tree on interruption or timeout."""

import os
import signal
import socket
import subprocess
from contextvars import ContextVar

BATCH = ContextVar("collection_batch", default=None)


def run(command, *, timeout, **kwargs):
    batch = BATCH.get()
    if batch:
        kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), batch[1])
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        active = None
        try:
            if batch:
                from .resume import write_json

                active = batch[0] / "active-process.json"
                write_json(active, {"host": socket.gethostname(), "pgid": process.pid})
            result = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            # The parent may exit first while a descendant still ignores SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if active:
                active.unlink(missing_ok=True)
            raise
        if result:
            # A failed wrapper can leave its native child behind.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if active:
                active.unlink(missing_ok=True)
            raise subprocess.CalledProcessError(result, command)
        if active:
            active.unlink(missing_ok=True)
