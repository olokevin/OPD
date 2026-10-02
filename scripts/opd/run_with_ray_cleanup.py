#!/usr/bin/env python3
"""Own a local OPD launch and reap its Ray daemons on exit (Linux).

Ray CLI daemons outlive ``ray start`` and can start new process groups. Becoming
a child subreaper keeps those orphaned descendants attached to this supervisor,
so cleanup needs neither process-name matching nor a host-wide ``ray stop``.
Local systemd launches additionally clean up if this supervisor is hard-killed.
"""

import ctypes
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import psutil


def log(message, *, file=sys.stdout):
    # A closed terminal or output pipe must not prevent process cleanup.
    try:
        print(message, file=file, flush=True)
    except OSError:
        pass


def become_subreaper():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The OPD Ray supervisor requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def stop_descendants(grace_seconds=5):
    """Terminate our descendants, including adopted daemons, then reap them.

    Refresh the tree after TERM: a service may fork while it is shutting down.
    psutil's Process handles check process identity before sending signals.
    """
    parent = psutil.Process()
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=grace_seconds)
    deadline = time.monotonic() + grace_seconds
    while children := parent.children(recursive=True):
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=max(0, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            return not parent.children(recursive=True)
    return True


def allocate_run_dir():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "6,7").split(",")[0] or "0"
    # RAY_TMPDIR is a prefix, never an existing directory to erase.
    base = Path(os.environ.get("RAY_TMPDIR", f"/tmp/ray_opd_gpu{gpu}")).absolute()
    return tempfile.mkdtemp(prefix=base.name + "_", dir=base.parent)


def main(command, run_dir=None):
    become_subreaper()
    received_signal = None

    def handle_signal(signum, _frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, handle_signal)

    # systemd preallocates this directory so ExecStopPost can remove it even
    # when the supervisor receives SIGKILL before its first instruction.
    run_dir = run_dir if run_dir is not None else allocate_run_dir()
    env = dict(os.environ, RAY_TMPDIR=run_dir, OPD_RAY_SUPERVISOR_PID=str(os.getpid()))
    exit_code = 1
    log(f"OPD Ray temporary directory: {run_dir}")
    log(f"OPD cleanup supervisor PID: {os.getpid()}")
    try:
        if received_signal is None:
            # Isolate the child from terminal signals; the supervisor handles
            # those once and shuts down the complete process tree in finally.
            child = subprocess.Popen(command, env=env, start_new_session=True)
            while received_signal is None:
                try:
                    exit_code = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
    finally:
        log("Stopping this OPD run's Ray processes...")
        if stop_descendants():
            shutil.rmtree(run_dir)
            log(f"Removed OPD Ray temporary directory: {run_dir}")
        else:
            log(f"Ray processes did not exit; retaining {run_dir}", file=sys.stderr)
            exit_code = exit_code or 1
    if received_signal is not None:
        return 128 + received_signal
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: run_with_ray_cleanup.py COMMAND [ARGS...]")
    sys.exit(main(sys.argv[1:]))
