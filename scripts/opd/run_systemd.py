#!/usr/bin/env python3
"""Run a local OPD job as a transient user service with final Ray cleanup."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

from run_with_ray_cleanup import allocate_run_dir, log, main as supervise


MARKER = ".opd-systemd-unit"
STATE = ".opd-launch.json"


def validate_run_dir(run_dir, unit):
    path = Path(run_dir)
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid():
        raise RuntimeError(f"Refusing cleanup of unowned or invalid directory: {path}")
    if (path / MARKER).read_text() != unit:
        raise RuntimeError(f"Service ownership marker does not match: {path}")
    return path


def cleanup(run_dir, unit):
    if not os.path.lexists(run_dir):
        return
    path = validate_run_dir(run_dir, unit)
    shutil.rmtree(path)
    log(f"systemd removed OPD Ray temporary directory: {path}")


def worker(run_dir, unit):
    path = validate_run_dir(run_dir, unit)
    state = json.loads((path / STATE).read_text())
    (path / STATE).unlink()
    os.chdir(state["cwd"])
    os.environ.clear()
    os.environ.update(state["env"])
    os.environ["OPD_SYSTEMD_UNIT"] = unit
    return supervise(state["command"], run_dir=run_dir)


def exec_quote(value):
    """Quote one systemd ExecStopPost argument, including expansion literals."""
    return (
        '"'
        + value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        + '"'
    )


def service_command(run_dir, unit, run_id):
    script = str(Path(__file__).resolve())
    cleanup_command = " ".join(
        exec_quote(arg) for arg in (sys.executable, script, "--cleanup", run_dir, unit)
    )
    return [
        "systemd-run",
        "--user",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={unit}",
        "--service-type=exec",
        # TERM goes only to the supervisor, giving its finally block time to
        # run. On main-process death/timeout, systemd kills the entire cgroup.
        "--property=KillMode=mixed",
        "--property=TimeoutStopSec=30s",
        "--property=SendSIGKILL=yes",
        "--property=Restart=no",
        f"--property=ExecStopPost={cleanup_command}",
        "--setenv=PYTHONUNBUFFERED=1",
        f"--setenv=OPD_SYSTEMD_RUN_ID={run_id}",
        sys.executable,
        script,
        "--worker",
        run_dir,
        unit,
    ]


def unit_state(unit, run_id):
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=LoadState",
            "--property=Environment",
        ],
        capture_output=True,
        text=True,
    )
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if values.get("LoadState") == "not-found":
        return "absent"
    if result.returncode or "LoadState" not in values:
        raise RuntimeError(f"Cannot check service ownership: {result.stderr.strip()}")
    if f"OPD_SYSTEMD_RUN_ID={run_id}" in shlex.split(values.get("Environment", "")):
        return "owned"
    return "foreign"


def stop_unit(unit, runner, run_id):
    # A terminal signal can arrive before systemd-run has submitted the unit.
    deadline = time.monotonic() + 30
    while True:
        state = unit_state(unit, run_id)
        if state == "foreign":
            raise RuntimeError(f"Refusing to stop another run's service: {unit}")
        if state == "absent":
            if runner.poll() is not None:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {unit} to be submitted")
            time.sleep(0.1)
            continue
        result = subprocess.run(
            ["systemctl", "--user", "stop", unit], capture_output=True, text=True
        )
        if result.returncode == 0:
            return
        if runner.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError(f"Could not stop {unit}: {result.stderr.strip()}")
        time.sleep(0.1)


def launch(command):
    # Do not silently fall back to unprotected execution when the bus is down.
    check = subprocess.run(
        ["systemctl", "--user", "show", "--property=Version", "--value"],
        capture_output=True,
        text=True,
    )
    if check.returncode:
        raise RuntimeError(
            "User systemd manager is unavailable. Run from a login session with a user manager, "
            "or explicitly use OPD_SYSTEMD=0 for supervisor-only cleanup. "
            + check.stderr.strip()
        )
    unit = (
        os.environ.get("OPD_SYSTEMD_UNIT")
        or f"opd-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"opd-[A-Za-z0-9_-]+(?:\.service)?", unit):
        raise ValueError(
            "OPD_SYSTEMD_UNIT must start with opd- and contain only letters, digits, _ or -"
        )
    if not unit.endswith(".service"):
        unit += ".service"
    # A named unit is useful for stopping jobs. Never take over an existing one.
    existing = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=LoadState", "--value"],
        capture_output=True,
        text=True,
    )
    if existing.stdout.strip() != "not-found":
        raise RuntimeError(
            f"Cannot launch {unit}: unit already exists or its state could not be checked"
        )

    received_signal = None

    def handle_signal(signum, _frame):
        nonlocal received_signal
        if received_signal is None:
            received_signal = signum

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, handle_signal)
    run_dir = allocate_run_dir()
    run_id = uuid.uuid4().hex
    path = Path(run_dir)
    (path / MARKER).write_text(unit)
    runner = None
    try:
        # Transfer the exact launch environment without exposing credentials in
        # systemd-run argv or modifying the user manager's shared environment.
        with (path / STATE).open(
            "x", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as state_file:
            json.dump(
                {"command": command, "cwd": os.getcwd(), "env": dict(os.environ)},
                state_file,
            )
        log(f"OPD systemd unit: {unit}")
        log(f"Stop and wait for cleanup: systemctl --user stop {unit}")
        runner = subprocess.Popen(
            service_command(run_dir, unit, run_id), start_new_session=True
        )
        while True:
            if received_signal is not None:
                stop_unit(unit, runner, run_id)
                runner.wait(timeout=10)
                return 128 + received_signal
            try:
                return runner.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Normally ExecStopPost has already removed the directory. A failed
        # submission or a crashed systemd-run client still needs reconciliation.
        if path.exists():
            state = unit_state(unit, run_id)
            if state in ("absent", "foreign"):
                cleanup(run_dir, unit)
            elif runner is not None:
                stop_unit(unit, runner, run_id)
                cleanup(run_dir, unit)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 4 and sys.argv[1] == "--worker":
            sys.exit(worker(sys.argv[2], sys.argv[3]))
        if len(sys.argv) == 4 and sys.argv[1] == "--cleanup":
            cleanup(sys.argv[2], sys.argv[3])
            sys.exit(0)
        if len(sys.argv) < 2:
            sys.exit("Usage: run_systemd.py COMMAND [ARGS...]")
        sys.exit(launch(sys.argv[1:]))
    except (OSError, RuntimeError, ValueError) as error:
        log(str(error), file=sys.stderr)
        sys.exit(1)
