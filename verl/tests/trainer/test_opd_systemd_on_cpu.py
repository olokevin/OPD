"""Unit coverage plus opt-in tests against a real user systemd manager."""

import importlib.util
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from .test_opd_ray_cleanup_on_cpu import assert_cleaned, wait_for
from .test_opd_ray_cleanup_on_cpu import launch as launch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/opd"))
    spec = importlib.util.spec_from_file_location("opd_systemd", ROOT / "scripts/opd/run_systemd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_requires_matching_owner_marker_and_rejects_symlinks(manager, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / manager.MARKER).write_text("opd-owned.service")
    with pytest.raises(RuntimeError, match="marker"):
        manager.cleanup(str(run_dir), "opd-other.service")
    alias = tmp_path / "alias"
    alias.symlink_to(run_dir, target_is_directory=True)
    with pytest.raises(RuntimeError, match="invalid"):
        manager.cleanup(str(alias), "opd-owned.service")
    assert run_dir.exists()
    manager.cleanup(str(run_dir), "opd-owned.service")
    manager.cleanup(str(run_dir), "opd-owned.service")
    assert not run_dir.exists()


def test_stop_refuses_another_runs_service(manager, monkeypatch):
    monkeypatch.setattr(manager, "unit_state", lambda *_: "foreign")
    monkeypatch.setattr(manager.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Must not stop foreign unit"))
    with pytest.raises(RuntimeError, match="another run"):
        manager.stop_unit("opd-other.service", SimpleNamespace(poll=lambda: None), "our-id")


def test_unavailable_user_manager_fails_without_starting_job(manager, monkeypatch):
    monkeypatch.setattr(manager.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=1, stderr="no bus"))
    monkeypatch.setattr(manager, "allocate_run_dir", lambda: pytest.fail("Must check manager before allocating"))
    with pytest.raises(RuntimeError, match="OPD_SYSTEMD=0"):
        manager.launch(["false"])


def test_stop_hook_escapes_systemd_expansions(manager):
    command = manager.service_command('/tmp/ray space%value$dollar"quote', "opd-test.service", "abc")
    stop = next(arg for arg in command if arg.startswith("--property=ExecStopPost="))
    assert 'space%%value$$dollar\\"quote' in stop
    assert "--property=KillMode=mixed" in command


real_systemd = pytest.mark.skipif(
    os.environ.get("OPD_TEST_SYSTEMD") != "1", reason="Set OPD_TEST_SYSTEMD=1 outside sandbox"
)


@pytest.fixture
def managed_launch(launch):
    units = []

    def start(**overrides):
        unit = f"opd-test-{uuid.uuid4().hex[:12]}.service"
        units.append(unit)
        options = dict(OPD_SYSTEMD="1", SLURM_JOB_ID="", OPD_SYSTEMD_UNIT=unit)
        options.update(overrides)
        process, state = launch(**options)
        return process, state, unit

    yield start
    for unit in units:
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)


@real_systemd
@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_systemd_completion_and_failure(managed_launch, exit_code):
    process, state, _ = managed_launch(TEST_DRIVER_EXIT=str(exit_code))
    assert_cleaned(process, state, exit_code)


@real_systemd
def test_real_stop_runs_supervisor_and_preserves_other_job(managed_launch):
    sibling, sibling_state, sibling_unit = managed_launch(TEST_BLOCK="1")
    wait_for(sibling_state / "driver.json", sibling)
    process, state, unit = managed_launch(TEST_BLOCK="1")
    wait_for(state / "driver.json", process)
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert_cleaned(process, state, 143)
    assert "Removed OPD Ray temporary directory:" in (state / "launcher.log").read_text()
    assert sibling.poll() is None
    daemon = json.loads((sibling_state / "daemon.json").read_text())
    assert psutil.pid_exists(daemon["pid"])
    subprocess.run(["systemctl", "--user", "stop", sibling_unit], check=True)
    assert_cleaned(sibling, sibling_state, 143)


@real_systemd
def test_real_sigkill_supervisor_uses_systemd_cleanup(managed_launch):
    process, state, unit = managed_launch(TEST_BLOCK="1", TEST_IGNORE_TERM="1")
    wait_for(state / "driver.json", process)
    pid = int(
        subprocess.check_output(["systemctl", "--user", "show", unit, "--property=MainPID", "--value"], text=True)
    )
    assert pid > 1
    os.kill(pid, signal.SIGKILL)
    code = process.wait(timeout=40)
    assert code != 0
    assert_cleaned(process, state, code)
    log = (state / "launcher.log").read_text()
    assert "systemd removed OPD Ray temporary directory:" in log


@real_systemd
def test_real_ctrl_c_stops_service_through_supervisor(managed_launch):
    process, state, _ = managed_launch(TEST_BLOCK="1")
    wait_for(state / "driver.json", process)
    process.send_signal(signal.SIGINT)
    assert_cleaned(process, state, 130)


@real_systemd
def test_real_duplicate_unit_does_not_stop_existing_job(managed_launch, launch):
    process, state, unit = managed_launch(TEST_BLOCK="1")
    wait_for(state / "driver.json", process)
    duplicate, duplicate_state = launch(OPD_SYSTEMD="1", SLURM_JOB_ID="", OPD_SYSTEMD_UNIT=unit)
    assert duplicate.wait(timeout=15) != 0
    assert not (duplicate_state / "driver.json").exists()
    assert process.poll() is None
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert_cleaned(process, state, 143)
