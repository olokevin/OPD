"""Exercise the actual shell launcher without loading models or starting Ray."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux child subreaper")

FAKE_COMMAND = r"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(os.environ["TEST_STATE"])
name = Path(sys.argv[0]).name
if name == "ray":
    assert sys.argv[1] == "start", "Must never run host-wide ray stop"
    run_dir = Path(os.environ["RAY_TMPDIR"])
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "ray_client_server.err").write_text("test log")
    subprocess.Popen([sys.executable, str(root / "daemon.py")], start_new_session=True)
    deadline = time.monotonic() + 10
    while not (root / "daemon.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    sys.exit(int(os.environ.get("TEST_START_EXIT", "0")))
elif sys.argv[1:3] == ["-m", "verl.trainer.main_ppo"]:
    (root / "driver.json").write_text(json.dumps({
        "pid": os.getpid(), "address": os.environ.get("RAY_ADDRESS")
    }))
    if os.environ.get("TEST_BLOCK") == "1":
        time.sleep(120)
    sys.exit(int(os.environ.get("TEST_DRIVER_EXIT", "0")))
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
"""

DAEMON = r"""
import json
import os
from pathlib import Path
import signal
import time

if os.environ.get("TEST_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
root = Path(os.environ["TEST_STATE"])
(root / "daemon.json").write_text(json.dumps({
    "pid": os.getpid(), "run_dir": os.environ["RAY_TMPDIR"]
}))
time.sleep(120)
"""


def wait_for(path, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        assert process.poll() is None, (path.parent / "launcher.log").read_text()
        time.sleep(0.02)
    pytest.fail(f"Timed out waiting for {path}")


@pytest.fixture
def launch(tmp_path):
    launches = []

    def start(**overrides):
        state = tmp_path / str(len(launches))
        state.mkdir()
        binaries = state / "bin"
        binaries.mkdir()
        for name in ("ray", "python3"):
            command = binaries / name
            command.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
            command.chmod(0o755)
        sleep = binaries / "sleep"
        sleep.write_text("#!/bin/sh\nexit 0\n")
        sleep.chmod(0o755)
        (state / "daemon.py").write_text(DAEMON)
        prefix = state / "ray"
        prefix.mkdir()
        (prefix / "keep").write_text("Existing session must be preserved")
        env = dict(os.environ)
        env.pop("OPD_RAY_SUPERVISOR_PID", None)
        env.update(
            PATH=f"{binaries}:{env['PATH']}",
            TEST_STATE=str(state),
            RAY_TMPDIR=str(prefix),
            RAY_ISOLATE="1",
            RAY_EXTERNAL="0",
            CUDA_VISIBLE_DEVICES="6",
            SLURM_JOB_ID="test",
            OPD_SYSTEMD="0",
            LOG_DIR=str(state / "logs"),
            PEFT_MODE="none",
            CALIB_MODE="none",
        )
        env.update(overrides)
        with (state / "launcher.log").open("w") as log:
            process = subprocess.Popen(
                ["bash", str(ROOT / "on_policy_distillation.sh")],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        launches.append((process, state))
        return process, state

    yield start
    for process, state in launches:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        # Bound leaks even if a regression makes one of the assertions fail.
        for marker in ("daemon.json", "driver.json"):
            if (state / marker).exists():
                pid = json.loads((state / marker).read_text())["pid"]
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass


def assert_cleaned(process, state, expected_code):
    assert process.wait(timeout=15) == expected_code, (state / "launcher.log").read_text()
    daemon = json.loads((state / "daemon.json").read_text())
    assert not psutil.pid_exists(daemon["pid"])
    assert not Path(daemon["run_dir"]).exists()
    assert (state / "ray" / "keep").exists()
    if (state / "driver.json").exists():
        driver = json.loads((state / "driver.json").read_text())
        assert not psutil.pid_exists(driver["pid"])


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("isolate", ["0", "1"])
def test_cleans_detached_services_and_preserves_driver_status(launch, exit_code, isolate):
    process, state = launch(TEST_DRIVER_EXIT=str(exit_code), RAY_ISOLATE=isolate)
    assert_cleaned(process, state, exit_code)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signals_clean_services_without_stopping_concurrent_run(launch, signum):
    sibling, sibling_state = launch(TEST_BLOCK="1")
    wait_for(sibling_state / "driver.json", sibling)
    process, state = launch(TEST_BLOCK="1")
    wait_for(state / "driver.json", process)
    process.send_signal(signum)
    assert_cleaned(process, state, 128 + signum)
    assert sibling.poll() is None
    daemon = json.loads((sibling_state / "daemon.json").read_text())
    assert psutil.pid_exists(daemon["pid"])
    assert Path(daemon["run_dir"]).exists()
    sibling.terminate()
    assert_cleaned(sibling, sibling_state, 143)


def test_failed_ray_start_cleans_partial_cluster_and_skips_training(launch):
    process, state = launch(TEST_START_EXIT="9")
    assert_cleaned(process, state, 9)
    assert not (state / "driver.json").exists()


def test_kills_daemon_that_ignores_sigterm(launch):
    process, state = launch(TEST_IGNORE_TERM="1")
    assert_cleaned(process, state, 0)


def test_killed_training_process_cleans_local_run_with_file_logging(launch):
    process, state = launch(TEST_BLOCK="1", SLURM_JOB_ID="")
    driver = wait_for(state / "driver.json", process)
    os.kill(driver["pid"], signal.SIGKILL)
    assert_cleaned(process, state, 137)
    assert list((state / "logs").glob("run_*.log"))


def test_external_cluster_is_not_started_or_removed(launch):
    process, state = launch(RAY_EXTERNAL="1", RAY_ADDRESS="127.0.0.1:9999", TEST_DRIVER_EXIT="3")
    assert process.wait(timeout=15) == 3
    assert not (state / "daemon.json").exists()
    assert (state / "ray" / "keep").exists()
    driver = json.loads((state / "driver.json").read_text())
    assert driver["address"] == "127.0.0.1:9999"
