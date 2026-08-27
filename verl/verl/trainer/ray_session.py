"""Private per-run Ray sessions for the ES entry points.

Each ES job starts its own local Ray instance in a private temp dir so that a
concurrent job on another GPU is unaffected -- a host-global `ray stop --force`
would kill it (see the note in scripts/es/run_es_math.sh).  The flip side is
that nothing else ever reclaims those dirs or those raylets, so this module
ties both to the lifetime of the process that created them.
"""

import atexit
import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import ray

# main_es.py and main_es_token.py use different prefixes: the plasma socket
# lives under the session dir and AF_UNIX paths are capped at 107 bytes.
SESSION_PREFIXES = ("ray_es_session_", "ray_es_")


def _live_temp_dirs():
    """Temp dirs claimed by a raylet currently running on this host.

    Returns None if the process list cannot be read, in which case callers must
    not delete anything.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    live = set()
    for line in out.splitlines():
        if "raylet/raylet" not in line:
            continue
        for tok in line.split():
            if tok.startswith("--temp-dir="):
                live.add(tok[len("--temp-dir=") :])
    return live


def sweep_stale_sessions(min_age_s=3600):
    """Remove session dirs left behind by runs that were hard-killed.

    A SIGKILL (or the OOM killer) bypasses every in-process hook, so those runs
    can only be reclaimed by a later one.  A dir is removed only when no live
    raylet claims it *and* it has not been touched for `min_age_s` -- the age
    guard avoids racing a concurrent job that has mkdtemp'd but not yet started
    its raylet.
    """
    live = _live_temp_dirs()
    if live is None:
        return
    now = time.time()
    root = tempfile.gettempdir()
    for prefix in SESSION_PREFIXES:
        for d in glob.glob(os.path.join(root, prefix + "*")):
            if d in live or not os.path.isdir(d):
                continue
            try:
                if now - os.path.getmtime(d) < min_age_s:
                    continue
            except OSError:
                continue
            shutil.rmtree(d, ignore_errors=True)


def init_ray(prefix):
    """ray.init into a private temp dir that is torn down when we exit."""
    for k in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
        os.environ.pop(k, None)

    sweep_stale_sessions()

    unique_dir = tempfile.mkdtemp(prefix=prefix)
    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=unique_dir,
        dashboard_port=None,
    )

    def _cleanup():
        try:
            ray.shutdown()
        except Exception:
            pass
        shutil.rmtree(unique_dir, ignore_errors=True)

    atexit.register(_cleanup)
    # SIGTERM otherwise skips atexit and leaves the raylet reparented to init;
    # SystemExit unwinds normally so the handler above still runs.
    signal.signal(signal.SIGTERM, lambda s, _f: sys.exit(128 + s))
    return unique_dir
