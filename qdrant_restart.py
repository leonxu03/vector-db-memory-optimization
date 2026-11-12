"""Restart the Qdrant Docker container and drop the OS page cache (shared by main.py and query.py)."""

import shlex
import subprocess
import time
import urllib.request

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER = "qdrant"
READY_TIMEOUT_S = 300  # give up waiting for Qdrant to come back after restart


class RestartError(Exception):
    """Restart or cache drop failed; carries the reason to print."""


def run(cmd):
    """Run a command, echoing it first; raise RestartError if it fails. Returns its stdout."""
    print(f"$ {shlex.join(cmd)}", flush=True)  # flush so it shows before the command's output
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RestartError(f"'{shlex.join(cmd)}' failed (exit {exc.returncode}): "
                           f"{exc.stderr.strip()}") from exc
    except OSError as exc:
        raise RestartError(f"'{shlex.join(cmd)}' failed: {exc}") from exc


def wait_until_ready():
    """Poll Qdrant's /readyz until it answers 200 (all collections loaded after restart)."""
    start = time.monotonic()
    while time.monotonic() - start < READY_TIMEOUT_S:
        try:
            with urllib.request.urlopen(f"{QDRANT_URL}/readyz", timeout=5) as resp:
                if resp.status == 200:
                    print(f"Qdrant ready after {time.monotonic() - start:.0f}s", flush=True)
                    return
        except OSError:
            pass  # still starting up
        time.sleep(1)
    raise RestartError(f"Qdrant not ready {READY_TIMEOUT_S}s after restart.")


def restart_qdrant():
    """Restart the container, check Docker reports it running, then wait until Qdrant is ready."""
    run(["sudo", "docker", "restart", QDRANT_CONTAINER])
    state = run(["sudo", "docker", "inspect", "-f", "{{.State.Status}}", QDRANT_CONTAINER])
    if state != "running":
        raise RestartError(f"Container '{QDRANT_CONTAINER}' is '{state}' after restart, not running.")
    wait_until_ready()


def drop_caches():
    """Flush dirty pages and drop the OS page cache, dentries and inodes."""
    run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
