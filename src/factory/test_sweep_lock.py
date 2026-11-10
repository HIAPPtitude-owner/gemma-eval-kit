"""Tests for factory.sweep_lock, including a real two-process contention test."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from factory.sweep_lock import LockHeldError, StaleLockError, pid_alive, sweep_lock

HOLDER = (
    "import sys\n"
    "from pathlib import Path\n"
    "from factory.sweep_lock import sweep_lock\n"
    "with sweep_lock(Path(sys.argv[1]), 'holder-sweep'):\n"
    "    print('HELD', flush=True)\n"
    "    sys.stdin.readline()\n"
)
CONTENDER = (
    "import sys\n"
    "from pathlib import Path\n"
    "from factory.sweep_lock import LockHeldError, sweep_lock\n"
    "try:\n"
    "    with sweep_lock(Path(sys.argv[1]), 'contender'):\n"
    "        print('ACQUIRED', flush=True)\n"
    "except LockHeldError as exc:\n"
    "    print(exc, flush=True)\n"
    "    sys.exit(3)\n"
)


def test_second_real_process_is_refused(tmp_path: Path) -> None:
    """Holder process takes the lock and blocks; a second real process must be refused."""
    lock = tmp_path / "locks" / "local_eval.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "HELD"
        rec = json.loads(lock.read_text(encoding="utf-8"))
        # The venv python.exe is a launcher on Windows, so the recorded pid is the real
        # interpreter's (a child of holder.pid), not holder.pid itself.
        assert pid_alive(rec["pid"]) and rec["started_utc"] and rec["token"]
        contender = subprocess.run(
            [sys.executable, "-c", CONTENDER, str(lock)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert contender.returncode == 3, contender.stdout + contender.stderr
        assert "LOCKED" in contender.stdout and "holder-sweep" in contender.stdout
        assert f"pid {rec['pid']}" in contender.stdout
    finally:
        assert holder.stdin is not None
        holder.stdin.write("\n")
        holder.stdin.close()
        assert holder.wait(timeout=30) == 0
    assert not lock.exists(), "the holder must remove its lock on release"
    with sweep_lock(lock, "after"):
        assert lock.exists()
    assert not lock.exists()


def _dead_pid() -> int:
    r = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
    )
    pid = int(r.stdout.strip())
    assert not pid_alive(pid)
    return pid


def test_stale_lock_is_reported_not_stolen(tmp_path: Path) -> None:
    lock = tmp_path / "l.lock"
    pid = _dead_pid()
    body = {"pid": pid, "started_utc": "2026-10-16T00:00:00Z", "label": "old", "token": "t"}
    lock.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(StaleLockError, match=f"pid {pid}"), sweep_lock(lock, "new"):
        pass
    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "t", "never stolen"


def test_unreadable_lock_refuses(tmp_path: Path) -> None:
    lock = tmp_path / "l.lock"
    lock.write_text("", encoding="utf-8")
    with pytest.raises(LockHeldError, match="cannot be read"), sweep_lock(lock, "new"):
        pass
    assert lock.exists()


def test_release_leaves_a_lock_that_is_not_ours(tmp_path: Path) -> None:
    lock = tmp_path / "l.lock"
    with sweep_lock(lock, "mine"):
        other = {"pid": os.getpid(), "started_utc": "x", "label": "other", "token": "other"}
        lock.write_text(json.dumps(other), encoding="utf-8")
    assert lock.exists()


def test_pid_alive() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(_dead_pid())
