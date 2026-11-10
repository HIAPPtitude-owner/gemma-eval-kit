"""A single-holder lock file for local sweeps, safe on Windows.

The lock is created atomically with ``O_CREAT | O_EXCL`` and records the holder's pid, start
time, label and a random token. A second acquirer is refused: ``LockHeldError`` while the
recorded pid is alive (or the file cannot be read yet), ``StaleLockError`` when that pid is no
longer running. A stale lock is reported, never stolen: a dead pid can be a crash that left
the distro half-repaired, and a recycled pid cannot be told apart from the holder, so a human
deletes the file. Release removes the file only if it still carries this holder's token.
"""

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_RETRIES = 5
_RETRY_SLEEP_S = 0.2


class LockHeldError(RuntimeError):
    """Raised when another live process holds the sweep lock."""


class StaleLockError(RuntimeError):
    """Raised when the lock file names a process that is no longer running."""


def pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is a running process, without signalling it.

    On Windows ``os.kill(pid, 0)`` would call TerminateProcess, so this uses OpenProcess and
    GetExitCodeProcess instead.
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def _refuse(path: Path) -> None:
    """Raise the right error for an existing lock file at ``path``."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        pid, started, label = int(rec["pid"]), rec["started_utc"], rec["label"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LockHeldError(
            f"LOCKED: {path} exists but cannot be read ({exc}); another sweep may be starting. "
            "If no sweep is running, delete the file."
        ) from exc
    if pid_alive(pid):
        raise LockHeldError(f"LOCKED: sweep {label!r} held by pid {pid} since {started} ({path})")
    raise StaleLockError(
        f"STALE-LOCK: {path} names pid {pid} ({label!r}, started {started}), which is not "
        "running. Confirm no sweep is running and check the distro state, then delete the file."
    )


def _create(path: Path, rec: dict[str, Any]) -> None:
    """Atomically create the lock file holding ``rec``, or raise why it cannot be taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _ in range(_RETRIES):
        try:
            fd = os.open(path, flags)
        except FileExistsError:
            _refuse(path)
        except PermissionError:  # Windows: a just-released lock can be delete-pending.
            time.sleep(_RETRY_SLEEP_S)
            continue
        try:
            os.write(fd, (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
        except OSError:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise
        os.close(fd)
        return
    raise LockHeldError(f"LOCKED: {path} stayed delete-pending; retry in a moment")


def _release(path: Path, token: str) -> None:
    """Delete the lock if it still carries ``token``; report (never raise) if that fails."""
    for _ in range(_RETRIES):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("token") != token:
                print(f"LOCK-NOT-OURS: {path} no longer carries this sweep's token; left as is")
                return
            path.unlink()
            return
        except FileNotFoundError:
            print(f"LOCK-MISSING: {path} was removed while this sweep held it")
            return
        except (PermissionError, ValueError):  # WinError 32: a reader has it open.
            time.sleep(_RETRY_SLEEP_S)
    print(f"LOCK-RELEASE-FAILED: delete {path} by hand", flush=True)


@contextmanager
def sweep_lock(path: Path, label: str) -> Iterator[dict[str, Any]]:
    """Hold the sweep lock at ``path`` for the duration of the ``with`` block.

    Args:
        path: The lock file.
        label: What the holder is doing (shown to a refused acquirer).

    Yields:
        The record written into the lock file.

    Raises:
        LockHeldError: If a live process holds the lock (or it is unreadable).
        StaleLockError: If the lock names a process that is no longer running.
    """
    rec = {
        "pid": os.getpid(),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "token": uuid.uuid4().hex,
    }
    _create(path, rec)
    try:
        yield rec
    finally:
        _release(path, rec["token"])
