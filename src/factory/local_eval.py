"""Local gold/empty sweeps on the WSL1 distro with the ``as_shipped`` and ``diagnostic`` harnesses.

Neither harness is the scorer (the scorer is Docker with a private wheel set). Both run the
public harness's subprocess backend inside ``gemma-wsl1``: gold mode through
``harness/shared/gold_driver.py`` (``task.patch`` supplied as ``agent_patch`` under
``--skip-agent-patch``), empty mode through the plain ``swegemma`` CLI. ``diagnostic`` first
applies ``harness/diagnostic/repair.sh``; ``as_shipped`` removes that repair's ``.pth``
switch so it measures the distro as provisioned by ``harness/shared/provision.sh`` alone.
Distro and pin handling lives in ``factory.local_env``; ``public_docker`` (the public Docker
kit, a third non-scorer route) lives in ``factory.docker_kit``.
"""

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from factory import docker_kit
from factory.local_env import (
    HARNESS_ROUTES,
    PINNED_FILES,
    Harness,
    Mode,
    SweepDriftError,
    assert_harness_pins,
    assert_repair_logged,
    end_of_sweep_check,
    launcher,
    lock_path,
    measure_env,
    sha256_file,
    to_wsl_path,
    wsl_logged,
)
from factory.paths import competition_dir, data_root, results_dir
from factory.pins import REPO_ROOT
from factory.sweep_lock import sweep_lock

__all__ = [
    "HARNESS_ROUTES",
    "PINNED_FILES",
    "REPO_ROOT",
    "GoldRow",
    "Harness",
    "IngestError",
    "Mode",
    "SweepDriftError",
    "SweepError",
    "append_ledger_rows",
    "assert_harness_pins",
    "assert_repair_logged",
    "c5_verdict",
    "classify",
    "eval_argv",
    "ingest_sweep",
    "resolve_task_ids",
    "sha256_file",
    "sweep",
    "to_wsl_path",
    "write_env_file",
]

RootCause = Literal[
    "missing_dependency",
    "wheel_shadowing",
    "host_venv_shadowing",
    "network_tls",
    "version_drift",
    "invalid_task",
    "unclassified",
]
UNCLASSIFIED = "unclassified"
ENV_FILE = "factory_env.json"
C5_KEYS = ("empty", "malformed", "wrong")
TASK_TIMEOUT_S = 3600

# One regex per root cause; first match wins, so the specific signatures come first.
CLASSIFY_TABLE: list[tuple[str, re.Pattern[str]]] = [
    (
        "host_venv_shadowing",
        re.compile(
            r"Router\.__init__\(\) got an unexpected keyword argument 'on_startup'"
            r"|module 'httpx' has no attribute 'Stream'"
        ),
    ),
    (
        "missing_dependency",
        re.compile(
            r"ModuleNotFoundError: No module named"
            r"|recursive dependency involving fixture '[^']+' detected"
            # No async plugin: the scorer's pytest line has -p no:anyio (verification.py:631)
            # and pytest_asyncio is never injected (container_setup.py:155-165).
            r"|async def functions are not natively supported"
        ),
    ),
    ("wheel_shadowing", re.compile(r"import file mismatch:")),
    (
        "network_tls",
        re.compile(
            r"CERTIFICATE_VERIFY_FAILED|\[SSL: |Max retries exceeded with url"
            r"|Temporary failure in name resolution|Name or service not known"
        ),
    ),
    # ORDER-DEPENDENT: this pattern also matches the httpx host-venv signature
    # ("module 'httpx' has no attribute 'Stream'"); only host_venv_shadowing coming first in
    # this table keeps that excerpt correct (pinned by test_httpx_excerpt_needs_table_order).
    (
        "version_drift",
        re.compile(
            # site-packages only: a name missing from the WORKSPACE is the expected empty-mode
            # failure (real excerpt s6_fastapi_15030_empty_workspace_import.log).
            r"ImportError: cannot import name '[^']+' from '[^']+' \([^)]*site-packages"
            r"|AttributeError: module '[\w.]+' has no attribute"
        ),
    ),
    (
        "invalid_task",
        re.compile(
            r"patch does not apply|error: patch failed|corrupt patch|Hunk #\d+ FAILED"
            # exit 0 but a test the test_patch adds was skipped (e.g. skipif py<3.11 on 3.13)
            r"|Required test node did not pass"
        ),
    ),
]


class IngestError(ValueError):
    """Raised when a harness results directory is missing a file or a required field."""


class SweepError(RuntimeError):
    """Raised after a sweep in which at least one task produced no harness result."""

    def __init__(self, failed: list[str], rows: list["GoldRow"]) -> None:
        super().__init__(f"{len(failed)} task run(s) produced no result: {', '.join(failed)}")
        self.failed = failed
        self.rows = rows


class GoldRow(BaseModel):
    """One task outcome under one harness, mode and run (a GOLD_VALIDATION ledger row)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    env_hash: str
    wheelhouse_sha: str
    mode: Mode
    run_index: int
    resolved: bool
    root_cause: RootCause | None
    log_path: str
    harness_route: str

    @field_validator("env_hash", "task_id", "wheelhouse_sha")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v

    @field_validator("harness_route")
    @classmethod
    def _known_route(cls, v: str) -> str:
        if v not in HARNESS_ROUTES.values():
            raise ValueError(f"harness_route {v!r} not in {sorted(HARNESS_ROUTES.values())}")
        return v


def classify(log_text: str, harness: Harness = "as_shipped") -> str:
    """Map a harness test log to a root cause.

    The diagnostic route builds isolated sandbox venvs (docker_semantics.py,
    ``system_site_packages=False``), so host-venv shadowing cannot occur there by
    construction (R31): the same signature is the scorer's dedup-to-highest version
    (e.g. starlette 1.6.0 without ``on_startup``) and is recorded as ``version_drift``.

    Args:
        log_text: The task's test output (``test_outputs/<task>.log``), plus any harness error.
        harness: The harness that produced the log.

    Returns:
        The first matching root cause from ``CLASSIFY_TABLE``, else ``"unclassified"``.
    """
    for cause, pattern in CLASSIFY_TABLE:
        if pattern.search(log_text):
            if cause == "host_venv_shadowing" and harness in ("diagnostic", "public_docker"):
                return "version_drift"
            return cause
    return UNCLASSIFIED


def c5_verdict(results: dict[str, bool]) -> str:
    """Return the C5 verdict over the empty, malformed and wrong-task-gold submissions.

    Args:
        results: ``{"empty": resolved, "malformed": resolved, "wrong": resolved}``.

    Returns:
        ``"PASS"`` only if all three scored False, ``"FAIL"`` if any scored True, and
        ``"NOT-VERIFIED"`` if any of the three was not scored.
    """
    if any(k not in results for k in C5_KEYS):
        return "NOT-VERIFIED"
    return "FAIL" if any(results[k] for k in C5_KEYS) else "PASS"


def eval_argv(
    harness: Harness,
    mode: Mode,
    task_id: str,
    results_wsl: str,
    launch: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Build the in-distro command line for one task's eval run.

    The launcher comes from ``local_env.launcher``: as_shipped gold runs ``gold_driver.py``
    (verify_task wrapped to use ``task.patch``) and empty the plain ``swegemma`` CLI;
    diagnostic runs both through ``docker_semantics.py``. All use ``--sandbox subprocess
    --skip-agent-patch``. ``launch`` replaces that launcher and ``extra_env`` adds
    ``NAME=value`` settings (both used only by ``factory.c5``).
    """
    comp = to_wsl_path(competition_dir())
    env = [f"{k}={v}" for k, v in (extra_env or {}).items()]
    return [
        "/usr/bin/env",
        "TEST_TMPDIR=/tmp",
        *env,
        *(launch if launch is not None else launcher(harness, mode)),
        "eval",
        "--tasks",
        f"{comp}/tasks.jsonl",
        "--snapshots-dir",
        f"{comp}/snapshots",
        "--submission-dir",
        f"{comp}/sample_submission",
        "--results-dir",
        results_wsl,
        "--sandbox",
        "subprocess",
        "--skip-agent-patch",
        "--task-id",
        task_id,
        "--concurrency",
        "1",
        "--display",
        "quiet",
    ]


def resolve_task_ids(spec: str, tasks_jsonl: Path | None = None) -> list[str]:
    """Resolve ``--tasks``: ``all`` or a file of task ids (one per line, ``#`` comments).

    Raises:
        ValueError: If the file names a task id absent from ``tasks.jsonl``.
    """
    tasks_jsonl = tasks_jsonl or competition_dir() / "tasks.jsonl"
    with tasks_jsonl.open(encoding="utf-8") as f:
        known = [json.loads(line)["instance_id"] for line in f if line.strip()]
    if spec == "all":
        return known
    lines = Path(spec).read_text(encoding="utf-8").splitlines()
    ids = [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]
    unknown = [t for t in ids if t not in set(known)]
    if unknown:
        raise ValueError(f"unknown task id(s) in {spec}: {', '.join(unknown)}")
    return ids


def write_env_file(results: Path, env: dict[str, Any]) -> None:
    """Write the sweep identity (harness, route, mode, run, env_hash) next to its results."""
    results.mkdir(parents=True, exist_ok=True)
    text = json.dumps(env, indent=2, sort_keys=True) + "\n"
    (results / ENV_FILE).write_text(text, encoding="utf-8", newline="\n")


def _read_env_file(results: Path, expect: dict[str, Any]) -> dict[str, Any]:
    path = results / ENV_FILE
    if not path.is_file():
        raise IngestError(f"{path}: missing {ENV_FILE} (written by sweep before each run)")
    env = json.loads(path.read_text(encoding="utf-8"))
    for key in ("env_hash", "wheelhouse_sha", *expect):
        if key not in env:
            raise IngestError(f"{path}: missing field {key!r}")
    for key, want in expect.items():
        if env[key] != want:
            raise IngestError(f"{path}: {key} is {env[key]!r}, ingest asked for {want!r}")
    return env


def _root_cause(harness: Harness, mode: Mode, resolved: bool, text: str) -> str | None:
    """Root cause for a row: env signatures in any mode; an unexplained gold failure is
    ``unclassified``; an empty-mode pass means the tests cannot discriminate."""
    if resolved:
        return "invalid_task" if mode == "empty" else None
    cause = classify(text, harness)
    if cause != UNCLASSIFIED or mode == "gold":
        return cause
    return None


def _task_rows(results: Path) -> list[dict[str, Any]]:
    path = results / "task_results.jsonl"
    if not path.is_file():
        raise IngestError(f"{path}: missing (harness wrote no results)")
    recs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        for field in ("instance_id", "resolved"):
            if field not in rec:
                raise IngestError(f"{path} line {n}: missing field {field!r}")
        if not isinstance(rec["resolved"], bool):
            raise IngestError(f"{path} line {n}: field 'resolved' is not a bool")
        recs.append(rec)
    return recs


def ingest_sweep(
    results_dir: Path, harness: Harness, harness_route: str, mode: Mode, run_index: int
) -> list[GoldRow]:
    """Turn one harness results directory into GoldRows.

    Args:
        results_dir: A ``swegemma eval --results-dir`` output holding ``factory_env.json``.
        harness: ``as_shipped``, ``diagnostic`` or ``public_docker``.
        harness_route: Must equal ``HARNESS_ROUTES[harness]``.
        mode: ``gold`` or ``empty``.
        run_index: Repeat index of this run.

    Returns:
        One row per ``task_results.jsonl`` line, classified from its test log.

    Raises:
        ValueError: If ``harness_route`` does not belong to ``harness``.
        IngestError: If the env file, the results file or a required field is missing.
    """
    if HARNESS_ROUTES.get(harness) != harness_route:
        raise ValueError(f"harness_route {harness_route!r} does not belong to {harness!r}")
    expect = {"harness": harness, "harness_route": harness_route, "mode": mode}
    env = _read_env_file(results_dir, {**expect, "run_index": run_index})
    rows = []
    for rec in _task_rows(results_dir):
        tid = rec["instance_id"]
        log = results_dir / "test_outputs" / f"{tid.replace('/', '__')}.log"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        if rec.get("error"):
            text += f"\n{rec['error']}"
        cause = _root_cause(harness, mode, rec["resolved"], text)
        rows.append(
            GoldRow(
                task_id=tid,
                env_hash=env["env_hash"],
                wheelhouse_sha=env["wheelhouse_sha"],
                mode=mode,
                run_index=run_index,
                resolved=rec["resolved"],
                root_cause=cause,  # type: ignore[arg-type]
                log_path=str(log) if log.is_file() else "",
                harness_route=harness_route,
            )
        )
    return rows


def gold_ledger_path() -> Path:
    """Return the gold-validation ledger: ``<data_root>/ledger/gold_validation.jsonl``."""
    return data_root() / "ledger" / "gold_validation.jsonl"


def append_ledger_rows(
    rows: list[GoldRow], path: Path | None = None, sweep_id: str | None = None
) -> None:
    """Append rows to the gold-validation ledger: utf-8, one JSON object per line, fsync.

    Each record carries the GoldRow fields plus the spec's GOLD_VALIDATION columns
    ``gold_passes`` / ``empty_fails`` (null for the other mode) and ``checked_at_utc``, and
    ``sweep_id`` (joins against the drift ledger, ``local_env.drifted_sweep_ids``).
    """
    path = path or gold_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            rec = row.model_dump()
            rec["gold_passes"] = row.resolved if row.mode == "gold" else None
            rec["empty_fails"] = (not row.resolved) if row.mode == "empty" else None
            rec["checked_at_utc"] = now
            rec["sweep_id"] = sweep_id
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _run_one(
    task_id: str, harness: Harness, mode: Mode, run_index: int, sweep_dir: Path, env: dict
) -> list[GoldRow] | None:
    """Run one task once; append its row to the ledger and print one progress line."""
    route = HARNESS_ROUTES[harness]
    task_dir = sweep_dir / f"r{run_index}" / task_id
    write_env_file(
        task_dir,
        {**env, "harness": harness, "harness_route": route, "mode": mode, "run_index": run_index},
    )
    log = sweep_dir / "logs" / f"{task_id}.{mode}.r{run_index}.log"
    start = time.monotonic()
    if harness == "public_docker":
        argv, run = docker_kit.kit_argv(mode, task_id, task_dir), docker_kit.kit_logged
    else:
        argv, run = eval_argv(harness, mode, task_id, to_wsl_path(task_dir)), wsl_logged
    rc = run(argv, log, TASK_TIMEOUT_S)
    secs = time.monotonic() - start
    try:
        rows = ingest_sweep(task_dir, harness, route, mode, run_index) if rc == 0 else []
    except IngestError as exc:
        print(f"INGEST-ERROR: {task_id}: {exc}", flush=True)
        rows = []
    if len(rows) != 1 or rows[0].task_id != task_id:
        print(
            f"TASK {task_id} mode={mode} run={run_index} HARNESS-ERROR exit={rc} "
            f"seconds={secs:.1f} log={log}",
            flush=True,
        )
        return None
    append_ledger_rows(rows, sweep_id=sweep_dir.name)
    r = rows[0]
    print(
        f"TASK {task_id} mode={mode} run={run_index} resolved={r.resolved} "
        f"root_cause={r.root_cause} seconds={secs:.1f}",
        flush=True,
    )
    return rows


def sweep(task_ids: list[str], harness: Harness, mode: Mode, repeats: int = 1) -> list[GoldRow]:
    """Run ``task_ids`` x ``repeats`` under one harness and mode (WSL1, or the Docker kit).

    Each task runs as its own ``swegemma eval`` (``--concurrency 1``) into
    ``<results_dir>/local_eval/<stamp>-<harness>-<mode>/r<k>/<task>``, with the distro
    output in ``logs/``. Rows append to the ledger as each task finishes.

    Returns:
        All rows, in run order.

    Raises:
        LockHeldError, StaleLockError: If another sweep holds (or left) the sweep lock.
        PinMismatchError: Before any run, if a pin no longer holds.
        SweepDriftError: After the sweep, if the env re-measured at the end differs from the
            start (the sweep is marked in the drift ledger; its rows stay as appended).
        SweepError: After the sweep, if any task run produced no result.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = results_dir() / "local_eval" / f"{stamp}-{harness}-{mode}"
    rows: list[GoldRow] = []
    failed: list[str] = []
    kit = harness == "public_docker"
    if kit:  # refuses (KitError) before the lock and the sweep dir: no daemon/socket/image
        docker_kit.kit_preflight()
    with sweep_lock(lock_path(), f"{harness}-{mode} {sweep_dir.name}"):
        sweep_dir.mkdir(parents=True, exist_ok=True)
        env = docker_kit.measure_kit_env(sweep_dir) if kit else measure_env(harness, sweep_dir)
        print(
            f"SWEEP {harness} route={HARNESS_ROUTES[harness]} mode={mode} "
            f"tasks={len(task_ids)} repeats={repeats} env_hash={env['env_hash']} dir={sweep_dir}",
            flush=True,
        )
        for k in range(repeats):
            for tid in task_ids:
                got = _run_one(tid, harness, mode, k, sweep_dir, env)
                if got is None:
                    failed.append(f"{tid}@r{k}")
                else:
                    rows.extend(got)
        if kit:
            drift = docker_kit.end_of_kit_check(env, sweep_dir)
        else:
            drift = end_of_sweep_check(harness, env, sweep_dir)
    resolved = sum(r.resolved for r in rows)
    print(f"SWEEP-DONE rows={len(rows)} resolved={resolved} failed={len(failed)}", flush=True)
    if drift:
        raise SweepDriftError(f"{drift} ({len(failed)} task run(s) also failed)", rows)
    if failed:
        raise SweepError(failed, rows)
    return rows
