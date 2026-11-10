"""E2: re-score the stored Kaggle agent patches on the three local routes (plan T3).

R2 (config ``v1``) and R3 (config ``v1_calls50``) ran the agent on Kaggle; their results dirs
hold ``task_results.jsonl`` (the Kaggle-route verdict) and ``patches/<task>.patch`` (the
agent's patch; absent when the agent produced none, ``agent_patch_size`` 0). Each stored patch
is scored again on ``as_shipped`` and ``diagnostic`` (WSL1) and ``public_docker`` (the Docker
kit) through ``harness/shared/c5_driver.py``, which substitutes the file named by
``$FACTORY_C5_PATCH_FILE`` as agent_patch. A missing patch is scored as an empty file.

The per-task loop follows ``factory.c5._run_one``; the env measure, sweep lock and drift
check follow ``factory.local_eval.sweep``. Rows go to ``<data_root>/ledger/rescore.jsonl``.
None of the three routes is the scorer, and neither is the Kaggle route.
"""

import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from factory import docker_kit
from factory.c5 import C5_DRIVER, PATCH_ENV
from factory.local_env import (
    DOCKER_SEMANTICS,
    HARNESS_ROUTES,
    WSL_PY,
    Harness,
    SweepDriftError,
    end_of_sweep_check,
    lock_path,
    measure_env,
    to_wsl_path,
    wsl_logged,
)
from factory.local_eval import (
    TASK_TIMEOUT_S,
    IngestError,
    SweepError,
    _task_rows,
    classify,
    eval_argv,
)
from factory.paths import data_root, results_dir
from factory.sweep_lock import sweep_lock

CONFIGS = {"v1": "R2/v1/results/v1", "v1_calls50": "R3/v1/results/v1_calls50"}


class StoredOutcome(BaseModel):
    """One task of a stored Kaggle run: its verdict there and its agent patch (if any)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    repo: str
    kaggle_resolved: bool
    patch_path: Path | None


class RescoreRow(BaseModel):
    """One stored patch re-scored on one local route (a rescore.jsonl ledger row)."""

    model_config = ConfigDict(frozen=True)

    config: str
    task_id: str
    repo: str
    harness_route: str
    env_hash: str
    kaggle_resolved: bool
    local_resolved: bool
    root_cause: str | None
    input_bytes: int
    log_path: str


def load_stored_from(results: Path) -> list[StoredOutcome]:
    """Read a stored Kaggle results dir into one StoredOutcome per ``task_results`` row.

    A row's patch is ``patches/<task_id>.patch``; its byte size must equal the row's
    ``agent_patch_size``, and a row with size 0 must have no file.

    Raises:
        IngestError: On a missing ``repo``/``agent_patch_size`` field, a duplicate task, a
            patch whose task has no row, or a patch that disagrees with ``agent_patch_size``.
    """
    patches = results / "patches"
    files = {p.name.removesuffix(".patch"): p for p in patches.glob("*.patch")}
    outcomes: list[StoredOutcome] = []
    for rec in _task_rows(results):
        tid = rec["instance_id"]
        for field in ("repo", "agent_patch_size"):
            if field not in rec:
                raise IngestError(f"{results}: task {tid}: missing field {field!r}")
        if any(o.task_id == tid for o in outcomes):
            raise IngestError(f"{results}: task {tid} appears twice in task_results.jsonl")
        path = files.pop(tid, None)
        size = path.stat().st_size if path is not None else 0
        if size != rec["agent_patch_size"]:
            raise IngestError(
                f"{results}: task {tid}: agent_patch_size {rec['agent_patch_size']} but "
                f"patch file {'missing' if path is None else f'has {size} bytes'}"
            )
        outcomes.append(
            StoredOutcome(
                task_id=tid,
                repo=rec["repo"],
                kaggle_resolved=rec["resolved"],
                patch_path=path if size else None,
            )
        )
    if files:
        raise IngestError(f"{patches}: patch(es) with no task_results row: {sorted(files)}")
    return outcomes


def load_stored(config: str) -> list[StoredOutcome]:
    """Load a stored config (``v1`` or ``v1_calls50``) from under ``results_dir()``.

    Raises:
        ValueError: If ``config`` is not in ``CONFIGS``.
        IngestError: As ``load_stored_from``.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {sorted(CONFIGS)}")
    return load_stored_from(results_dir() / CONFIGS[config])


def input_bytes_for(o: StoredOutcome) -> int:
    """Return the byte size of the patch scored for ``o`` (0: the empty input)."""
    return o.patch_path.stat().st_size if o.patch_path is not None else 0


def rescore_launch(harness: Harness) -> list[str]:
    """In-distro launcher for a WSL route: ``c5_driver.py``, wrapped on ``diagnostic``.

    Raises:
        ValueError: For ``public_docker``, which launches through ``docker_kit.kit_argv``.
    """
    driver = to_wsl_path(C5_DRIVER)
    if harness == "as_shipped":
        return [WSL_PY, driver]
    if harness == "diagnostic":
        return [WSL_PY, to_wsl_path(DOCKER_SEMANTICS), "--gold", driver]
    raise ValueError(f"{harness}: no in-distro launcher (public_docker uses kit_argv)")


def rescore_ledger_path() -> Path:
    """Return the E2 ledger: ``<data_root>/ledger/rescore.jsonl``."""
    return data_root() / "ledger" / "rescore.jsonl"


def append_rescore_row(row: RescoreRow, sweep_id: str) -> None:
    """Append one row (plus ``sweep_id`` and ``checked_at_utc``) to the ledger, with fsync.

    ``sweep_id`` joins against the drift ledger (``local_env.drifted_sweep_ids``).
    """
    path = rescore_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        **row.model_dump(),
        "sweep_id": sweep_id,
        "checked_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_input(o: StoredOutcome, sweep_dir: Path) -> Path:
    """Copy the stored patch byte for byte (an empty file when there is none)."""
    patch_file = sweep_dir / "inputs" / f"{o.task_id}.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    if o.patch_path is None:
        patch_file.write_bytes(b"")
    else:
        shutil.copyfile(o.patch_path, patch_file)
    return patch_file


def _run_task(o: StoredOutcome, harness: Harness, task_dir: Path, patch: Path, log: Path) -> int:
    if harness == "public_docker":
        argv = docker_kit.kit_argv("gold", o.task_id, task_dir, C5_DRIVER, patch)
        return docker_kit.kit_logged(argv, log, TASK_TIMEOUT_S)
    argv = eval_argv(
        harness,
        "gold",
        o.task_id,
        to_wsl_path(task_dir),
        launch=rescore_launch(harness),
        extra_env={PATCH_ENV: to_wsl_path(patch)},
    )
    return wsl_logged(argv, log, TASK_TIMEOUT_S)


def _run_one(
    o: StoredOutcome, config: str, harness: Harness, sweep_dir: Path, env: dict[str, Any]
) -> RescoreRow | None:
    """Score one stored patch; append its row and print one ``TASK`` line (None: no result)."""
    task_dir = sweep_dir / "runs" / o.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    patch = _write_input(o, sweep_dir)
    log = sweep_dir / "logs" / f"{o.task_id}.log"
    start = time.monotonic()
    rc = _run_task(o, harness, task_dir, patch, log)
    secs = time.monotonic() - start
    try:
        recs = _task_rows(task_dir) if rc == 0 else []
    except IngestError as exc:
        print(f"INGEST-ERROR: {o.task_id}: {exc}", flush=True)
        recs = []
    if len(recs) != 1 or recs[0]["instance_id"] != o.task_id:
        print(f"TASK {o.task_id} HARNESS-ERROR exit={rc} seconds={secs:.1f} log={log}", flush=True)
        return None
    rec = recs[0]
    test_log = task_dir / "test_outputs" / f"{o.task_id.replace('/', '__')}.log"
    text = test_log.read_text(encoding="utf-8", errors="replace") if test_log.is_file() else ""
    if rec.get("error"):
        text += f"\n{rec['error']}"
    row = RescoreRow(
        config=config,
        task_id=o.task_id,
        repo=o.repo,
        harness_route=HARNESS_ROUTES[harness],
        env_hash=env["env_hash"],
        kaggle_resolved=o.kaggle_resolved,
        local_resolved=rec["resolved"],
        root_cause=None if rec["resolved"] else classify(text, harness),
        input_bytes=input_bytes_for(o),
        log_path=str(test_log) if test_log.is_file() else "",
    )
    append_rescore_row(row, sweep_dir.name)
    print(
        f"TASK {o.task_id} config={config} kaggle={o.kaggle_resolved} "
        f"local={row.local_resolved} root_cause={row.root_cause} "
        f"input_bytes={row.input_bytes} seconds={secs:.1f} log={log}",
        flush=True,
    )
    return row


def rescore(config: str, harness: Harness) -> list[RescoreRow]:
    """Re-score every stored patch of ``config`` on ``harness``; print ``RESCORE-DONE``.

    Runs into ``<results_dir>/local_eval/<stamp>-<harness>-rescore-<config>``.

    Raises:
        ValueError, IngestError: Before any run, on an unknown config or bad stored data.
        KitError: Before the lock (public_docker), if the daemon, an image or the socket fails.
        LockHeldError, StaleLockError: If another sweep holds (or left) the sweep lock.
        PinMismatchError: Before any run, if a pin no longer holds.
        SweepDriftError: After the run, if the env re-measured at the end differs.
        SweepError: After the run, if any task produced no result.
    """
    stored = load_stored(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = results_dir() / "local_eval" / f"{stamp}-{harness}-rescore-{config}"
    kit = harness == "public_docker"
    if kit:
        docker_kit.kit_preflight()
    rows: list[RescoreRow] = []
    failed: list[str] = []
    with sweep_lock(lock_path(), f"rescore-{harness}-{config} {sweep_dir.name}"):
        sweep_dir.mkdir(parents=True, exist_ok=True)
        env = docker_kit.measure_kit_env(sweep_dir) if kit else measure_env(harness, sweep_dir)
        print(
            f"RESCORE {harness} route={HARNESS_ROUTES[harness]} config={config} "
            f"tasks={len(stored)} env_hash={env['env_hash']} dir={sweep_dir}",
            flush=True,
        )
        for o in stored:
            row = _run_one(o, config, harness, sweep_dir, env)
            if row is None:
                failed.append(o.task_id)
            else:
                rows.append(row)
        if kit:
            drift = docker_kit.end_of_kit_check(env, sweep_dir)
        else:
            drift = end_of_sweep_check(harness, env, sweep_dir)
    print(f"RESCORE-DONE rows={len(rows)} failed={len(failed)} dir={sweep_dir}", flush=True)
    if drift:
        raise SweepDriftError(f"{drift} ({len(failed)} task run(s) also failed)", [])
    if failed:
        raise SweepError(failed, [])
    return rows
