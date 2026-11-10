"""C5 on the ``diagnostic`` harness: score patches that must NOT resolve (plan T3 Step 5).

C5 scores, on gold-valid tasks, the empty patch, a malformed patch (``"not a diff\\n"``) and
the gold patch of a different task. Each run goes through ``docker_semantics.py``'s
``--gold <driver>`` slot with ``harness/shared/c5_driver.py`` (ruling R35), which substitutes
the file named by ``$FACTORY_C5_PATCH_FILE`` as agent_patch. Runs reuse the sweep machinery
(lock, pins, env measure, end-of-sweep drift check) but do NOT append to the gold-validation
ledger: they are not gold/empty measurements.
"""

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from factory.local_env import (
    DOCKER_SEMANTICS,
    PINNED_FILES,
    WSL_PY,
    SweepDriftError,
    end_of_sweep_check,
    lock_path,
    measure_env,
    sha256_file,
    to_wsl_path,
    wsl_logged,
)
from factory.local_eval import (
    TASK_TIMEOUT_S,
    IngestError,
    SweepError,
    _task_rows,
    c5_verdict,
    eval_argv,
)
from factory.paths import competition_dir, results_dir
from factory.pins import PINS_PATH, REPO_ROOT, PinMismatchError
from factory.sweep_lock import sweep_lock

PIN_KEY = "c5_driver_py_sha256"
C5_DRIVER = REPO_ROOT / PINNED_FILES[PIN_KEY]
PATCH_ENV = "FACTORY_C5_PATCH_FILE"
MALFORMED_PATCH = "not a diff\n"
LABELS = ("empty", "malformed", "wrong")
EMPTY_LINE = re.compile(r"^TASK (\S+) mode=empty run=\d+ resolved=(True|False)\b", re.MULTILINE)


@dataclass(frozen=True)
class C5Run:
    """One scored input: ``label`` is empty, malformed or wrong (``donor``'s gold patch)."""

    task_id: str
    label: str
    donor: str | None = None


def parse_manifest(text: str, known: set[str]) -> list[C5Run]:
    """Parse ``<task_id><TAB><empty|malformed|wrong:<donor_id>>`` lines (``#`` comments).

    Raises:
        ValueError: On a malformed line, an unknown task or donor, or a donor equal to its task.
    """
    runs = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"manifest line {n}: expected '<task>\\t<input>', got {raw!r}")
        task, spec = parts[0].strip(), parts[1].strip()
        label, _, donor = spec.partition(":")
        if task not in known:
            raise ValueError(f"manifest line {n}: unknown task {task!r}")
        if label in ("empty", "malformed") and not donor:
            runs.append(C5Run(task, label))
        elif label == "wrong" and donor in known and donor != task:
            runs.append(C5Run(task, label, donor))
        else:
            raise ValueError(f"manifest line {n}: bad input {spec!r} for task {task!r}")
    return runs


def load_gold_patches(tasks_jsonl: Path | None = None) -> dict[str, str]:
    """Return ``{instance_id: gold patch}`` from ``tasks.jsonl``."""
    tasks_jsonl = tasks_jsonl or competition_dir() / "tasks.jsonl"
    with tasks_jsonl.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    return {r["instance_id"]: r["patch"] for r in recs}


def input_patch(run: C5Run, gold: dict[str, str]) -> str:
    """Return the patch text C5 scores for ``run``."""
    if run.label == "empty":
        return ""
    if run.label == "malformed":
        return MALFORMED_PATCH
    if run.donor is None:
        raise ValueError(f"{run}: 'wrong' needs a donor task")
    return gold[run.donor]


def c5_argv(task_id: str, results_wsl: str, patch_wsl: str, driver: Path) -> list[str]:
    """In-distro argv: diagnostic launcher with ``driver`` in the ``--gold`` slot."""
    launch = [WSL_PY, to_wsl_path(DOCKER_SEMANTICS), "--gold", to_wsl_path(driver)]
    return eval_argv(
        "diagnostic", "gold", task_id, results_wsl, launch=launch, extra_env={PATCH_ENV: patch_wsl}
    )


def empty_results(log_text: str) -> dict[str, bool]:
    """Read ``TASK <id> mode=empty ... resolved=<bool>`` lines from a sweep log."""
    return {m[1]: m[2] == "True" for m in EMPTY_LINE.finditer(log_text)}


def task_verdicts(empty: dict[str, bool], scored: dict[C5Run, bool]) -> dict[str, str]:
    """Per-task C5 verdict: empty from the empty sweep, malformed/wrong from ``scored``."""
    out = {}
    for task in sorted({r.task_id for r in scored}):
        res = {r.label: ok for r, ok in scored.items() if r.task_id == task}
        if task in empty:
            res.setdefault("empty", empty[task])
        out[task] = c5_verdict(res)
    return out


def _run_one(run: C5Run, gold: dict[str, str], sweep_dir: Path, driver: Path) -> bool | None:
    name = f"{run.task_id}.{run.label}"
    task_dir = sweep_dir / "runs" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    patch = input_patch(run, gold)
    patch_file = sweep_dir / "inputs" / f"{name}.patch"
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_bytes(patch.encode("utf-8"))
    log = sweep_dir / "logs" / f"{name}.log"
    start = time.monotonic()
    argv = c5_argv(run.task_id, to_wsl_path(task_dir), to_wsl_path(patch_file), driver)
    rc = wsl_logged(argv, log, TASK_TIMEOUT_S)
    secs = time.monotonic() - start
    try:
        recs = _task_rows(task_dir) if rc == 0 else []
    except IngestError as exc:
        print(f"INGEST-ERROR: {name}: {exc}", flush=True)
        recs = []
    if len(recs) != 1 or recs[0]["instance_id"] != run.task_id:
        print(f"TASK {name} HARNESS-ERROR exit={rc} seconds={secs:.1f} log={log}", flush=True)
        return None
    rec = recs[0]
    print(
        f"TASK {run.task_id} input={run.label} donor={run.donor} resolved={rec['resolved']} "
        f"input_bytes={len(patch.encode('utf-8'))} agent_patch_size={rec.get('agent_patch_size')} "
        f"test_exit_code={rec.get('test_exit_code')} seconds={secs:.1f} log={log}",
        flush=True,
    )
    return bool(rec["resolved"])


def driver_sha_checked(driver: Path, pinned: bool, pins_path: Path = PINS_PATH) -> str:
    """Return the sha256 of the driver bytes about to run.

    Raises:
        PinMismatchError: If ``pinned`` and that sha differs from pins.json's c5 pin.
    """
    sha = sha256_file(driver)
    if pinned:
        pin = json.loads(pins_path.read_text(encoding="utf-8"))["harness"][PIN_KEY]
        if sha != pin:
            raise PinMismatchError(f"{driver}: sha256 {sha}, pinned {pin}")
    return sha


def c5_sweep(runs: list[C5Run], driver: Path = C5_DRIVER) -> dict[C5Run, bool]:
    """Score every run under the diagnostic harness; write ``c5_results.json`` in the dir.

    ``driver`` is the pinned ``c5_driver.py``; another path is for mutant receipts only, and
    its sweep dir is tagged ``-unpinned-driver``.

    Raises:
        SweepDriftError: If the env drifted during the runs.
        SweepError: If any run produced no harness result.
    """
    tag = "c5" if driver.resolve() == C5_DRIVER.resolve() else "c5-unpinned-driver"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = results_dir() / "local_eval" / f"{stamp}-diagnostic-{tag}"
    gold = load_gold_patches()
    scored: dict[C5Run, bool] = {}
    failed: list[str] = []
    with sweep_lock(lock_path(), f"diagnostic-{tag} {sweep_dir.name}"):
        sweep_dir.mkdir(parents=True, exist_ok=True)
        # measure_env runs assert_harness_pins first, so a repo c5_driver.py whose bytes differ
        # from pins.json is refused before this line; driver_sha_checked re-hashes the bytes
        # that will actually run and, for the pinned driver, compares them with the pin again.
        env = measure_env("diagnostic", sweep_dir)
        driver_sha = driver_sha_checked(driver, pinned=tag == "c5")
        print(
            f"SWEEP diagnostic {tag} runs={len(runs)} env_hash={env['env_hash']} "
            f"driver_sha256={driver_sha} driver={driver} dir={sweep_dir}",
            flush=True,
        )
        for run in runs:
            got = _run_one(run, gold, sweep_dir, driver)
            if got is None:
                failed.append(f"{run.task_id}.{run.label}")
            else:
                scored[run] = got
        drift = end_of_sweep_check("diagnostic", env, sweep_dir)
        end_sha = sha256_file(driver)
        if end_sha != driver_sha:
            drift = f"{drift or ''} driver sha256 changed: start {driver_sha} end {end_sha}"
            print(f"DRIFT: {drift}", flush=True)
    out = [{**r.__dict__, "resolved": ok} for r, ok in scored.items()]
    record = {"env_hash": env["env_hash"], "driver": str(driver), "driver_sha256": driver_sha}
    text = json.dumps({**record, "runs": out}, indent=2)
    (sweep_dir / "c5_results.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"SWEEP-DONE runs={len(scored)} failed={len(failed)} dir={sweep_dir}", flush=True)
    if drift:
        raise SweepDriftError(drift, [])
    if failed:
        raise SweepError(failed, [])
    return scored
