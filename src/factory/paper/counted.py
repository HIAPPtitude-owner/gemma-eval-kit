"""The counted E1 gold/empty sweeps and their re-ingested rows (moved from reports/t3_coverage_tables.py).

Rows are re-ingested with ``factory.local_eval.ingest_sweep`` at the current commit, so
classification is consistent across sweeps; nothing is read from prose.
"""

from pathlib import Path

from factory.local_eval import HARNESS_ROUTES, GoldRow, ingest_sweep
from factory.paths import results_dir


def le_dir() -> Path:
    """``<results>/local_eval``, resolved per call: importing this module never needs the
    data root (the release's ``paths.py`` has no default and raises ``DataRootError``)."""
    return results_dir() / "local_eval"


# Counted measurements (R28/R29/R33): harness, mode -> (sweep dirs, sweep logs, end-check).
COUNTED = {
    ("as_shipped", "gold"): (
        ["20261017T012915Z-as_shipped-gold"],
        ["s6_as_shipped_gold.log"],
        "R29 orchestrator end-state receipt (sweep pre-dates the in-code check)",
    ),
    ("as_shipped", "empty"): (
        ["20261017T021756Z-as_shipped-empty", "20261017T163246Z-as_shipped-empty"],
        ["s6_as_shipped_empty.log", "s6_as_shipped_empty_patch18.log"],
        "R33: killed sweep (orchestrator end-state receipt) + 18-task patch (in-code check)",
    ),
    ("diagnostic", "gold"): (
        ["20261021T161943Z-diagnostic-gold"],
        ["s7_diag_gold.log"],
        "in-code end-of-sweep check",
    ),
    ("diagnostic", "empty"): (
        ["20261021T185850Z-diagnostic-empty"],
        ["s7_diag_empty.log"],
        "in-code end-of-sweep check",
    ),
    ("public_docker", "gold"): (
        ["20261108T051249Z-public_docker-gold"],
        ["s13b_kit_e1_gold.log"],
        "in-code end-of-sweep check",
    ),
    ("public_docker", "empty"): (
        ["20261108T065744Z-public_docker-empty"],
        ["s13b_kit_e1_empty.log"],
        "in-code end-of-sweep check",
    ),
}


def task_dirs(sweep: str) -> list[Path]:
    return sorted(
        p for p in (le_dir() / sweep / "r0").iterdir() if (p / "task_results.jsonl").is_file()
    )


def rows_for(harness: str, mode: str) -> dict[str, GoldRow]:
    out: dict[str, GoldRow] = {}
    for sweep in COUNTED[(harness, mode)][0]:
        for d in task_dirs(sweep):
            for row in ingest_sweep(d, harness, HARNESS_ROUTES[harness], mode, 0):  # type: ignore[arg-type]
                assert row.task_id not in out, f"duplicate {row.task_id} in {harness}/{mode}"
                out[row.task_id] = row
    assert len(out) == 129, f"{harness}/{mode}: {len(out)} rows"
    return out
