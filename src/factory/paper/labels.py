"""The released dev-129 labels: one single-run row per (task, route), from the counted sweeps.

This is the only source of ``labels/dev129_labels.csv`` (ledger ruling R2): the release
build (``scripts/build_release.py``) and the Kaggle dataset export both call
``write_labels_csv``. A row holds IDs and our own outcomes only: the task id, its repo, the
route (harness name), whether the gold patch passed, whether the empty patch failed, our
root-cause class for a failed gold run, and ``runs`` (always 1: the labels are single-run).
"""

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path

from factory.local_eval import GoldRow
from factory.paper.counted import rows_for
from factory.paths import competition_dir

ROUTES = ("as_shipped", "diagnostic", "public_docker")
LABEL_COLUMNS = ("task_id", "repo", "route", "gold_pass", "empty_fail", "root_cause", "runs")
RUNS = 1  # Single-run labels (spec 7b.3): no 3/3 stability label in this release.
N_TASKS = 129


def task_repos(comp: Path | None = None) -> dict[str, str]:
    """Return ``instance_id -> repo`` from the competition's ``tasks.jsonl``."""
    text = ((comp or competition_dir()) / "tasks.jsonl").read_text(encoding="utf-8")
    return {
        t["instance_id"]: t["repo"] for t in map(json.loads, filter(str.strip, text.split("\n")))
    }


class LabelError(ValueError):
    """Raised when the counted sweeps cannot give the full single-run label table."""


def label_rows(
    comp: Path | None = None, rows: Callable[[str, str], Mapping[str, GoldRow]] = rows_for
) -> list[dict[str, str]]:
    """Return the 387 label rows (129 tasks x 3 routes), sorted by route then task id.

    Args:
        comp: Competition dir for the repo column (default ``factory.paths``).
        rows: ``(harness, mode) -> {task_id: GoldRow}`` (``counted.rows_for``; tests fake it).

    Raises:
        LabelError: If a route's gold and empty rows differ in tasks or are not 129, or a task
            id is missing from ``tasks.jsonl``.
    """
    repos = task_repos(comp)
    out: list[dict[str, str]] = []
    for route in ROUTES:
        gold, empty = rows(route, "gold"), rows(route, "empty")
        if set(gold) != set(empty) or len(gold) != N_TASKS:
            raise LabelError(f"{route}: gold {len(gold)} / empty {len(empty)} tasks differ")
        missing = sorted(set(gold) - set(repos))
        if missing:
            raise LabelError(f"{route}: task ids not in tasks.jsonl: {missing[:5]}")
        for task in sorted(gold):
            g, e = gold[task], empty[task]
            out.append(
                {
                    "task_id": task,
                    "repo": repos[task],
                    "route": route,
                    "gold_pass": str(g.resolved).lower(),
                    "empty_fail": str(not e.resolved).lower(),
                    "root_cause": "" if g.resolved else (g.root_cause or ""),
                    "runs": str(RUNS),
                }
            )
    return out


def write_labels_csv(path: Path, comp: Path | None = None) -> int:
    """Write ``label_rows`` to ``path`` as CSV (LF line ends) and return the row count."""
    rows = label_rows(comp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
