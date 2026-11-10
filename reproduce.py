"""Recompute every number in the paper from the Kaggle dataset and check it (the notebook).

Run: ``python reproduce.py [DATASET_DIR]``. ``DATASET_DIR`` holds ``labels/dev129_labels.csv``,
``e2_rows.csv`` and ``paper_numbers.json``; on Kaggle it defaults to the first folder under
``/kaggle/input`` that holds ``paper_numbers.json``. Every key is recomputed by
``factory.paper.numbers.build`` (the same code that printed the paper's numbers), fed from the
two CSVs instead of our ledgers, and compared with ``paper_numbers.json``. Prints
``REPRODUCED N keys`` and exits 0 when all N match, else one ``MISMATCH:`` line per key and 1.

Run-time needs: Python 3.12 or newer (the release's ``requires-python``), and ``factory`` from
the gemma-eval-kit release. When ``factory`` is not importable the script pip-installs it from
``CODE_URL`` (the public GitHub repo), so the Kaggle notebook runs with internet on, after that
repo is public. Nothing from the competition download is needed or installed.
"""

from __future__ import annotations  # so an old Python reaches check_python's message

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

CODE_URL = "git+https://github.com/HIAPPtitude-owner/gemma-eval-kit"
MIN_PYTHON = (3, 12)
KAGGLE_INPUT = Path("/kaggle/input")
LABELS = "labels/dev129_labels.csv"


class Outcome(NamedTuple):
    """The two fields ``numbers.e1_numbers`` reads from a gold or empty row."""

    resolved: bool
    root_cause: str | None


def check_python(version: tuple[int, ...] = tuple(sys.version_info[:2])) -> None:
    """Exit with a clear message on a Python older than the release supports.

    Raises:
        SystemExit: If ``version`` is below ``MIN_PYTHON``.
    """
    if tuple(version[:2]) < MIN_PYTHON:
        need = ".".join(map(str, MIN_PYTHON))
        have = ".".join(map(str, version[:2]))
        raise SystemExit(f"REPRODUCE-ERROR: gemma-eval-kit needs Python {need}+, this is {have}")


def _ensure_factory() -> None:
    try:
        importlib.import_module("factory.paths")
        return
    except ImportError:
        pass
    print(f"installing gemma-eval-kit from {CODE_URL}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", CODE_URL], check=True)
    importlib.invalidate_caches()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _bool(text: str) -> bool:
    if text not in ("true", "false"):
        raise ValueError(f"not a boolean cell: {text!r}")
    return text == "true"


def e1_inputs(dataset: Path) -> tuple[dict[tuple[str, str], dict[str, Outcome]], dict[str, str]]:
    """``(route, mode) -> {task_id: Outcome}`` and ``task_id -> repo`` from the labels CSV."""
    rows: dict[tuple[str, str], dict[str, Outcome]] = {}
    repos: dict[str, str] = {}
    for r in _read_csv(dataset / LABELS):
        gold = Outcome(_bool(r["gold_pass"]), r["root_cause"] or None)
        rows.setdefault((r["route"], "gold"), {})[r["task_id"]] = gold
        empty = Outcome(not _bool(r["empty_fail"]), None)
        rows.setdefault((r["route"], "empty"), {})[r["task_id"]] = empty
        repos[r["task_id"]] = r["repo"]
    return rows, repos


def e2_inputs(dataset: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, bool]]]:
    """The E2 rows (booleans parsed) and ``config -> task_id -> kaggle_resolved`` from them."""
    rows: list[dict[str, Any]] = []
    kaggle: dict[str, dict[str, bool]] = {}
    for r in _read_csv(dataset / "e2_rows.csv"):
        row: dict[str, Any] = {**r, "kaggle_resolved": _bool(r["kaggle_resolved"])}
        row["local_resolved"] = _bool(r["local_resolved"])
        rows.append(row)
        kaggle.setdefault(row["config"], {})[row["task_id"]] = row["kaggle_resolved"]
    return rows, kaggle


def recompute(dataset: Path) -> dict[str, Any]:
    """Every paper key, recomputed by ``numbers.build`` from the dataset's two CSVs."""
    from factory.paper import numbers

    e1_rows, repos = e1_inputs(dataset)
    e2, kaggle = e2_inputs(dataset)
    return numbers.build(lambda r, m: e1_rows[(r, m)], repos, e2, kaggle)


def compare(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """One ``MISMATCH:`` line per key that is missing, extra or different."""
    return [
        f"MISMATCH: {k} recomputed={got.get(k)!r} paper={want.get(k)!r}"
        for k in sorted(got.keys() | want.keys())
        if k not in got or k not in want or got[k] != want[k]
    ]


def find_dataset(argv: list[str]) -> Path:
    if argv:
        return Path(argv[0])
    for d in sorted(KAGGLE_INPUT.glob("**/paper_numbers.json")):
        return d.parent
    raise FileNotFoundError(f"no paper_numbers.json under {KAGGLE_INPUT}; pass DATASET_DIR")


def main(argv: list[str] | None = None) -> int:
    check_python()
    dataset = find_dataset(sys.argv[1:] if argv is None else argv)
    _ensure_factory()  # no data root needed: every input comes from the dataset's CSVs
    want = json.loads((dataset / "paper_numbers.json").read_text(encoding="utf-8"))
    bad = compare(recompute(dataset), want)
    for line in bad:
        print(line, flush=True)
    if bad:
        print(f"NOT-REPRODUCED: {len(bad)} of {len(want)} keys differ", flush=True)
        return 1
    print(f"REPRODUCED {len(want)} keys", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
