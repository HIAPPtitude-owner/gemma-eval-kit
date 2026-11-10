"""Tests for factory.paper.labels on synthetic rows (no data root needed)."""

import json
from pathlib import Path

import pytest

from factory.local_eval import HARNESS_ROUTES, GoldRow
from factory.paper.labels import LABEL_COLUMNS, LabelError, label_rows

IDS = [f"demo_{i}" for i in range(129)]


def _comp(tmp_path: Path, ids: list[str] = IDS) -> Path:
    lines = [json.dumps({"instance_id": t, "repo": "o/demo"}) for t in ids]
    (tmp_path / "tasks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _row(task: str, harness: str, mode: str, resolved: bool) -> GoldRow:
    return GoldRow(
        task_id=task,
        env_hash="e",
        wheelhouse_sha="w",
        mode=mode,
        run_index=0,
        resolved=resolved,
        root_cause=None if resolved else "missing_dependency",
        log_path="x.log",
        harness_route=HARNESS_ROUTES[harness],
    )


def _rows(ids: list[str] = IDS, empty_ids: list[str] | None = None):
    def rows(harness: str, mode: str) -> dict[str, GoldRow]:
        pick = ids if mode == "gold" or empty_ids is None else empty_ids
        return {t: _row(t, harness, mode, mode == "gold" and t != "demo_0") for t in pick}

    return rows


def test_full_table_and_csv(tmp_path: Path) -> None:
    rows = label_rows(_comp(tmp_path), _rows())
    assert len(rows) == 387 and tuple(rows[0]) == LABEL_COLUMNS
    first = {r["route"]: r for r in rows if r["task_id"] == "demo_0"}
    assert first["public_docker"]["gold_pass"] == "false"
    assert first["public_docker"]["root_cause"] == "missing_dependency"
    assert {r["runs"] for r in rows} == {"1"} and {r["empty_fail"] for r in rows} == {"true"}


def test_gold_and_empty_task_sets_must_match(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="differ"):
        label_rows(_comp(tmp_path), _rows(empty_ids=IDS[:-1] + ["other"]))


def test_fewer_than_129_tasks_refused(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="128"):
        label_rows(_comp(tmp_path), _rows(ids=IDS[:-1]))


def test_task_missing_from_tasks_jsonl_refused(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="not in tasks.jsonl"):
        label_rows(_comp(tmp_path, IDS[:-1]), _rows())
