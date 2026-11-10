"""The paper's number printer: E1 and E2 ledgers -> reports/paper_numbers.json + paper-tables.md.

Run: ``uv run python -m factory.paper.numbers``. Every number the writeup cites is a flat key of
``build()``; T6 renders the prose from these keys. Percentages are rounded once, here, to 1 dp.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from functools import cache
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Protocol

from factory.local_eval import HARNESS_ROUTES
from factory.paper.counted import rows_for
from factory.paths import competition_dir
from factory.pins import REPO_ROOT
from factory.rescore import CONFIGS, load_stored, rescore_ledger_path

Numbers = dict[str, int | float | str]

ROUTES = ("as_shipped", "diagnostic", "public_docker")
CAUSES = [
    "missing_dependency",
    "wheel_shadowing",
    "host_venv_shadowing",
    "network_tls",
    "version_drift",
    "invalid_task",
    "unclassified",
]
# The six full E2 sweeps (40 rows each). The three 20261108T1644xxZ-*-rescore-v1 sweeps are
# smoke rows and are excluded.
E2_SWEEPS = (
    "20261109T015448Z-as_shipped-rescore-v1",
    "20261109T023637Z-diagnostic-rescore-v1",
    "20261109T032710Z-public_docker-rescore-v1",
    "20261109T041515Z-as_shipped-rescore-v1_calls50",
    "20261109T045335Z-diagnostic-rescore-v1_calls50",
    "20261109T054248Z-public_docker-rescore-v1_calls50",
)
E2_ROWS = 240
# Third-party public-kit gold pass count, cited only as a comparison (C+S 747248).
THIRD_PARTY_GOLD_PASS = 43
SIGN_NOTE = (
    "Sign test p is descriptive, not the pre-registered promotion rule. "
    "All outcomes are single-run."
)
JSON_OUT = REPO_ROOT / "reports" / "paper_numbers.json"
MD_OUT = REPO_ROOT / "reports" / "paper-tables.md"


class E1Row(Protocol):
    """What ``e1_numbers`` reads from a row: a ``GoldRow`` or the notebook's label ``Outcome``."""

    @property
    def resolved(self) -> bool: ...

    @property
    def root_cause(self) -> str | None: ...


def sign_test(b: int, c: int) -> float:
    """Exact two-sided sign test on ``b`` vs ``c`` discordant pairs (1.0 when there are none)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1)


def e1_numbers(
    rows: Callable[[str, str], Mapping[str, E1Row]] = rows_for,
    repos: Mapping[str, str] | None = None,
) -> Numbers:
    """E1 keys: per-route gold/empty coverage, gold-row causes, pairwise gold disagreement.

    ``rows`` (``(route, mode) -> {task_id: row}``) defaults to the counted sweeps and ``repos``
    (task_id -> repo) to ``tasks.jsonl``; the release notebook passes both from the exported
    labels CSV. Only a row's ``resolved`` and ``root_cause`` are read.
    """
    n: Numbers = {"e1.third_party_gold_pass": THIRD_PARTY_GOLD_PASS}
    gold: dict[str, Mapping[str, E1Row]] = {}
    for r in ROUTES:
        g, e = rows(r, "gold"), rows(r, "empty")
        assert g.keys() == e.keys(), f"{r}: gold and empty task sets differ"
        gold[r] = g
        n[f"e1.{r}.tasks"] = len(g)
        n[f"e1.{r}.gold_pass"] = sum(x.resolved for x in g.values())
        n[f"e1.{r}.empty_fail"] = sum(not x.resolved for x in e.values())
        n[f"e1.{r}.valid"] = sum(g[t].resolved and not e[t].resolved for t in g)
        causes = Counter(x.root_cause for x in g.values() if x.root_cause is not None)
        assert set(causes) <= set(CAUSES), f"{r}: unknown cause(s) {set(causes) - set(CAUSES)}"
        for c in CAUSES:
            n[f"e1.{r}.cause.{c}"] = causes.get(c, 0)
    for a, b in combinations(ROUTES, 2):
        ga, gb = gold[a], gold[b]
        assert ga.keys() == gb.keys(), f"{a}/{b}: task sets differ"
        n[f"e1.disagree.{a}__{b}.pass_fail"] = sum(
            ga[t].resolved and not gb[t].resolved for t in ga
        )
        n[f"e1.disagree.{a}__{b}.fail_pass"] = sum(
            not ga[t].resolved and gb[t].resolved for t in ga
        )
    repo = repos if repos is not None else _task_repos()
    n["scope.e1_repos"] = len({repo[t] for t in gold[ROUTES[0]]})
    return n


def _task_repos() -> dict[str, str]:
    """instance_id -> repo, from the competition's tasks.jsonl."""
    lines = (competition_dir() / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    return {t["instance_id"]: t["repo"] for t in map(json.loads, filter(str.strip, lines))}


def e2_rows(ledger: Path | None = None) -> list[dict[str, Any]]:
    """The 240 E2 rows: E2_SWEEPS only, the last row per (config, task_id, harness_route)."""
    path = ledger or rescore_ledger_path()
    last: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sweep_id"] in E2_SWEEPS:
            last[(row["config"], row["task_id"], row["harness_route"])] = row
    assert len(last) == E2_ROWS, f"{path}: {len(last)} E2 rows, expected {E2_ROWS}"
    return list(last.values())


def _route_of(harness_route: str) -> str:
    names = [h for h, v in HARNESS_ROUTES.items() if v == harness_route]
    assert len(names) == 1, f"harness_route {harness_route!r} maps to {names}"
    return names[0]


def _kaggle() -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for config in CONFIGS:
        out[config] = {o.task_id: o.kaggle_resolved for o in load_stored(config)}
    return out


def _paired(prefix: str, v1: dict[str, bool], c50: dict[str, bool]) -> Numbers:
    assert v1.keys() == c50.keys(), f"{prefix}: v1 and v1_calls50 task sets differ"
    only_v1 = sum(v1[t] and not c50[t] for t in v1)
    only_c50 = sum(c50[t] and not v1[t] for t in v1)
    return {
        f"{prefix}.both": sum(v1[t] and c50[t] for t in v1),
        f"{prefix}.only_v1": only_v1,
        f"{prefix}.only_c50": only_c50,
        f"{prefix}.sign_p": sign_test(only_v1, only_c50),
    }


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def e2_numbers(
    rows: list[dict[str, Any]], kaggle: dict[str, dict[str, bool]] | None = None
) -> Numbers:
    """E2 keys: per config x route agreement with Kaggle, per repo, and the paired verdict.

    ``kaggle`` (config -> task_id -> resolved) defaults to the stored outcomes (``load_stored``);
    every row's ``kaggle_resolved`` must equal it.
    """
    kaggle = kaggle if kaggle is not None else _kaggle()
    local: dict[tuple[str, str], dict[str, bool]] = {}
    repos = sorted({r["repo"] for r in rows})
    n: Numbers = {}
    for r in rows:
        assert r["kaggle_resolved"] == kaggle[r["config"]][r["task_id"]], f"kaggle drift: {r}"
        local.setdefault((r["config"], _route_of(r["harness_route"])), {})[r["task_id"]] = r[
            "local_resolved"
        ]
    by_repo = Counter(
        (r["config"], _route_of(r["harness_route"]), r["repo"])
        for r in rows
        if r["local_resolved"] != r["kaggle_resolved"]
    )
    for config in CONFIGS:
        n[f"e2.{config}.kaggle.resolved"] = sum(kaggle[config].values())
        for route in ROUTES:
            loc = local[(config, route)]
            assert loc.keys() == kaggle[config].keys(), f"{config}/{route}: task set differs"
            agree = sum(loc[t] == kaggle[config][t] for t in loc)
            p = f"e2.{config}.{route}"
            n[f"{p}.resolved"] = sum(loc.values())
            n[f"{p}.agree"] = agree
            n[f"{p}.disagree"] = len(loc) - agree
            n[f"{p}.disagree_pct"] = _pct(len(loc) - agree, len(loc))
            for repo in repos:
                n[f"{p}.repo.{repo}.disagree"] = by_repo[(config, route, repo)]
    n.update(_paired("e2.paired.kaggle", kaggle["v1"], kaggle["v1_calls50"]))
    kd = _sign(int(n["e2.paired.kaggle.only_v1"]) - int(n["e2.paired.kaggle.only_c50"]))
    flips = 0
    for route in ROUTES:
        n.update(_paired(f"e2.paired.{route}", local[("v1", route)], local[("v1_calls50", route)]))
        d = int(n[f"e2.paired.{route}.only_v1"]) - int(n[f"e2.paired.{route}.only_c50"])
        flips += _sign(d) != kd
    n["e2.paired.flips"] = flips
    n["scope.repos"] = len(repos)
    n["scope.tasks"] = len(kaggle["v1"])
    return n


GOLD_JOIN = (
    "agree_on_gold_pass",
    "disagree_on_gold_pass",
    "disagree_on_gold_fail",
    "kaggle_pass_local_fail",
    "kaggle_fail_local_pass",
)


def e2_gold_numbers(
    rows: Callable[[str, str], Mapping[str, E1Row]], e2: list[dict[str, Any]]
) -> Numbers:
    """E2 outcomes joined on task id with the SAME route's E1 gold row (ruling R-P2a).

    ``e2.gold_pass_tasks.<route>`` / ``e2.gold_fail_tasks.<route>`` count the E2 tasks whose
    gold patch passes / fails on that route; per config x route, ``GOLD_JOIN`` splits
    agreement with Kaggle by that gold status, and disagreement by direction. ``rows`` is the
    same ``(route, mode) -> {task_id: row}`` source ``e1_numbers`` reads (mode ``gold`` only).
    """
    gold = {r: {t: x.resolved for t, x in rows(r, "gold").items()} for r in ROUTES}
    tasks = {row["task_id"] for row in e2}
    n: Numbers = {}
    for route in ROUTES:
        missing = sorted(tasks - gold[route].keys())
        assert not missing, f"{route}: E2 task(s) with no E1 gold row: {missing[:5]}"
        n[f"e2.gold_pass_tasks.{route}"] = sum(gold[route][t] for t in tasks)
        n[f"e2.gold_fail_tasks.{route}"] = sum(not gold[route][t] for t in tasks)
    count = Counter[str]()
    for row in e2:
        route = _route_of(row["harness_route"])
        g, k, loc = gold[route][row["task_id"]], row["kaggle_resolved"], row["local_resolved"]
        p = f"e2.{row['config']}.{route}"
        count[f"{p}.agree_on_gold_pass"] += g and k == loc
        count[f"{p}.disagree_on_gold_pass"] += g and k != loc
        count[f"{p}.disagree_on_gold_fail"] += not g and k != loc
        count[f"{p}.kaggle_pass_local_fail"] += k and not loc
        count[f"{p}.kaggle_fail_local_pass"] += loc and not k
    for config in CONFIGS:
        for route in ROUTES:
            for key in GOLD_JOIN:
                n[f"e2.{config}.{route}.{key}"] = count[f"e2.{config}.{route}.{key}"]
    return n


def build(
    rows: Callable[[str, str], Mapping[str, E1Row]] = rows_for,
    repos: Mapping[str, str] | None = None,
    e2: list[dict[str, Any]] | None = None,
    kaggle: dict[str, dict[str, bool]] | None = None,
) -> Numbers:
    """Every paper number as a flat ``key -> int | float | str`` dict.

    The defaults read our ledgers; the release notebook (``reproduce.py``) passes the exported
    labels and E2 rows instead (see ``e1_numbers``, ``e2_numbers`` and ``e2_gold_numbers``).
    """
    rows = cache(rows)  # e1_numbers and e2_gold_numbers read the same gold rows
    e2 = e2 if e2 is not None else e2_rows()
    n: Numbers = {"scope.configs": len(CONFIGS)}
    n.update(e1_numbers(rows, repos))
    n.update(e2_numbers(e2, kaggle))
    n.update(e2_gold_numbers(rows, e2))
    return n


def render_tables(n: Numbers) -> str:
    """The Markdown tables for reports/paper-tables.md, read only from ``n``."""
    out = ["# Paper tables", "", "Generated by `uv run python -m factory.paper.numbers`.", ""]
    return "\n".join([*out, *_e1_tables(n), *_e2_tables(n), *_gold_join_table(n)])


def _gold_join_table(n: Numbers) -> list[str]:
    out = ["## E2 by gold validity (each route joined with its own E1 gold row)", ""]
    out += [
        "| route | E2 tasks gold pass | E2 tasks gold fail |",
        "|---|---|---|",
        *(
            f"| {r} | {n[f'e2.gold_pass_tasks.{r}']} | {n[f'e2.gold_fail_tasks.{r}']} |"
            for r in ROUTES
        ),
        "",
        (
            "| config | route | agree, gold pass | disagree, gold pass | disagree, gold fail "
            "| Kaggle pass, local fail | Kaggle fail, local pass |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for config in CONFIGS:
        for r in ROUTES:
            cells = " | ".join(str(n[f"e2.{config}.{r}.{k}"]) for k in GOLD_JOIN)
            out.append(f"| {config} | {r} | {cells} |")
    return [*out, ""]


def _e1_tables(n: Numbers) -> list[str]:
    out = [f"## E1: gold and empty coverage ({n['e1.as_shipped.tasks']} tasks)", ""]
    out += [
        "| route | gold pass | empty fail | valid (gold pass, empty fail) |",
        "|---|---|---|---|",
    ]
    for r in ROUTES:
        out.append(
            f"| {r} | {n[f'e1.{r}.gold_pass']} | {n[f'e1.{r}.empty_fail']} | {n[f'e1.{r}.valid']} |"
        )
    out += ["", f"Third-party public kit gold pass: {n['e1.third_party_gold_pass']}.", ""]
    out += ["| route A | route B | A pass, B fail | A fail, B pass |", "|---|---|---|---|"]
    for a, b in combinations(ROUTES, 2):
        k = f"e1.disagree.{a}__{b}"
        out.append(f"| {a} | {b} | {n[f'{k}.pass_fail']} | {n[f'{k}.fail_pass']} |")
    out += ["", "| gold-row root cause | " + " | ".join(ROUTES) + " |", "|---|---|---|---|"]
    for c in CAUSES:
        out.append(f"| {c} | " + " | ".join(str(n[f"e1.{r}.cause.{c}"]) for r in ROUTES) + " |")
    return [*out, ""]


def _e2_tables(n: Numbers) -> list[str]:
    out = [f"## E2: stored-patch re-score ({n['scope.tasks']} tasks per config)", ""]
    out += ["| config | route | resolved | Kaggle resolved | agree | disagree | disagree % |"]
    out.append("|---|---|---|---|---|---|---|")
    for config in CONFIGS:
        for r in ROUTES:
            p = f"e2.{config}.{r}"
            out.append(
                f"| {config} | {r} | {n[f'{p}.resolved']} | {n[f'e2.{config}.kaggle.resolved']} | "
                f"{n[f'{p}.agree']} | {n[f'{p}.disagree']} | {n[f'{p}.disagree_pct']} |"
            )
    out += ["", "| paired (v1 vs v1_calls50) | both | only v1 | only calls50 | sign p |"]
    out.append("|---|---|---|---|---|")
    for r in ("kaggle", *ROUTES):
        p = f"e2.paired.{r}"
        out.append(
            f"| {r} | {n[f'{p}.both']} | {n[f'{p}.only_v1']} | {n[f'{p}.only_c50']} | "
            f"{n[f'{p}.sign_p']} |"
        )
    out += ["", SIGN_NOTE]
    out += ["", f"Routes whose paired sign differs from Kaggle's: {n['e2.paired.flips']}.", ""]
    return out


def to_json(n: Numbers) -> str:
    """``n`` exactly as reports/paper_numbers.json holds it (sorted keys, trailing newline)."""
    return json.dumps(n, sort_keys=True, indent=2) + "\n"


def main() -> None:
    n = build()
    with JSON_OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write(to_json(n))
    with MD_OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_tables(n))
    print(f"wrote {JSON_OUT} ({len(n)} keys) and {MD_OUT}")


if __name__ == "__main__":
    main()
