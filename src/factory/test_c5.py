"""Tests for factory.c5 and harness/shared/c5_driver.py (plan T3 Step 5, ruling R35)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from factory import c5, local_env
from factory.c5 import C5Run, parse_manifest
from factory.pins import PinMismatchError

DRIVER = c5.C5_DRIVER
KNOWN = {"rich_3063", "rich_3061", "requests_7205"}


def _driver_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c5_driver_under_test", DRIVER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spawn_driver(env_value: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != c5.PATCH_ENV}
    if env_value is not None:
        env[c5.PATCH_ENV] = env_value
    return subprocess.run(
        [sys.executable, str(DRIVER), "eval"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_driver_env_unset_exits_nonzero() -> None:
    with pytest.raises(SystemExit, match="unset"):
        _driver_module().load_patch({})
    proc = _spawn_driver(None)
    assert proc.returncode != 0
    assert "FACTORY_C5_PATCH_FILE is unset" in proc.stderr
    assert "swegemma" not in proc.stderr  # refused before importing the harness


def test_driver_missing_file_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "nope.patch"
    with pytest.raises(SystemExit, match="cannot read"):
        _driver_module().load_patch({c5.PATCH_ENV: str(missing)})
    proc = _spawn_driver(str(missing))
    assert proc.returncode != 0
    assert "cannot read" in proc.stderr
    assert "swegemma" not in proc.stderr


def test_driver_empty_file_passes_through_as_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.patch"
    empty.write_bytes(b"")
    assert _driver_module().load_patch({c5.PATCH_ENV: str(empty)}) == ""


def test_driver_keeps_bytes_exactly(tmp_path: Path) -> None:
    p = tmp_path / "x.patch"
    p.write_bytes(b"diff --git a/f b/f\r\n+x\n")
    assert _driver_module().load_patch({c5.PATCH_ENV: str(p)}) == "diff --git a/f b/f\r\n+x\n"


def test_driver_pin_detects_change(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for rel in local_env.PINNED_FILES.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_env.REPO_ROOT / rel, root / rel)
    local_env.assert_harness_pins(repo_root=root)
    with (root / local_env.PINNED_FILES["c5_driver_py_sha256"]).open("ab") as f:
        f.write(b"# tampered\n")
    with pytest.raises(PinMismatchError, match="c5_driver"):
        local_env.assert_harness_pins(repo_root=root)


def test_driver_sha_checked(tmp_path: Path) -> None:
    driver = tmp_path / "c5_driver.py"
    driver.write_bytes(b"print('x')\n")
    sha = local_env.sha256_file(driver)
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"harness": {c5.PIN_KEY: sha}}), encoding="utf-8")
    assert c5.driver_sha_checked(driver, pinned=True, pins_path=pins) == sha
    driver.write_bytes(b"print('mutant')\n")
    with pytest.raises(PinMismatchError, match="pinned"):
        c5.driver_sha_checked(driver, pinned=True, pins_path=pins)
    assert c5.driver_sha_checked(driver, pinned=False, pins_path=pins) != sha


def test_repo_driver_sha_matches_pin() -> None:
    assert c5.driver_sha_checked(DRIVER, pinned=True) == local_env.sha256_file(DRIVER)


def test_parse_manifest() -> None:
    text = "# c5\nrich_3063\tmalformed\nrich_3063\twrong:rich_3061\n\nrequests_7205\tempty\n"
    assert parse_manifest(text, KNOWN) == [
        C5Run("rich_3063", "malformed"),
        C5Run("rich_3063", "wrong", "rich_3061"),
        C5Run("requests_7205", "empty"),
    ]
    for bad in (
        "rich_3063 malformed",  # no tab
        "nope_1\tmalformed",  # unknown task
        "rich_3063\twrong:rich_3063",  # donor is the task itself
        "rich_3063\twrong:nope_1",  # unknown donor
        "rich_3063\twrong",  # no donor
        "rich_3063\tgold",  # not a C5 input
    ):
        with pytest.raises(ValueError, match="manifest line 1"):
            parse_manifest(bad, KNOWN)


def test_input_patch() -> None:
    gold = {"rich_3061": "diff --git a/x b/x\n"}
    assert c5.input_patch(C5Run("rich_3063", "empty"), gold) == ""
    assert c5.input_patch(C5Run("rich_3063", "malformed"), gold) == "not a diff\n"
    assert c5.input_patch(C5Run("rich_3063", "wrong", "rich_3061"), gold) == gold["rich_3061"]


def test_c5_argv_uses_driver_in_gold_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    # The argv names <data root>/competition but reads nothing there; a fixed root lets the
    # release's tests run without FACTORY_DATA_ROOT (its paths.py has no default).
    monkeypatch.setenv("FACTORY_DATA_ROOT", "D:/kit-data")  # not a REWRITES target
    argv = c5.c5_argv("rich_3063", "/mnt/d/r", "/mnt/d/p.patch", DRIVER)
    assert argv[:3] == ["/usr/bin/env", "TEST_TMPDIR=/tmp", "FACTORY_C5_PATCH_FILE=/mnt/d/p.patch"]
    assert argv[3] == local_env.WSL_PY
    assert argv[4].endswith("/harness/diagnostic/docker_semantics.py")
    assert argv[5] == "--gold" and argv[6].endswith("/harness/shared/c5_driver.py")
    joined = " ".join(argv)
    for part in ("--sandbox subprocess", "--skip-agent-patch", "--concurrency 1"):
        assert part in joined
    assert argv[argv.index("--task-id") + 1] == "rich_3063"


def test_empty_results_and_task_verdicts() -> None:
    log = (
        "TASK rich_3063 mode=empty run=0 resolved=False root_cause=None seconds=26.3\n"
        "TASK requests_7205 mode=empty run=0 resolved=True root_cause=invalid_task seconds=2\n"
    )
    empty = c5.empty_results(log)
    assert empty == {"rich_3063": False, "requests_7205": True}
    ok = {C5Run("rich_3063", "malformed"): False, C5Run("rich_3063", "wrong", "x"): False}
    assert c5.task_verdicts(empty, ok) == {"rich_3063": "PASS"}
    bad = {**ok, C5Run("rich_3063", "wrong", "x"): True}
    assert c5.task_verdicts(empty, bad) == {"rich_3063": "FAIL"}
    assert c5.task_verdicts({}, ok) == {"rich_3063": "NOT-VERIFIED"}
    other = {
        C5Run("requests_7205", "malformed"): False,
        C5Run("requests_7205", "wrong", "x"): False,
    }
    assert c5.task_verdicts(empty, other) == {"requests_7205": "FAIL"}  # empty resolved
