"""WSL1 distro state for the local sweeps: pins, the repair switch, and the env_hash.

Split out of ``factory.local_eval`` (500-line file cap). Everything that talks to the
``gemma-wsl1`` distro or asserts a pin before a sweep lives here.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from factory.archive import _sha256
from factory.paths import competition_dir, data_root
from factory.pins import HARNESS_PINS, PINS_PATH, REPO_ROOT, PinMismatchError, dir_digest

Harness = Literal["as_shipped", "diagnostic", "public_docker"]
Mode = Literal["gold", "empty"]
HARNESS_ROUTES: dict[str, str] = {
    "as_shipped": "wsl1-subprocess-py3.13-as-shipped",
    "diagnostic": "wsl1-subprocess-py3.13-diagnostic",
    # Not a WSL route: factory.docker_kit runs the competition's public Docker kit.
    "public_docker": "docker-desktop-public-kit-py3.13",
}
PINNED_FILES: dict[str, str] = {key: rel for key, (rel, _) in HARNESS_PINS.items()}
# Pins that define a gold/empty sweep environment (folded into env_hash). c5_driver.py is
# asserted with the others before every sweep but is not part of that environment, so adding
# it kept the as_shipped/diagnostic env_hash values (e5d086df / d0e52852) comparable.
ENV_HASH_PINS = ("gold_driver_py_sha256", "provision_sh_sha256")
REPAIR_SH = REPO_ROOT / "harness" / "diagnostic" / "repair.sh"
CHANGELOG = REPO_ROOT / "harness" / "diagnostic" / "CHANGELOG.md"
DOCKER_SEMANTICS = REPO_ROOT / "harness" / "diagnostic" / "docker_semantics.py"
WRAPPER_PIN = re.compile(r"^WRAPPER_SHA256=([0-9a-f]{64})$", re.MULTILINE)

# WSL1 distro layout (pins.json wsl1; T1 Step 1).
WSL_PY = "/root/venv-official/bin/python"
WSL_SWEGEMMA = "/root/venv-official/bin/swegemma"
WSL_UV = "/root/.local/bin/uv"
DIAG_TARGET = "/root/factory-diagnostic/site-packages"
DIAG_PTH = "factory_diagnostic.pth"


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return _sha256(path)


def assert_harness_pins(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Assert every ``HARNESS_PINS`` file (``provision.sh``, ``gold_driver.py``,
    ``c5_driver.py``) matches its sha256 pin in pins.json.

    Args:
        repo_root: Tree holding the ``harness/`` files to check (the repo by default).

    Returns:
        The parsed pins.json mapping.

    Raises:
        PinMismatchError: If a pin is missing or a file's sha256 differs from its pin.
    """
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    recorded = pins.get("harness", {})
    for key, rel in PINNED_FILES.items():
        if key not in recorded:
            raise PinMismatchError(f"pins.json harness.{key} missing; regenerate pins.json")
        actual = sha256_file(repo_root / rel)
        if actual != recorded[key]:
            raise PinMismatchError(f"{rel}: sha256 {actual}, pinned {recorded[key]}")
    return pins


def assert_repair_logged(repair: Path = REPAIR_SH, changelog: Path = CHANGELOG) -> str:
    """Assert ``repair.sh``'s current sha256 (first 12 hex) has a CHANGELOG line.

    Returns:
        The full sha256 of ``repair``.

    Raises:
        PinMismatchError: If no CHANGELOG line names the current sha.
    """
    sha = sha256_file(repair)
    if sha[:12] not in changelog.read_text(encoding="utf-8"):
        raise PinMismatchError(
            f"{repair.name} sha256 {sha[:12]} has no line in {changelog.name}; "
            "log the change (cause, change, tasks affected) before sweeping"
        )
    return sha


def assert_wrapper_pinned(repair: Path = REPAIR_SH, wrapper: Path = DOCKER_SEMANTICS) -> str:
    """Assert ``docker_semantics.py`` matches the ``WRAPPER_SHA256=`` pin inside ``repair.sh``.

    The pin makes repair.sh's own sha (logged in CHANGELOG.md, folded into env_hash) cover
    the wrapper, and repair.sh re-checks it inside the distro.

    Returns:
        The full sha256 of ``wrapper``.

    Raises:
        PinMismatchError: If repair.sh carries no pin or the wrapper's sha differs from it.
    """
    m = WRAPPER_PIN.search(repair.read_text(encoding="utf-8"))
    if not m:
        raise PinMismatchError(f"{repair.name} has no WRAPPER_SHA256=<64 hex> line")
    sha = sha256_file(wrapper)
    if sha != m.group(1):
        raise PinMismatchError(f"{wrapper.name}: sha256 {sha}, pinned in {repair.name} {m[1]}")
    return sha


def launcher(harness: Harness, mode: Mode) -> list[str]:
    """Return the in-distro launcher argv for one eval run.

    ``as_shipped``: ``gold_driver.py`` (gold) or the plain ``swegemma`` CLI (empty).
    ``diagnostic``: ``docker_semantics.py`` in both modes, wrapping ``gold_driver.py`` for gold.
    """
    _wsl_only(harness)
    gold = to_wsl_path(REPO_ROOT / PINNED_FILES["gold_driver_py_sha256"])
    if harness == "as_shipped":
        return [WSL_PY, gold] if mode == "gold" else [WSL_SWEGEMMA]
    wrapper = [WSL_PY, to_wsl_path(DOCKER_SEMANTICS)]
    return [*wrapper, "--gold", gold] if mode == "gold" else wrapper


def _wsl_only(harness: Harness) -> None:
    if harness not in ("as_shipped", "diagnostic"):
        raise ValueError(f"{harness!r} is not a WSL1 route; use factory.docker_kit")


def to_wsl_path(path: Path) -> str:
    """Translate a Windows drive path to its WSL ``/mnt/<drive>/...`` form."""
    text = str(path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/?(.*)$", text)
    if not m:
        raise ValueError(f"not a drive path: {path}")
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}"


def _wsl(argv: list[str]) -> list[str]:
    distro = json.loads(PINS_PATH.read_text(encoding="utf-8"))["wsl1"]["distro"]
    return ["wsl.exe", "-d", distro, "--cd", "/root", "--exec", *argv]


def wsl_capture(argv: list[str], timeout: int = 600) -> str:
    """Run a command in the distro and return its stdout; raise with stderr on failure."""
    r = subprocess.run(_wsl(argv), check=False, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"wsl {argv[0]} exit {r.returncode}: {err[-800:]}")
    return r.stdout.decode("utf-8", "replace")


def wsl_logged(argv: list[str], log: Path, timeout: int) -> int:
    """Run a command in the distro with stdout+stderr sent to ``log``; return its exit code.

    A timeout is written into the log and returned as exit code 124.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as f:
        try:
            return subprocess.run(
                _wsl(argv), check=False, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nFACTORY: timed out after {timeout}s\n".encode())
            return 124


def _base_site() -> str:
    code = (
        "import os, sys; v = 'python%d.%d' % sys.version_info[:2]; "
        "print(os.path.join(sys.base_prefix, 'lib', v, 'site-packages'))"
    )
    return wsl_capture([WSL_PY, "-c", code]).strip()


def _apply_harness(harness: Harness, sweep_dir: Path, base_site: str) -> str | None:
    """Apply repair.sh (diagnostic) or remove its .pth switch (as_shipped)."""
    if harness == "diagnostic":
        repair_sha = assert_repair_logged()
        assert_wrapper_pinned()
        log = sweep_dir / "logs" / "repair.log"
        rc = wsl_logged(["/bin/bash", to_wsl_path(REPAIR_SH)], log, 1800)
        if rc != 0:
            raise RuntimeError(f"repair.sh exit {rc}; see {log}")
        print(f"REPAIR-APPLIED: repair.sh {repair_sha[:12]} log={log}", flush=True)
        return repair_sha
    wsl_capture(["/bin/rm", "-f", f"{base_site}/{DIAG_PTH}"])
    return None


def _env_material(harness: Harness, pins: dict[str, Any], repair_sha: str | None) -> dict:
    """Measure the distro state that defines a harness environment."""
    base_site = _base_site()
    freeze = wsl_capture([WSL_UV, "pip", "freeze", "--python", WSL_PY])
    freeze_sha = hashlib.sha256(freeze.encode("utf-8")).hexdigest()
    if freeze_sha != pins["wsl1"]["freeze_sha256"]:
        raise PinMismatchError(f"harness venv freeze {freeze_sha} != pinned; rerun provision.sh")
    listing = wsl_capture(["/bin/ls", "-1", base_site]).split()
    # repair.sh v1 removed the v0 switch: neither harness may have it (sandbox venvs of the
    # diagnostic harness are isolated and never read the base site-packages).
    if DIAG_PTH in listing:
        raise PinMismatchError(f"{DIAG_PTH} present in {base_site}; {harness} expects it gone")
    diag_freeze = ""
    extra: dict[str, str] = {}
    if harness == "diagnostic":
        diag_freeze = wsl_capture([WSL_UV, "pip", "freeze", "--target", DIAG_TARGET])
        extra["docker_semantics_sha256"] = assert_wrapper_pinned()
    return {
        **extra,
        "harness_route": HARNESS_ROUTES[harness],
        "harness_pins": {k: pins["harness"][k] for k in ENV_HASH_PINS},
        "repair_sh_sha256": repair_sha,
        "harness_freeze_sha256": freeze_sha,
        "base_site_listing": sorted(listing),
        "diagnostic_target_freeze": diag_freeze,
    }


def measure_env(harness: Harness, sweep_dir: Path) -> dict[str, Any]:
    """Assert the pins, put the distro in the harness's state, and measure its env_hash.

    The hash covers the route, the pinned script shas, repair.sh (diagnostic), the harness
    venv freeze (asserted equal to pins.json wsl1.freeze_sha256), the base interpreter's
    site-packages listing and, for diagnostic, the repair target's freeze. The material is
    written to ``<sweep_dir>/env_material.json``.

    Returns:
        ``{"env_hash": ..., "wheelhouse_sha": ..., "repair_sh_sha256": ... | None}``.

    Raises:
        PinMismatchError: If a pinned file, the harness freeze or the wheel set has drifted.
    """
    _wsl_only(harness)
    pins = assert_harness_pins()
    wheels = dir_digest(competition_dir() / "wheels", "*.whl")
    if wheels._asdict() != pins["competition_wheels"]:
        raise PinMismatchError(f"competition wheels {wheels} != pinned")
    repair_sha = _apply_harness(harness, sweep_dir, _base_site())
    material = _env_material(harness, pins, repair_sha)
    (sweep_dir / "env_material.json").write_text(
        json.dumps(material, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "env_hash": _hash_material(material),
        "wheelhouse_sha": wheels.sha256,
        "repair_sh_sha256": repair_sha,
    }


def _hash_material(material: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def remeasure_env(harness: Harness, start_repair_sha: str | None) -> str:
    """Re-measure the env_hash WITHOUT changing distro state (no repair, no .pth removal).

    Raises:
        PinMismatchError: If a pin, the harness freeze, the .pth switch or (diagnostic)
            repair.sh no longer matches what the sweep started with.
    """
    _wsl_only(harness)
    pins = assert_harness_pins()
    if harness == "diagnostic":
        now = sha256_file(REPAIR_SH)
        if now != start_repair_sha:
            raise PinMismatchError(f"repair.sh sha256 {now[:12]} != sweep start {start_repair_sha}")
    return _hash_material(_env_material(harness, pins, start_repair_sha))


class SweepDriftError(RuntimeError):
    """Raised when the distro env at the end of a sweep differs from its start."""

    def __init__(self, reason: str, rows: list[Any]) -> None:
        super().__init__(reason)
        self.rows = rows


def lock_path() -> Path:
    """Return the single-sweep lock file: ``<data_root>/locks/local_eval.lock``."""
    return data_root() / "locks" / "local_eval.lock"


def drift_ledger_path() -> Path:
    """Return the drift marker ledger: ``<data_root>/ledger/gold_validation_drift.jsonl``.

    Rows in ``gold_validation.jsonl`` whose ``sweep_id`` appears here are untrustworthy;
    the row lines themselves are never rewritten.
    """
    return data_root() / "ledger" / "gold_validation_drift.jsonl"


def drifted_sweep_ids(path: Path | None = None) -> set[str]:
    """Return the sweep ids marked as drifted (their ledger rows must be excluded)."""
    path = path or drift_ledger_path()
    if not path.is_file():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["sweep_id"] for line in lines if line.strip()}


def end_of_sweep_check(harness: Harness, env: dict[str, Any], sweep_dir: Path) -> str | None:
    """Re-measure the env at sweep end; on drift, mark the sweep and return the reason.

    The marker is appended to the drift ledger (utf-8, fsync) and written to
    ``<sweep_dir>/DRIFT.json``; a ``DRIFT:`` line is printed.
    """
    try:
        end_hash = remeasure_env(harness, env.get("repair_sh_sha256"))
    except (PinMismatchError, RuntimeError, OSError) as exc:
        reason = f"end-of-sweep re-measure failed: {exc}"
    else:
        if end_hash == env["env_hash"]:
            return None
        reason = f"env_hash changed: start {env['env_hash']} end {end_hash}"
    mark_drift(sweep_dir, harness, env["env_hash"], reason)
    return reason


def mark_drift(sweep_dir: Path, harness: Harness, start_env_hash: str, reason: str) -> None:
    """Mark a sweep as drifted: ``<sweep_dir>/DRIFT.json`` plus a line in the drift ledger.

    Shared by every route's end-of-sweep check, so ``drifted_sweep_ids`` excludes them all.
    The ledger append is utf-8 and fsync'd; a ``DRIFT:`` line is printed.
    """
    rec = {"sweep_id": sweep_dir.name, "harness": harness, "start_env_hash": start_env_hash}
    rec["reason"] = reason
    text = json.dumps(rec, sort_keys=True)
    (sweep_dir / "DRIFT.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    path = drift_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"DRIFT: sweep {sweep_dir.name}: {reason}; its rows are untrustworthy", flush=True)
