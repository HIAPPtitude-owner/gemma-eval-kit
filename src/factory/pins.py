"""Environment pins: wheel-set digests, interpreters, metric entry point and quota.

``python -m factory.pins`` regenerates ``pins.json`` at the repo root. Values that were
measured by hand in T1 (Steps 1-6, 8) are recorded here as constants with their receipts;
everything that can be re-measured on this machine (wheel digests, freeze and rootfs hashes)
is re-measured at generation time and checked against the recorded value: the hand-measured
hashes against the constants below (``PinMismatchError``), and the whole re-measured mapping,
wheel digests included, against the existing ``pins.json``, which is not overwritten when it
differs (``PINS-REFUSED``, exit 1) unless ``--force`` is given.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

from factory.archive import _sha256
from factory.paths import competition_dir, data_root, wheelhouse_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
PINS_PATH = REPO_ROOT / "pins.json"

WHEELHOUSE_COUNT = 41
COMPETITION_WHEELS_COUNT = 124

# Step 2 receipt: `uvx kaggle --version` -> "Kaggle CLI 2.2.4".
KAGGLE_CLI_VERSION = "2.2.4"
# Step 6 receipt: swegemma/harness/verification.py:345 `async def verify_task(`.
METRIC_ENTRYPOINT = "swegemma.harness.verification:verify_task"

# Step 4 receipts (T2 validator interpreter, Windows).
OFFICIAL_PYTHON_PATH = "<data-root>/venv-probe/Scripts/python.exe"
OFFICIAL_PYTHON_VERSION = "3.12.10"
OFFICIAL_FREEZE = "venv-probe-freeze.txt"
OFFICIAL_FREEZE_SHA256 = "fcdf33f38b223d2c4b77f9fa538d07ae4679e1176715a59f564c32adcefd7365"

# Step 1 receipts (both T3 sweeps, WSL1).
WSL1_DISTRO = "gemma-wsl1"
WSL1_ROOTFS_TAR = "wsl/dl/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz"
WSL1_ROOTFS_SHA256 = "2a790896740b14d637dbdc583cce1ba081ac53b9e9cdb46dc09a2f73abbd9934"
WSL1_PYTHON = "3.13.16"
WSL1_HARNESS_VENV = "/root/venv-official"
WSL1_FREEZE = "wsl/probe/wsl1-freeze-313-mirrored.txt"
WSL1_FREEZE_SHA256 = "d91832576c3fd06189acb3fe4f24dbe9e416f9335d041fe60493a8ae0ad9786d"

# T3 Step 2: harness scripts used unchanged by both local sweeps (asserted before every sweep
# by factory.local_eval). gold_driver.py is a byte copy of <data-root>/wsl/probe/.
HARNESS_PINS: dict[str, tuple[str, str]] = {
    "provision_sh_sha256": (
        "harness/shared/provision.sh",
        "0d61e69f06f769ee662b8fa99aa8e7ebc0b084710a6e841d75f2e52068adf421",
    ),
    "gold_driver_py_sha256": (
        "harness/shared/gold_driver.py",
        "0196ea76e0aee2ef3035f57dac614df71a57af10b2ededea73a6015198859758",
    ),
    # T3 Step 5 (ruling R35): substitutes $FACTORY_C5_PATCH_FILE as agent_patch for C5.
    "c5_driver_py_sha256": (
        "harness/shared/c5_driver.py",
        "212902f24dad7b972a8205577567b4a5bc3e83e312ee84c5b4147efa640c6a1f",
    ),
}

# T5b: the organizer notebook kaggle/run_template.py is derived from (cells 2, 4, 8, 10).
# Receipt: sha256sum of the file pulled into data/research/notebooks/ on 2026-10-15.
TEMPLATE_NOTEBOOK = (
    REPO_ROOT
    / "data/research/notebooks/getting-started-gemma-4-developer-agent"
    / "getting-started-gemma-4-developer-agent.ipynb"
)
TEMPLATE_SHA = "8e7d84a326acae509fb596163ba085831961b588b86d51af31e037c5319b917c"



class DirDigest(NamedTuple):
    """Digest of the files in a directory matching a glob pattern."""

    count: int
    sha256: str


class PinMismatchError(RuntimeError):
    """Raised when a re-measured value does not match its recorded pin."""


def dir_digest(root: Path, pattern: str) -> DirDigest:
    """Digest every file under ``root`` matching ``pattern``.

    The digest is a sha256 over the sorted lines ``relpath\\0filehash\\n``, where
    ``relpath`` is the forward-slash path relative to ``root``. A rename, an added or removed
    file, or a content change all change the digest.

    Args:
        root: Directory to scan.
        pattern: Glob pattern passed to ``Path.glob`` (e.g. ``"*.whl"``).

    Returns:
        The number of matching files and the hex sha256 digest.
    """
    files = [p for p in root.glob(pattern) if p.is_file()]
    lines = sorted(f"{p.relative_to(root).as_posix()}\0{_sha256(p)}\n" for p in files)
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return DirDigest(count=len(lines), sha256=digest)


def _counted_digest(root: Path, expected_count: int) -> dict[str, Any]:
    """Return ``{count, sha256}`` for a wheel directory, asserting its file count."""
    found = dir_digest(root, "*.whl")
    if found.count != expected_count:
        raise PinMismatchError(f"{root}: {found.count} wheels, expected {expected_count}")
    return found._asdict()


def _checked_sha(path: Path, expected: str) -> str:
    """Return ``expected`` after confirming it equals the sha256 of ``path``."""
    actual = _sha256(path)
    if actual != expected:
        raise PinMismatchError(f"{path}: sha256 {actual}, recorded {expected}")
    return expected


def build_pins() -> dict[str, Any]:
    """Re-measure the measurable pins and assemble the full pins mapping.

    Returns:
        The mapping written to ``pins.json``.

    Raises:
        PinMismatchError: If a wheel count or a recorded hash no longer matches disk.
    """
    root = data_root()
    return {
        "wheelhouse": _counted_digest(wheelhouse_dir(), WHEELHOUSE_COUNT),
        "competition_wheels": _counted_digest(
            competition_dir() / "wheels", COMPETITION_WHEELS_COUNT
        ),
        "kaggle_cli_version": KAGGLE_CLI_VERSION,
        "official_python": {
            "path": OFFICIAL_PYTHON_PATH,
            "version": OFFICIAL_PYTHON_VERSION,
            "freeze_sha256": _checked_sha(root / OFFICIAL_FREEZE, OFFICIAL_FREEZE_SHA256),
        },
        "wsl1": {
            "distro": WSL1_DISTRO,
            "rootfs_tar_sha256": _checked_sha(root / WSL1_ROOTFS_TAR, WSL1_ROOTFS_SHA256),
            "python": WSL1_PYTHON,
            "harness_venv": WSL1_HARNESS_VENV,
            "freeze_sha256": _checked_sha(root / WSL1_FREEZE, WSL1_FREEZE_SHA256),
        },
        "metric_entrypoint": METRIC_ENTRYPOINT,
        "harness": {
            key: _checked_sha(REPO_ROOT / rel, sha) for key, (rel, sha) in HARNESS_PINS.items()
        },
        "template_sha": _checked_sha(TEMPLATE_NOTEBOOK, TEMPLATE_SHA),
    }


def _changed_keys(old_text: str, new: dict[str, Any]) -> list[str]:
    """Return the top-level pins keys whose recorded value differs from the re-measured one."""
    try:
        old = json.loads(old_text)
    except json.JSONDecodeError:
        return ["<unparseable>"]
    if not isinstance(old, dict):
        return ["<not an object>"]
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


def main(argv: list[str] | None = None) -> int:
    """Regenerate ``pins.json``, refusing to overwrite a recorded file that differs.

    Returns:
        0 when the file was written or already matches; 1 (``PINS-REFUSED``) when the file
        exists with different content and ``--force`` was not given.
    """
    parser = argparse.ArgumentParser(prog="python -m factory.pins")
    parser.add_argument("--out", type=Path, default=PINS_PATH, help="pins JSON to write")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a pins file whose content differs"
    )
    args = parser.parse_args(argv)
    pins = build_pins()
    text = json.dumps(pins, indent=2, sort_keys=True) + "\n"
    out: Path = args.out
    if out.is_file():
        recorded = out.read_text(encoding="utf-8")
        if recorded == text:
            print(f"PINS-UNCHANGED: {out}")
            return 0
        if not args.force:
            changed = ", ".join(_changed_keys(recorded, pins)) or "formatting only"
            print(
                f"PINS-REFUSED: {out} differs from the re-measured pins (changed: {changed}); "
                "nothing was written (pass --force to overwrite)"
            )
            return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
