"""Integrity check for archived research files against a sha256 manifest."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    """Return the hex sha256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: Path, root: Path) -> list[str]:
    """Check every file named in a manifest against its recorded sha256.

    The manifest is a JSON object mapping forward-slash paths relative to ``root``
    to full hex sha256 digests.

    Args:
        manifest: Path to the JSON manifest.
        root: Directory the manifest's relative paths are resolved against.

    Returns:
        One message per missing or mismatched file; an empty list means all match.
    """
    entries: dict[str, str] = json.loads(manifest.read_text(encoding="utf-8"))
    messages: list[str] = []
    for rel, expected in sorted(entries.items()):
        path = root / rel
        if not path.is_file():
            messages.append(f"{rel}: missing under {root}")
            continue
        actual = _sha256(path)
        if actual != expected.lower():
            messages.append(f"{rel}: sha256 {actual} does not match manifest {expected}")
    return messages
