"""Probe-only C5 driver: runs the unmodified swegemma CLI with --skip-agent-patch,
substituting the contents of the file named by $FACTORY_C5_PATCH_FILE as agent_patch at the
verify_task call (sibling of gold_driver.py; plan T3 Step 5, ruling R35).

Fails loudly (nonzero exit, before swegemma is imported) when the variable is unset or the
file cannot be read. It never falls back to task.patch or to "". An empty file is a real
input: it is the empty-patch case and is passed through as "".

The official package is not edited; the wrap lives in this process only.
"""

import os
import sys
from collections.abc import Mapping

PATCH_ENV = "FACTORY_C5_PATCH_FILE"


def load_patch(environ: Mapping[str, str] = os.environ) -> str:
    """Return the substitute patch text, or exit nonzero with a message naming the cause."""
    path = environ.get(PATCH_ENV)
    if not path:
        raise SystemExit(f"c5_driver: {PATCH_ENV} is unset; refusing to run (no fallback)")
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"c5_driver: cannot read {PATCH_ENV}={path!r}: {exc}") from exc


def run(patch: str, argv: list[str]) -> int:
    """Install the verify_task wrap that always supplies ``patch``, then run the CLI."""
    import swegemma.evaluate as ev
    from swegemma.cli import main

    orig_verify = ev.verify_task

    async def _c5_verify(docker, config, task, snapshot_path, **kwargs):
        kwargs["agent_patch"] = patch
        return await orig_verify(docker, config, task, snapshot_path, **kwargs)

    ev.verify_task = _c5_verify
    return main(argv)


if __name__ == "__main__":
    sys.exit(run(load_patch(), sys.argv[1:]))
