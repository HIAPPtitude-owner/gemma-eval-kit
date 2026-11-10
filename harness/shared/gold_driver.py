"""Probe-only gold driver: runs the unmodified swegemma CLI with --skip-agent-patch,
substituting each task's gold `patch` as agent_patch at the verify_task call.

The official package is not edited; the wrap lives in this process only.
"""

import sys

import swegemma.evaluate as ev
from swegemma.cli import main

_orig_verify = ev.verify_task


async def _gold_verify(docker, config, task, snapshot_path, **kwargs):
    if not kwargs.get("agent_patch"):
        kwargs["agent_patch"] = task.patch
    return await _orig_verify(docker, config, task, snapshot_path, **kwargs)


ev.verify_task = _gold_verify

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
