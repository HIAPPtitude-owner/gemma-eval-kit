"""Probe-side launcher for the `diagnostic` harness: Docker-scorer setup on the subprocess backend.

Run as ``python docker_semantics.py [--gold <gold_driver.py>] eval ...`` with the harness
interpreter (/root/venv-official/bin/python). No installed swegemma / adk_submission /
adk_eval_core file is edited; every change lives in this process only, like gold_driver.py.

What it changes, each with the scorer source it mirrors (installed swegemma copy):

1. Isolated sandbox venv. ``evaluate.py``:76-78 builds ``SubprocessManager(timeout_seconds=...)``
   whose default ``system_site_packages=True`` (``sandbox/subprocess.py``:37) plus the
   ``_host_env.pth`` writer (:389-400) puts the harness venv (starlette 1.8.0, httpx 0.28.1,
   attrs, ...) on every sandbox path. Here the factory builds the same class with
   ``system_site_packages=False``; the instance is still a real ``SubprocessManager``, so every
   type-name gate in the harness behaves as before.
2. The image layer. The scorer's container already holds the Dockerfile.sandbox:7 packages and
   gets the highest-version base wheels tar-extracted into its site-packages
   (``harness/container_setup.py``:337-342, :300-315, dedup :107-135). ``repair.sh`` builds that
   tree once at ``IMAGE_SITE``; each sandbox venv gets a hard-linked copy (a copy on
   cross-device), which is the per-container extraction.
3. Editable install and setup.py. The subprocess backend returns early from
   ``install_editable_package`` and ``install_test_dependencies`` (``container_setup.py``:512-513,
   :529-530). Here both run their Docker steps: the same pip command (:514-517), then the
   competition's own ``sandbox/setup.py`` with the same arguments (:532-573).
"""

from __future__ import annotations

import logging
import os
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import swegemma.evaluate as ev
import swegemma.harness.container_setup as cs
import swegemma.harness.verification as ver
from swegemma.sandbox.subprocess import SubprocessManager

IMAGE_SITE = Path("/root/factory-diagnostic/site-packages")
IMAGE_MARKER = ".factory_image_injected"
# Verbatim from container_setup.py:516 (install_editable_package, Docker branch).
EDITABLE_CMD = (
    "pip install --no-index --find-links=/wheels --no-build-isolation --no-deps "
    "-e /workspace 2>/dev/null || true"
)
logger = logging.getLogger("factory.docker_semantics")
_ORIG_EDITABLE = cs.install_editable_package
_ORIG_TEST_DEPS = cs.install_test_dependencies


def isolated_manager(*args: Any, **kwargs: Any) -> SubprocessManager:
    """Build the harness's own SubprocessManager with ``system_site_packages=False``."""
    kwargs["system_site_packages"] = False
    return SubprocessManager(*args, **kwargs)


def sandbox_site_packages(docker: SubprocessManager, cid: str) -> Path:
    """Return the one ``lib/python*/site-packages`` dir of sandbox ``cid``'s venv."""
    venv = docker.sandboxes[cid]["venv"]
    dirs = sorted(venv.glob("lib/python*/site-packages"))
    if len(dirs) != 1:
        raise RuntimeError(f"FACTORY: expected one site-packages under {venv}, found {dirs}")
    return dirs[0]


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def inject_image(docker: SubprocessManager, cid: str, image: Path | None = None) -> Path:
    """Populate the sandbox venv's site-packages from ``image`` once (marker-guarded).

    Hard links keep it cheap; a pip uninstall in the sandbox (the editable install replacing
    the repo's own wheel copy) unlinks the sandbox's names only, never the shared tree.

    Args:
        docker: The sandbox manager.
        cid: Sandbox id.
        image: Image tree to copy (``IMAGE_SITE`` when None).

    Raises:
        RuntimeError: If ``image`` is missing (repair.sh has not run).
    """
    image = image or IMAGE_SITE
    site = sandbox_site_packages(docker, cid)
    if (site / IMAGE_MARKER).exists():
        return site
    if not image.is_dir():
        raise RuntimeError(f"FACTORY: image layer {image} missing; run repair.sh first")
    for entry in sorted(image.iterdir()):
        if entry.name == "bin":
            continue
        dst = site / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, copy_function=_link_or_copy, dirs_exist_ok=True)
        else:
            _link_or_copy(str(entry), str(dst))
    (site / IMAGE_MARKER).write_text(str(image) + "\n", encoding="utf-8")
    return site


def install_editable_package(docker: Any, cid: str) -> None:
    """Docker branch of ``install_editable_package`` for the subprocess backend."""
    if not isinstance(docker, SubprocessManager):
        _ORIG_EDITABLE(docker, cid)
        return
    inject_image(docker, cid)
    docker.exec(cid, EDITABLE_CMD)


def install_test_dependencies(
    docker: Any, cid: str, repo: str = "", *, fast_path: bool = True, config: Any = None
) -> None:
    """Docker branch of ``install_test_dependencies`` (setup.py run) for the subprocess backend.

    Raises:
        RuntimeError: If the competition ``sandbox/setup.py`` cannot be found.
    """
    if not isinstance(docker, SubprocessManager):
        _ORIG_TEST_DEPS(docker, cid, repo, fast_path=fast_path, config=config)
        return
    inject_image(docker, cid)
    script = cs.resolve_sandbox_setup_script(config)
    if script is None or not script.is_file():
        raise RuntimeError("FACTORY: sandbox/setup.py not found (container_setup.py:532-549)")
    docker.copy_to(cid, script, "/tmp/setup.py")
    cmd = f"python3 /tmp/setup.py {'--fast-path' if fast_path else ''} {repo}".strip()
    try:
        res = docker.exec(cid, cmd)
        if res.exit_code != 0:
            logger.warning("setup.py exit %d for %s: %s", res.exit_code, repo, res.stderr)
    finally:
        docker.exec(cid, "rm -f /tmp/setup.py")


def apply() -> None:
    """Install the three replacements in this process (module attributes only)."""
    ev.SubprocessManager = isolated_manager
    for mod in (cs, ver):
        mod.install_editable_package = install_editable_package
        mod.install_test_dependencies = install_test_dependencies


def main(argv: list[str]) -> int:
    """Apply the replacements, then run ``gold_driver.py`` (``--gold``) or the swegemma CLI."""
    apply()
    if argv[:1] == ["--gold"]:
        driver = argv[1]
        sys.argv = [driver, *argv[2:]]
        runpy.run_path(driver, run_name="__main__")
        return 0
    from swegemma.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
