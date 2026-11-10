"""The ``public_docker`` route: the competition's own public Docker kit on Docker Desktop.

A runner container (``factory-kit-runner``, built from ``harness/docker_kit/Dockerfile.runner``)
runs ``swegemma eval --sandbox docker`` with the host's ``/var/run/docker.sock`` mounted, so
the sandbox (``swebench-sandbox:latest``, built from the kit's ``Dockerfile.public``) is a
sibling container. swegemma stages files into the sandbox by tar archive only, so the runner
mounts (``/comp``, ``/results``, ``/inputs``) never reach the sandbox.

Neither image is the scorer: this is a third non-scorer environment. The images are built from
the user's own Kaggle downloads (competition files and wheelhouse); nothing is vendored.
"""

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from factory.local_env import (
    HARNESS_ROUTES,
    PINNED_FILES,
    Mode,
    _hash_material,
    assert_harness_pins,
    mark_drift,
)
from factory.paths import competition_dir, data_root
from factory.pins import REPO_ROOT, PinMismatchError, dir_digest

SANDBOX_TAG = "swebench-sandbox:latest"
RUNNER_TAG = "factory-kit-runner:0.2.11"
ROUTE = "public_docker"
KIT_DIR = REPO_ROOT / "harness" / "docker_kit"
RUNNER_DOCKERFILE = KIT_DIR / "Dockerfile.runner"
CONSTRAINTS = KIT_DIR / "runner-constraints.txt"
SHARED_DIR = REPO_ROOT / "harness" / "shared"
SOCKET = "/var/run/docker.sock"
PATCH_ENV = "FACTORY_C5_PATCH_FILE"  # read by harness/shared/c5_driver.py
# Drivers baked into the runner at /factory/ (unchanged copies of harness/shared/), by pin key.
DRIVERS = {Path(rel).name: key for key, rel in PINNED_FILES.items() if rel.endswith("driver.py")}
STAGED_MARKER = ".factory-staged"


class KitError(RuntimeError):
    """Raised when the Docker kit cannot run: no daemon, no socket, a missing image."""


def _docker(
    args: list[str], timeout: int = 600, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run ``docker <args>``; a missing CLI or a timeout becomes a KitError."""
    try:
        return subprocess.run(
            ["docker", *args], check=False, capture_output=capture, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise KitError("docker CLI not found on PATH; install/start Docker Desktop") from exc
    except subprocess.TimeoutExpired as exc:
        raise KitError(f"docker {args[0]} timed out after {timeout}s") from exc


def _err(r: subprocess.CompletedProcess) -> str:
    return (r.stderr or b"").decode("utf-8", "replace").strip()[-800:]


def _out(r: subprocess.CompletedProcess) -> str:
    return (r.stdout or b"").decode("utf-8", "replace")


def stage_sandbox_context(comp: Path, out: Path) -> Path:
    """Stage the sandbox build context: the kit's Dockerfile, its two shims and the wheels.

    ``out`` is rebuilt from scratch on every call, so a wheel removed from ``comp/wheels``
    cannot survive. It is deleted only if an earlier call staged it (marker file).

    Raises:
        KitError: If ``out`` exists and is not a staging dir made by this function.
    """
    if out.exists():
        if not (out / STAGED_MARKER).is_file():
            raise KitError(f"{out} exists and is not a staging dir (no {STAGED_MARKER}); refusing")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(comp / "docker" / "Dockerfile.public", out / "Dockerfile")
    for name in ("imp.py", "telnetlib.py"):
        shutil.copyfile(comp / "docker" / name, out / name)
    shutil.copytree(comp / "wheels", out / "wheels")
    (out / STAGED_MARKER).write_text(f"staged from {comp}\n", encoding="utf-8")
    return out


def image_id(tag: str) -> str:
    """Return ``docker image inspect --format {{.Id}} <tag>``.

    Raises:
        KitError: Naming the tag, if the image is missing.
    """
    r = _docker(["image", "inspect", "--format", "{{.Id}}", tag], timeout=60)
    if r.returncode != 0:
        raise KitError(f"image {tag} missing ({_err(r)}); run `factory eval kit-build`")
    return _out(r).strip()


def build_images(comp: Path, wheelhouse: Path) -> dict[str, str]:
    """Build the sandbox and runner images and return their ids.

    The sandbox context is staged under ``<data_root>/docker_kit/sandbox_ctx`` (it holds
    competition wheels, so it never lives in the repo). The runner builds from
    ``harness/docker_kit`` with named contexts ``shared`` (the drivers) and ``wheelhouse``.

    Raises:
        KitError: If either build fails.
    """
    ctx = stage_sandbox_context(comp, data_root() / "docker_kit" / "sandbox_ctx")
    builds = {
        SANDBOX_TAG: ["build", "-t", SANDBOX_TAG, str(ctx)],
        RUNNER_TAG: [
            "build",
            "-t",
            RUNNER_TAG,
            "-f",
            str(RUNNER_DOCKERFILE),
            "--build-context",
            f"shared={SHARED_DIR}",
            "--build-context",
            f"wheelhouse={wheelhouse}",
            str(KIT_DIR),
        ],
    }
    for tag, args in builds.items():
        r = _docker(args, timeout=3600, capture=False)
        if r.returncode != 0:
            raise KitError(f"docker build {tag} exit {r.returncode}")
    return {"sandbox_image_id": image_id(SANDBOX_TAG), "runner_image_id": image_id(RUNNER_TAG)}


def kit_preflight() -> None:
    """Refuse before any task unless the daemon, both images and the socket are usable.

    Raises:
        KitError: Naming the cause (daemon, image tag, or ``/var/run/docker.sock``).
    """
    r = _docker(["version", "--format", "{{.Server.Version}}"], timeout=60)
    if r.returncode != 0:
        raise KitError(f"Docker daemon not reachable: {_err(r)}")
    image_id(SANDBOX_TAG)
    image_id(RUNNER_TAG)
    ping = "import docker; docker.from_env().ping()"
    r = _docker(["run", "--rm", "-v", f"{SOCKET}:{SOCKET}", RUNNER_TAG, "python", "-c", ping])
    if r.returncode != 0:
        raise KitError(f"runner cannot reach the daemon through {SOCKET}: {_err(r)}")


def kit_argv(
    mode: Mode,
    task_id: str,
    task_dir: Path,
    driver: Path | None = None,
    patch_file: Path | None = None,
) -> list[str]:
    """Build the full ``docker run`` argv for one task's eval run in the runner.

    Gold runs ``gold_driver.py``, empty the plain ``swegemma`` CLI, and ``driver`` (a file
    baked into the runner, e.g. ``c5_driver.py``) replaces both. ``patch_file`` is mounted
    read-only at ``/inputs`` and named by ``FACTORY_C5_PATCH_FILE``. The eval flags match
    ``local_eval.eval_argv`` except ``--sandbox docker`` and the runner-side paths.

    Raises:
        ValueError: If ``driver`` is not one of the drivers baked into the runner.
    """
    if driver is not None and driver.name not in DRIVERS:
        raise ValueError(f"driver {driver.name} is not baked into {RUNNER_TAG}: {sorted(DRIVERS)}")
    name = f"factory-kit-{re.sub(r'[^A-Za-z0-9_.-]', '_', task_id)}-{uuid.uuid4().hex[:8]}"
    mounts = ["-v", f"{task_dir}:/results", "-v", f"{competition_dir()}:/comp:ro"]
    mounts += ["-v", f"{SOCKET}:{SOCKET}"]
    env = ["-e", "TEST_TMPDIR=/tmp"]
    if patch_file is not None:
        mounts += ["-v", f"{patch_file.parent}:/inputs:ro"]
        env += ["-e", f"{PATCH_ENV}=/inputs/{patch_file.name}"]
    if driver is not None:
        launch = ["python", f"/factory/{driver.name}"]
    else:
        launch = ["python", "/factory/gold_driver.py"] if mode == "gold" else ["swegemma"]
    return [
        "docker", "run", "--rm", "--name", name, *mounts, *env, RUNNER_TAG, *launch,
        "eval",
        "--tasks", "/comp/tasks.jsonl",
        "--snapshots-dir", "/comp/snapshots",
        "--submission-dir", "/comp/sample_submission",
        "--results-dir", "/results",
        "--sandbox", "docker",
        "--skip-agent-patch",
        "--task-id", task_id,
        "--concurrency", "1",
        "--display", "quiet",
    ]  # fmt: skip


def _sandbox_ids() -> set[str]:
    """Return the full ids of every container, in any state, created from ``SANDBOX_TAG``."""
    args = ["ps", "-a", "-q", "--no-trunc", "--filter", f"ancestor={SANDBOX_TAG}"]
    r = _docker(args, timeout=60)
    if r.returncode != 0:
        raise KitError(f"docker ps (sandbox containers) exit {r.returncode}: {_err(r)}")
    return set(_out(r).split())


def _reap(before: set[str]) -> list[str]:
    """Force-remove the sandbox containers that did not exist in the ``before`` snapshot."""
    new = sorted(_sandbox_ids() - before)
    if new:
        r = _docker(["rm", "-f", *new], timeout=120)
        if r.returncode != 0:
            raise KitError(f"could not remove leaked sandbox container(s) {new}: {_err(r)}")
    return new


def kit_logged(argv: list[str], log: Path, timeout: int) -> int:
    """Run a ``kit_argv`` command with stdout+stderr sent to ``log``; return its exit code.

    swegemma removes its sandbox only in a ``finally`` (``cleanup_all``). The runner's
    python runs as PID 1 with no SIGTERM handler, so a stopped or killed runner leaks its
    sibling sandbox. So on every exit path (normal, crash, timeout, KeyboardInterrupt), a
    runner that may still be alive is killed (``--name`` from ``kit_argv``). Then every
    ``ancestor=SANDBOX_TAG`` container absent from a snapshot taken just before the run is
    removed. Containers that existed before the run, or that come from any other image,
    are never touched. A timeout is written into the log and returned as exit code 124.

    Raises:
        KitError: If the snapshot fails (the run never starts) or a leak cannot be removed.
    """
    name = argv[argv.index("--name") + 1]
    before = _sandbox_ids()
    log.parent.mkdir(parents=True, exist_ok=True)
    rc: int | None = None
    with log.open("wb") as f:
        try:
            rc = subprocess.run(
                argv, check=False, stdout=f, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nFACTORY: timed out after {timeout}s\n".encode())
        finally:
            if rc is None:
                _docker(["kill", name], timeout=60)  # fails harmlessly if already gone
            reaped = _reap(before)
            if reaped:
                ids = " ".join(c[:12] for c in reaped)
                f.write(f"\nFACTORY: removed leaked sandbox container(s): {ids}\n".encode())
    return 124 if rc is None else rc


def _canon(line: str) -> str:
    """Normalise a ``name==version`` pin: pip and uv spell project names differently."""
    name, _, version = line.strip().partition("==")
    return f"{re.sub(r'[-_.]+', '-', name).lower()}=={version}"


def _runner_freeze() -> list[str]:
    """Return the runner's installed set (``pip freeze --all`` minus pip), canonicalised."""
    r = _docker(["run", "--rm", RUNNER_TAG, "pip", "freeze", "--all"])
    if r.returncode != 0:
        raise KitError(f"pip freeze in {RUNNER_TAG} exit {r.returncode}: {_err(r)}")
    lines = {_canon(ln) for ln in _out(r).splitlines() if ln.strip()}
    return sorted(ln for ln in lines if not ln.startswith("pip=="))


def _runner_driver_shas(pins: dict[str, Any]) -> dict[str, str]:
    """Return the sha256 of each driver baked into the runner, asserted equal to its pin."""
    paths = [f"/factory/{n}" for n in DRIVERS]
    r = _docker(["run", "--rm", "--entrypoint", "sha256sum", RUNNER_TAG, *paths])
    if r.returncode != 0:
        raise KitError(f"sha256sum in {RUNNER_TAG} exit {r.returncode}: {_err(r)}")
    got = {Path(p).name: s for s, p in (ln.split() for ln in _out(r).splitlines() if ln.strip())}
    shas = {}
    for name, key in DRIVERS.items():
        if got.get(name) != pins["harness"][key]:
            raise PinMismatchError(
                f"{RUNNER_TAG} /factory/{name} sha256 {got.get(name)} != pinned "
                f"{pins['harness'][key]}; rebuild with `factory eval kit-build`"
            )
        shas[key] = got[name]
    return shas


def _measure() -> tuple[dict[str, Any], dict[str, str]]:
    """Measure the kit environment: ``(material, extras)``; ``env_hash`` hashes material.

    Raises:
        PinMismatchError: If a driver, the competition wheels or the runner freeze drifted.
        KitError: If an image is missing or a docker call fails.
    """
    pins = assert_harness_pins()
    wheels = dir_digest(competition_dir() / "wheels", "*.whl")
    if wheels._asdict() != pins["competition_wheels"]:
        raise PinMismatchError(f"competition wheels {wheels} != pinned")
    ids = {"sandbox_image_id": image_id(SANDBOX_TAG), "runner_image_id": image_id(RUNNER_TAG)}
    want = sorted({_canon(ln) for ln in CONSTRAINTS.read_text(encoding="utf-8").splitlines()})
    freeze = _runner_freeze()
    if freeze != want:
        extra = sorted(set(freeze) - set(want))
        missing = sorted(set(want) - set(freeze))
        raise PinMismatchError(
            f"{RUNNER_TAG} freeze != {CONSTRAINTS.name}: extra {extra[:10]} missing {missing[:10]}"
        )
    material = {
        **ids,
        "harness_route": HARNESS_ROUTES[ROUTE],
        "runner_freeze": freeze,
        "wheelhouse_sha": wheels.sha256,
    }
    return material, {**ids, **_runner_driver_shas(pins), "wheelhouse_sha": wheels.sha256}


def measure_kit_env(sweep_dir: Path) -> dict[str, Any]:
    """Measure the kit's env_hash at sweep start; write ``<sweep_dir>/env_material.json``.

    The hash covers both image ids, the runner's freeze (asserted equal, as a line set, to
    ``runner-constraints.txt``) and the competition wheels sha (``pins.dir_digest``, as
    ``local_env.measure_env`` computes it).

    Returns:
        ``env_hash``, ``wheelhouse_sha``, both image ids and the driver shas.
    """
    material, extras = _measure()
    (sweep_dir / "env_material.json").write_text(
        json.dumps(material, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {"env_hash": _hash_material(material), **extras}


def end_of_kit_check(env: dict[str, Any], sweep_dir: Path | None = None) -> str | None:
    """Re-measure the kit env at sweep end; return the drift reason, or None if unchanged.

    With ``sweep_dir``, a drift is marked through ``local_env.mark_drift`` (``DRIFT.json``
    plus the drift ledger), exactly as the WSL routes mark theirs.
    """
    try:
        end_hash = _hash_material(_measure()[0])
    except (PinMismatchError, RuntimeError, OSError) as exc:
        reason = f"end-of-sweep re-measure failed: {exc}"
    else:
        if end_hash == env["env_hash"]:
            return None
        reason = f"env_hash changed: start {env['env_hash']} end {end_hash}"
    if sweep_dir is not None:
        mark_drift(sweep_dir, ROUTE, env["env_hash"], reason)
    return reason
