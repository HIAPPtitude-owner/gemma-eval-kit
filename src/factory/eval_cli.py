"""The ``factory eval local|kit-build|c5|rescore`` commands (moved out of ``factory.cli``).

This module imports none of the Kaggle/submit code (``kaggle_run``, ``submit``, ``bundle``,
``quota``), so it ships in the public gemma-eval-kit release, whose ``factory`` script is
``factory.eval_cli:main``. ``factory.cli`` registers these commands next to ``eval kaggle``.

The handlers call their collaborators through a ``seams`` module (``factory.cli`` when
reached from there, this module otherwise), so a test that replaces ``factory.cli.sweep``
still reaches the handler.
"""

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import get_args

# The ``noqa: F401`` names are this module's own seams (SEAM_NAMES), reached via sys.modules.
from factory.c5 import c5_sweep, empty_results, parse_manifest, task_verdicts  # noqa: F401
from factory.docker_kit import KitError, build_images  # noqa: F401
from factory.local_eval import (  # noqa: F401
    Harness,
    Mode,
    SweepDriftError,
    SweepError,
    resolve_task_ids,
    sweep,
)
from factory.paths import DataRootError, competition_dir, wheelhouse_dir  # noqa: F401
from factory.pins import PinMismatchError
from factory.rescore import CONFIGS, rescore  # noqa: F401

# The collaborators a handler reaches through ``seams`` (see the module docstring).
SEAM_NAMES = (
    "build_images",
    "c5_sweep",
    "competition_dir",
    "empty_results",
    "parse_manifest",
    "rescore",
    "resolve_task_ids",
    "sweep",
    "task_verdicts",
    "wheelhouse_dir",
)
EVAL_COMMANDS = ("local", "kit-build", "c5", "rescore")


def add_eval_commands(eval_sub: argparse._SubParsersAction) -> None:
    """Add ``local``, ``kit-build``, ``c5`` and ``rescore`` to the ``eval`` subparsers."""
    p_local = eval_sub.add_parser(
        "local", help="sweep tasks on the WSL1 distro or the public Docker kit (not the scorer)"
    )
    p_local.add_argument(
        "--harness",
        required=True,
        choices=get_args(Harness),
        help="as_shipped/diagnostic: WSL1 distro; public_docker: Docker kit (kit-build first)",
    )
    eval_sub.add_parser("kit-build", help="build the public Docker kit's sandbox + runner images")
    p_local.add_argument("--mode", required=True, choices=get_args(Mode))
    p_local.add_argument("--tasks", required=True, help="'all' or a file of task ids")
    p_local.add_argument("--repeats", type=int, default=1, help="runs per task (default 1)")
    p_c5 = eval_sub.add_parser("c5", help="C5: score non-gold patches on the diagnostic harness")
    p_c5.add_argument("--manifest", required=True, type=Path, help="<task>\\t<input> lines")
    p_c5.add_argument("--empty-log", required=True, type=Path, help="diagnostic empty sweep log")
    p_rescore = eval_sub.add_parser(
        "rescore", help="E2: re-score stored Kaggle agent patches on one local route"
    )
    p_rescore.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p_rescore.add_argument("--harness", required=True, choices=get_args(Harness))


def _cmd_eval_local(args: argparse.Namespace, s: ModuleType) -> int:
    try:
        task_ids = s.resolve_task_ids(args.tasks)
        rows = s.sweep(task_ids, args.harness, args.mode, repeats=args.repeats)
    except SweepError as exc:
        print(f"EVAL-INCOMPLETE: {exc} ({len(exc.rows)} row(s) recorded)")
        return 1
    except SweepDriftError as exc:
        print(f"EVAL-DRIFT: {exc} ({len(exc.rows)} row(s) recorded, marked untrustworthy)")
        return 1
    except (PinMismatchError, ValueError, RuntimeError) as exc:
        print(f"EVAL-REFUSED: {exc}")
        return 1
    print(f"EVAL-OK: {len(rows)} row(s), {sum(r.resolved for r in rows)} resolved")
    return 0


def _cmd_eval_kit_build(_args: argparse.Namespace, s: ModuleType) -> int:
    try:
        ids = s.build_images(s.competition_dir(), s.wheelhouse_dir())
    except (KitError, DataRootError, OSError) as exc:
        print(f"KIT-BUILD-REFUSED: {exc}")
        return 1
    for key, value in ids.items():
        print(f"KIT-IMAGE: {key}={value}")
    return 0


def _cmd_eval_c5(args: argparse.Namespace, s: ModuleType) -> int:
    try:
        known = set(s.resolve_task_ids("all"))
        runs = s.parse_manifest(args.manifest.read_text(encoding="utf-8"), known)
        empty = s.empty_results(args.empty_log.read_text(encoding="utf-8"))
        scored = s.c5_sweep(runs)
    except (SweepError, SweepDriftError) as exc:
        print(f"C5-INCOMPLETE: {exc}")
        return 1
    except (PinMismatchError, ValueError, RuntimeError, OSError) as exc:
        print(f"C5-REFUSED: {exc}")
        return 1
    verdicts = s.task_verdicts(empty, scored)
    for task, verdict in verdicts.items():
        print(f"C5-VERDICT {task} {verdict} (empty from {args.empty_log})")
    overall = "PASS" if verdicts and all(v == "PASS" for v in verdicts.values()) else "FAIL"
    if any(v == "NOT-VERIFIED" for v in verdicts.values()):
        overall = "NOT-VERIFIED"
    print(f"C5-OVERALL {overall} tasks={len(verdicts)}")
    return 0 if overall == "PASS" else 1


def _cmd_eval_rescore(args: argparse.Namespace, s: ModuleType) -> int:
    try:
        rows = s.rescore(args.config, args.harness)
    except (SweepError, SweepDriftError) as exc:
        print(f"RESCORE-INCOMPLETE: {exc}")
        return 1
    except (PinMismatchError, ValueError, RuntimeError, OSError) as exc:
        print(f"RESCORE-REFUSED: {exc}")
        return 1
    print(f"RESCORE-OK: {len(rows)} row(s), {sum(r.local_resolved for r in rows)} resolved")
    return 0


_HANDLERS = {
    "local": _cmd_eval_local,
    "kit-build": _cmd_eval_kit_build,
    "c5": _cmd_eval_c5,
    "rescore": _cmd_eval_rescore,
}


def run_eval(args: argparse.Namespace, seams: ModuleType | None = None) -> int:
    """Run one of the ``EVAL_COMMANDS`` named by ``args.eval_command``.

    Args:
        args: Parsed arguments from a parser built with ``add_eval_commands``.
        seams: Module whose ``SEAM_NAMES`` attributes the handler calls; this module if None.

    Returns:
        The handler's exit code (0 on success, 1 on any refusal or incomplete sweep).
    """
    return _HANDLERS[args.eval_command](args, seams or sys.modules[__name__])


def _build_parser() -> argparse.ArgumentParser:
    """Build a ``factory`` parser holding only the ``eval`` group."""
    parser = argparse.ArgumentParser(prog="factory")
    sub = parser.add_subparsers(dest="command", required=True)
    p_eval = sub.add_parser("eval", help="local gold/empty evaluation sweeps (not the scorer)")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)
    add_eval_commands(eval_sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ``factory eval local|kit-build|c5|rescore`` (the release's ``factory`` script).

    Args:
        argv: Arguments excluding the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code from ``run_eval``.
    """
    args = _build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
