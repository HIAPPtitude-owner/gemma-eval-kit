"""Filesystem locations under the data root named by ``FACTORY_DATA_ROOT`` (required)."""

import os
from pathlib import Path

DATA_ROOT_ENV = "FACTORY_DATA_ROOT"


class DataRootError(ValueError):
    """Raised when ``FACTORY_DATA_ROOT`` is not set."""


def data_root() -> Path:
    """Return the data root named by ``FACTORY_DATA_ROOT``; there is no default.

    Returns:
        The data root as a Path.

    Raises:
        DataRootError: If ``FACTORY_DATA_ROOT`` is unset or empty.
    """
    raw = os.environ.get(DATA_ROOT_ENV, "").strip()
    if not raw:
        raise DataRootError(
            f"set {DATA_ROOT_ENV} to the folder holding competition/ and wheelhouse/ "
            "(see README.md, Quickstart)"
        )
    return Path(raw)


def competition_dir() -> Path:
    """Return the directory holding competition files: ``<data_root>/competition``."""
    return data_root() / "competition"


def wheelhouse_dir() -> Path:
    """Return the directory holding offline wheels: ``<data_root>/wheelhouse``."""
    return data_root() / "wheelhouse"


def results_dir() -> Path:
    """Return the directory holding run results: ``<data_root>/results``."""
    return data_root() / "results"


def ledger_path() -> Path:
    """Return the run ledger file: ``<data_root>/ledger/runs.jsonl``."""
    return data_root() / "ledger" / "runs.jsonl"
