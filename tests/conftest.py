"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest


def find_repo_root(start: Path = Path(__file__)) -> Path:
    """The nearest ancestor holding pyproject.toml, wherever the test sits."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"no pyproject.toml above {start}")


REPO_ROOT = find_repo_root()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
