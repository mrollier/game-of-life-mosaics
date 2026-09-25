"""Repository conventions that are easy to break by accident."""

import json

import pytest

from tests.conftest import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))
RESEARCH = [p for p in NOTEBOOKS if "research" in p.parts]


def _ids(paths):
    return [p.relative_to(REPO_ROOT).as_posix() for p in paths]


@pytest.mark.parametrize("path", RESEARCH, ids=_ids(RESEARCH))
def test_research_notebooks_have_no_outputs(path):
    """They are run to be read, not shipped with megabytes of figures."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{path.name} has saved outputs"
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_ids(NOTEBOOKS))
def test_notebooks_hold_no_machine_paths(path):
    """No absolute paths from someone's machine, in code or in outputs."""
    text = path.read_text(encoding="utf-8")
    for marker in ("/Users/", "C:\\Users", "/home/", "OneDrive"):
        assert marker not in text, f"{path.name} contains {marker!r}"

