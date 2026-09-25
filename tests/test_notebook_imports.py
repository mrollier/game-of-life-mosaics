"""Every project name a notebook imports must exist.

Notebooks are not executed in the test suite (they download models, run
long solves and write figures), so a rename in the package would otherwise
break them silently. This test parses each code cell, collects the imports
from this project's modules and checks that every imported name resolves.
"""

import ast
import importlib
import json

import pytest

from tests.conftest import REPO_ROOT

PROJECT_MODULES = ("gol_mosaics", "beyond_tiles")

# Optional dependencies a module may need at import time; a notebook that
# uses them is skipped rather than failed when they are not installed.
OPTIONAL = {"ortools", "pysat", "gurobipy", "numba", "rembg", "gradio"}

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").rglob("*.ipynb"))


def project_imports(path):
    """(module, name or None) for every project import in a notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython magics and shell escapes are not Python.
        lines = [line for line in source.splitlines()
                 if not line.lstrip().startswith(("%", "!"))]
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith(PROJECT_MODULES):
                found += [(node.module, alias.name) for alias in node.names]
            elif isinstance(node, ast.Import):
                found += [(alias.name, None) for alias in node.names
                          if alias.name.startswith(PROJECT_MODULES)]
    return found


@pytest.mark.parametrize("path", NOTEBOOKS,
                         ids=[p.relative_to(REPO_ROOT).as_posix()
                              for p in NOTEBOOKS])
def test_notebook_imports_resolve(path):
    missing = []
    for module, name in project_imports(path):
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            if exc.name and exc.name.split(".")[0] in OPTIONAL:
                pytest.skip(f"{module} needs optional {exc.name}")
            missing.append(f"{module} ({exc})")
            continue
        if name is not None and name != "*" and not hasattr(mod, name):
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"{module}.{name}")
    assert not missing, f"{path.name} imports names that do not exist: {missing}"
