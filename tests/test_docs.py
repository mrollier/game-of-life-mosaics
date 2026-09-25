"""The documentation's links resolve and its example runs."""

import re
import subprocess

import pytest

from tests.conftest import REPO_ROOT

LINK = re.compile(r"\]\(([^)\s]+)\)")


def _tracked_markdown():
    files = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT,
                           check=True, capture_output=True, text=True).stdout
    return [REPO_ROOT / f for f in files.split()]


MARKDOWN = _tracked_markdown()


@pytest.mark.parametrize("path", MARKDOWN,
                         ids=[p.relative_to(REPO_ROOT).as_posix() for p in MARKDOWN])
def test_relative_links_resolve(path):
    broken = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part = target.split("#")[0]
        if file_part and not (path.parent / file_part).exists():
            broken.append(target)
    assert not broken, f"{path.name}: broken links {broken}"


def test_readme_example_runs(tmp_path, monkeypatch):
    """The README's first Python block works as printed."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    monkeypatch.chdir(REPO_ROOT)
    code = code.replace("'john_mosaic.png'", repr(str(tmp_path / "out.png")))
    exec(compile(code, "README.md", "exec"), {})
    assert (tmp_path / "out.png").stat().st_size > 0


def test_no_stray_files_at_the_root():
    """The root holds project files only; code and docs live in folders."""
    allowed = {".gitattributes", ".gitignore", "CHANGELOG.md",
               "CONTRIBUTING.md", "LICENSE", "MANIFEST.in", "README.md",
               "pyproject.toml"}
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True).stdout.split()
    root_files = {f for f in tracked if "/" not in f}
    assert root_files <= allowed, sorted(root_files - allowed)
