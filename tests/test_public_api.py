"""The 3.0 public API: what the package exports, and that 2.x names are gone.

3.0 renamed the public API without compatibility shims (see CHANGELOG.md and
docs/api.md for the 2.x -> 3.0 table). These tests pin the exports and
fail on any 2.x identifier left behind in code, tests, notebooks or scripts.
"""

import re

import pytest

import gol_mosaics
from tests.conftest import REPO_ROOT

EXPORTS = {
    "MosaicGenerator", "ColourScheme", "GollyExporter", "ImageProcessor",
    "ECABackground", "MosaicRenderer", "compose", "agar_background",
    "density_band", "life_safe_pattern", "mosaic_background",
    "filled_background", "fill_layer_count", "scatter_background",
    "PatternLibrary",
}

# 2.x identifiers that must not appear anywhere in code any more.
BANNED = [
    r"\bColorScheme\b",
    r"\bcolor_scheme\b",
    r"\bgol_mosaics\.colors\b",
    r"\balpha_color\b",
    r"\bfill_color\b",
    r"\b_hex_to_rgb\b",
]

SEARCHED = ["src/**/*.py", "tests/**/*.py", "app/*.py", "experiments/**/*.py",
            "studies/**/*.py", "search/**/*.py", "tools/*.py",
            "notebooks/**/*.ipynb"]


def test_exports_are_exactly_the_public_api():
    assert set(gol_mosaics.__all__) == EXPORTS
    for name in EXPORTS:
        assert hasattr(gol_mosaics, name), name


def _code_files():
    for pattern in SEARCHED:
        for path in REPO_ROOT.glob(pattern):
            if path.name != "test_public_api.py":
                yield path


@pytest.mark.parametrize("pattern", BANNED)
def test_no_2x_names_left(pattern):
    hits = []
    regex = re.compile(pattern)
    for path in _code_files():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not hits, f"{pattern} still used at {hits[:10]}"
