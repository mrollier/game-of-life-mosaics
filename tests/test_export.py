"""Tests for GollyExporter class."""

import pytest
import numpy as np
from gol_mosaics import GollyExporter


def test_export_to_cells_exact_text(tmp_path):
    """A known 3x3 pattern produces the exact .cells text."""
    mosaic = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    path = tmp_path / "plus.cells"

    GollyExporter.export_to_cells(mosaic, str(path))

    assert path.read_text() == (
        "!Generated from Game of Life Mosaic\n"
        ".O.\n"
        "OOO\n"
        ".O.\n"
    )


def test_export_to_cells_validates_input(tmp_path):
    """Non-2D or non-binary arrays and unknown corners raise ValueError."""
    path = str(tmp_path / "bad.cells")
    with pytest.raises(ValueError):
        GollyExporter.export_to_cells(np.zeros(4), path)
    with pytest.raises(ValueError):
        GollyExporter.export_to_cells(np.array([[0, 2]]), path)
    with pytest.raises(ValueError):
        GollyExporter.export_to_cells(np.zeros((5, 5)), path,
                                      add_glider='centre')


def test_export_to_cells_glider_does_not_mutate_input(tmp_path):
    """Adding a glider writes it to the file but leaves the caller's array
    untouched."""
    mosaic = np.zeros((10, 10))
    path = tmp_path / "glider.cells"

    GollyExporter.export_to_cells(mosaic, str(path), add_glider='top left')

    assert not mosaic.any()  # input unchanged
    lines = path.read_text().splitlines()
    assert lines[1:4] == ['.O........', '..O.......', 'OOO.......']


def test_export_to_rle_exact_text(tmp_path):
    """A known glider produces the exact RLE header and encoding."""
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
    path = tmp_path / "glider.rle"

    GollyExporter.export_to_rle(glider, str(path), name='Glider',
                                comments='line one\nline two')

    assert path.read_text() == (
        "#N Glider\n"
        "#C line one\n"
        "#C line two\n"
        "x = 3, y = 3, rule = B3/S23\n"
        "bob$2bo$3o!\n"
    )


def test_export_to_rle_wraps_long_patterns(tmp_path):
    """The RLE body is wrapped into lines of at most 70 characters."""
    row = np.tile([0, 1], 100).reshape(1, -1)  # alternating -> 200-char RLE
    path = tmp_path / "wide.rle"

    GollyExporter.export_to_rle(row, str(path))

    body = path.read_text().splitlines()[1:]  # skip the size line
    assert len(body) > 1
    assert all(len(line) <= 70 for line in body)
    assert ''.join(body) == 'bo' * 100 + '!'


def test_export_to_rle_validates_input(tmp_path):
    """Non-2D or non-binary arrays raise ValueError."""
    path = str(tmp_path / "bad.rle")
    with pytest.raises(ValueError):
        GollyExporter.export_to_rle(np.zeros(4), path)
    with pytest.raises(ValueError):
        GollyExporter.export_to_rle(np.array([[0, 3]]), path)


def test_encode_rle_row_units():
    """Run-length encoding of single rows."""
    assert GollyExporter._encode_rle_row(np.array([])) == ''
    assert GollyExporter._encode_rle_row(np.array([0])) == 'b'
    assert GollyExporter._encode_rle_row(np.array([1, 1, 1])) == '3o'
    assert GollyExporter._encode_rle_row(np.array([1, 0, 0, 1, 1])) == 'o2b2o'
