"""Tests for the crash-safe checkpoint writer."""

import numpy as np

from gol_mosaics._atomic import atomic_save


def test_atomic_save_writes_and_overwrites(tmp_path):
    path = tmp_path / "cube_00001.npy"
    atomic_save(path, np.arange(5))
    atomic_save(path, np.arange(3))  # Path.rename would fail here on Windows
    assert np.load(path).tolist() == [0, 1, 2]
    assert [p.name for p in tmp_path.iterdir()] == ["cube_00001.npy"]


def test_atomic_save_leaves_no_npy_named_temp_file(tmp_path, monkeypatch):
    """Resuming runs list *.npy; a half-written temp file must not match."""
    import os

    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(os.path.basename(src))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    atomic_save(tmp_path / "leaf.npy", np.zeros(2))
    assert seen and not seen[0].endswith(".npy")
