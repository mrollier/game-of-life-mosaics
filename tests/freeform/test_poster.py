"""Tests for the production pipeline, gol_mosaics.freeform.poster."""

import numpy as np
import pytest

pytest.importorskip("ortools")

from gol_mosaics.freeform.lns import window_devs  # noqa: E402
from gol_mosaics.freeform.poster import PosterConfig, solve_poster  # noqa: E402


def _canvas(size=32):
    """A dark disc on white: dense and empty windows, a ragged mask edge."""
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(y - size / 2, x - size / 2)
    grey = np.clip(r / (size / 2) * 255, 0, 255).astype(np.uint8)
    free = r < size * 0.45
    return grey, free


def _small(**overrides):
    base = dict(d_max=0.3, strip_rows=16, strip_time=30.0, strip_procs=2,
                strip_workers=1, polish_rounds=1, polish_budget=3.0,
                patch_time=1.0, polish_procs=1, seam_rounds=1,
                seam_budget=3.0)
    base.update(overrides)
    return PosterConfig(**base)


def test_solve_poster_end_to_end(tmp_path):
    grey, free = _canvas()
    result = solve_poster(grey, free, _small(), out=tmp_path)
    pattern = result.pattern
    assert pattern.shape == grey.shape
    assert pattern[~free].sum() == 0
    assert result.report["verify"] == {"bounded": True, "toroidal": True}
    assert result.objective == int(window_devs(pattern, free, result.windows,
                                               result.targets).sum())
    assert result.report["max_diagonal_run"] <= 5
    assert (tmp_path / "strips_pattern.npy").exists()


def test_isolated_strips_match_the_shared_pool():
    """One process per strip changes robustness, not the answer."""
    grey, free = _canvas()
    plain = solve_poster(grey, free, _small(polish_rounds=0, seam_rounds=0))
    isolated = solve_poster(grey, free, _small(polish_rounds=0, seam_rounds=0,
                                               isolate=True))
    assert plain.report["solve"]["statuses"] == ["OPTIMAL", "OPTIMAL"]
    assert np.array_equal(plain.pattern, isolated.pattern)


def test_resume_picks_up_the_saved_pattern(tmp_path):
    grey, free = _canvas()
    first = solve_poster(grey, free, _small(seam_rounds=0), out=tmp_path)
    again = solve_poster(grey, free, _small(seam_rounds=0), out=tmp_path,
                         resume=True)
    assert again.report["solve"]["statuses"] == ["resumed", "resumed"]
    assert again.objective <= first.objective
