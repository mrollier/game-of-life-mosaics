"""Tests for the beyond-tiles spike: free-form density-matched still lifes.

The numpy-only layer (targets, metrics) is tested unconditionally; the
CP-SAT layer is skipped when ortools is not installed, mirroring the
pysat gating in test_nosym_tiles.py.
"""

import numpy as np
import pytest

from beyond_tiles.targets import (
    cell_targets,
    ramp_grey,
    uniform_grey,
    window_slices,
    window_targets,
)
from beyond_tiles.metrics import deviation_stats, motif_stats, tile_db_overlap


# ---------------------------------------------------------------------------
# targets: window geometry
# ---------------------------------------------------------------------------


def test_window_slices_exact_tiling():
    windows = window_slices((100, 100), k=8, stride=4)
    assert len(windows) == 576  # 24 x 24
    starts = sorted({ws[0].start for ws in windows})
    assert starts[0] == 0 and starts[-1] == 92
    for si, sj in windows:
        assert 0 <= si.start and si.stop <= 100
        assert 0 <= sj.start and sj.stop <= 100
        assert si.stop - si.start == 8 and sj.stop - sj.start == 8


def test_window_slices_clamped_final_window():
    windows = window_slices((150, 150), k=8, stride=4)
    starts = sorted({ws[0].start for ws in windows})
    # range(0, 143, 4) ends at 140; a clamped window at 142 must be added
    assert starts[-1] == 142
    assert len(windows) == 37 * 37
    assert all(ws[0].stop <= 150 and ws[1].stop <= 150 for ws in windows)


def test_window_slices_full_coverage():
    covered = np.zeros((20, 20), dtype=bool)
    for si, sj in window_slices((20, 20), k=8, stride=4):
        covered[si, sj] = True
    assert covered.all()


# ---------------------------------------------------------------------------
# targets: greyscale -> per-cell density
# ---------------------------------------------------------------------------


def test_cell_targets_inversion_and_cap():
    grey = np.array([[0, 128, 255]], dtype=np.uint8)
    d = cell_targets(grey, d_max=0.45)
    assert d[0, 0] == pytest.approx(0.45)  # black -> densest
    assert d[0, 2] == pytest.approx(0.0)  # white -> empty
    assert d[0, 0] > d[0, 1] > d[0, 2]
    assert (d >= 0).all() and (d <= 0.45).all()


def test_ramp_and_uniform_grey_helpers():
    ramp = ramp_grey(16)
    assert ramp.shape == (16, 16)
    assert ramp[0, 0] == 0 and ramp[0, -1] == 255
    assert (uniform_grey(8, 100) == 100).all()


# ---------------------------------------------------------------------------
# targets: per-window integer targets with masking
# ---------------------------------------------------------------------------


def test_window_targets_masking_scales_and_drops():
    grey = uniform_grey(8, 0)  # black everywhere -> d = d_max per cell
    cell_t = cell_targets(grey, d_max=0.4)
    free = np.ones((8, 8), dtype=bool)
    free[:, 4:] = False  # right half masked

    full = window_slices((8, 8), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, full)
    assert kept == full
    assert targets[0] == round(32 * 0.4)  # only free cells contribute

    all_masked_targets, all_masked_kept = window_targets(
        cell_t, np.zeros((8, 8), dtype=bool), full
    )
    assert all_masked_kept == [] and len(all_masked_targets) == 0


def test_gradient_monotonic_densities():
    si = _solver()
    from scipy.stats import spearmanr

    grey = ramp_grey(20)  # black on the left -> dense on the left
    free = np.ones((20, 20), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si, time_limit_s=20.0))
    col_density = [
        result.pattern[:, j : j + 4].mean() for j in range(0, 20, 4)
    ]
    rho, _ = spearmanr(col_density, range(5))
    assert rho < -0.8  # density falls left to right


def test_masked_cells_stay_dead():
    si = _solver()
    grey = uniform_grey(12, 0)
    free = np.ones((12, 12), dtype=bool)
    free[4:8, 4:8] = False
    result = si.solve_image(grey, free, _test_config(si))
    assert result.pattern[4:8, 4:8].sum() == 0
    assert result.pattern.any()


def test_same_seed_same_pattern():
    si = _solver()
    grey = ramp_grey(12)
    free = np.ones((12, 12), dtype=bool)
    a = si.solve_image(grey, free, _test_config(si))
    b = si.solve_image(grey, free, _test_config(si))
    assert np.array_equal(a.pattern, b.pattern)


def test_end_to_end_artifacts(tmp_path):
    si = _solver()
    from PIL import Image

    from beyond_tiles.artifacts import save_run
    from beyond_tiles.targets import grey_and_mask_from_image

    img = Image.fromarray(np.tile(ramp_grey(64), (1, 1)), mode="L").convert("RGB")
    grey, free = grey_and_mask_from_image(img, size=16, remove_background=False)
    assert grey.shape == (16, 16) and free.shape == (16, 16)
    assert free.all()  # no alpha channel -> everything is subject

    result = si.solve_image(grey, free, _test_config(si))
    outdir = tmp_path / "run"
    save_run(outdir, result, grey, free)
    for name in (
        "pattern.npy",
        "render.png",
        "density_maps.png",
        "convergence.csv",
        "metrics.json",
        "pattern.cells",
    ):
        assert (outdir / name).exists(), name

    # the .cells export must round-trip to the same pattern
    rows = [
        line
        for line in (outdir / "pattern.cells").read_text().splitlines()
        if line and not line.startswith("!")
    ]
    parsed = np.array(
        [[1 if c == "O" else 0 for c in row] for row in rows], dtype=np.uint8
    )
    assert np.array_equal(parsed, result.pattern)


# ---------------------------------------------------------------------------
# metrics: deviation statistics
# ---------------------------------------------------------------------------


def _solver():
    """Import the CP-SAT layer, skipping when ortools is absent."""
    pytest.importorskip("ortools")
    from beyond_tiles import still_image

    return still_image


def _test_config(still_image, **overrides):
    """Deterministic small-scale solver settings for tests."""
    defaults = dict(k=4, stride=2, d_max=0.45, time_limit_s=10.0, workers=1, seed=0)
    defaults.update(overrides)
    return still_image.SpikeConfig(**defaults)


# ---------------------------------------------------------------------------
# CP-SAT encoder
# ---------------------------------------------------------------------------


def test_white_image_solves_to_all_dead():
    si = _solver()
    grey = uniform_grey(12, 255)
    free = np.ones((12, 12), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si))
    assert result.objective == 0
    assert result.pattern.sum() == 0
    assert result.pattern.shape == (12, 12)


def test_uniform_target_yields_verified_still_life():
    si = _solver()
    grey = uniform_grey(12, int(round(255 * (1 - 0.35 / 0.45))))  # d = 0.35
    free = np.ones((12, 12), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si))
    checks = si.verify_still_life(result.pattern)
    assert checks["bounded"] and checks["toroidal"]
    assert result.pattern.mean() > 0.15


def test_ring_no_birth_regression():
    # d_max targets push live mass to the grid edge; the pattern must
    # still be stable when embedded in a dead plane (pad-1 check).
    si = _solver()
    from gol_mosaics.life import is_still_life

    grey = uniform_grey(10, 0)  # black -> d = d_max everywhere
    free = np.ones((10, 10), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si))
    assert result.pattern.any()
    assert is_still_life(np.pad(result.pattern, 1))


def test_deviation_stats_hand_computed():
    # Two 8x8 windows side by side; left achieves 0.25 vs target 0.20,
    # right achieves 0.0 vs target 0.10.
    pattern = np.zeros((8, 16), dtype=np.uint8)
    pattern[:4, :4] = 1  # 16 live cells in the left window
    cell_t = np.concatenate(
        [np.full((8, 8), 0.20), np.full((8, 8), 0.10)], axis=1
    )
    free = np.ones((8, 16), dtype=bool)
    windows = window_slices((8, 16), k=8, stride=8)
    assert len(windows) == 2

    stats = deviation_stats(pattern, cell_t, free, windows)
    assert stats["mad"] == pytest.approx((0.05 + 0.10) / 2)
    assert stats["max"] == pytest.approx(0.10)
    # darkest quartile = windows with the highest targets -> the left one
    assert stats["mad_darkest_quartile"] == pytest.approx(0.05)
    # achieved (0.25, 0.0) vs target (0.20, 0.10) correlate positively
    assert stats["pearson"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# metrics: texture statistics (E5)
# ---------------------------------------------------------------------------


def test_motif_stats_uniform_pattern():
    stats = motif_stats(np.ones((8, 8), dtype=np.uint8), block=4)
    assert stats["n_distinct"] == 1
    assert stats["entropy_bits"] == pytest.approx(0.0)


def test_motif_stats_two_distinct_blocks():
    pattern = np.zeros((4, 8), dtype=np.uint8)
    pattern[:, :4] = 1  # block 1: all live
    pattern[0, 4] = 1  # block 2: single live corner
    stats = motif_stats(pattern, block=4)
    assert stats["n_distinct"] == 2
    assert stats["entropy_bits"] == pytest.approx(1.0)
    assert stats["n_blocks"] == 2


def test_motif_stats_skips_empty_blocks():
    pattern = np.zeros((4, 8), dtype=np.uint8)
    pattern[:, :4] = 1
    stats = motif_stats(pattern, block=4)  # right block empty -> excluded
    assert stats["n_blocks"] == 1


def test_tile_db_overlap_detects_level1_pond():
    import gol_mosaics
    from pathlib import Path

    pond = np.load(
        Path(gol_mosaics.__file__).parent / "data" / "solutions_pattern_level_1.npy"
    )[0].astype(np.uint8)
    pattern = np.zeros((12, 12), dtype=np.uint8)
    pattern[0:6, 0:6] = pond  # one block IS the level-1 tile
    pattern[6:8, 6:8] = 1  # one block is a 2x2 block still life (not a tile)
    assert tile_db_overlap(pattern, level=1) == pytest.approx(0.5)
    assert tile_db_overlap(pond, level=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# targets: subject-relative tone normalization
# ---------------------------------------------------------------------------


def test_normalize_grey_stretches_subject_range():
    from beyond_tiles.targets import normalize_grey

    grey = np.full((10, 10), 255, dtype=np.uint8)
    free = np.zeros((10, 10), dtype=bool)
    free[:, :5] = True
    # subject occupies a narrow midtone band 100..160
    grey[:, :5] = np.linspace(100, 160, 50).reshape(10, 5).astype(np.uint8)

    out = normalize_grey(grey, free, p_lo=0.0, p_hi=100.0)
    subject = out[free]
    assert subject.min() == 0 and subject.max() == 255  # full range used
    assert (out[~free] == 255).all()  # background stays white
    # monotone: order of grey values preserved
    order_in = np.argsort(grey[free], kind="stable")
    assert (np.diff(subject[order_in].astype(int)) >= 0).all()


def test_normalize_grey_flat_subject_unchanged():
    from beyond_tiles.targets import normalize_grey

    grey = uniform_grey(8, 77)
    free = np.ones((8, 8), dtype=bool)
    out = normalize_grey(grey, free)
    assert (out == 77).all()


def test_equalize_grey_flattens_distribution():
    from beyond_tiles.targets import equalize_grey

    rng = np.random.default_rng(0)
    grey = np.full((20, 20), 255, dtype=np.uint8)
    free = np.zeros((20, 20), dtype=bool)
    free[:, :10] = True
    # high-key subject: values clustered at the bright end
    grey[free] = rng.integers(200, 250, free.sum()).astype(np.uint8)

    out = equalize_grey(grey, free)
    med = np.median(out[free])
    assert 100 <= med <= 155  # roughly centred after equalization
    assert (out[~free] == 255).all()
    # monotone: brighter in stays brighter (or equal) out
    a, b = grey[free], out[free]
    order = np.argsort(a, kind="stable")
    assert (np.diff(b[order].astype(int)) >= 0).all()


# ---------------------------------------------------------------------------
# shipped high-resolution assets
# ---------------------------------------------------------------------------


def test_pattern_asset_round_trip(tmp_path):
    from beyond_tiles.artifacts import load_pattern_asset, save_pattern_asset

    rng = np.random.default_rng(3)
    pattern = rng.integers(0, 2, (37, 53)).astype(np.uint8)
    path = save_pattern_asset(tmp_path / "p.npz", pattern)
    back = load_pattern_asset(path)
    assert back.shape == pattern.shape
    assert back.dtype == np.uint8
    assert (back == pattern).all()


@pytest.mark.parametrize(
    "name,size", [("marilyn_200_optimal.npz", 200), ("marilyn_400_feasible.npz", 400)]
)
def test_shipped_assets_are_still_lifes(name, size):
    from gol_mosaics.life import is_still_life

    from beyond_tiles.artifacts import ASSETS, load_pattern_asset

    pattern = load_pattern_asset(ASSETS / name)
    assert pattern.shape == (size, size)
    assert pattern.sum() > 0
    assert is_still_life(np.pad(pattern, 1))
