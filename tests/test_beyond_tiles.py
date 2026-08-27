"""Tests for the beyond-tiles spike: free-form density-matched still lifes.

The numpy-only layer (targets, metrics) is tested unconditionally; the
CP-SAT layer is skipped when ortools is not installed, mirroring the
pysat gating in test_nosym_tiles.py.
"""

from pathlib import Path

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
    "name,size",
    [
        ("marilyn_200_optimal.npz", 200),
        ("marilyn_400_feasible.npz", 400),
        ("marilyn_400_pipeline.npz", 400),
        ("marilyn_1000_pipeline.npz", 1000),
    ],
)
def test_shipped_assets_are_still_lifes(name, size):
    from gol_mosaics.life import is_still_life

    from beyond_tiles.artifacts import ASSETS, load_pattern_asset

    pattern = load_pattern_asset(ASSETS / name)
    assert pattern.shape == (size, size)
    assert pattern.sum() > 0
    assert is_still_life(np.pad(pattern, 1))


def test_agar_background_on_shipped_asset_is_a_still_life():
    """A life-safe backdrop on real portrait geometry stays stable.

    The synthetic fixtures in test_compose.py check the gap rule; this one
    checks it against a mask with the ragged edges a real alpha cut has.
    """
    from gol_mosaics.compose import life_safe_pattern
    from gol_mosaics.life import is_still_life

    from beyond_tiles.artifacts import ASSETS, load_pattern_asset
    from beyond_tiles.still_image import verify_still_life
    from beyond_tiles.targets import grey_and_mask_from_image

    repo = Path(__file__).resolve().parents[1]
    pattern = load_pattern_asset(ASSETS / "marilyn_400_pipeline.npz")
    _, free = grey_and_mask_from_image(repo / "input/images/marilyn.png", 400)

    whole = life_safe_pattern(pattern, ~free)
    assert whole.sum() > pattern.sum(), "the agar should add cells"
    assert is_still_life(np.pad(whole, 1))
    assert verify_still_life(whole) == {"bounded": True, "toroidal": True}


# ---------------------------------------------------------------------------
# convergence movies
# ---------------------------------------------------------------------------


def test_select_frames_keeps_endpoints_and_budget():
    from beyond_tiles.animate import select_frames

    times = list(np.linspace(0.0, 100.0, 500))
    picks = select_frames(times, max_frames=10)
    assert picks[0] == 0 and picks[-1] == len(times) - 1
    assert len(picks) <= 10
    assert picks == sorted(set(picks))  # strictly increasing, no repeats


def test_select_frames_passes_short_runs_through():
    from beyond_tiles.animate import select_frames

    assert select_frames([0.0, 1.0, 2.0], max_frames=10) == [0, 1, 2]
    assert select_frames([], max_frames=10) == []


def test_select_frames_time_pacing_spreads_over_wall_time():
    from beyond_tiles.animate import select_frames

    # 90 incumbents in the first second, 10 spread over the next 99
    times = list(np.linspace(0, 1, 90)) + list(np.linspace(2, 100, 10))
    picks = select_frames(times, max_frames=10, pacing="time")
    late = sum(times[i] > 10 for i in picks)
    assert late >= 5  # uniform in time, not dominated by the early burst
    index_picks = select_frames(times, max_frames=10, pacing="index")
    assert sum(times[i] > 10 for i in index_picks) < late


def test_snapshot_round_trip(tmp_path):
    from beyond_tiles.artifacts import load_snapshots, save_snapshots

    rng = np.random.default_rng(7)
    snaps = [
        (float(t), int(100 - t), rng.integers(0, 2, (12, 20)).astype(np.uint8))
        for t in range(4)
    ]
    save_snapshots(tmp_path / "s.npz", snaps)
    back = load_snapshots(tmp_path / "s.npz")
    assert len(back) == len(snaps)
    for (t0, o0, p0), (t1, o1, p1) in zip(snaps, back):
        assert (t0, o0) == (t1, o1)
        assert (p0 == p1).all()


def test_write_gif_frame_count(tmp_path):
    from PIL import Image

    from beyond_tiles.animate import frame_image, write_gif

    history = [(1.0, 30), (2.0, 20), (3.0, 10)]
    snaps = [
        (t, o, np.zeros((8, 8), dtype=np.uint8)) for t, o in history
    ]
    frames = [frame_image(s, history, "8x8", px=200) for s in snaps]
    path = write_gif(tmp_path / "m.gif", frames, duration_ms=50)
    with Image.open(path) as gif:
        assert gif.n_frames == 3


def test_snapshots_off_by_default():
    solver = _solver()
    grey = uniform_grey(16, 128)
    free = np.ones((16, 16), dtype=bool)
    cfg = solver.SpikeConfig(k=8, stride=8, time_limit_s=5.0, workers=1, seed=0)
    result = solver.solve_image(grey, free, cfg)
    assert result.snapshots == []
    assert result.obj_history  # the cheap log is always kept


# ---------------------------------------------------------------------------
# model-build optimizations (Stage 1 of the optimization campaign)
# ---------------------------------------------------------------------------


def test_partial_edge_windows_cover_and_disjoint():
    windows = window_slices((150, 150), k=8, stride=8, edge="partial")
    covered = np.zeros((150, 150), dtype=np.int32)
    for si, sj in windows:
        covered[si, sj] += 1
    assert (covered == 1).all()  # full coverage, pairwise disjoint
    row_spans = sorted({(ws[0].start, ws[0].stop) for ws in windows})
    assert row_spans[-1] == (144, 150)  # final short window, not a clamped one
    assert all(stop - start == 8 for start, stop in row_spans[:-1])


def test_edge_mode_rejects_unknown():
    with pytest.raises(ValueError):
        window_slices((16, 16), k=8, stride=8, edge="wrap")


def test_forced_dead_constraint_pruning():
    si = _solver()
    grey = uniform_grey(12, 0)
    free = np.ones((12, 12), dtype=bool)
    free[:, 6:] = False  # right half masked
    cfg = _test_config(si)
    full = si.build_model(cell_targets(grey, cfg.d_max), np.ones((12, 12), bool), cfg)
    masked = si.build_model(cell_targets(grey, cfg.d_max), free, cfg)
    n_full = len(full.model.Proto().constraints)
    n_masked = len(masked.model.Proto().constraints)
    # Interior fixed-dead cells away from the free region lose both
    # stability branches, so the masked model must be strictly smaller
    # even though it adds x == 0 pins.
    assert n_masked < n_full
    result = si.solve(masked, cfg)
    assert result.pattern[:, 6:].sum() == 0
    assert si.verify_still_life(result.pattern)["bounded"]


def test_reported_objective_matches_pattern():
    # Guards the bulk solution extraction: the pattern read back from the
    # response proto must reproduce the objective CP-SAT reported.
    si = _solver()
    grey = ramp_grey(16)
    free = np.ones((16, 16), dtype=bool)
    free[2:6, 9:14] = False
    result = si.solve_image(grey, free, _test_config(si))
    assert result.windows is not None and result.targets is not None
    recomputed = 0
    for t, (subgrid_i, subgrid_j) in zip(result.targets, result.windows):
        window_free = free[subgrid_i, subgrid_j]
        live = int(result.pattern[subgrid_i, subgrid_j][window_free].sum())
        recomputed += abs(live - int(t))
    assert recomputed == result.objective


def test_save_run_uses_result_windows(tmp_path):
    si = _solver()
    import dataclasses as dc
    import json

    from beyond_tiles.artifacts import save_run

    grey = ramp_grey(16)
    free = np.ones((16, 16), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si))
    with_fields = save_run(tmp_path / "a", result, grey, free)
    stripped = dc.replace(result, windows=None, targets=None)
    fallback = save_run(tmp_path / "b", stripped, grey, free)
    assert with_fields["deviation"] == fallback["deviation"]
    assert with_fields["objective"] == fallback["objective"]
    assert json.loads((tmp_path / "a" / "metrics.json").read_text())["deviation"] == \
        with_fields["deviation"]


# ---------------------------------------------------------------------------
# constructive seeds + rectangular LNS (Stage 3)
# ---------------------------------------------------------------------------


def test_seed_is_still_life_under_any_mask():
    from gol_mosaics.life import is_still_life

    from beyond_tiles.seeds import build_seed
    from beyond_tiles.targets import window_targets

    rng = np.random.default_rng(5)
    for trial in range(4):
        free = rng.random((48, 48)) > 0.3
        cell_t = rng.uniform(0, 0.45, (48, 48))
        windows = window_slices((48, 48), k=8, stride=8)
        targets, kept = window_targets(cell_t, free, windows)
        seed = build_seed(free, kept, targets, phase=(trial % 3, trial % 3))
        assert is_still_life(np.pad(seed, 1)), f"trial {trial}"
        assert seed[~free].sum() == 0  # masked cells stay dead


def test_seed_density_tracks_targets():
    from beyond_tiles.seeds import build_seed
    from beyond_tiles.targets import window_targets

    # left half dark (0.4), right half light (0.05)
    cell_t = np.concatenate(
        [np.full((32, 16), 0.40), np.full((32, 16), 0.05)], axis=1
    )
    free = np.ones((32, 32), dtype=bool)
    windows = window_slices((32, 32), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed = build_seed(free, kept, targets)
    dark = seed[:, :16].mean()
    light = seed[:, 16:].mean()
    assert dark > 0.25 and light < 0.15
    assert dark > light + 0.15


def test_best_seed_reports_consistent_objective():
    from beyond_tiles.seeds import best_seed, seed_objective
    from beyond_tiles.targets import window_targets

    rng = np.random.default_rng(9)
    cell_t = rng.uniform(0, 0.45, (24, 24))
    free = np.ones((24, 24), dtype=bool)
    windows = window_slices((24, 24), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed, obj = best_seed(free, kept, targets)
    assert obj == seed_objective(seed, free, kept, targets)
    for phase in [(0, 0), (1, 2)]:
        from beyond_tiles.seeds import build_seed

        other = build_seed(free, kept, targets, phase)
        assert obj <= seed_objective(other, free, kept, targets)


def test_window_devs_hand_computed():
    from beyond_tiles.lns import window_devs

    pattern = np.zeros((8, 16), dtype=np.uint8)
    pattern[:2, :2] = 1  # 4 live in the left window
    free = np.ones((8, 16), dtype=bool)
    windows = window_slices((8, 16), k=8, stride=8)
    targets = np.array([1, 3])
    assert window_devs(pattern, free, windows, targets).tolist() == [3, 3]
    assert window_devs(pattern, free, windows, targets, slack=3).tolist() == [0, 0]


def test_select_disjoint_keeps_gap():
    from beyond_tiles.lns import _select_disjoint

    scored = [
        (10, (0, 1, 0, 1), (0, 8, 0, 8)),
        (9, (0, 1, 1, 2), (0, 8, 8, 16)),  # touches the first: must be dropped
        (8, (0, 1, 3, 4), (0, 8, 24, 32)),  # 8-cell gap: fine
    ]
    chosen = _select_disjoint(scored, limit=4)
    boxes = [c[2] for c in chosen]
    assert (0, 8, 0, 8) in boxes and (0, 8, 24, 32) in boxes
    assert (0, 8, 8, 16) not in boxes


def test_agar_hint_bounds_first_incumbent():
    si = _solver()
    import dataclasses as dc

    grey = ramp_grey(24)
    free = np.ones((24, 24), dtype=bool)
    cfg = dc.replace(_test_config(si, k=8, stride=8), hint_mode="agar")
    result = si.solve_image(grey, free, cfg)
    assert result.seed_objective is not None
    # The solver must do at least as well as the hint it was given.
    assert result.objective <= result.seed_objective


def test_lns_improves_seed_and_stays_still_life():
    si = _solver()
    from beyond_tiles.lns import LnsConfig, improve, window_devs
    from beyond_tiles.seeds import build_seed
    from beyond_tiles.targets import cell_targets, window_targets

    grey = ramp_grey(24)
    free = np.ones((24, 24), dtype=bool)
    free[8:16, 8:16] = False
    cell_t = cell_targets(grey, 0.45)
    windows = window_slices((24, 24), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed = build_seed(free, kept, targets)
    before = int(window_devs(seed, free, kept, targets).sum())

    lcfg = LnsConfig(
        patch_windows=2, patch_time_s=2.0, budget_s=30.0, n_procs=1, seed=0
    )
    res = improve(seed, free, kept, targets, lcfg, log=lambda *_: None)
    objs = [o for _, o in res.obj_history]
    assert objs == sorted(objs, reverse=True)  # monotone, never worsens
    assert res.objective <= before
    checks = si.verify_still_life(res.pattern)
    assert checks["bounded"] and checks["toroidal"]
    assert res.pattern[~free].sum() == 0


def test_lns_rejects_overlapping_windows():
    from beyond_tiles.lns import LnsConfig, improve

    windows = window_slices((100, 100), k=8, stride=8)  # clamp: overlap at 92
    with pytest.raises(ValueError):
        improve(
            np.zeros((100, 100), np.uint8),
            np.ones((100, 100), bool),
            windows,
            np.zeros(len(windows), np.int64),
            LnsConfig(budget_s=1.0),
        )


# ---------------------------------------------------------------------------
# review fixes (guard rails)
# ---------------------------------------------------------------------------


def test_dither_carries_residual_across_segments():
    # Isolated kept windows (all diffusion neighbours masked): residuals
    # must be handed down the scan instead of dropped per segment.
    from beyond_tiles.targets import window_targets

    free = np.zeros((48, 48), dtype=bool)
    for r in range(0, 6, 2):
        for c in range(0, 6, 2):
            free[r * 8 : r * 8 + 8, c * 8 : c * 8 + 8] = True
    cell_t = np.full((48, 48), 2.5 / 64)
    windows = window_slices((48, 48), k=8, stride=8)
    fs, kept = window_targets(cell_t, free, windows, dither="fs")
    assert len(kept) == 9
    assert abs(fs.sum() - 9 * 2.5) <= 1.0


def test_strip_guards_reject_unsound_configs():
    si = _solver()
    from beyond_tiles.decompose import StripPlan, lower_bound_strips, solve_strips

    grey = uniform_grey(16, 100)
    free = np.ones((16, 16), dtype=bool)
    plan = StripPlan(spans=[(0, 8), (8, 16)], gap=1)
    with pytest.raises(ValueError, match="gap"):
        solve_strips(grey, free, _test_config(si, k=8, stride=8), plan)
    with pytest.raises(ValueError, match="stride"):
        solve_strips(grey, free, _test_config(si), StripPlan([(0, 8), (8, 16)], 2))
    with pytest.raises(ValueError, match="force_dead"):
        lower_bound_strips(
            grey, free, _test_config(si, k=8, stride=8, mask_mode="none"),
            StripPlan([(0, 8), (8, 16)], 2),
        )


def test_bench_rejects_protocol_overrides():
    from beyond_tiles.bench import quick_suite

    with pytest.raises(ValueError, match="protocol-fixed"):
        quick_suite({"seed": 7})
    with pytest.raises(ValueError, match="protocol-fixed"):
        quick_suite({"time_limit_s": 5.0})


def test_anneal_rejects_overlapping_windows():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import AnnealConfig, anneal

    windows = window_slices((100, 100), k=8, stride=8)  # clamp overlap at 92
    with pytest.raises(ValueError, match="disjoint"):
        anneal(
            np.zeros((100, 100), np.uint8),
            np.ones((100, 100), bool),
            windows,
            np.zeros(len(windows), np.int64),
            AnnealConfig(sweeps=1),
        )


def test_single_replica_uses_cold_endpoints():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import AnnealConfig, anneal
    from beyond_tiles.targets import window_targets

    cell_t = np.full((16, 16), 0.25)
    free = np.ones((16, 16), dtype=bool)
    windows = window_slices((16, 16), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed = np.zeros((16, 16), dtype=np.uint8)
    cfg = AnnealConfig(sweeps=200, replicas=1, seed=3, report_every=0)
    pattern, info = anneal(seed, free, kept, targets, cfg, log=lambda *_: None)
    # At the cold endpoints the block moves alone must make real progress.
    assert info["best_energy"] < 64


# ---------------------------------------------------------------------------
# annealing engine (Stage 5)
# ---------------------------------------------------------------------------


def test_energy_zero_iff_stable_and_on_target():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import energy, instability
    from beyond_tiles.targets import window_targets

    # A block exactly meets a target of 4 -> energy 0.
    pattern = np.zeros((8, 8), dtype=np.uint8)
    pattern[3:5, 3:5] = 1
    free = np.ones((8, 8), dtype=bool)
    windows = window_slices((8, 8), k=8, stride=8)
    targets = np.array([4])
    assert instability(np.pad(pattern, 1)) == 0
    assert energy(np.pad(pattern, 1), free, windows, targets, 4.0, 0) == 0.0
    # Off target by 2 -> energy 2; unstable single cell -> lam per violation.
    assert energy(np.pad(pattern, 1), free, windows, np.array([6]), 4.0, 0) == 2.0
    lonely = np.zeros((8, 8), dtype=np.uint8)
    lonely[4, 4] = 1
    assert instability(np.pad(lonely, 1)) == 1
    assert energy(np.pad(lonely, 1), free, windows, np.array([1]), 4.0, 0) == 4.0


def test_incremental_energy_matches_recompute():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import AnnealConfig, anneal, energy
    from beyond_tiles.targets import window_targets

    rng = np.random.default_rng(3)
    cell_t = rng.uniform(0, 0.45, (24, 24))
    free = np.ones((24, 24), dtype=bool)
    windows = window_slices((24, 24), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed = np.zeros((24, 24), dtype=np.uint8)
    cfg = AnnealConfig(sweeps=30, replicas=2, seed=1, report_every=0, swap_every=10)
    pattern, info = anneal(seed, free, kept, targets, cfg, log=lambda *_: None)
    # The kernel's incremental energy must agree with a full recompute of
    # the returned best grid.
    assert info["best_energy"] == pytest.approx(
        energy(np.pad(pattern, 1), free, kept, targets, cfg.lam, cfg.slack)
    )


def test_anneal_reduces_energy_and_is_deterministic():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import AnnealConfig, anneal, energy
    from beyond_tiles.targets import window_targets

    cell_t = np.full((16, 16), 0.25)
    free = np.ones((16, 16), dtype=bool)
    windows = window_slices((16, 16), k=8, stride=8)
    targets, kept = window_targets(cell_t, free, windows)
    seed = np.zeros((16, 16), dtype=np.uint8)
    e0 = energy(np.pad(seed, 1), free, kept, targets, 4.0, 0)
    cfg = AnnealConfig(sweeps=200, replicas=2, seed=7, report_every=0)
    a, info_a = anneal(seed, free, kept, targets, cfg, log=lambda *_: None)
    b, info_b = anneal(seed, free, kept, targets, cfg, log=lambda *_: None)
    assert info_a["best_energy"] < e0
    assert np.array_equal(a, b)  # same config + seed -> same result


def test_kill_repair_yields_exact_still_life():
    pytest.importorskip("numba")
    from beyond_tiles.anneal import instability, kill_repair

    rng = np.random.default_rng(2)
    for _ in range(5):
        noisy = (rng.random((20, 20)) < 0.3).astype(np.uint8)
        repaired = kill_repair(noisy)
        assert instability(np.pad(repaired, 1)) == 0
        # repair only removes cells
        assert (repaired <= noisy).all()


# ---------------------------------------------------------------------------
# strip decomposition (Stage 4)
# ---------------------------------------------------------------------------


def test_plan_strips_alignment():
    from beyond_tiles.decompose import plan_strips

    plan = plan_strips(400, k=8, strip_rows=48, gap=2)
    assert plan.spans[0] == (0, 48)
    assert plan.spans[-1][1] == 400
    assert all((r1 - r0) % 8 == 0 for r0, r1 in plan.spans)
    # spans tile the height exactly
    assert all(
        plan.spans[i][1] == plan.spans[i + 1][0]
        for i in range(len(plan.spans) - 1)
    )
    # no runt strip shorter than one window
    assert all(r1 - r0 >= 8 for r0, r1 in plan.spans)


def test_strip_solve_stitches_to_still_life():
    si = _solver()
    from beyond_tiles.decompose import StripPlan, solve_strips

    grey = uniform_grey(32, 100)
    free = np.ones((32, 32), dtype=bool)
    cfg = _test_config(si, k=8, stride=8, time_limit_s=20.0)
    plan = StripPlan(spans=[(0, 16), (16, 32)], gap=2)
    out = solve_strips(grey, free, cfg, plan, n_procs=1)
    pattern = out["pattern"]
    assert pattern.shape == (32, 32)
    checks = si.verify_still_life(pattern)
    assert checks["bounded"] and checks["toroidal"]
    assert pattern[14:16].sum() == 0  # the dead separator rows
    assert pattern[:14].sum() > 0 and pattern[16:].sum() > 0


def test_strip_solve_parallel_matches_serial():
    si = _solver()
    from beyond_tiles.decompose import StripPlan, solve_strips

    grey = ramp_grey(24)
    free = np.ones((24, 24), dtype=bool)
    cfg = _test_config(si, k=8, stride=8, time_limit_s=15.0)
    plan = StripPlan(spans=[(0, 8), (8, 16), (16, 24)], gap=2)
    serial = solve_strips(grey, free, cfg, plan, n_procs=1)
    parallel = solve_strips(grey, free, cfg, plan, n_procs=2)
    assert np.array_equal(serial["pattern"], parallel["pattern"])
    assert serial["objective"] == parallel["objective"]


def test_lower_bound_is_valid_on_solved_instance():
    si = _solver()
    from beyond_tiles.decompose import StripPlan, lower_bound_strips

    grey = ramp_grey(16)
    free = np.ones((16, 16), dtype=bool)
    cfg = _test_config(si, k=8, stride=8, time_limit_s=20.0)
    exact = si.solve_image(grey, free, cfg)
    assert exact.status == "OPTIMAL"
    bound = lower_bound_strips(
        grey, free, cfg, StripPlan(spans=[(0, 8), (8, 16)], gap=0), n_procs=1
    )
    assert bound["all_optimal"]
    assert bound["lower_bound"] <= exact.objective


def test_relaxed_model_has_fewer_constraints():
    si = _solver()
    grey = uniform_grey(16, 60)
    free = np.ones((16, 16), dtype=bool)
    cfg = _test_config(si, k=8, stride=8)
    from beyond_tiles.targets import cell_targets

    cell_t = cell_targets(grey, cfg.d_max)
    full = si.build_model(cell_t, free, cfg)
    relaxed = si.build_model(cell_t, free, cfg, relax_top=True, relax_bottom=True)
    assert len(relaxed.model.Proto().constraints) < len(
        full.model.Proto().constraints
    )


# ---------------------------------------------------------------------------
# objective slack, dithered targets, solver parameters (Stage 2)
# ---------------------------------------------------------------------------


def test_slack_absorbs_small_deviation():
    # One 8x8 window with target 2: the nearest still lifes have 0 or 4
    # live cells (the minimum non-empty still life is 4 cells), so the
    # optimum is 2 without slack and 0 with slack 2.
    si = _solver()
    grey = uniform_grey(8, 237)  # cell target 0.0318 -> window target 2
    free = np.ones((8, 8), dtype=bool)
    base = _test_config(si, k=8, stride=8)
    tight = si.solve_image(grey, free, base)
    assert tight.targets.tolist() == [2]
    assert tight.objective == 2
    import dataclasses as dc

    slacked = si.solve_image(grey, free, dc.replace(base, slack=2))
    assert slacked.objective == 0
    assert slacked.status == "OPTIMAL"


def test_dither_preserves_total_mass():
    from beyond_tiles.targets import window_targets

    rng = np.random.default_rng(11)
    cell_t = rng.uniform(0, 0.45, (40, 40))
    free = np.ones((40, 40), dtype=bool)
    windows = window_slices((40, 40), k=8, stride=8)
    exact_total = cell_t.sum()
    fs, _ = window_targets(cell_t, free, windows, dither="fs")
    assert abs(fs.sum() - exact_total) <= 1.0
    for t, (si, sj) in zip(fs, windows):
        assert 0 <= t <= 64


def test_dither_beats_rounding_on_biased_field():
    # Every window sums to x.4: plain rounding drops 0.4 cells per window,
    # error diffusion keeps the total within half a cell.
    from beyond_tiles.targets import window_targets

    cell_t = np.full((32, 32), 2.4 / 64)
    free = np.ones((32, 32), dtype=bool)
    windows = window_slices((32, 32), k=8, stride=8)
    exact_total = 2.4 * 16
    rounded, _ = window_targets(cell_t, free, windows, dither="round")
    fs, _ = window_targets(cell_t, free, windows, dither="fs")
    assert abs(rounded.sum() - exact_total) > 5  # systematic deficit
    assert abs(fs.sum() - exact_total) <= 1.0


def test_dither_skips_dropped_windows():
    from beyond_tiles.targets import window_targets

    cell_t = np.full((16, 16), 0.3)
    free = np.ones((16, 16), dtype=bool)
    free[:8, :8] = False  # first window fully masked -> dropped
    windows = window_slices((16, 16), k=8, stride=8)
    fs, kept = window_targets(cell_t, free, windows, dither="fs")
    assert len(kept) == 3 and len(fs) == 3
    assert abs(fs.sum() - 0.3 * 64 * 3) <= 1.0


def test_apply_solver_params():
    si = _solver()
    from ortools.sat.python import cp_model

    cfg = si.SpikeConfig(
        time_limit_s=12.0,
        workers=3,
        seed=7,
        add_lb_subsolvers=True,
        num_violation_ls=2,
        symmetry_level=0,
        max_det_time=5.0,
    )
    solver = cp_model.CpSolver()
    si._apply_solver_params(solver, cfg)
    p = solver.parameters
    assert p.max_time_in_seconds == 12.0
    assert p.num_workers == 3 and p.random_seed == 7
    assert list(p.extra_subsolvers) == ["lb_tree_search", "objective_lb_search"]
    assert p.num_violation_ls == 2
    assert p.symmetry_level == 0
    assert p.max_deterministic_time == 5.0


# ---------------------------------------------------------------------------
# benchmark harness (Stage 0 of the optimization campaign)
# ---------------------------------------------------------------------------


def test_parse_overrides_types():
    from beyond_tiles.bench import parse_overrides

    out = parse_overrides(["workers=4", "d_max=0.4", "mask_mode=none", "flag=true"])
    assert out == {"workers": 4, "d_max": 0.4, "mask_mode": "none", "flag": True}
    with pytest.raises(ValueError):
        parse_overrides(["notapair"])


def test_derive_timings_pure():
    from beyond_tiles.bench import derive_timings

    history = [(1.5, 30), (2.0, 10), (7.5, 2)]
    t = derive_timings(history, "OPTIMAL", 8.0)
    assert t == {"time_to_first_s": 1.5, "time_to_optimal_s": 8.0}
    t = derive_timings(history, "FEASIBLE", 8.0)
    assert t == {"time_to_first_s": 1.5, "time_to_optimal_s": None}
    assert derive_timings([], "FEASIBLE", 8.0)["time_to_first_s"] is None


def test_build_time_recorded():
    si = _solver()
    grey = uniform_grey(12, 128)
    free = np.ones((12, 12), dtype=bool)
    result = si.solve_image(grey, free, _test_config(si))
    assert result.build_time_s > 0


def test_bench_case_roundtrip(tmp_path):
    _solver()
    import json

    from beyond_tiles.bench import BenchCase, compare, run_case

    grey = uniform_grey(16, 200)
    free = np.ones((16, 16), dtype=bool)
    case = BenchCase("tiny", 16, 5.0, 0, {"workers": 1})
    metrics = run_case(case, tmp_path / "tag", grey=grey, free=free)
    saved = json.loads((tmp_path / "tag" / "tiny" / "metrics.json").read_text())
    for key in ("build_time_s", "time_to_first_s", "time_to_optimal_s", "bench"):
        assert key in saved, key
    assert saved["bench"]["case"]["name"] == "tiny"
    assert saved["objective"] == metrics["objective"]
    table = compare([tmp_path / "tag"])
    assert "tag/tiny" in table and "| run |" in table


def test_snapshots_recorded_and_end_on_the_final_pattern():
    solver = _solver()
    grey = ramp_grey(24)
    free = np.ones((24, 24), dtype=bool)
    cfg = solver.SpikeConfig(
        k=8, stride=8, time_limit_s=10.0, workers=1, seed=0, snapshot_gap_s=0.01
    )
    result = solver.solve_image(grey, free, cfg)
    assert result.snapshots
    times = [t for t, _, _ in result.snapshots]
    assert times == sorted(times)
    assert (result.snapshots[-1][2] == result.pattern).all()
    assert all(p.shape == result.pattern.shape for _, _, p in result.snapshots)
