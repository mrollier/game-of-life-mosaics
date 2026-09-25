"""Tests for the beyond-tiles research harness in experiments/beyond_tiles:
the benchmark protocol, run artefacts, convergence movies and the
(archived) parallel-tempering annealer.
"""

from pathlib import Path

import numpy as np
import pytest

from gol_mosaics.freeform.metrics import deviation_stats, motif_stats, tile_db_overlap
from gol_mosaics.freeform.targets import (
    cell_targets,
    ramp_grey,
    uniform_grey,
    window_slices,
    window_targets,
)
from tests.conftest import REPO_ROOT
from tests.freeform.helpers import _solver, _test_config

# Committed extracts of the headline solves (bit-packed patterns)
ASSETS = REPO_ROOT / "experiments" / "beyond_tiles" / "assets"


def test_end_to_end_artifacts(tmp_path):
    si = _solver()
    from PIL import Image

    from beyond_tiles.artifacts import save_run
    from gol_mosaics.freeform.targets import grey_and_mask_from_image

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
    from gol_mosaics.freeform.targets import window_targets

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
    from gol_mosaics.freeform.targets import window_targets

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
    from gol_mosaics.freeform.targets import window_targets

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
    from gol_mosaics.freeform.targets import window_targets

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
