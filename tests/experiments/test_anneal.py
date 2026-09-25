"""Tests for the archived parallel-tempering annealer (needs numba)."""

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


def test_anneal_rejects_overlapping_windows():
    pytest.importorskip("numba")
    from archive.anneal import AnnealConfig, anneal

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
    from archive.anneal import AnnealConfig, anneal
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
    from archive.anneal import energy, instability
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
    from archive.anneal import AnnealConfig, anneal, energy
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
    from archive.anneal import AnnealConfig, anneal, energy
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
    from archive.anneal import instability, kill_repair

    rng = np.random.default_rng(2)
    for _ in range(5):
        noisy = (rng.random((20, 20)) < 0.3).astype(np.uint8)
        repaired = kill_repair(noisy)
        assert instability(np.pad(repaired, 1)) == 0
        # repair only removes cells
        assert (repaired <= noisy).all()
