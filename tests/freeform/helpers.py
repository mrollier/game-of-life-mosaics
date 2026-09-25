"""Shared helpers for the free-form tests."""

import pytest


def _solver():
    """Import the CP-SAT layer, skipping when ortools is absent."""
    pytest.importorskip("ortools")
    from gol_mosaics.freeform import solver as still_image

    return still_image


def _test_config(still_image, **overrides):
    """Deterministic small-scale solver settings for tests."""
    defaults = dict(k=4, stride=2, d_max=0.45, time_limit_s=10.0, workers=1, seed=0)
    defaults.update(overrides)
    return still_image.SolveConfig(**defaults)
