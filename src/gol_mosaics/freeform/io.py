"""Compact storage for solved patterns and solver snapshots.

A free-form still life costs minutes to hours of CP-SAT, so solves are
kept: bit-packed and compressed, a 1000 x 1000 pattern is about 20 kB, small
enough to version next to the code that made it.
"""

from pathlib import Path

import numpy as np


def save_pattern_asset(path, pattern: np.ndarray) -> Path:
    """Store a binary pattern bit-packed, for versioning alongside the code."""
    pattern = np.asarray(pattern)
    np.savez_compressed(
        path,
        packed=np.packbits(pattern.astype(bool), axis=None),
        shape=np.asarray(pattern.shape, dtype=np.int64),
    )
    return Path(path)


def load_pattern_asset(path) -> np.ndarray:
    """Inverse of `save_pattern_asset`: uint8 array of the original shape."""
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        n = int(np.prod(shape))
        return np.unpackbits(data["packed"])[:n].reshape(shape).astype(np.uint8)


def save_snapshots(path, snapshots) -> Path:
    """Store a run's incumbent patterns (time, objective, pattern) bit-packed."""
    times = np.array([t for t, _, _ in snapshots], dtype=np.float64)
    objectives = np.array([o for _, o, _ in snapshots], dtype=np.int64)
    frames = np.stack([np.asarray(p) for _, _, p in snapshots])
    np.savez_compressed(
        path,
        packed=np.packbits(frames.astype(bool), axis=-1),
        shape=np.asarray(frames.shape[1:], dtype=np.int64),
        times=times,
        objectives=objectives,
    )
    return Path(path)


def load_snapshots(path):
    """Inverse of `save_snapshots`: list of (time, objective, pattern)."""
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        frames = np.unpackbits(data["packed"], axis=-1)[..., : shape[1]]
        return [
            (float(t), int(o), frame.astype(np.uint8))
            for t, o, frame in zip(data["times"], data["objectives"], frames)
        ]
