"""Golden digests that pin behaviour across refactors.

Each test hashes the output of one deterministic pipeline stage and compares
it with a digest recorded from a known-good version. A refactor that changes
performance or structure but not behaviour must keep every digest; a change
that is *meant* to alter output re-pins the digest here and says why in the
commit message and CHANGELOG.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from tests.conftest import REPO_ROOT


def sha(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(repr((part.shape, str(part.dtype))).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, bytes):
            digest.update(part)
        else:
            digest.update(repr(part).encode())
    return digest.hexdigest()[:16]


# ------------------------------------------------------------------ CNFs

CNF = {
    3: "d4a1e80102921a3f",
    4: "83120d119db93fbd",
    5: "afe3924af9054d3f",
}
SCHEME_CNF = {
    3: "01cddb71da6c4a6a",
    4: "1dd506be9f473930",
    5: "5aa76762614b2c86",
}
NOSYM_CNF_3 = "56a0fef32c88ed06"


def cnf_digest(level):
    from gol_mosaics.sat_search import build_cnf
    return build_cnf(level).sha256[:16]


def scheme_cnf_digest(level):
    from gol_mosaics.tile_scheme import build_scheme_cnf, square_scheme
    return build_scheme_cnf(square_scheme(level)).sha256[:16]


def nosym_cnf_digest():
    from gol_mosaics.nosym_tiles import build_nosym_cnf
    return build_nosym_cnf(3).sha256[:16]


@pytest.mark.parametrize("level", sorted(CNF))
def test_cnf_fingerprint(level):
    assert cnf_digest(level) == CNF[level]


@pytest.mark.parametrize("level", sorted(SCHEME_CNF))
def test_scheme_cnf_fingerprint(level):
    assert scheme_cnf_digest(level) == SCHEME_CNF[level]


def test_nosym_cnf_fingerprint():
    assert nosym_cnf_digest() == NOSYM_CNF_3


# ------------------------------------------------------------ databases

DATABASES = {
    ("diamond", 1): "fccbf1942367830a",
    ("diamond", 2): "fc3f341c4fa06b03",
    ("diamond", 3): "c88b3a73c1464a0d",
    ("diamond", 4): "2c6b72b1a854d0f1",
    ("diamond", 5): "2e7df2b50508e752",
    ("diamond", 6): "ce1392bbc369204c",
    ("square", 3): "c1bc7367dfd9143a",
    ("square", 4): "f30fca76a628b704",
    ("square", 5): "0b1fa383a042ef25",
}
NOSYM_DATABASE_3 = "de3e3835a0b970d9"


def database_digest(layout, level):
    from gol_mosaics import TileLibrary
    tiles = TileLibrary.load(level, layout=layout).tiles
    return sha(np.asarray(tiles, dtype=np.uint8))


def nosym_database_digest():
    from gol_mosaics.nosym_tiles import load_nosym_tiles
    return sha(np.asarray(load_nosym_tiles(3), dtype=np.uint8))


@pytest.mark.parametrize("layout,level", sorted(DATABASES))
def test_database_digest(layout, level):
    """Row order matters: it drives tie-breaking and seeded tile draws."""
    assert database_digest(layout, level) == DATABASES[(layout, level)]


def test_nosym_database_digest():
    assert nosym_database_digest() == NOSYM_DATABASE_3


# ------------------------------------------------------------------ ECA

ECA_ALL_RULES = "ca05467fdfe5662f"


def eca_digest():
    from gol_mosaics import ECABackground
    parts = []
    for rule in range(256):
        np.random.seed(rule)
        parts.append(ECABackground(rule).generate(width=37, height=23,
                                                  supersample=3))
    return sha(*parts)


def test_eca_all_rules():
    assert eca_digest() == ECA_ALL_RULES


# -------------------------------------------------------------- mosaics

def synthetic_portrait(size=(90, 70)) -> Image.Image:
    """A soft dark blob with a transparent surround: exercises tones, the
    alpha mask and the aspect crop without rembg or image files."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot((y - h / 2) / (h / 2), (x - w / 2) / (w / 2))
    grey = np.clip(255 * r ** 1.5, 0, 255).astype(np.uint8)
    alpha = np.where(r < 0.85, 255, 0).astype(np.uint8)
    rgba = np.dstack([grey, grey, grey, alpha])
    return Image.fromarray(rgba, mode="RGBA")


MOSAICS = {
    ("diamond", False): "d71d9c982c92dd92",
    ("diamond", True): "e05d9edf80698e0b",  # 2.x: f6f734f0a4eba4de
    ("square", False): "f32de4ae48d20329",
    ("square", True): "dbefb6e5c7e4c156",  # 2.x: e6b07429eb0975fb
}


def mosaic_digest(layout, random_tiles):
    from gol_mosaics import MosaicGenerator
    generator = MosaicGenerator(level=3, grid_size=10, eca_rule=30,
                                random_tiles=random_tiles,
                                layout=layout)
    image, cells, mask = generator.generate_from_pil(
        synthetic_portrait(), remove_background=False, seed=11,
        return_arrays=True)
    return sha(np.asarray(cells, dtype=np.uint8),
               np.asarray(mask, dtype=np.uint8), np.asarray(image))


@pytest.mark.parametrize("layout,random_tiles", sorted(MOSAICS))
def test_mosaic_digest(layout, random_tiles):
    assert mosaic_digest(layout, random_tiles) == MOSAICS[(layout,
                                                           random_tiles)]


# ------------------------------------------------------------- renderer

RENDER = "414268ec3c282c43"


def render_digest():
    from gol_mosaics import ColourScheme, MosaicRenderer
    rng = np.random.default_rng(5)
    cells = rng.integers(0, 2, (19, 23))
    backdrop = rng.integers(0, 7, (19, 23))  # 5 and 6 sit above layers=3
    renderer = MosaicRenderer(ColourScheme(fill_pixel="#123456"))
    return sha(np.asarray(renderer.render(cells, backdrop,
                                                      layers=3)),
               np.asarray(renderer.render(cells, backdrop)))


def test_render_digest():
    assert render_digest() == RENDER


# --------------------------------------------------------------- export

EXPORT = "2c8d8080f07683f2"


def export_digest(tmp_path):
    from gol_mosaics import GollyExporter
    rng = np.random.default_rng(9)
    cells = rng.integers(0, 2, (13, 17))
    cells[:, :3] = 0  # a long leading run
    cells[4] = 1      # a full live row
    out = []
    GollyExporter.export_to_cells(cells, str(tmp_path / "a.cells"))
    GollyExporter.export_to_cells(cells, str(tmp_path / "b.cells"),
                                  add_glider="top left")
    GollyExporter.export_to_rle(cells, str(tmp_path / "c.rle"),
                                name="golden", comments="one\ntwo")
    big = rng.integers(0, 2, (3, 400))  # RLE lines wrap at 70 characters
    GollyExporter.export_to_rle(big, str(tmp_path / "d.rle"))
    for name in ("a.cells", "b.cells", "c.rle", "d.rle"):
        # Line endings are normalised: the digest pins content, and the
        # writer's newline convention is covered by test_export.py.
        out.append((tmp_path / name).read_bytes().replace(b"\r\n", b"\n"))
    return sha(*out)


def test_export_digest(tmp_path):
    assert export_digest(tmp_path) == EXPORT


# ------------------------------------------------------------- freeform

FREEFORM = "8c941dd73c777c94"


def freeform_digest():
    pytest.importorskip("ortools")
    from gol_mosaics.freeform.io import load_packed
    from gol_mosaics.freeform.lns import LnsConfig, window_devs
    from gol_mosaics.freeform.metrics import deviation_stats
    from gol_mosaics.freeform.seeds import seed_objective
    from gol_mosaics.freeform.targets import (cell_targets, grey_and_mask_from_image,
                                      window_slices, window_targets)

    pattern = load_packed(
        REPO_ROOT / "experiments/beyond_tiles/assets/marilyn_400_pipeline.npz")
    grey, free = grey_and_mask_from_image(
        Image.open(REPO_ROOT / "input/images/marilyn.png"), size=400)
    cell_t = cell_targets(grey, 0.4)
    windows = window_slices(grey.shape, k=8, stride=8, edge="partial")
    targets, kept = window_targets(cell_t, free, windows, dither="fs")
    devs = window_devs(pattern, free, kept, targets)
    seams = window_devs(pattern, free, kept, targets,
                        lcfg=LnsConfig(seam_rows=(46, 94), seam_cols=(142,)))
    obj = seed_objective(pattern, free, kept, targets)
    stats = deviation_stats(pattern, cell_t, free, kept)
    return sha(targets, devs, seams, obj,
               {k: round(v, 9) for k, v in stats.items()})


def test_freeform_digest():
    assert freeform_digest() == FREEFORM


# ------------------------------------------------------------- recording

if __name__ == "__main__":
    # python -m tests.test_goldens  prints the current digests, for
    # re-pinning after an intended behaviour change.
    import tempfile
    from pathlib import Path

    print("CNF =", {lv: cnf_digest(lv) for lv in CNF})
    print("SCHEME_CNF =", {lv: scheme_cnf_digest(lv) for lv in SCHEME_CNF})
    print("NOSYM_CNF_3 =", repr(nosym_cnf_digest()))
    print("DATABASES =", {key: database_digest(*key) for key in DATABASES})
    print("NOSYM_DATABASE_3 =", repr(nosym_database_digest()))
    print("ECA_ALL_RULES =", repr(eca_digest()))
    print("MOSAICS =", {key: mosaic_digest(*key) for key in MOSAICS})
    print("RENDER =", repr(render_digest()))
    with tempfile.TemporaryDirectory() as tmp:
        print("EXPORT =", repr(export_digest(Path(tmp))))
    print("FREEFORM =", repr(freeform_digest()))
