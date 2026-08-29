"""LinkedIn banner art: two free-form still lifes, ten variations each.

The still lifes come from `poster.py` at 800x200 (see REPORT.md); this
script only does the post-hoc colouring, so it needs no solver and runs in
seconds:

    python experiments/beyond_tiles/linkedin_banners.py

Two geometry notes, both learned here:

* **800x200, not 792x198.** LinkedIn's banner is 1584x396, exactly 4:1, but
  the pipeline needs both axes to be multiples of the window size 8 and 198
  is not. 800x200 is the nearest 4:1 grid that qualifies; rendered at
  `scale=2` it gives 1600x400, four rows of sky more than the banner needs.
* **The two masks want different tile levels.** Banner 1 is a wide range
  with only a thin strip of sky (18.6% of the canvas), and `mosaic_background`
  keeps a site only when the tile's whole footprint is clear, so level 6 seats
  almost nothing there. Banner 2 is a single peak under 51.8% sky and takes
  every level. The two variation lists reflect that rather than pretending
  one set of parameters suits both.
* **A plain mosaic leaves a ragged halo.** A tile is seated only where its
  whole footprint fits, so the distance from the ridgeline to the nearest tile
  is whatever the lattice allows — tens of cells at the larger levels.
  `filled_background` packs that gap with smaller tiles and then loose still
  lifes and brings the worst case to single digits. The FILLED set below is
  that comparison, and `rule_figure` below is the one for the fit test itself.
"""

import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image

from beyond_tiles.artifacts import (load_pattern_asset,
                                    save_pattern_asset)
from beyond_tiles.poster import load_rect
from beyond_tiles.still_image import verify_still_life
from gol_mosaics import (ColorScheme, MosaicRenderer, compose,
                         filled_background)

WIDTH, HEIGHT, SCALE = 800, 200, 2
BANNER = (1584, 396)  # LinkedIn's own size; the render is 1600x400

ICE = ColorScheme.monochrome(foreground="#1E64C8", background="#FFFFFF")
SLATE = ColorScheme.monochrome(foreground="#2B2B2B", background="#F2EFE9")

# The two palettes picked from the avatar-matched set, kept fixed so the
# BACKGROUNDS list below varies nothing but the backdrop.
TEAL_RUST = ColorScheme(gol_background="#F4EDE2", gol_pixel="#2B3538",
                        eca_background="#1B5E5E", eca_pixel="#E8D9C0")
CREAM_AUBURN = ColorScheme(gol_background="#F2E9DA", gol_pixel="#8C4A2F",
                           eca_background="#F2E9DA", eca_pixel="#8C4A2F")

# (name, compose kwargs). Seeds are fixed so every render is reproducible.
BANNER_1 = [
    ("mosaic-l2-open", dict(style="mosaic", level=2, density=(0.0, 1.0),
                            seed=1, scheme=ColorScheme.ugent())),
    ("mosaic-l2-dense", dict(style="mosaic", level=2, density=(0.65, 1.0),
                             seed=2, scheme=ColorScheme.warhol(seed=3))),
    ("mosaic-l2-ramp", dict(style="mosaic", level=2, tone="linear",
                            tone_angle=0.0, seed=3, scheme=ICE)),
    ("mosaic-l3-sparse", dict(style="mosaic", level=3, density=(0.0, 0.40),
                              seed=4, scheme=SLATE)),
    ("mosaic-l3-square", dict(style="mosaic", level=3, shape="square",
                              density=(0.5, 1.0), seed=5,
                              scheme=ColorScheme.warhol(seed=11))),
    ("mosaic-l2-vignette", dict(style="mosaic", level=2, tone="vignette",
                                seed=6, scheme=ColorScheme.inverted())),
    ("mosaic-l3-rise", dict(style="mosaic", level=3, tone="linear",
                            tone_angle=90.0, seed=7,
                            scheme=ColorScheme.warhol(seed=21))),
    ("eca", dict(style="eca", rule=110, scheme=ColorScheme.ugent())),
    ("agar", dict(style="agar", scheme=ICE)),
    ("flat", dict(style="flat", scheme=ColorScheme.warhol(seed=5))),
]

BANNER_2 = [
    ("mosaic-l2-open", dict(style="mosaic", level=2, density=(0.0, 1.0),
                            seed=1, scheme=ColorScheme.ugent())),
    ("mosaic-l3-vignette", dict(style="mosaic", level=3, density=(0.6, 1.0),
                                tone="vignette", seed=2,
                                scheme=ColorScheme.warhol(seed=3))),
    ("mosaic-l4-rise", dict(style="mosaic", level=4, tone="linear",
                            tone_angle=90.0, seed=3, scheme=ICE)),
    ("mosaic-l5-sparse", dict(style="mosaic", level=5, density=(0.0, 0.40),
                              seed=4, scheme=SLATE)),
    ("mosaic-l6-radial", dict(style="mosaic", level=6, tone="radial", seed=5,
                              scheme=ColorScheme.warhol(seed=11))),
    ("mosaic-l4-square", dict(style="mosaic", level=4, shape="square",
                              seed=6, scheme=ColorScheme.ugent())),
    ("mosaic-l5-square", dict(style="mosaic", level=5, shape="square",
                              density=(0.5, 1.0), seed=7,
                              scheme=ColorScheme.inverted())),
    ("eca", dict(style="eca", rule=90, scheme=ColorScheme.ugent())),
    ("agar", dict(style="agar", scheme=ICE)),
    ("flat", dict(style="flat", scheme=ColorScheme.warhol(seed=5))),
]

# --- Avatar-matched set -----------------------------------------------------
#
# Ten recolourings of banner 2's `mosaic-l2-open`, chosen against the owner's
# profile photo: auburn hair #8C4A2F, copper beard #A5583A, warm skin #C98D6B,
# slate shirt #3F4750, weathered-wood taupe #9A8E82, dark teal window glass
# #223433. Two properties of that photo drive every choice below.
#
# * It is **dark overall**, so a light banner separates the avatar disc and a
#   dark one deliberately merges it. Both are used, on purpose.
# * It is **low-saturation and filmic**, so the palettes stay muted. The UGent
#   yellow of the original reads as neon beside it; `ochre-ink` is that same
#   idea pulled down about a third in saturation.
#
# The four scheme slots map to: sky field, sky tiles, mountain field, mountain
# cells. The sky is the largest area, so it carries the statement colour.
AVATAR = [
    # Direct complement. Auburn sits near hue 18 degrees; teal near 198 is
    # opposite it. The photo's window glass already carries a dark teal, so
    # this extends a colour that is literally in the picture.
    ("teal-rust", dict(style="mosaic", level=2, seed=1, scheme=TEAL_RUST)),
    # Analogous warm: everything within about 25 degrees of the hair. Nothing
    # separates by hue, so the dark avatar separates by value instead. The
    # calm option.
    ("sand-copper", dict(style="mosaic", level=3, seed=2, scheme=ColorScheme(
        gol_background="#FAF5EC", gol_pixel="#4A3628",
        eca_background="#E3CBA8", eca_pixel="#A5583A"))),
    # The shirt lifted to a field colour — the largest non-skin area of the
    # photo — with the hair's amber as the accent that ties them together.
    ("slate-amber", dict(style="mosaic", level=2, density=(0.55, 1.0), seed=3,
                         scheme=ColorScheme(
                             gol_background="#EDEAE4", gol_pixel="#3F4750",
                             eca_background="#4A5560", eca_pixel="#D8A24A"))),
    # Monochromatic: one hue family at three values. The hair belongs to the
    # same family, so avatar and banner read as a single image.
    ("terracotta", dict(style="mosaic", level=4, seed=4, scheme=ColorScheme(
        gol_background="#F6EBE3", gol_pixel="#6B3324",
        eca_background="#B5654A", eca_pixel="#E0AE96"))),
    # Inverted value: pale mountain on a dark sky, so the dark avatar sinks
    # into the banner instead of sitting on it. Closest to the photo's own
    # filmic grade.
    ("forest-dusk", dict(style="mosaic", level=2, tone="linear",
                         tone_angle=90.0, seed=5, scheme=ColorScheme(
                             gol_background="#141A18", gol_pixel="#CBD8C6",
                             eca_background="#1E3A34", eca_pixel="#6E8C6A"))),
    # High-key and near-monochrome. Maximum value contrast against a dark
    # avatar, and the most legible of the ten at thumbnail size.
    ("cream-auburn", dict(style="mosaic", level=3, density=(0.0, 0.45), seed=6,
                          scheme=CREAM_AUBURN)),
    # The original UGent pairing you picked, desaturated to match the photo's
    # muted grade. Same geometry and seed as `banner2-mosaic-l2-open`, so it
    # is a pure recolour and the fair comparison.
    ("ochre-ink", dict(style="mosaic", level=2, seed=1, scheme=ColorScheme(
        gol_background="#F7F2E6", gol_pixel="#2E4756",
        eca_background="#D9B25C", eca_pixel="#2E4756"))),
    # One colour only, matched to the beard, on large level-5 tiles. Structure
    # carries the image rather than palette.
    ("copper-mono", dict(style="mosaic", level=5, seed=7,
                         scheme=ColorScheme.monochrome("#9C5233", "#F5EFE6"))),
    # Split complement: instead of the teal directly opposite the hair, the
    # two hues flanking it. Less obvious than teal-and-orange.
    ("indigo-rust", dict(style="mosaic", level=3, shape="square", seed=8,
                         scheme=ColorScheme(
                             gol_background="#EFEDE8", gol_pixel="#3A4661",
                             eca_background="#2F3E5C", eca_pixel="#B06A3E"))),
    # Neutral field, single warm accent: the wood slats' taupe as the sky, the
    # hair's ember as the only saturated note. Leaves the avatar as the most
    # colourful thing on the page.
    ("taupe-ember", dict(style="mosaic", level=2, tone="vignette", seed=9,
                         scheme=ColorScheme(
                             gol_background="#F0EDE7", gol_pixel="#4B463F",
                             eca_background="#A9A096", eca_pixel="#C4633A"))),
]

# --- Backdrop study ---------------------------------------------------------
#
# The two chosen palettes held fixed while the backdrop varies, so any
# difference between these ten is structural rather than chromatic. Five per
# scheme, spanning all three generated backdrops:
#
# * **mosaic** — `level` sets tile size (6*level cells), `density` picks a band
#   of the tile bank, `tone` grades that band across the canvas, `shape`
#   switches lattice.
# * **agar** — `pitch` is the 2x2 block lattice period. (3, 4) is the tightest
#   still-life-safe value; anything larger is sparser and equally safe.
# * **eca** — `rule` picks the automaton, `supersample` its cell size in
#   pixels. The default 15 gives roughly 13 chunky rows on a 200-cell canvas;
#   dropping it to 5 or 6 turns the same rule into a fine texture.
BACKGROUNDS = [
    # Tile size sweep: four times the area of the L2 you picked, so each
    # diamond reads as an ornament instead of a texture.
    ("teal-l4", dict(style="mosaic", level=4, seed=11, scheme=TEAL_RUST)),
    # Largest tiles in the bank, restricted to the emptiest third of it — a
    # wide open lattice with very few cells in it.
    ("teal-l6-sparse", dict(style="mosaic", level=6, density=(0.0, 0.35),
                            seed=12, scheme=TEAL_RUST)),
    # Square lattice instead of diamond, graded so the sky packs towards the
    # top and thins as it meets the ridgeline.
    ("teal-l5-square-ramp", dict(style="mosaic", level=5, shape="square",
                                 tone="linear", tone_angle=90.0, seed=13,
                                 scheme=TEAL_RUST)),
    # No tiles at all: the tightest safe block agar, a uniform fine weave.
    ("teal-agar", dict(style="agar", pitch=(3, 4), scheme=TEAL_RUST)),
    # Rule 30 is the chaotic one; supersample 6 keeps it fine-grained rather
    # than the blocky default.
    ("teal-eca30", dict(style="eca", rule=30, supersample=6,
                        scheme=TEAL_RUST)),
    # Densest third of the level-2 bank: small tiles packed tight, closest to
    # a woven cloth.
    ("cream-l2-dense", dict(style="mosaic", level=2, density=(0.65, 1.0),
                            seed=14, scheme=CREAM_AUBURN)),
    # Radial grade peaks at the canvas centre, which on this image is directly
    # behind the summit.
    ("cream-l4-radial", dict(style="mosaic", level=4, tone="radial", seed=15,
                             scheme=CREAM_AUBURN)),
    # Vignette is the inverse: dense at the edges, opening out over the peak.
    ("cream-l3-square-vignette", dict(style="mosaic", level=3, shape="square",
                                      tone="vignette", seed=16,
                                      scheme=CREAM_AUBURN)),
    # Agar at double pitch — the same construction as `teal-agar`, four times
    # sparser, so it reads as scattered dots rather than a weave.
    ("cream-agar-sparse", dict(style="agar", pitch=(6, 8),
                               scheme=CREAM_AUBURN)),
    # Rule 110 is the complex one: structured triangles rather than noise.
    ("cream-eca110", dict(style="eca", rule=110, supersample=5,
                          scheme=CREAM_AUBURN)),
]

# --- Filling the halo -------------------------------------------------------
#
# Same two palettes again, now with a third colour for the filler cells. The
# haze tones sit between each scheme's field and its tiles, so the fringe reads
# as the mosaic thinning into the sky rather than as a separate material.
#
# The knobs are `fill` (which smaller levels to cascade through before
# scattering), `fill_band` (how far from the ridgeline the loose still lifes
# reach — None, the default, is everywhere) and `fill_fade` (whether they thin
# out towards its far edge).
TEAL_HAZE = replace(TEAL_RUST, fill_pixel="#7FA8A0")
CREAM_HAZE = replace(CREAM_AUBURN, fill_pixel="#C98F6B")

# Banner 1 is where filling changes the most: its sky is a thin band, so a
# plain mosaic is one ragged row of diamonds over a wide empty field.
FILLED_1 = [
    ("l3", dict(style="mosaic", level=3, fill="auto", seed=21,
                scheme=TEAL_HAZE)),
    ("l2-tight", dict(style="mosaic", level=2, fill="auto", fill_band=10,
                      seed=22, scheme=CREAM_HAZE)),
    # A band instead of the default: the filler hugs the ridgeline and the
    # sky above it stays open.
    ("l3-band", dict(style="mosaic", level=3, fill="auto", fill_band=18,
                     seed=23, scheme=TEAL_HAZE)),
    # fill=() skips the tile cascade — loose still lifes only, so the fringe is
    # all grain with no intermediate sizes.
    ("l3-scatter", dict(style="mosaic", level=3, fill=(), seed=24,
                        scheme=CREAM_HAZE)),
    # Fading off: the band ends on a line instead of dissolving. The honest
    # comparison for whether the fade earns its keep.
    ("l2-hard", dict(style="mosaic", level=2, fill="auto", fill_band=14,
                     fill_fade=False, seed=25, scheme=TEAL_HAZE)),
]

FILLED_2 = [
    # The largest tiles in the bank, whose halo is the worst of all, with every
    # smaller level cascaded in behind them.
    ("l6", dict(style="mosaic", level=6, fill="auto", seed=31,
                scheme=TEAL_HAZE)),
    ("l4", dict(style="mosaic", level=4, fill="auto", seed=32,
                scheme=CREAM_HAZE)),
    # Your favourite geometry, filled. L2 already had the tightest halo of the
    # three, so this is the smallest change of the set.
    ("l2", dict(style="mosaic", level=2, fill="auto", seed=33,
                scheme=TEAL_HAZE)),
    # Square tiles cascade down to level 3 only — the bank stops there — so the
    # scatter has more to do.
    ("l5-square", dict(style="mosaic", level=5, shape="square", fill="auto",
                       seed=34, scheme=CREAM_HAZE)),
    # The same geometry as `l4`, with the loose still lifes pulled back to a
    # fringe. The honest comparison for whether filling the whole sky is right.
    ("l4-band", dict(style="mosaic", level=4, fill="auto", fill_band=18,
                     seed=35, scheme=TEAL_HAZE)),
]

# (variation-set name, source image, solve name, variations). The solve name
# keys both the gitignored run directory and the committed asset, so a fresh
# clone renders everything without the run.
SOURCES = [
    ("banner1", "input/images/linkedin-background.png", "banner1", BANNER_1),
    ("banner2", "input/images/linkedin-background-2.png", "banner2", BANNER_2),
    ("banner2-avatar", "input/images/linkedin-background-2.png", "banner2",
     AVATAR),
    ("banner2-bg", "input/images/linkedin-background-2.png", "banner2",
     BACKGROUNDS),
    ("banner1-filled", "input/images/linkedin-background.png", "banner1",
     FILLED_1),
    ("banner2-filled", "input/images/linkedin-background-2.png", "banner2",
     FILLED_2),
]


def load_solve(here: Path, solve: str, saved: set) -> np.ndarray:
    """The 800x200 still life, from the run directory or from the asset.

    `results/` is gitignored, so a fresh clone has only the bit-packed asset.
    When the run *is* present it wins and the asset is rewritten from it, once
    per solve however many variation sets share it.
    """
    asset = here / f"assets/{solve}_800x200_pipeline.npz"
    run = here / f"results/linkedin{solve[-1]}/pattern.npy"
    if not run.exists():
        return load_pattern_asset(asset)

    pattern = np.load(run)
    if solve not in saved:
        saved.add(solve)
        save_pattern_asset(asset, pattern)
    return pattern


def to_banner(image: Image.Image) -> Image.Image:
    """Crop a 1600x400 render to LinkedIn's 1584x396.

    Symmetric on the width, but the four surplus rows all come off the top:
    the mountains sit on the bottom edge in both sources, and the sky can
    spare them.
    """
    w, h = image.size
    left = (w - BANNER[0]) // 2
    return image.crop((left, h - BANNER[1], left + BANNER[0], h))


def main() -> None:
    here = Path(__file__).resolve().parent
    out = REPO / "output/images/linkedin"
    out.mkdir(parents=True, exist_ok=True)
    sheet, saved = {}, set()

    for name, src, solve, variations in SOURCES:
        pattern = load_solve(here, solve, saved)
        assert pattern.shape == (HEIGHT, WIDTH), pattern.shape
        ver = verify_still_life(pattern)
        assert ver["bounded"] and ver["toroidal"], f"{name}: {ver}"
        _, free = load_rect(str(REPO / src), WIDTH, HEIGHT, "eq", 5.0)
        print(f"{name}: {int(pattern.sum()):,} live cells, "
              f"background {float((~free).mean()):.3f}, verify OK")

        column = []
        for label, kwargs in variations:
            image = to_banner(compose(pattern, ~free, scale=SCALE, **kwargs))
            image.save(out / f"{name}-{label}.png")
            column.append((f"{name} · {label}", image))
            print(f"   {label:20s} -> {name}-{label}.png")
        sheet[name] = column

    halo_figure(here, out)
    rule_figure(here)

    contact_sheet([sheet["banner1"], sheet["banner2"]],
                  out / "contact-sheet.png", width=16, panel_h=1.35)
    # The study sets are the ones being chosen from, so each also gets its own
    # sheet at 2 x 5 — big enough to judge a palette or a backdrop by.
    studies = {
        "avatar-sheet": _halve(sheet["banner2-avatar"]),
        "bg-sheet": _halve(sheet["banner2-bg"]),
        # This one pairs by source rather than by palette: the point of the
        # filled set is how differently the two skies behave.
        "filled-sheet": [sheet["banner1-filled"], sheet["banner2-filled"]],
    }
    for stem, columns in studies.items():
        contact_sheet(columns, out / f"{stem}.png", width=17, panel_h=2.4)
        print(f"{stem:14s} -> {out / f'{stem}.png'}")
    print(f"contact sheet -> {out / 'contact-sheet.png'}")


def _halve(column):
    """Split a ten-panel set into two columns of five."""
    return [column[:5], column[5:]]


def halo(field, background):
    """Chebyshev distance from each silhouette cell to the nearest field cell.

    The number the filler exists to bring down. A plain mosaic seats a tile
    only where the tile's whole footprint fits, so this is set by where the
    lattice falls rather than by the subject's outline — and it is the
    *variation*, not the width, that reads as untidy.
    """
    from scipy.ndimage import binary_dilation, distance_transform_cdt

    silhouette = (~background) & binary_dilation(background,
                                                 np.ones((3, 3), bool))
    empty = np.asarray(field) == 0
    return distance_transform_cdt(empty, metric="chessboard")[silhouette]


def halo_figure(here: Path, out: Path) -> None:
    """The curated before/after: banner 2 at level 6, plain against filled.

    Level 6 is the worst case and the chosen render, so it is the honest
    figure. Written to `figures/` rather than to `out`, because everything in
    `out` is regenerable bulk and this one is documentation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gol_mosaics import mosaic_background

    pattern = load_solve(here, "banner2", set())
    _, free = load_rect(
        str(REPO / "input/images/linkedin-background-2.png"),
        WIDTH, HEIGHT, "eq", 5.0)
    background = ~free

    panels = [("plain mosaic", dict(fill=None)),
              ("filled: smaller tiles graded by size, then loose still lifes",
               dict(fill="auto"))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.4))
    for ax, (title, kwargs) in zip(axes, panels):
        image = to_banner(compose(pattern, background, scale=SCALE,
                                  style="mosaic", level=6, seed=31,
                                  scheme=TEAL_HAZE, **kwargs))
        field = (mosaic_background(background, level=6, seed=31)
                 if kwargs["fill"] is None
                 else filled_background(background, level=6, seed=31))
        spread = halo(field, background)
        ax.imshow(np.asarray(image.convert("RGB")))
        ax.set_title(f"{title} — halo mean {spread.mean():.1f}, "
                     f"max {spread.max()} cells", fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    path = REPO / "experiments/beyond_tiles/figures/linkedin_halo_filling.png"
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"halo figure   -> {path.relative_to(REPO)}")

    # The chosen render, kept as a curated figure so the repository carries
    # the deliverable without the whole regenerable variation set.
    chosen = out / "banner2-filled-l6.png"
    Image.open(chosen).save(
        REPO / "experiments/beyond_tiles/figures/linkedin_banner_filled_l6.png")


def box_rule_field(background, level, seed=None, shape="diamond"):
    """`mosaic_background` as it stood before 2026-08-29, for the figure.

    The superseded rule: a site was kept only when its whole `6*level` box
    plus the gap was clear, rather than the tile's own footprint. Kept here
    rather than in the library because its only remaining use is to draw the
    comparison — see REPORT.md section 8.
    """
    from gol_mosaics.compose import density_band
    from gol_mosaics.patterns import PatternLibrary
    from gol_mosaics.tile_scheme import assemble, diamond_scheme

    gap = 2
    scheme = diamond_scheme(level)
    tiles = np.asarray(PatternLibrary.load(level, shape=shape).solutions,
                       dtype=np.uint8)
    indices, _ = density_band(level, shape, (0.0, 1.0))
    height, width = background.shape
    n = scheme.n
    (u_i, u_j), (v_i, v_j) = scheme.u, scheme.v

    det = u_i * v_j - v_i * u_j
    ab = [((i * v_j - j * v_i) / det, (j * u_i - i * u_j) / det)
          for i in (-n, height) for j in (-n, width)]
    a_lo, a_hi = int(np.floor(min(a for a, _ in ab))), int(np.ceil(max(a for a, _ in ab)))
    b_lo, b_hi = int(np.floor(min(b for _, b in ab))), int(np.ceil(max(b for _, b in ab)))

    grid = np.full((a_hi - a_lo + 1, b_hi - b_lo + 1), -1, dtype=np.int64)
    rows, cols = [], []
    for a in range(a_lo, a_hi + 1):
        for b in range(b_lo, b_hi + 1):
            ci, cj = a * u_i + b * v_i, a * u_j + b * v_j
            if ci - gap < 0 or cj - gap < 0:
                continue
            if ci + n + gap > height or cj + n + gap > width:
                continue
            if background[ci - gap:ci + n + gap, cj - gap:cj + n + gap].all():
                rows.append(a - a_lo)
                cols.append(b - b_lo)
    if rows:
        grid[rows, cols] = np.random.default_rng(seed).choice(indices,
                                                              size=len(rows))
    origin_i = min(a_lo * u_i, a_hi * u_i) + min(b_lo * v_i, b_hi * v_i)
    origin_j = min(a_lo * u_j, a_hi * u_j) + min(b_lo * v_j, b_hi * v_j)
    pad = 2
    whole = assemble(scheme, grid, tiles, pad=pad)
    return np.ascontiguousarray(
        whole[pad - origin_i:pad - origin_i + height,
              pad - origin_j:pad - origin_j + width])


def rule_figure(here: Path) -> None:
    """Box rule against support rule, banner 2 at level 6.

    Both panels are plain mosaics with no filling at all, so the only
    difference on show is which lattice sites the fit test accepts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gol_mosaics import mosaic_background

    pattern = load_solve(here, "banner2", set())
    _, free = load_rect(
        str(REPO / "input/images/linkedin-background-2.png"),
        WIDTH, HEIGHT, "eq", 5.0)
    background = ~free

    fields = [("box rule: the whole 6*level box must be clear",
               box_rule_field(background, 6, seed=31)),
              ("support rule: only the tile's own footprint must be",
               mosaic_background(background, level=6, seed=31))]
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.4))
    for ax, (title, field) in zip(axes, fields):
        backdrop = background.astype(np.uint8)
        image = MosaicRenderer(TEAL_HAZE).render_full_mosaic(
            pattern, backdrop * (field + backdrop))
        spread = halo(field, background)
        ax.imshow(np.asarray(to_banner(image.resize(
            (WIDTH * SCALE, HEIGHT * SCALE), Image.Resampling.NEAREST)
        ).convert("RGB")))
        ax.set_title(f"{title} — {int(field.sum()):,} cells, "
                     f"halo mean {spread.mean():.1f}, max {spread.max()}",
                     fontsize=10)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    path = REPO / "experiments/beyond_tiles/figures/linkedin_fit_rule.png"
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"fit-rule fig  -> {path.relative_to(REPO)}")


def contact_sheet(columns, path, width=16, panel_h=1.35) -> None:
    """One sheet, a column per variation set, so sets compare row by row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = max(len(c) for c in columns)
    fig, axes = plt.subplots(rows, len(columns),
                             figsize=(width, panel_h * rows), squeeze=False)
    for j, column in enumerate(columns):
        for i in range(rows):
            ax = axes[i, j]
            ax.axis("off")
            if i < len(column):
                title, image = column[i]
                ax.imshow(np.asarray(image.convert("RGB")))
                ax.set_title(title, fontsize=8, pad=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
