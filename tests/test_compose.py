"""
Tests for gol_mosaics.compose (post-hoc backgrounds and recolouring).
"""

import numpy as np
import pytest

from gol_mosaics.colors import ColorScheme
from gol_mosaics.compose import (
    BLOCK,
    ELEMENTARY,
    agar_background,
    compose,
    density_band,
    fill_layer_count,
    filled_background,
    life_safe_pattern,
    mosaic_background,
    scatter_background,
)
from gol_mosaics.life import is_still_life
from gol_mosaics.renderer import hex_to_rgb

SCHEME = ColorScheme(gol_background='#FFFFFF', gol_pixel='#000000',
                     eca_background='#FFD200', eca_pixel='#1E64C8')


def _subject_scene(size=40, radius=9):
    """A circular subject holding one block, on a background region."""
    yy, xx = np.mgrid[0:size, 0:size]
    subject = np.hypot(yy - size / 2, xx - size / 2) <= radius
    pattern = np.zeros((size, size), dtype=np.uint8)
    pattern[size // 2:size // 2 + 2, size // 2:size // 2 + 2] = 1
    return pattern, ~subject


def test_subject_keeps_gol_colours():
    """Whatever the backdrop, the subject region is painted in GoL colours."""
    pattern, background = _subject_scene(size=120, radius=18)
    for style in ('none', 'flat', 'eca', 'agar', 'mosaic'):
        rgb = np.asarray(compose(pattern, background, SCHEME, style=style,
                                 level=3))[:, :, :3]
        subject = ~background
        assert (rgb[subject & (pattern == 1)] == hex_to_rgb('#000000')).all(), style
        assert (rgb[subject & (pattern == 0)] == hex_to_rgb('#FFFFFF')).all(), style


def test_flat_background_is_uniform():
    """style='flat' fills the whole background with one colour."""
    pattern, background = _subject_scene()
    rgb = np.asarray(compose(pattern, background, SCHEME, style='flat'))[:, :, :3]
    assert (rgb[background] == hex_to_rgb('#FFD200')).all()


def test_eca_background_uses_both_field_colours():
    """style='eca' paints the background with the two ECA colours only."""
    pattern, background = _subject_scene()
    rgb = np.asarray(compose(pattern, background, SCHEME,
                             style='eca', rule=110, supersample=3))[:, :, :3]
    painted = rgb[background]
    is_field = (painted == hex_to_rgb('#FFD200')).all(axis=1)
    is_pixel = (painted == hex_to_rgb('#1E64C8')).all(axis=1)
    assert (is_field | is_pixel).all()
    assert is_pixel.any(), "rule 110 should put some pixels down"


def test_transparent_background_punches_alpha():
    """style='none' leaves the background fully transparent, subject opaque."""
    pattern, background = _subject_scene()
    alpha = np.asarray(compose(pattern, background, SCHEME, style='none'))[:, :, 3]
    assert (alpha[background] == 0).all()
    assert (alpha[~background] == 255).all()


def test_agar_keeps_its_distance():
    """Every agar cell is `gap` cells clear of the subject and of the border."""
    pattern, background = _subject_scene()
    gap = 2
    agar = agar_background(background, gap=gap)
    assert agar.any(), "the background is roomy enough for blocks"
    for i, j in zip(*np.nonzero(agar)):
        window = background[i - gap:i + 1 + gap, j - gap:j + 1 + gap]
        assert window.shape == (2 * gap + 1, 2 * gap + 1), "block ran off the edge"
        assert window.all(), "agar cell too close to the subject"


def test_agar_blocks_are_whole():
    """The agar is made of complete 2x2 blocks."""
    agar = agar_background(np.ones((40, 40), dtype=bool))
    labelled = agar.copy()
    for i, j in zip(*np.nonzero(agar)):
        if labelled[i, j]:
            assert agar[i:i + BLOCK, j:j + BLOCK].sum() == BLOCK * BLOCK
            labelled[i:i + BLOCK, j:j + BLOCK] = 0


def test_agar_alone_is_a_still_life():
    """The bare block agar is stable, with no subject involved."""
    assert is_still_life(agar_background(np.ones((60, 60), dtype=bool)))


def test_agar_union_is_a_still_life():
    """Subject plus agar background is stable over the whole grid."""
    pattern, background = _subject_scene()
    assert is_still_life(pattern), "the fixture's subject must be stable first"
    assert is_still_life(life_safe_pattern(pattern, background))


@pytest.mark.parametrize('pitch', [(3, 4), (4, 4), (5, 7)])
def test_wider_pitches_stay_still_lifes(pitch):
    """Sparser lattices are safe too — only the density changes."""
    pattern, background = _subject_scene()
    whole = life_safe_pattern(pattern, background, pitch=pitch)
    assert is_still_life(whole)
    assert whole.sum() > pattern.sum()


def test_agar_rejects_tight_pitch():
    """A period-3 column lattice would allow births, so it is refused."""
    with pytest.raises(ValueError, match='at least'):
        agar_background(np.ones((20, 20), dtype=bool), pitch=(3, 3))


def test_agar_rejects_small_gap():
    """A gap below 2 would let the agar interact with the subject."""
    with pytest.raises(ValueError, match='at least'):
        agar_background(np.ones((20, 20), dtype=bool), gap=1)


def test_life_safe_pattern_rejects_stray_live_cells():
    """Live cells outside the subject would break the still-life guarantee."""
    pattern, background = _subject_scene()
    pattern[0, 0] = 1
    with pytest.raises(ValueError, match='live cells inside the background'):
        life_safe_pattern(pattern, background)


def test_scale_upscales_in_uniform_blocks():
    """scale=3 gives a 3x image whose cells are solid 3x3 blocks."""
    pattern, background = _subject_scene(size=20, radius=5)
    image = compose(pattern, background, SCHEME, style='flat', scale=3)
    assert image.size == (60, 60)
    rgb = np.asarray(image)[:, :, :3]
    cells = rgb.reshape(20, 3, 20, 3, 3)
    assert (cells == cells[:, :1, :, :1]).all()


def test_compose_rejects_bad_input():
    """Shape, style and scale are all validated."""
    pattern, background = _subject_scene(size=20, radius=5)
    with pytest.raises(ValueError, match='2D'):
        compose(np.zeros((2, 2, 2)), background, SCHEME)
    with pytest.raises(ValueError, match='does not match'):
        compose(pattern, background[:10], SCHEME)
    with pytest.raises(ValueError, match='Unknown style'):
        compose(pattern, background, SCHEME, style='sparkles')
    with pytest.raises(ValueError, match='positive integer'):
        compose(pattern, background, SCHEME, scale=0)


def test_compose_defaults_to_ugent():
    """Omitting the scheme paints in the default UGent colours."""
    pattern, background = _subject_scene(size=20, radius=5)
    rgb = np.asarray(compose(pattern, background, style='flat'))[:, :, :3]
    assert (rgb[background] == hex_to_rgb(ColorScheme.ugent().eca_background)).all()


# --------------------------------------------------------- tile backgrounds

def _mosaic_scene(size=120, radius=18):
    """A subject with enough room around it to seat whole tiles."""
    return _subject_scene(size=size, radius=radius)


SHAPE_LEVELS = [('diamond', 2), ('diamond', 3), ('diamond', 4),
                ('square', 3), ('square', 4)]


@pytest.mark.parametrize('shape,level', SHAPE_LEVELS)
def test_mosaic_background_matches_the_mask(shape, level):
    """The field is cropped onto the pattern's own grid, whatever the basis.

    The lattice never divides the canvas, so this is really a check on the
    offset arithmetic that places `assemble`'s output.
    """
    pattern, background = _mosaic_scene()
    field = mosaic_background(background, level=level, shape=shape, seed=0)
    assert field.shape == background.shape
    assert field.any(), "the background is roomy enough for tiles"
    assert set(np.unique(field)) <= {0, 1}


def test_mosaic_background_handles_a_rectangular_canvas():
    """Nothing in the placement assumes a square grid."""
    yy, xx = np.mgrid[0:90, 0:150]
    subject = np.hypot(yy - 45, xx - 75) <= 15
    field = mosaic_background(~subject, level=3, seed=0)
    assert field.shape == (90, 150)
    assert field.any()


@pytest.mark.parametrize('shape,level', SHAPE_LEVELS)
def test_mosaic_keeps_its_distance(shape, level):
    """Every mosaic cell is `gap` cells clear of the subject and the border."""
    pattern, background = _mosaic_scene()
    gap = 2
    field = mosaic_background(background, level=level, shape=shape, gap=gap,
                              seed=0)
    for i, j in zip(*np.nonzero(field)):
        window = background[i - gap:i + 1 + gap, j - gap:j + 1 + gap]
        assert window.shape == (2 * gap + 1, 2 * gap + 1), "tile ran off the edge"
        assert window.all(), "mosaic cell too close to the subject"


@pytest.mark.parametrize('shape,level', SHAPE_LEVELS)
def test_mosaic_union_is_a_still_life(shape, level):
    """Subject plus tile background is stable over the whole grid."""
    pattern, background = _mosaic_scene()
    assert is_still_life(pattern), "the fixture's subject must be stable first"
    field = mosaic_background(background, level=level, shape=shape, seed=0)
    whole = life_safe_pattern(pattern, background, field=field)
    assert whole.sum() > pattern.sum(), "the mosaic should add cells"
    assert is_still_life(whole)


def test_mosaic_background_seed_is_reproducible():
    """The same seed redraws the same field; a different one does not."""
    _, background = _mosaic_scene()
    first = mosaic_background(background, level=4, seed=7)
    assert np.array_equal(first, mosaic_background(background, level=4, seed=7))
    assert not np.array_equal(first,
                              mosaic_background(background, level=4, seed=8))


def test_mosaic_tone_grades_the_field():
    """A linear tone puts the dense tiles where the angle points.

    Compared against the opposite angle rather than against a flat field:
    the subject punches a hole in the middle, so absolute per-column counts
    are not comparable on their own.
    """
    _, background = _mosaic_scene(size=150, radius=20)
    kwargs = dict(level=4, shape='diamond', tone='linear', seed=3)
    rightwards = mosaic_background(background, tone_angle=0, **kwargs)
    leftwards = mosaic_background(background, tone_angle=180, **kwargs)
    half = background.shape[1] // 2
    assert rightwards[:, half:].sum() > rightwards[:, :half].sum()
    assert leftwards[:, :half].sum() > leftwards[:, half:].sum()


def test_mosaic_vignette_is_denser_than_radial_at_the_edges():
    """'radial' gathers density at the centre, 'vignette' at the edges."""
    _, background = _mosaic_scene(size=150, radius=20)
    kwargs = dict(level=4, shape='diamond', seed=3)
    radial = mosaic_background(background, tone='radial', **kwargs)
    vignette = mosaic_background(background, tone='vignette', **kwargs)
    border = np.zeros(background.shape, dtype=bool)
    border[:40] = border[-40:] = border[:, :40] = border[:, -40:] = True
    assert vignette[border].sum() > radial[border].sum()


def test_mosaic_background_rejects_bad_arguments():
    """Unknown geometry, unknown tone and an unsafe gap are all refused."""
    background = np.ones((80, 80), dtype=bool)
    with pytest.raises(ValueError, match='Unknown shape'):
        mosaic_background(background, shape='hexagon')
    with pytest.raises(ValueError, match='Unknown tone'):
        mosaic_background(background, tone='swirl')
    with pytest.raises(ValueError, match='at least'):
        mosaic_background(background, gap=1)
    with pytest.raises(ValueError, match='levels 3-5'):
        mosaic_background(background, level=2, shape='square')


def test_density_band_selects_the_band():
    """Only tiles inside the normalised band come back."""
    indices, _ = density_band(4, 'diamond', (0.3, 0.7))
    from gol_mosaics.patterns import PatternLibrary
    normalised = PatternLibrary.load(4, shape='diamond').densities
    assert indices.size
    assert ((normalised[indices] >= 0.3) & (normalised[indices] <= 0.7)).all()
    assert indices.size < normalised.size, "the band should exclude something"


def test_density_band_reports_the_absolute_range():
    """The normalised band is relative per level, so the real fill is told."""
    from gol_mosaics.patterns import PatternLibrary
    indices, (low, high) = density_band(4, 'diamond', (0.3, 0.7))
    solutions = PatternLibrary.load(4, shape='diamond').solutions
    fill = solutions[indices].mean(axis=(1, 2))
    assert low == pytest.approx(fill.min())
    assert high == pytest.approx(fill.max())
    assert 0 < low <= high < 1


def test_density_band_rejects_an_empty_band():
    """An impossible band names the range that is actually available."""
    with pytest.raises(ValueError, match=r'spans \[0\.000, 1\.000\]'):
        density_band(4, 'diamond', (1.5, 2.0))


def test_density_band_rejects_an_inverted_band():
    with pytest.raises(ValueError, match='inverted'):
        density_band(4, 'diamond', (0.8, 0.2))


def test_mosaic_density_band_changes_the_fill():
    """A dense band lays down more cells than a sparse one."""
    _, background = _mosaic_scene()
    sparse = mosaic_background(background, level=4, density=(0.0, 0.2), seed=1)
    dense = mosaic_background(background, level=4, density=(0.8, 1.0), seed=1)
    assert dense.sum() > sparse.sum()


def test_compose_mosaic_uses_the_field_colours():
    """style='mosaic' paints the background with the two ECA colours only."""
    pattern, background = _mosaic_scene()
    rgb = np.asarray(compose(pattern, background, SCHEME, style='mosaic',
                             level=3, seed=0))[:, :, :3]
    painted = rgb[background]
    is_field = (painted == hex_to_rgb('#FFD200')).all(axis=1)
    is_pixel = (painted == hex_to_rgb('#1E64C8')).all(axis=1)
    assert (is_field | is_pixel).all()
    assert is_pixel.any(), "the tiles should put some pixels down"


def test_life_safe_pattern_rejects_a_crowding_field():
    """A field touching the subject is refused, not silently merged."""
    pattern, background = _mosaic_scene()
    field = np.zeros(pattern.shape, dtype=np.uint8)
    i, j = np.argwhere(pattern)[0]
    field[i - 3:i - 1, j:j + 2] = 1          # a block two cells off the subject
    with pytest.raises(ValueError, match='within 2 of the subject'):
        life_safe_pattern(pattern, background, field=field)


def test_life_safe_pattern_rejects_a_field_at_the_border():
    """Clearance from the border is what keeps the toroidal check passing."""
    pattern, background = _mosaic_scene()
    field = np.zeros(pattern.shape, dtype=np.uint8)
    field[0:2, 0:2] = 1
    with pytest.raises(ValueError, match='grid border'):
        life_safe_pattern(pattern, background, field=field)


def test_life_safe_pattern_rejects_a_mismatched_field():
    pattern, background = _mosaic_scene()
    with pytest.raises(ValueError, match='does not match'):
        life_safe_pattern(pattern, background,
                          field=np.zeros((10, 10), dtype=np.uint8))


@pytest.mark.parametrize('background', [
    np.zeros((60, 60), dtype=bool),      # no background at all
    np.ones((10, 10), dtype=bool),       # grid smaller than one tile
    np.ones((22, 22), dtype=bool),       # box fits, but no lattice site does
])
def test_mosaic_background_seats_nothing_gracefully(background):
    """No room is an empty field of the right shape, not a crash.

    The lattice is fixed rather than centred on the canvas, so the third case
    is real: a level-3 box needs its corner at exactly (2, 2) to fit a 22x22
    grid, and the diamond basis has no site there.
    """
    field = mosaic_background(background, level=3, seed=0)
    assert field.shape == background.shape
    assert not field.any()


# ------------------------------------------------------- filling the gap

def _halo(field, background):
    """Chebyshev distance from each silhouette cell to the nearest field cell.

    This is the number the filler exists to bring down: a plain mosaic seats
    tiles only where a whole footprint fits, so the halo width is set by where
    the lattice falls rather than by the subject's outline.
    """
    from scipy.ndimage import binary_dilation, distance_transform_cdt

    subject = ~background
    silhouette = subject & binary_dilation(background, np.ones((3, 3), bool))
    distance = distance_transform_cdt(~(np.asarray(field) != 0),
                                      metric='chessboard')
    return distance[silhouette]


def test_every_elementary_shape_is_a_still_life():
    """The filler bank is only safe if each of its shapes is stable alone."""
    for name, shape in ELEMENTARY.items():
        assert is_still_life(np.pad(shape, 2)), name


@pytest.mark.parametrize('shape,level', SHAPE_LEVELS)
def test_filled_union_is_a_still_life(shape, level):
    """The layering theorem: every layer keeps `gap` from all the others.

    Each pass sees only what the previous ones left, so no dead cell's 3x3
    neighbourhood can hold cells from two populations, and every population
    keeps the neighbour counts it was verified with.
    """
    pattern, background = _mosaic_scene()
    field = filled_background(background, level=level, shape=shape, seed=0)
    whole = life_safe_pattern(pattern, background, field=field)
    assert is_still_life(whole)
    assert set(np.unique(whole)) <= {0, 1}, "layer numbers must not leak out"


@pytest.mark.parametrize('band', [None, 8])
@pytest.mark.parametrize('shape,level', SHAPE_LEVELS)
def test_filled_union_is_a_still_life_at_every_band(shape, level, band):
    """The support fit test must hold with the scatter banded and unbanded.

    Unbanded is the default and the denser of the two, so it is the case that
    exercises the clearance rule hardest: a mistake there is silent in the
    render and only shows up as a birth beside a tile edge.
    """
    pattern, background = _mosaic_scene(size=200, radius=30)
    field = filled_background(background, level=level, shape=shape,
                              fill_band=band, seed=4)
    assert is_still_life(life_safe_pattern(pattern, background, field=field))


def test_mosaic_seats_a_tile_whose_box_corner_is_blocked():
    """The fit test is on the tile support, not on its 6*level box.

    A diamond fills under half its box, so a box test rejects sites over
    corners no tile can ever reach — which is what left wedges of bare
    background between neighbouring diamonds. Blocking exactly one such cell
    must not cost the tile.
    """
    from scipy.ndimage import binary_dilation

    from gol_mosaics.tile_scheme import diamond_scheme

    level, gap = 3, 2
    scheme = diamond_scheme(level)
    n = scheme.n
    footprint = binary_dilation(scheme.support | scheme.frame,
                                np.ones((2 * gap + 1,) * 2, dtype=bool))
    # Inside the box, outside the support's own clearance: unreachable by any
    # tile of the family, and the only thing a box test would trip over.
    corner = tuple(np.argwhere(~footprint)[0])

    # Lattice site (1, 0) of the diamond basis u = (3L, 3L).
    site = (3 * level, 3 * level)
    background = np.ones((6 * n, 6 * n), dtype=bool)
    background[site[0] + corner[0], site[1] + corner[1]] = False

    field = mosaic_background(background, level=level, seed=0)
    seated = field[site[0]:site[0] + n, site[1]:site[1] + n]
    assert seated[scheme.frame].all(), "the site should still take a tile"
    assert not (field.astype(bool)
                & binary_dilation(~background,
                                  np.ones((2 * gap + 1,) * 2, dtype=bool))
                ).any(), "and it must still keep its distance"


def test_filled_background_closes_the_halo():
    """The point of the feature: a much tighter, much more even halo."""
    pattern, background = _mosaic_scene(size=200, radius=30)
    plain = mosaic_background(background, level=4, seed=0)
    filled = filled_background(background, level=4, seed=0)
    before, after = _halo(plain, background), _halo(filled, background)
    assert after.max() < before.max() / 2
    assert after.max() <= 12, after.max()
    assert (filled != 0).sum() > plain.sum(), "filling should add cells"


def test_filled_background_numbers_its_layers():
    """0 empty, 1 the main mosaic, then one number per cascade level.

    The renderer grades the layers by tile size, so the numbering has to be
    dense and in descending order of level, with the scatter last.
    """
    _, background = _mosaic_scene()
    field = filled_background(background, level=4, seed=5)
    layers = fill_layer_count(4)
    assert layers == 5, "L4, L3, L2, L1, scatter"
    assert set(np.unique(field)) <= set(range(layers + 1))
    plain = mosaic_background(background, level=4, seed=5)
    assert np.array_equal(field == 1, plain.astype(bool))
    assert not ((field > 1) & plain.astype(bool)).any()
    assert (field > 1).any(), "there should be something left to fill"


def test_filled_background_layer_numbers_follow_tile_size():
    """Layer k+1 is drawn from a smaller bank than layer k.

    Checked through the cells rather than the labels: a smaller level's tiles
    live in a smaller box, so its connected clumps are smaller too.
    """
    from scipy.ndimage import label

    _, background = _mosaic_scene(size=200, radius=30)
    field = filled_background(background, level=4, seed=1)
    spans = []
    for layer in range(1, fill_layer_count(4)):
        cells = np.argwhere(field == layer)
        if not len(cells):
            continue
        spans.append(cells.max(axis=0)[0] - cells.min(axis=0)[0])
    assert len(spans) >= 3, "the cascade should reach at least three levels"


def test_fill_layer_count_matches_the_field():
    """The renderer trusts this count, so it must not exceed the numbering."""
    _, background = _mosaic_scene()
    for shape, level in SHAPE_LEVELS:
        field = filled_background(background, level=level, shape=shape, seed=3)
        assert int(field.max()) <= fill_layer_count(level, shape)
    assert fill_layer_count(4, fill=None) == 1
    assert fill_layer_count(4, fill=()) == 2


def test_filled_background_without_fill_is_the_plain_mosaic():
    """fill=None is the old behaviour, byte for byte."""
    _, background = _mosaic_scene()
    assert np.array_equal(filled_background(background, level=4, fill=None,
                                            seed=2),
                          mosaic_background(background, level=4, seed=2))


def test_filled_background_takes_explicit_levels():
    """A named cascade uses those levels and nothing else."""
    _, background = _mosaic_scene()
    everything = filled_background(background, level=4, seed=1)
    scatter_only = filled_background(background, level=4, fill=(), seed=1)
    assert (scatter_only != 0).sum() < (everything != 0).sum()
    assert is_still_life(np.pad(scatter_only != 0, 1))


def test_filled_background_rejects_an_unknown_fill():
    _, background = _mosaic_scene()
    with pytest.raises(ValueError, match='Unknown fill'):
        filled_background(background, level=4, fill='everything')


def test_scatter_keeps_its_distance():
    """Every scattered cell is `gap` clear of the subject and of the border."""
    _, background = _mosaic_scene()
    gap = 2
    field = scatter_background(background, gap=gap, seed=0)
    assert field.any(), "the background is roomy enough for small still lifes"
    for i, j in zip(*np.nonzero(field)):
        window = background[i - gap:i + 1 + gap, j - gap:j + 1 + gap]
        assert window.shape == (2 * gap + 1, 2 * gap + 1), "ran off the edge"
        assert window.all(), "scattered cell too close to the subject"


def test_scatter_respects_occupied_cells():
    """Cells another layer already placed are kept clear by the same gap."""
    from scipy.ndimage import binary_dilation

    _, background = _mosaic_scene()
    gap = 2
    tiles = mosaic_background(background, level=4, seed=0)
    loose = scatter_background(background, occupied=tiles, gap=gap, seed=0)
    assert loose.any()
    near = binary_dilation(tiles.astype(bool),
                           structure=np.ones((2 * gap + 1,) * 2, dtype=bool))
    assert not (loose.astype(bool) & near).any()


def test_scatter_alone_is_a_still_life():
    """The bare scatter is stable, with no subject involved."""
    assert is_still_life(scatter_background(np.ones((80, 80), dtype=bool),
                                            seed=0))


def test_scatter_band_confines_the_filler():
    """A band keeps the filler as a fringe instead of snow over the field."""
    from scipy.ndimage import distance_transform_cdt

    _, background = _mosaic_scene(size=200, radius=30)
    band = 12
    fringe = scatter_background(background, band=band, fade=False, seed=0)
    everywhere = scatter_background(background, seed=0)
    assert fringe.any()
    assert fringe.sum() < everywhere.sum()
    # The band is tested at each shape's anchor cell, so a shape seated on
    # the boundary reaches up to its own size (4 at most) beyond it.
    distance = distance_transform_cdt(background, metric='chessboard')
    assert (distance[fringe.astype(bool)] <= band + 4).all()


def test_scatter_fade_thins_towards_the_far_edge():
    """Fading trades cells for a fringe that dissolves instead of ending."""
    _, background = _mosaic_scene(size=200, radius=30)
    kwargs = dict(band=20, seed=0)
    assert (scatter_background(background, fade=True, **kwargs).sum()
            < scatter_background(background, fade=False, **kwargs).sum())


def test_scatter_seed_is_reproducible():
    """The same seed redraws the same filler; a different one does not."""
    _, background = _mosaic_scene()
    first = scatter_background(background, seed=7)
    assert np.array_equal(first, scatter_background(background, seed=7))
    assert not np.array_equal(first, scatter_background(background, seed=8))


@pytest.mark.parametrize('background', [
    np.zeros((60, 60), dtype=bool),   # no background at all
    np.ones((6, 6), dtype=bool),      # no room for a shape plus its margin
])
def test_scatter_seats_nothing_gracefully(background):
    field = scatter_background(background, seed=0)
    assert field.shape == background.shape
    assert not field.any()


def test_scatter_handles_a_mask_with_no_subject():
    """An all-background mask has no zero for the distance transform.

    A band then means nothing to measure against, so it must not be applied
    rather than crash or silently empty the field.
    """
    field = scatter_background(np.ones((80, 80), dtype=bool), band=10, seed=0)
    assert field.any()
    assert is_still_life(field)


def test_scatter_rejects_bad_arguments():
    background = np.ones((40, 40), dtype=bool)
    with pytest.raises(ValueError, match='at least'):
        scatter_background(background, gap=1)
    with pytest.raises(ValueError, match='does not match'):
        scatter_background(background, occupied=np.zeros((10, 10), dtype=bool))
    with pytest.raises(ValueError, match='No shapes'):
        scatter_background(background, shapes={})


def test_compose_grades_the_filler_levels():
    """Each filler level gets its own step along the ramp to `fill_pixel`."""
    pattern, background = _mosaic_scene()
    scheme = ColorScheme(gol_background='#FFFFFF', gol_pixel='#000000',
                         eca_background='#FFD200', eca_pixel='#1E64C8',
                         fill_pixel='#FF0000')
    rgb = np.asarray(compose(pattern, background, scheme, style='mosaic',
                             level=4, fill='auto', seed=0))[:, :, :3]
    painted = rgb[background]
    # The field colour, the main mosaic and the far end of the ramp are the
    # three the palette names outright; the levels between them are mixtures.
    for colour in ('#FFD200', '#1E64C8', '#FF0000'):
        assert (painted == hex_to_rgb(colour)).all(axis=1).any(), colour
    assert len(np.unique(painted, axis=0)) == 1 + fill_layer_count(4)


def test_compose_without_fill_keeps_two_field_colours():
    """A plain mosaic has one layer, so no ramp and nothing new to paint."""
    pattern, background = _mosaic_scene()
    rgb = np.asarray(compose(pattern, background, SCHEME, style='mosaic',
                             level=4, fill=None, seed=0))[:, :, :3]
    assert len(np.unique(rgb[background], axis=0)) == 2


def test_compose_filler_ramp_ends_on_the_derived_haze():
    """With no `fill_pixel` the ramp still runs, ending on the derived haze."""
    pattern, background = _mosaic_scene()
    rgb = np.asarray(compose(pattern, background, SCHEME, style='mosaic',
                             level=4, fill='auto', seed=0))[:, :, :3]
    painted = rgb[background]
    assert len(np.unique(painted, axis=0)) == 1 + fill_layer_count(4)
    assert (painted == hex_to_rgb(SCHEME.fill)).all(axis=1).any()


def test_life_safe_pattern_accepts_a_layered_field():
    """`filled_background`'s layer numbers must not leak into the pattern."""
    pattern, background = _mosaic_scene()
    field = filled_background(background, level=4, seed=0)
    assert field.max() > 2, "the fixture must exercise more than one layer"
    whole = life_safe_pattern(pattern, background, field=field)
    assert set(np.unique(whole)) <= {0, 1}
    assert whole.sum() == pattern.sum() + (field != 0).sum()
