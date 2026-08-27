"""
Tests for gol_mosaics.compose (post-hoc backgrounds and recolouring).
"""

import numpy as np
import pytest

from gol_mosaics.colors import ColorScheme
from gol_mosaics.compose import (
    BLOCK,
    agar_background,
    compose,
    life_safe_pattern,
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
    pattern, background = _subject_scene()
    for style in ('none', 'flat', 'eca', 'agar'):
        rgb = np.asarray(compose(pattern, background, SCHEME, style=style))[:, :, :3]
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
