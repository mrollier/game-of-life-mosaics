# Thesis cover

The background of the cover of a joint UGent / Universidade de São Paulo PhD
thesis (September 2026), 160 x 240 mm, front and back. Both are mosaics of
whole tiles of levels 1 to 7, packed without gaps and coloured by level,
and each is one verified still life. The title, its see-through boxes, the
bibliographic block and the logos are laid over them in Canva.

| Side | What | Cells |
|---|---|---|
| front | the drowned coast of Paraty Mirim, traced from three satellite silhouettes, tiles on the land or in the water (six candidates) | 360 x 540, 0.44 mm |
| back | the defence flyer's Ghent and São Paulo skyline, lifted 25 mm over a band of tiles for the bibliographic block | 540 x 810, 0.30 mm |

```bash
python studies/cover/cover.py map --out studies/cover/renders/front --pairs denim-apricot   # ~30 s
python studies/cover/cover.py lifted levels-deep --out studies/cover/renders/back          # ~20 s
```

Both commands rebuild the delivered files exactly. They need the level-7
tile census at the repository root (`solutions_pattern_level_7_orbits.npy`,
1.2 GB, not in git; see [search/tiles](../../search/tiles/README.md)), of
which a seeded sample of 20,000 tiles is read. The back cover also reads
the flyer's solve from [`../flyer/solves/`](../flyer/README.md).

| Folder | Versioned | Contents |
|---|---|---|
| `renders/front/` | yes | `paraty-{a,b,c}-{land,sea}_denim-apricot.{png,svg}` and a contact sheet |
| `renders/back/` | yes | `back_levels-deep.{png,svg}`, the chosen back cover |
| `assets/silhouettes/` | yes | the three silhouettes (land green, sea blue, 2:3) the front is traced from |
| `assets/palettes.json` | yes | the print-safe ramps and fields (output of `palettes`) |
| `output/` | no | label maps, the study rounds (`round7/`, `round9/`, `back-round7/`), sheets |

## Print files

Each design comes as a PNG with a whole number of pixels per cell (front
10 px, 3600 x 5400 at 571.5 dpi; back 7 px, 3780 x 5670 at 600 dpi) and as
an SVG with one path per colour, the outline of that colour's cells traced
into polygons. The SVG is exact at any size, has no seams between cells and
stays well under Canva's 3 MB limit (0.3 to 0.6 MB). Every SVG was checked
against its PNG cell by cell.

Colours are Denim on apricot: the field `#FFD9A8`, and per level a tile
ground and a tone-on-tone ink one step darker, all inside the Coated
FOGRA39 gamut. The covers use the ramp's `COAST` steps: the two palest
blues are dropped, so ponds and small tiles still stand off the apricot and
the coastline reads from a distance. A small tile in a hole among bigger
ones takes its host's colour, and the grout inside the hole closes, so the
hole reads as part of the host.

## How it works

The method is in the docstring of [`cover.py`](cover.py). In short, all
diamond tiles sit on one **pond lattice**, a level-L tile covers an L x L
square of its vertices, and any vertex-disjoint set of tiles is a still
life. A cover is therefore a mask and a level field: tiles are packed
biggest first, each where it touches the most, and ponds fill every vertex
left over.

- **Front (`map`).** Land is where the silhouette's green outweighs its
  blue, averaged over each cell. The level field rises with the distance
  from the shore (the `-land` versions) or from it into the water (the
  `-sea` versions, which read like a depth chart). The page is padded by
  48 cells with the coast mirrored into the padding, packed as a torus and
  cut at the trim, so tiles run off the page and the part inside the trim
  is a still life. The coast is exact to the 3-cell pond lattice, 1.3 to
  2.7 mm.
- **Back (`lifted`).** The flyer's `skyline2` (the solved free-form grain
  and its tile region) with 60 empty rows taken out above the city and a
  band of 84 rows added under its ground, so the page stays 540 x 810 and
  no neighbourhood changes. The tile region, from the top ribbon through the
  buildings to the band, is repacked like the front, with no tile cell
  within two cells of the grain (so no cell sees both, and each is a still
  life on its own). Each region is first filled in its ponds' colour, so the
  buildings keep their outline where no tile fits. The level field is held
  under about 6.5, at random give or take a level, so the deep band mixes
  the upper levels instead of turning solid level 7. No tile is cut except by
  the trim.

## Design history

Nine review rounds with the author, 2026-09-28:

1. Concept sketches. Favourites: strata, slope, terraces, margin.
2. Bands of whole tiles, one colour per level, all verified still lifes.
   Too empty; small tiles should creep into the nooks of big ones.
3. Zone maps on a 360 x 540 torus, each level in its own zone. Only light
   fields; tone-on-tone ink; tighter packing wanted.
4. FFT-scored packing and OKLCH ramps proofed against FOGRA39. Denim on
   apricot and Ember on sky chosen; Rolling hills and Archipelago liked.
5. The pond-lattice insight: packing becomes gapless. Designs built
   around the text boxes.
6. Full-bleed designs: the text is laid over the pattern, not cut out of
   it. Favourites: open sea, lagoons, corner shoals, island arc; a Brazilian
   link wanted, ideally Paraty Mirim.
7. Ten designs including invented rias of Paraty Mirim; print files; the
   flyer skyline as a first back cover (`back`).
8. The real coast of Paraty Mirim traced from the author's silhouettes, cut
   at the trim instead of wrapping; Denim on apricot only.
9. The back cover lifted over a band of tiles for the bibliographic block,
   in six variants; `levels-deep` chosen and the band cut from 30 mm to
   25 mm.
