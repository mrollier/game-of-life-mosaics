# Sacred-geometry posters

Five A2 posters after photographs of sacred geometry: three Himalayan
mandala thangkas, the star dome of a Persian ceiling and a panel of Persian
tilework. Each is one Game of Life still life: the picture packed with the
thesis cover's tiles (`studies/cover/cover.py`), levels 1 to 7, without a
gap, exactly symmetric, on white paper.

| Print | Source | Sheet | Mosaic |
|---|---|---|---|
| `sacred-mandala-gold_A2` | green and gold mandala | A2 portrait | 372 x 436 mm |
| `sacred-mandala-blue_A2` | red and lapis mandala | A2 portrait | 372 x 431 mm |
| `sacred-mandala-dark_A2` | black and vermilion mandala | A2 portrait | 372 x 509 mm |
| `sacred-dome-calm_A2` | star dome (16-fold), traced calm | A2 landscape | 504 x 372 mm |
| `sacred-tilework_A2` | tile panel, black grout | A2 portrait | 372 x 467 mm |

All five are in "fresh pigment": the sources' own inks, brighter and more
saturated. Cells are 1/60 inch (0.42 mm), the jury print's cell. Every
print is a one-page vector PDF of the A2 page with 3 mm bleed (426 x 600
mm, the dome 600 x 426 mm) and a 600 dpi PNG. Each PDF was rasterised
again and checked cell by cell against the cells it was drawn from.

## Rebuilding

The script reuses the cover's packer and the postcards' colour and PDF
helpers, so it needs the level-7 census at the repository root and the
Coated FOGRA39 profile (`Coated_Fogra39L_VIGC_300.icc` from color.org, not
in git). The print check needs `pdftoppm` (poppler) on the PATH.

```bash
python studies/sacred/sacred.py build mandala-gold mandala-blue mandala-dark dome-calm tilework --profile ICC
python studies/sacred/sacred.py print --profile ICC
```

`build` traces, packs and renders a sheet in every palette (about 1 minute
to pack, 30 s per dye palette), into `output/round2`; `print` writes the
chosen sheets (`PRINTS`) to `output/print`. The sources are kept out of git
(their rights are unknown): put them in `assets/sources/` as
`mandala-blue.png`, `mandala-gold.png`, `mandala-dark.png`, `dome.png` and
`tilework.png`.

## Method

- **Frame.** Each mosaic keeps its source's proportions and sits on the
  sheet with at least 24 mm of paper round it. Tiles sit only where they
  lie wholly inside the mosaic's rectangle, 2 cells in from its edge, so the
  sheet is a finite still life, and a 2-cell lead line frames it.
- **Tracing.** Each source is reduced to 9 to 14 of its own inks (k-means
  in OKLab) and carried onto the cells, then made exactly symmetric: D4
  inside the mandala's square, mirrored left and right (and up and down for
  the dome and the tilework) outside it. The dome is a kaleidoscope of one
  11.25-degree wedge of the photo, a strap pointing up; its centre drifts
  down with the radius, as the photo is a little off-axis.
- **Levels by detail.** A level-L tile may sit where one ink covers at
  least THETA[L] (0.74 to 0.78) of its cells; levels 7 to 2 are packed with
  the cover's contact-greedy rule, a whole symmetric orbit at a time, then
  every free pond vertex gets a pond. Each tile draws its own still life, so
  only the layout is symmetric.
- **Colour.** Each tile is one pane of its ink, its live cells a tone
  darker; seams are a dark lead from the darkest ink (the dome's are its
  turquoise), and seams among small tiles close in the ink around them.
  All colours are moved into Coated FOGRA39.
- **Print check.** `pdftoppm` without anti-aliasing: poppler 20.10's
  anti-aliased renderer draws white slivers and drops seams on these
  pages, while matplotlib's own renderer and poppler without it both give
  every cell exactly.

## How the designs were chosen

- **Round 1** had six sources on the full A2, their borders extended to
  the taller sheet (bands repeated, reflected or shifted by the pattern's
  period). The user rejected the extensions ("the copying does not really
  work well"), asked for true still lifes with nothing cut at the edge and
  for more big tiles, dropped the muqarnas vault (a photo of carved
  plaster, which does not trace into inks), and chose black grout for the
  tile panel and a strap up for the dome.
- **Round 2** kept the sources' proportions and tried six palettes per
  sheet: the source inks, fresh pigment, and the postcards' liquid-light
  dyes (liquid, cathedral, seventies, Brazil), each ink moved to the dye
  nearest its hue at its lightness, balanced by area and lit from the
  centre. The user chose fresh pigment for all five, and the calm tracing
  of the dome (180 tiles of level 7 instead of 4).

| Folder | Versioned | Contents |
|---|---|---|
| `sacred.py` | yes | the whole pipeline |
| `assets/sources/` | no | the source photographs (rights unknown) |
| `output/` | no | rounds, packs, palettes and the prints |
