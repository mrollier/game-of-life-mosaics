# Defence postcards and jury print

Handouts for a joint UGent / Universidade de São Paulo PhD defence
(2 October 2026): five A6 postcards, one per guest, and an A3 print of
Paraty Mirim for the ten jury members. Everything is made of Game of Life
still lifes.

| Piece | What | Cells |
|---|---|---|
| `postcard-conway` | John Conway, the README mosaic's recipe (level-3 tiles on an automaton in UGent yellow and blue) | 528 x 732, 0.21 mm |
| `postcard-dragon` | the gilded dragon of the Ghent Belfry, for the card held sideways, tiles of levels 1-4 each in one gold, on a night-blue automaton | 264 x 366, 0.42 mm |
| `postcard-paraty` | the thesis cover's Paraty Mirim, tiles in the sea | 264 x 366, 0.42 mm |
| `postcard-liquid-vortex`, `-cathedral` | stained glass: the whole card packed with tiles of levels 1-7 in dark lead, coloured from a liquid-light dye field | 264 x 366, 0.42 mm |
| `jury-print-paraty_A3` | Paraty Mirim with land and sea both packed with tiles: the level rises with the altitude on land and with the distance from the shore at sea | 720 x 1008, 0.42 mm |

Every card is a two-page vector PDF of 111 x 154 mm (A6 with 3 mm bleed):
the front, and a back with the thesis title and a thank-you in English,
Dutch and Portuguese, a dateline, the website and a QR code, set in the
thesis cover's inline-code boxes. The jury print is one page of
303 x 426 mm (A3 with 3 mm bleed). All PDFs were checked cell by cell
against their 600 dpi renders.

## Rebuilding

Both scripts reuse the cover's gapless packer (`studies/cover/cover.py`),
so they need the level-7 census at the repository root, and they keep
colours inside the Coated FOGRA39 gamut, so they need that profile
(`Coated_Fogra39L_VIGC_300.icc` from color.org, not in git):

```bash
python studies/postcards/postcards.py fronts --round 3 --out studies/postcards/output/final --profile ICC
python studies/postcards/postcards.py cards conway dragon paraty liquid-vortex liquid-cathedral \
    --src studies/postcards/output/final --out studies/postcards/output/final --profile ICC
```

The dragon front needs the photo it is cut from at `assets/dragon.png`,
which is kept out of git (its rights are unknown); run `dragon-layout`
first.

The jury print needs elevation: AWS's open Terrarium tiles at zoom 14
around the Pico do Pão de Açúcar, fetched into a folder of your choice:

```bash
for x in $(seq 6157 6165); do for y in $(seq 9276 9286); do
  curl -sS -o DIR/t_${x}_${y}.png https://s3.amazonaws.com/elevation-tiles-prod/terrarium/14/$x/$y.png
done; done
python studies/postcards/jury.py dem --dem DIR
python studies/postcards/jury.py register --dem DIR
python studies/postcards/jury.py refine --dem DIR
python studies/postcards/jury.py layout --dem DIR
python studies/postcards/jury.py tilings t2-calm --profile ICC     # about 2 minutes
```

The tiles' land (SRTM) is clean, but their sea is not (blended
bathymetry leaves false shallows everywhere), so the coast is the cover's
silhouette c and only the land's altitude comes from the tiles. The
silhouette is registered on the hills: a window 16.3 km wide.

| Folder | Versioned | Contents |
|---|---|---|
| `renders/` | yes | the six print PDFs |
| `assets/fonts/` | yes | Courier Prime (SIL Open Font License, `OFL.txt`) |
| `assets/qr-michielrollier.png` | yes | the QR code's modules, one pixel each (segno, version 3, level Q) |
| `assets/dragon.png`, `assets/refs/` | no | the dragon photo and the colour references the user supplied |
| `output/` | no | fronts, cards, rounds, the jury solves and tilings |

## How the designs were chosen

- **Cells.** 1/60 inch (0.42 mm), a whole number of printer pixels at any
  standard resolution, so no cell is stretched more than its neighbour;
  visible close up only. Conway's face fell apart at that size, so his card
  has cells of 1/120 inch; it is solid black on white, so no halftone
  screen touches it.
- **Dragon.** Tiles coloured cell by cell from the photo read as a mottled
  photo; one gold per tile ("gold panes") reads as the Game of Life and
  won. A free-form still life of the dragon (`freeform_dragon.py`, a
  subagent's solve) was a verified still life but only a speckled
  silhouette at A6. The final card is drawn in landscape and turned onto
  the portrait card, with every part 6 mm inside the trim.
- **Liquid light.** Round 1 had only small panes, because a bug made the
  distance to the dye's shore zero everywhere; with it fixed, panes of all
  seven levels appear. Eight dye fields and palettes were compared (vortex,
  fingers, bubbles and rays; liquid, seventies, cathedral and Brazil); the
  user kept the vortex and the cathedral bubbles.
- **Backs.** The user's corrections: "het weelderige landschap van
  herhaalde eenvoud", "autômatos de rede", "bedankt om erbij te zijn!", the
  gender-neutral "valeu pela presença!", a dateline, and a QR code.
- **Jury print.** A free-form land (density by altitude, `solve`) matched
  its target closely but read as a light haze next to the tiled sea; tiles
  on land too won. Every level keeps one colour of its family (seven steps
  of denim at sea, of terracotta on land), the relief is smoothed a little,
  the seams are pale and the sea deepens slowly; of five tilings from
  ordered to wild the user chose T2 "calm" (12 tiles per level, little
  jitter).
