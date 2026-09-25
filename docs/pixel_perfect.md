# Keeping mosaics pixel-perfect (and why they sometimes look blurry)

Every mosaic this project saves is **pixel-perfect**: one Game of Life cell maps
to exactly one pixel in a lossless PNG, with no interpolation or resizing. The
file on disk is as sharp as it can possibly be.

So if a mosaic looks **blurry when you zoom in**, the file is almost certainly
fine: the blur is added by whatever app is *displaying* it. This page explains
why, and how to view, export and reuse the images without re-blurring them.

## Why it looks blurry

Any app that shows an image at a size other than 100 % (one image pixel on one
screen pixel) has to **resample** it. Almost every consumer app does this with
*smooth* resampling (bilinear or bicubic) by default, so when you zoom past
100 % each one-pixel cell is blended with its neighbours. That is the blur you
see: the app drawing the picture, not the picture degrading.

To see crisp cells you need either:

- to view at **exactly 100 %** (or an integer multiple: 200 %, 300 %), or
- a viewer set to **nearest-neighbour** resampling (also called "no
  resample", "pixelated" or "point").

## Prove the file is fine (do this once)

1. Check the PNG's pixel size (right-click, Properties, Details), or in Python:
   ```python
   from PIL import Image
   im = Image.open("output.png")
   print(im.size, im.mode)   # e.g. (1224, 1584) RGBA
   ```
2. Open it at **100 % or an integer zoom** in any viewer: crisp.
3. Open it in a **nearest-neighbour** viewer (see below) and zoom right in:
   still crisp, hard-edged squares.

If steps 2 and 3 are sharp but Microsoft Photos or Canva look blurry, the file
is perfect and the softness is purely the app.

## Microsoft Photos

The Windows **Photos** app always smooths when you zoom past 100 %, and there
is no setting to turn that off. Blur on zoom there is expected and is **not** a
file defect. For crisp zooming on Windows, use a nearest-neighbour viewer:

- **IrfanView**: turn off *Options → Properties/Settings → Viewing → "Use
  resample for zooming"*.
- **GIMP**, **Krita**, **Paint.NET** or **Photoshop** all show hard pixels
  when zoomed in.

## Canva (the one to be careful with)

Canva is convenient but works against pixel art in a few ways:

- **The editor canvas always smooths on zoom.** The art looks soft *while you
  design* even though the asset is sharp; that is display only.
- **Never drag the placed image larger than its native pixel size.** Enlarging
  it forces Canva to upscale with smoothing, which bakes the blur into the
  export. Place it at **100 % or less** of its pixel dimensions.
- **Export as PNG, never JPG.** JPEG blurs one-pixel edges and adds ringing
  around the high-contrast cell borders, the worst case for pixel art. Leave
  "Compress file" unticked.
- **Match the export resolution to the asset.** Canva rasterises the whole page
  at the page's resolution; a page smaller than the mosaic's native size
  downsamples it. Size the design to the mosaic, or use **Download → PNG →
  size ×2 / ×3**.

If pixel-perfection is critical, compose in a pixel-aware tool (Photoshop,
GIMP, Krita at nearest-neighbour) or use the raw PNG.

## General gotchas with any pixel-perfect image

- **Never re-save as JPEG.** Stay lossless: PNG (or lossless WebP, BMP, TIFF).
- **Scale only by integer factors, and only with nearest-neighbour** (2×, 3×,
  4×). Non-integer scaling (×1.5, "fit to 1920") forces interpolation.
- **Downscaling is destructive.** Below native size, cells are lost and alias.
- **Social and chat apps recompress.** Most re-encode uploads to JPEG and may
  downscale; use "send as file / document" where the option exists.
- **Display scaling (150 %, 200 %) and thumbnails resample.** Judge sharpness at
  100 % display scaling and integer zoom, not from a thumbnail.
- **Printing:** make sure there are enough pixels for the physical size times
  the print resolution; some print drivers smooth. For visibly square cells in
  print, upscale with nearest-neighbour first.

## A copy that stays crisp longer in smoothing apps

If you regularly place mosaics in Canva or other smoothing tools, save an
integer-upscaled copy with **nearest-neighbour**. Cell edges stay hard, and
because the file has far more native pixels, smoothing apps must zoom much
further before any blur appears:

```python
from PIL import Image

w, h = mosaic.size
mosaic.resize((w * 4, h * 4), Image.Resampling.NEAREST).save("output_4x.png")
```

Nothing about the artwork changes: every cell becomes a clean 4x4 block.
`compose(..., scale=4)` does the same for backgrounds made with
`gol_mosaics.compose`.
