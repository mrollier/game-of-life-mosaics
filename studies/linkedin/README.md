# LinkedIn banners

Two 800x200 free-form still lifes of mountain ridges
(`input/images/linkedin-background{,-2}.png`), recoloured post hoc into
banner variations at LinkedIn's 1584x396. No solving happens here:

```bash
python studies/linkedin/linkedin_banners.py   # ~40 s
```

- `assets/banner{1,2}_800x200_pipeline.npz`: the two solved patterns,
  bit-packed. When a fresh `poster.py` run exists under
  `results/linkedin{1,2}/`, the script uses it and rewrites the asset.
- `figures/`: the halo-filling comparison, the level-6 filled banner and the
  fit-rule figure cited in REPORT.md sections 6-8 (rewritten by the script).
- `output/` (ignored): every variation, plus contact sheets per set.

To re-solve a banner:

```bash
python experiments/beyond_tiles/poster.py input/images/linkedin-background-2.png \
    --width 800 --height 200 --out ../../studies/linkedin/results/linkedin2
```
