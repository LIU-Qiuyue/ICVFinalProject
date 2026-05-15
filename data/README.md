# Data directory

Large binaries are **gitignored**. Create the following structure locally (names should match `configs/eval_config.json`):

```text
data/
  videos/              # optional source MP4 files
    tennis.mp4
    bmx-trees.mp4
    wild_001.mp4
  frames/              # PNG sequences extracted via src/extract_frames.py
    tennis/
      000000.png
      ...
    bmx-trees/
    wild_001/
  GroundTruth/         # binary masks for evaluation (course layout)
    tennis/
    bmx-trees/
    wild_001/
  gt_frames/           # optional clean-background GT frames for PSNR/SSIM
    tennis/
```

**Mandatory course datasets** (`bmx-trees`, `tennis`, plus your **Wild Video**) should follow the same `<dataset>` folder naming so evaluation commands stay copy-pasteable.

Ground-truth mask naming must align frame-by-frame with predicted masks (same stem ordering as PNG sequences).
