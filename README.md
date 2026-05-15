# Video Object Removal & Inpainting (AIAA 3201 / Introduction to Computer Vision)

Spring 2026 term project: remove dynamic objects from video and inpaint clean backgrounds using temporal cues. This repository bundles **our pipeline scripts**, **quantitative evaluation**, and **pinned third-party implementations** (SAM 2/3, ProPainter, utilities).

https://github.com/LIU-Qiuyue/ICVFinalProject.git

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Project scripts: frame extraction, Part 1 baseline, SAM 3 masks, mask merging |
| `evaluation/` | Mask/video metrics (IoU / JM–JR-style reporting, PSNR, SSIM), batch eval, qualitative grids |
| `configs/eval_config.json` | Default paths for batch evaluation |
| `third_party/sam2/` | SAM 2 (video segmentation) — **checkpoints not included** (see below) |
| `third_party/sam3/` | SAM 3 / SAM 3.1 code vendored for `src/sam3_video_masks.py` |
| `third_party/propainter/` | ProPainter inference code — **weights not included** (GitHub file-size limits) |
| `third_party/cc_torch/` | Connected-components helper used by some segmentation stacks |
| `data/` | Placeholder layout for frames, videos, and ground truth (see `data/README.md`) |
| `docs/assets/` | Recommended location for **visual results** showcased in this README |
| `outputs/` | Default directory for masks, inpainted frames/videos, and metric JSON (gitignored) |

Third-party projects retain their original licenses under `third_party/*/LICENSE` or upstream notices.

---

## Dependencies

We recommend **separate conda environments** because SAM 3, SAM 2, ProPainter, and Ultralytics often pin different PyTorch builds.

### Core (evaluation + light utilities)

From the repository root:

```bash
pip install -r requirements.txt
```

This installs NumPy/OpenCV/Matplotlib/Pandas/Ultralytics (Ultralytics pulls PyTorch suitable for many setups).

### SAM 3 (Part 3 extension — text/point/box video masks)

Follow `third_party/sam3/README.md`. Typical steps:

```bash
pip install -e third_party/sam3
huggingface-cli login   # access facebook/sam3 or facebook/sam3.1
```

### SAM 2 (Part 2 — interactive video masks)

Install per `third_party/sam2/README.md` or `INSTALL.md`. Download checkpoints into `third_party/sam2/checkpoints/` (not shipped here).

### ProPainter (video inpainting)

Use the environment instructions in `third_party/propainter/README.md`, then download weights into `third_party/propainter/weights/` (see below).

---

## Model weights (mandatory to obtain locally)

| Component | What to download | Where it goes |
|-----------|------------------|----------------|
| YOLOv8-seg | Pulled automatically on first run (`yolov8s-seg.pt` default) | Ultralytics cache or explicit `--yolo_model` path |
| SAM 3 / 3.1 | Hugging Face `facebook/sam3` or `facebook/sam3.1` | HF hub cache after `huggingface-cli login` |
| SAM 2 | Official checkpoint files | `third_party/sam2/checkpoints/` |
| ProPainter | `ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` | `third_party/propainter/weights/` |

See `third_party/propainter/weights/README.md` for the expected filenames.

---

## Data layout

Place course and custom assets under `data/` as described in **`data/README.md`**.

Default evaluation config (`configs/eval_config.json`) expects:

- **Predicted masks:** `outputs/masks/{part1|part2|part3}/<dataset>/`
- **Predicted inpainting:** `outputs/inpaint/{part1|part2|part3}/<dataset>/` (PNG sequences or paths your evaluation commands use)
- **Ground-truth masks:** `data/GroundTruth/<dataset>/`
- **Optional clean GT frames** (for PSNR/SSIM): `data/gt_frames/<dataset>/` — if missing, video metrics are skipped with a clear message.

---

## Usage

Run all commands from the **repository root** (`ICVFinalProject/`).

### 1) Extract frames from an MP4

```bash
python src/extract_frames.py \
  --video data/videos/tennis.mp4 \
  --out_dir data/frames/tennis
```

### 2) Part 1 — classical baseline (YOLOv8-seg + optical flow + OpenCV inpaint)

```bash
python src/part1_baseline.py \
  --frames_dir data/frames/tennis \
  --out_mask_dir outputs/masks/part1/tennis \
  --out_inpaint_dir outputs/inpaint/part1/tennis
```

### 3) Part 2 — SAM 2 masks (example)

Use the notebook/script shipped with SAM 2, for example under `third_party/sam2/notebooks/`, pointing `--video-dir` at `data/frames/<dataset>` and `--mask-out-dir` at `outputs/masks/part2/<dataset>`.

### 4) Part 3 — SAM 3 / SAM 3.1 masks (text prompt)

```bash
python src/sam3_video_masks.py \
  --resource data/frames/tennis \
  --text "person" \
  --frame-index 0 \
  --mask-out-dir outputs/masks/sam3/tennis \
  --version sam3.1
```

You can pass a video file instead of a frame folder with `--resource data/videos/tennis.mp4`.

### 5) Merge two mask directories (e.g., union)

```bash
python src/merge_masks.py \
  --mask-dir-a outputs/masks/part3/tennis \
  --mask-dir-b outputs/masks/sam3/tennis \
  --output-dir outputs/masks/merged/tennis \
  --mode union
```

### 6) ProPainter inpainting

From `third_party/propainter/` (see upstream README for exact flags):

```bash
python inference_propainter.py \
  --video ../data/frames/tennis \
  --mask ../outputs/masks/part3/tennis \
  --output ../outputs/inpaint/part3
```

### 7) Evaluation — single run

```bash
python -m evaluation.evaluate \
  --method-name part2 \
  --dataset-name bmx-trees \
  --pred-mask-dir outputs/masks/part2/bmx-trees \
  --pred-frame-dir outputs/inpaint/part2/bmx-trees \
  --gt-mask-dir data/GroundTruth/bmx-trees \
  --gt-frame-dir data/gt_frames/bmx-trees \
  --output-json outputs/evaluation/part2_bmx-trees_metrics.json
```

Add `--masked-psnr` for masked PSNR on the inpainted region.

### 8) Evaluation — batch (`configs/eval_config.json`)

```bash
python -m evaluation.evaluate_all --config configs/eval_config.json
```

Edit the JSON to swap datasets, methods, roots, IoU threshold, or enable `"masked_psnr": true`.

### 9) Qualitative comparison figure

Requires PNG sequences for results (not only MP4):

```bash
python -m evaluation.visualize_comparison \
  --dataset-name bmx-trees \
  --input-frame-dir data/frames/bmx-trees \
  --mask-dirs outputs/masks/part1/bmx-trees outputs/masks/part2/bmx-trees outputs/masks/part3/bmx-trees \
  --result-frame-dirs outputs/inpaint/part1/bmx-trees outputs/inpaint/part2/bmx-trees outputs/inpaint/part3/bmx-trees \
  --method-names Part1 Part2 Part3 \
  --frame-indices 0 10 20 30 \
  --output-dir docs/assets/comparisons
```

---

## Visual results for this README

Add side-by-side frames or short GIFs under **`docs/assets/`** and link them here, for example:

```markdown
![Comparison](docs/assets/teaser.png)
```

Course submissions also require processed videos for mandatory datasets via Canvas (`videos.zip`); hosting large binaries on GitHub is optional (prefer Releases or external links).

---

## Metrics (course alignment)

- **Mask quality:** **JM** (mean per-frame IoU) and **JR** (fraction of frames with IoU ≥ threshold, default 0.5) — see `evaluation/metrics.py` and `evaluation/evaluate.py`.
- **Video quality:** PSNR / SSIM when clean GT frames exist under `data/gt_frames/`.

---

## Troubleshooting

- **CUDA / PyTorch mismatches:** use separate conda envs per third-party README.
- **`ModuleNotFoundError: sam3`:** `pip install -e third_party/sam3` or ensure `third_party/sam3` is on `PYTHONPATH`.
- **Large artifacts:** never commit videos, extracted frames, or `.pth` checkpoints; keep them local or use Git LFS/releases.

---

## Citation

Cite the original papers for YOLO, SAM 2/3, ProPainter, and other methods you use; see the course project handout reference list.
