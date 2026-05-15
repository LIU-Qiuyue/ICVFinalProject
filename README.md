# Video Object Removal and Inpainting

**AIAA 3201 — Introduction to Computer Vision (Spring 2026)**

Public repository: [https://github.com/LIU-Qiuyue/ICVFinalProject](https://github.com/LIU-Qiuyue/ICVFinalProject)

---

## 1. Project Overview

This project addresses **video object removal and inpainting**: given an input video, we

1. **Extract frames** from the video,
2. **Generate dynamic object masks** across time,
3. **Remove** the target object(s), and
4. **Restore** missing regions with inpainting.

The codebase is organized into three course parts:

| Part | Goal |
|------|------|
| **Part 1** | Classical / hand-crafted baseline (YOLOv8-seg, optical flow, OpenCV inpainting) |
| **Part 2** | SOTA reproduction — SAM 2 masks + ProPainter inpainting |
| **Part 3** | Exploration — SAM 3 / SAM 3.1 prompts, mask merging, and extensions |

All first-party scripts live under `src/` and `evaluation/`. Heavy models are vendored under `third_party/` and must be installed and downloaded separately (see below).

---

## 2. Pipeline Overview

```text
input video
    → frame extraction          (src/extract_frames.py)
    → mask generation           (Part 1 / SAM 2 / SAM 3)
    → optional mask refinement  (src/merge_masks.py)
    → inpainting                (Part 1 OpenCV / ProPainter)
    → evaluation                (evaluation/)
    → visualization             (evaluation/visualize_comparison.py)
```

- **Part 1:** `src/part1_baseline.py` — end-to-end masks + inpainted PNGs in one command.
- **Part 2:** Interactive **SAM 2** tracking (notebook under `third_party/sam2/notebooks/`) → masks under `outputs/masks/part2/<dataset>/`, then **ProPainter** (`third_party/propainter/inference_propainter.py`).
- **Part 3:** **SAM 3 / 3.1** open-vocabulary masks via `src/sam3_video_masks.py`; optional union with other masks via `src/merge_masks.py`.

> **TODO:** There is no dedicated `src/part2_*.py` wrapper. Part 2 mask export is performed through the SAM 2 notebook/script; document your exact export steps in your report if you customize them.

---

## 3. Repository Structure

```text
ICVFinalProject/
├── README.md
├── LICENSE
├── requirements.txt          # Core Python deps (Part 1 + evaluation)
├── configs/
│   └── eval_config.json      # Batch evaluation paths and thresholds
├── src/
│   ├── extract_frames.py     # Video → PNG sequence
│   ├── part1_baseline.py     # Part 1: YOLO + flow + OpenCV inpaint
│   ├── sam3_video_masks.py     # Part 3: SAM 3 / 3.1 video masks
│   └── merge_masks.py        # Union / intersection of two mask folders
├── evaluation/
│   ├── metrics.py            # IoU, JM, JR, PSNR, SSIM
│   ├── evaluate.py           # Single method × dataset CLI
│   ├── evaluate_all.py       # Batch eval from JSON config
│   └── visualize_comparison.py
├── data/
│   └── README.md             # Expected local data layout (gitignored binaries)
├── docs/
│   └── assets/               # README figures (place PNGs here)
├── outputs/                  # Masks, inpaints, metrics (gitignored; .gitkeep only)
└── third_party/
    ├── sam2/                 # SAM 2 (Part 2 masks)
    ├── sam3/                 # SAM 3 / 3.1 (Part 3 masks)
    ├── propainter/           # ProPainter inference (Part 2 inpainting)
    └── cc_torch/             # Connected-components helper (upstream dep)
```

| Path | Role |
|------|------|
| `src/` | Runnable project pipelines (not third-party training code). |
| `evaluation/` | Quantitative and qualitative evaluation; mask binarization supports DAVIS-style indexed GT (`auto` mode). |
| `configs/eval_config.json` | Default dataset list, path templates, IoU threshold, mask threshold modes. |
| `data/` | Local videos, frames, and ground truth (not committed). See `data/README.md`. |
| `outputs/` | Generated masks, inpainted frames/videos, evaluation JSON, figures (gitignored). |
| `third_party/` | Upstream SAM 2/3 and ProPainter; install per their READMEs. |

---

## 4. Environment Setup

We recommend **separate conda environments** because Ultralytics, SAM 2, SAM 3, and ProPainter often require different PyTorch/CUDA builds.

### 4.1 Core environment (Part 1 + evaluation)

```bash
conda create -n icv-core python=3.10 -y
conda activate icv-core
pip install -r requirements.txt
```

`requirements.txt` installs: `numpy`, `opencv-python`, `scikit-image`, `pandas`, `matplotlib`, `pillow`, `ultralytics` (pulls PyTorch on many systems).

Run commands from the **repository root** (the folder that contains `src/` and `evaluation/`).

### 4.2 SAM 3 (Part 3)

Upstream notes: **Python 3.12+**, CUDA GPU, recent PyTorch. From repo root:

```bash
conda create -n icv-sam3 python=3.12 -y
conda activate icv-sam3
pip install -e third_party/sam3
huggingface-cli login    # access facebook/sam3 or facebook/sam3.1
```

See `third_party/sam3/README.md` for full install and GPU notes.

### 4.3 SAM 2 (Part 2)

Per `third_party/sam2/README.md` and `INSTALL.md`:

- Python ≥ 3.10, PyTorch ≥ 2.5.1 (install matching your CUDA build from [pytorch.org](https://pytorch.org/)).
- `pip install -e third_party/sam2` (may compile a small CUDA extension; see upstream FAQ if build warnings appear).
- Optional notebooks: `pip install -e "third_party/sam2[notebooks]"`.

### 4.4 ProPainter (Part 2 inpainting)

Follow `third_party/propainter/README.md` for its conda/env and PyTorch version. Weights go under `third_party/propainter/weights/` (see §5).

---

## 5. Model Weights / Checkpoints

**Do not commit large checkpoints to GitHub.** Download locally into the paths below (or use Hugging Face / Ultralytics caches where noted).

| Component | Files | Where to place |
|-----------|--------|----------------|
| **YOLOv8-seg** (Part 1) | `yolov8s-seg.pt` (default) | Auto-downloaded by Ultralytics on first run, or pass `--yolo_model /path/to/model.pt` |
| **SAM 2** (Part 2) | e.g. `sam2.1_hiera_tiny.pt`, `sam2.1_hiera_small.pt`, `sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_large.pt` | `third_party/sam2/checkpoints/` — run `cd third_party/sam2/checkpoints && ./download_ckpts.sh` per upstream README |
| **SAM 3 / 3.1** (Part 3) | HF weights `facebook/sam3` or `facebook/sam3.1` | Hugging Face cache after `huggingface-cli login`; optional local `.pt` via `--checkpoint` |
| **ProPainter** (Part 2) | `ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` | `third_party/propainter/weights/` — see `third_party/propainter/weights/README.md` |

Optional consolidated layout (if you prefer symlinks outside `third_party/`):

```text
checkpoints/          # optional; not created by default in this repo
  sam2/               # → symlink to third_party/sam2/checkpoints/
  sam3/               # → local .pt or HF cache
  propainter/         # → symlink to third_party/propainter/weights/
  yolo/               # → yolov8s-seg.pt
```

---

## 6. Dataset Preparation

Large assets are **gitignored**. Create this layout locally (details in `data/README.md`):

```text
data/
  videos/                    # optional source MP4s
    tennis.mp4
    bmx-trees.mp4
    wild_001.mp4
  frames/                    # extracted PNG sequences
    tennis/
    bmx-trees/
    wild_001/
    wild_002/
    wild_003/
    davis/                   # optional DAVIS sequences
      bus/
      bmx-trees/
  GroundTruth/               # GT masks for JM / JR (course layout)
    tennis/
    bmx-trees/
    bus/                     # optional DAVIS
  gt_frames/                 # optional clean frames for PSNR / SSIM
    tennis/
```

**Mandatory course datasets:** **Wild Video** (`wild_001`, `wild_002`, `wild_003`, or your chosen wild clip names), **bmx-trees**, **tennis**.

**Optional:** DAVIS sequences under `data/frames/davis/<sequence>/` with matching GT under `data/GroundTruth/<sequence>/` when available.

### Extract frames

```bash
python src/extract_frames.py \
  --video data/videos/tennis.mp4 \
  --out_dir data/frames/tennis
```

Output naming: `000000.png`, `000001.png`, … (six-digit stems).

> **Note:** Some GT mask folders use five-digit names (`00000.png`) while predictions may use six digits; `evaluation/` pairs frames by sorted numeric order and resizes masks when resolutions differ.

---

## 7. Usage

Run from the **repository root**.

### Part 1 — Classical baseline

**Method:** YOLOv8 instance segmentation on dynamic COCO classes, Lucas–Kanade optical-flow gating, mask dilation, temporal background borrowing, OpenCV inpainting.

```bash
python src/part1_baseline.py \
  --frames_dir data/frames/bmx-trees \
  --out_mask_dir outputs/masks/part1/bmx-trees \
  --out_inpaint_dir outputs/inpaint/part1/bmx-trees
```

**Outputs:**

- Masks: `outputs/masks/part1/<dataset>/*.png`
- Inpainted frames: `outputs/inpaint/part1/<dataset>/*.png`

Useful flags: `--yolo_model`, `--flow_thresh`, `--dilation_ksize`, `--temporal_radius` (see `python src/part1_baseline.py -h`).

---

### Part 2 — SAM 2 + ProPainter

#### Mask generation (SAM 2)

There is **no** `src/part2_*.py` in this repository. Use the upstream interactive video predictor:

- Notebook: `third_party/sam2/notebooks/video_predictor_example.ipynb`
- Script export: `third_party/sam2/notebooks/video_predictor_example.py`

**Workflow (high level):**

1. Install SAM 2 and download checkpoints (§4.3, §5).
2. Point the notebook at `data/frames/<dataset>` (JPEG folder or video).
3. Provide **point or box prompts** on a reference frame; propagate through the video.
4. Export binary/grayscale masks to:

   `outputs/masks/part2/<dataset>/`

**Prompt tips (from our experiments):**

- Point prompts on the moving object can **miss attached regions** (backpacks, vehicles).
- **Box prompts** are often more reliable for articulated or compound objects.

> **TODO:** Add a short subsection in your report documenting how you exported SAM 2 masks to PNG if you use a custom notebook cell.

#### Inpainting (ProPainter)

From repository root:

```bash
cd third_party/propainter

python inference_propainter.py \
  -i ../../data/frames/bmx-trees \
  -m ../../outputs/masks/part2/bmx-trees \
  -o ../../outputs/inpaint/part2/bmx-trees \
  --save_frames
```

| Flag | Meaning |
|------|---------|
| `-i` / `--video` | Frame folder or video path |
| `-m` / `--mask` | Mask folder (one mask per frame) |
| `-o` / `--output` | Output directory (video; add `--save_frames` for PNG sequence) |

**Outputs:** `outputs/inpaint/part2/<dataset>/` (MP4 and/or PNGs depending on flags).

---

### Part 3 — SAM 3 exploration

**Method:** Open-vocabulary video segmentation with text, point, or box prompts on a reference frame, then temporal propagation.

```bash
python src/sam3_video_masks.py \
  --resource data/frames/bmx-trees \
  --text "person" \
  --frame-index 0 \
  --mask-out-dir outputs/masks/part3/bmx-trees \
  --version sam3.1
```

You may also write to `outputs/masks/sam3/<dataset>/` for experiments outside the `part3` folder name.

**Alternative prompts** (see `python src/sam3_video_masks.py -h`):

- `--points` / `--point-labels` — relative coords by default (`--rel-coords`)
- `--boxes` / `--box-labels`
- `--resource data/videos/tennis.mp4` — video file instead of a frame folder

**Observations:**

- Simple text prompts (e.g. `"person"`, `"bus"`) often work well.
- **Compound** prompts (e.g. `"person and vehicle"`) may fail or segment inconsistently.
- **Box prompts** tend to be more stable than **point** prompts when objects have attached parts or multiple instances move together.

#### Optional mask merge (Part 3 extension)

```bash
python src/merge_masks.py \
  --mask-dir-a outputs/masks/part3/bmx-trees \
  --mask-dir-b outputs/masks/sam3/bmx-trees \
  --output-dir outputs/masks/merged/bmx-trees \
  --mode union
```

Then run ProPainter on `outputs/masks/merged/<dataset>/` if desired.

---

## 8. Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **JM** | Mean per-frame mask **IoU** (Jaccard) vs GT |
| **JR** | **IoU recall** — fraction of frames with IoU ≥ threshold (default **0.5**) |
| **PSNR / SSIM** | Full-frame video quality vs clean GT frames |
| **Qualitative** | Comparison grids via `visualize_comparison.py` |

**Important:**

- Datasets **without** GT masks: mask metrics are skipped (no forced numbers).
- **PSNR / SSIM** run only when `data/gt_frames/<dataset>/` exists; otherwise the CLI prints a skip message.
- GT masks may be **DAVIS-style indexed PNGs** (labels &lt; 128). Evaluation uses `mask_threshold_mode: auto` (see `configs/eval_config.json`) so foreground is `> 0` when appropriate.

### Single run

Mask-only example (no clean GT frames):

```bash
python -m evaluation.evaluate \
  --method-name part2 \
  --dataset-name bmx-trees \
  --pred-mask-dir outputs/masks/part2/bmx-trees \
  --pred-frame-dir data/frames/bmx-trees \
  --gt-mask-dir data/GroundTruth/bmx-trees \
  --output-json outputs/evaluation/part2_bmx-trees_metrics.json
```

With inpainted frames and optional PSNR/SSIM:

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

Optional: `--masked-psnr`, `--mask-threshold-mode auto|nonzero|fixed127`, `--gt-mask-threshold-mode`.

> **Note:** `evaluate.py` requires `--pred-frame-dir` to exist. For mask-only runs, point it at `data/frames/<dataset>` if inpaint output is missing.

### Batch evaluation

```bash
python -m evaluation.evaluate_all --config configs/eval_config.json
```

Writes per-pair JSON under `outputs/evaluation/` and `outputs/evaluation/summary_metrics.csv`.

> **Note:** `evaluate_all.py` skips a pair if **either** predicted mask **or** predicted frame directory is missing. For mask-only batch runs, ensure `pred_frame_template` points to an existing folder (e.g. `data/frames/...`) or create placeholder inpaint dirs.

### Qualitative comparison

Requires **PNG** sequences (not MP4-only folders):

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

## 9. Results

### Quantitative (mask quality, IoU threshold = 0.5)

JM / JR are from local evaluation (`outputs/evaluation/summary_metrics.csv` or `*_metrics.json`). PSNR / SSIM are **—** when clean GT frames under `data/gt_frames/` were not used for that run.

| Dataset | Method | JM | JR | 
|---------|--------|----|----|
| bmx-trees | Part 1 baseline | 0.6212035774056812 | 0.8375 |
| bmx-trees | Part 2 SAM2+ProPainter | 0.6392602001488565 | 0.875 | 
| bmx-trees | Part 3 SAM3+ProPainter | 0.6212035774056812 | 0.8375 | 
| tennis | Part 1 baseline | 0.4119005931256472 | 0.45714285714285713 | 
| tennis | Part 2 SAM2+ProPainter | 0.7025602564975957 | 0.8142857142857143 |
| tennis | Part 3 SAM3+ProPainter | 0.7049705302610878 | 0.8142857142857143 |
| davis-bus | Part 1 baseline | 0.7934773356344085 | 1.0 | 
| davis-bus | Part 2 SAM2+ProPainter | 0.9291797638068606 | 1.0 | 
| davis-bus | Part 3 SAM3+ProPainter | 0.9288134854387196 | 1.0 | 


Wild Video rows typically have **no GT masks** (qualitative only). 

### Qualitative figures
Included in our report and uploaded to Canvas.


## 10. Output Videos
Included in our submission and uploaded to Canvas.


---


## 11. Acknowledgements

This project builds on and vendors code from:

- **Ultralytics YOLOv8** (instance segmentation, Part 1)
- **OpenCV** inpainting and optical flow (Part 1)
- **[SAM 2](https://github.com/facebookresearch/sam2)** — video segmentation (Part 2)
- **[SAM 3](https://github.com/facebookresearch/sam3)** — open-vocabulary video masks (Part 3)
- **[ProPainter](https://github.com/sczhou/ProPainter)** — video inpainting (Part 2)
- **DAVIS** benchmark (optional evaluation sequences and indexed masks)

See upstream READMEs and licenses under `third_party/`. Cite the original papers in your course report as required by the project handout.

---

## License

See [LICENSE](LICENSE). Third-party components retain their respective licenses.
