#!/usr/bin/env python3
"""
Part 1 baseline: YOLOv8-seg + Lucas–Kanade optical-flow filtering,
mask dilation, temporal background borrowing, and OpenCV inpainting.

Example (run from repository root):

    python src/part1_baseline.py \\
      --frames_dir data/frames/wild_003 \\
      --out_mask_dir outputs/masks/part1/wild_003 \\
      --out_inpaint_dir outputs/inpaint/part1/wild_003
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# COCO class ids that can be dynamic (person, bicycle, car, motorcycle,
# bus, truck, bird, cat, dog, horse, sheep, cow, bear, ...)
DYNAMIC_CLASS_IDS: set[int] = {
    0, 1, 2, 3, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
}

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)

FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 1 baseline: dynamic object removal via YOLO + optical flow."
    )
    parser.add_argument(
        "--frames_dir",
        required=True,
        type=str,
        help="Input frame directory (PNG sequence), e.g. data/frames/tennis",
    )
    parser.add_argument(
        "--out_mask_dir",
        required=True,
        type=str,
        help="Output binary mask directory, e.g. outputs/masks/part1/tennis",
    )
    parser.add_argument(
        "--out_inpaint_dir",
        required=True,
        type=str,
        help="Output inpainted frame directory, e.g. outputs/inpaint/part1/tennis",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLOv8 seg model weight name or path (default: yolov8s-seg.pt)",
    )
    parser.add_argument(
        "--flow_thresh",
        type=float,
        default=1.5,
        help="Mean LK flow magnitude threshold for dynamic judgement (default: 1.5 px)",
    )
    parser.add_argument(
        "--dilation_ksize",
        type=int,
        default=15,
        help="Dilation kernel size for mask expansion (default: 15)",
    )
    parser.add_argument(
        "--temporal_radius",
        type=int,
        default=10,
        help="Max frame offset to search for temporal background borrowing (default: 10)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def load_frames(frames_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    exts = {".png", ".jpg", ".jpeg"}
    names = sorted(
        p.name for p in frames_dir.iterdir() if p.suffix.lower() in exts
    )
    if not names:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    frames = [cv2.imread(str(frames_dir / n)) for n in names]
    return frames, names


def detect_masks(
    frame: np.ndarray, model: YOLO, conf_thresh: float = 0.4
) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) of all DYNAMIC_CLASS_IDS detections."""
    h, w = frame.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    results = model.predict(source=frame, conf=conf_thresh, verbose=False)
    if not results:
        return combined

    r = results[0]
    if r.masks is None or r.boxes is None:
        return combined

    for i, cls_id in enumerate(r.boxes.cls.cpu().numpy().astype(int)):
        if cls_id not in DYNAMIC_CLASS_IDS:
            continue
        seg_mask = r.masks.data[i].cpu().numpy()  # float32, [0,1], model resolution
        seg_mask_resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, (seg_mask_resized > 0.5).astype(np.uint8) * 255)

    return combined


def is_dynamic(
    mask: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    flow_thresh: float,
) -> bool:
    """Estimate mean LK flow inside mask region; return True if dynamic."""
    ys, xs = np.where(mask > 127)
    if len(ys) < 5:
        return False

    # Sample feature points inside mask
    pts = np.column_stack([xs, ys]).astype(np.float32)
    step = max(1, len(pts) // 200)
    pts = np.ascontiguousarray(pts[::step].reshape(-1, 1, 2))

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS
    )
    good = status.ravel() == 1
    if good.sum() < 2:
        return False

    flow = next_pts[good] - pts[good]
    mean_mag = float(np.linalg.norm(flow.reshape(-1, 2), axis=1).mean())
    return mean_mag > flow_thresh


def refine_mask(mask: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Fill small holes + dilate mask edges."""
    # Fill small internal holes using closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    # Dilate to cover motion blur and boundary leakage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(closed, kernel_dilate)
    return dilated


def temporal_borrow(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    t: int,
    radius: int,
) -> np.ndarray:
    """
    For masked pixels in frame t, look for clean pixels from nearby frames.
    Returns a reconstructed frame (pixels not borrowed remain original).
    """
    result = frames[t].copy()
    mask_t = masks[t]

    ys, xs = np.where(mask_t > 127)
    if len(ys) == 0:
        return result

    n = len(frames)
    for offset in range(1, radius + 1):
        for sign in (-1, 1):
            src_idx = t + sign * offset
            if src_idx < 0 or src_idx >= n:
                continue
            src_mask = masks[src_idx]

            # Only borrow from positions that are clean in the source frame
            borrow_ok = (src_mask < 128)
            still_missing = mask_t > 127

            copy_here = borrow_ok & still_missing
            result[copy_here] = frames[src_idx][copy_here]

            # Update remaining unfilled pixels
            mask_t = mask_t.copy()
            mask_t[copy_here] = 0

        if (mask_t > 127).sum() == 0:
            break

    return result


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Spatial inpainting fallback using Telea algorithm."""
    return cv2.inpaint(frame, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    out_mask_dir = Path(args.out_mask_dir)
    out_inpaint_dir = Path(args.out_inpaint_dir)

    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_inpaint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading frames from: {frames_dir}")
    frames, names = load_frames(frames_dir)
    n = len(frames)
    print(f"[INFO] Total frames: {n}")

    print(f"[INFO] Loading YOLO model: {args.yolo_model}")
    model = YOLO(args.yolo_model)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # ── Stage 1: Detect masks for each frame ──────────────────────────────────
    print("[INFO] Detecting masks with YOLOv8-seg ...")
    raw_masks: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        mask = detect_masks(frame, model)
        raw_masks.append(mask)
        if (i + 1) % max(1, n // 10) == 0:
            print(f"  [{i + 1}/{n}] detection done")

    # ── Stage 2: Optical flow dynamic filter ──────────────────────────────────
    print("[INFO] Filtering dynamic objects via LK optical flow ...")
    dynamic_masks: list[np.ndarray] = []
    for i in range(n):
        mask = raw_masks[i]
        if mask.max() == 0:
            dynamic_masks.append(mask)
            continue

        # Use adjacent frame pair
        if i == 0:
            prev_g, curr_g = grays[0], grays[min(1, n - 1)]
        else:
            prev_g, curr_g = grays[i - 1], grays[i]

        if is_dynamic(mask, prev_g, curr_g, args.flow_thresh):
            dynamic_masks.append(mask)
        else:
            dynamic_masks.append(np.zeros_like(mask))

    # ── Stage 3: Refine masks ─────────────────────────────────────────────────
    print("[INFO] Refining masks (close + dilate) ...")
    refined_masks: list[np.ndarray] = [
        refine_mask(m, args.dilation_ksize) if m.max() > 0 else m
        for m in dynamic_masks
    ]

    # Save masks
    for i, (m, name) in enumerate(zip(refined_masks, names)):
        stem = Path(name).stem
        cv2.imwrite(str(out_mask_dir / f"{stem}.png"), m)

    print(f"[INFO] Masks saved to: {out_mask_dir}")

    # ── Stage 4: Temporal borrowing + cv2.inpaint fallback ───────────────────
    print("[INFO] Inpainting frames ...")
    for i, (frame, name) in enumerate(zip(frames, names)):
        if refined_masks[i].max() == 0:
            # Nothing to remove
            cv2.imwrite(str(out_inpaint_dir / name), frame)
            continue

        # Temporal borrowing first
        borrowed = temporal_borrow(frames, refined_masks, i, args.temporal_radius)

        # Remaining unfilled pixels after borrowing
        still_masked = refined_masks[i].copy()
        diff = np.abs(borrowed.astype(np.int32) - frame.astype(np.int32))
        filled = (diff.sum(axis=2) > 0) | (still_masked < 128)
        remaining = (~filled) & (still_masked > 127)
        remaining_u8 = remaining.astype(np.uint8) * 255

        if remaining_u8.max() > 0:
            result = inpaint_frame(borrowed, remaining_u8)
        else:
            result = borrowed

        cv2.imwrite(str(out_inpaint_dir / name), result)

        if (i + 1) % max(1, n // 10) == 0:
            print(f"  [{i + 1}/{n}] inpainting done")

    print(f"[INFO] Inpainted frames saved to: {out_inpaint_dir}")
    print("[INFO] Part 1 baseline complete.")


if __name__ == "__main__":
    main()
