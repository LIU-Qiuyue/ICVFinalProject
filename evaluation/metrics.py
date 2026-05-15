"""Metric helpers: masks (IoU, JM, JR) and video (PSNR, SSIM)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def binarize_mask(mask: np.ndarray, mode: str = "auto", fixed_threshold: int = 127) -> np.ndarray:
    """
    Foreground boolean mask.

    - ``nonzero``: ``mask > 0``
    - ``fixed127``: ``mask > fixed_threshold`` (default 127)
    - ``auto``:
        - ``mask.max() <= 1``: ``mask > 0``
        - few labels (<=20 unique) and ``max <= 20``: indexed map, ``mask > 0``
        - few labels (<=32 unique) and ``max < 128``: DAVIS-style indexed PNG (e.g. 0,38,75), ``mask > 0``
        - otherwise: ``mask > fixed_threshold`` (typical 0/255 binaries)
    """
    if mode not in ("auto", "nonzero", "fixed127"):
        raise ValueError(f"Unknown binarize mode {mode!r}; expected auto|nonzero|fixed127.")
    m = np.asarray(mask)
    if m.dtype == bool:
        return m.astype(bool)
    if mode == "nonzero":
        return m > 0
    if mode == "fixed127":
        return m > fixed_threshold

    mx = float(np.nanmax(m)) if m.size else 0.0
    nu = int(np.unique(m).size)
    if mx <= 1.0:
        return m > 0
    if nu <= 20 and mx <= 20.0:
        return m > 0
    if nu <= 32 and mx < 128.0:
        return m > 0
    return m > fixed_threshold


def natural_image_sort_key(path: Path) -> tuple:
    """Sort by numeric frame index when the stem is all digits; else lexicographic."""
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.suffix.lower())
    return (1, stem.lower(), path.suffix.lower())


def _list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Expected a directory of images at {directory.resolve()}, but it is missing or not a directory."
        )
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=natural_image_sort_key)
    return files


def sorted_image_paths(directory: str | Path) -> list[Path]:
    """List image files under ``directory`` using the same ordering as metrics."""
    return _list_image_files(Path(directory))


def load_binary_mask(
    path: str | Path,
    mode: str = "auto",
    threshold: int = 127,
) -> np.ndarray:
    """Load mask as grayscale (or first channel) and binarize with :func:`binarize_mask`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Mask file not found: {path.resolve()}. Cannot load binary mask."
        )
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"Failed to read mask image (OpenCV returned None): {path.resolve()}."
        )
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return binarize_mask(np.asarray(img), mode=mode, fixed_threshold=threshold)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and return RGB uint8 array, shape (H, W, 3)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {path.resolve()}. Cannot load RGB image."
        )
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"Failed to read image (OpenCV returned None): {path.resolve()}."
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_mask_gray(path: Path) -> np.ndarray:
    """Load single-channel mask array (no binarization)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path.resolve()}.")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"Failed to read mask image (OpenCV returned None): {path.resolve()}."
        )
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.asarray(img)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU for two boolean masks. Both-empty => 1.0; otherwise standard IoU."""
    if pred_mask.dtype != bool:
        pred_mask = pred_mask.astype(bool)
    if gt_mask.dtype != bool:
        gt_mask = gt_mask.astype(bool)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Mask shape mismatch for IoU: pred {pred_mask.shape} vs gt {gt_mask.shape}."
        )

    pred_empty = not np.any(pred_mask)
    gt_empty = not np.any(gt_mask)
    if pred_empty and gt_empty:
        return 1.0

    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def compute_jm_jr(
    pred_mask_dir: str | Path,
    gt_mask_dir: str | Path,
    iou_threshold: float = 0.5,
    pred_binarization_mode: str = "auto",
    gt_binarization_mode: str = "auto",
    fixed_threshold: int = 127,
) -> dict:
    """Pair frames by sorted stems; resize **pred** to **GT** shape with INTER_NEAREST, then binarize."""
    pred_mask_dir = Path(pred_mask_dir)
    gt_mask_dir = Path(gt_mask_dir)
    pred_files = _list_image_files(pred_mask_dir)
    gt_files = _list_image_files(gt_mask_dir)
    if len(pred_files) != len(gt_files):
        raise ValueError(
            f"Frame count mismatch: {len(pred_files)} prediction masks in {pred_mask_dir} "
            f"vs {len(gt_files)} GT masks in {gt_mask_dir}. Check filenames and padding."
        )

    mask_alignment: dict | None = None
    ious: list[float] = []
    for idx, (pred_path, gt_path) in enumerate(zip(pred_files, gt_files)):
        pred = _load_mask_gray(pred_path)
        gt = _load_mask_gray(gt_path)
        pred_hw = tuple(int(x) for x in pred.shape[:2])
        gt_hw = tuple(int(x) for x in gt.shape[:2])
        if pred_hw != gt_hw:
            gh, gw = gt_hw[0], gt_hw[1]
            pred = cv2.resize(pred, (gw, gh), interpolation=cv2.INTER_NEAREST)
            if mask_alignment is None:
                mask_alignment = {
                    "resized_pred_to_gt": True,
                    "interpolation": "INTER_NEAREST",
                    "dsize_wh": [gw, gh],
                    "target_shape_hw": [gh, gw],
                    "example_pair_index": idx,
                    "example_pred_path": str(pred_path),
                    "pred_shape_before_hw": list(pred_hw),
                }

        pred_mask = binarize_mask(
            pred, mode=pred_binarization_mode, fixed_threshold=fixed_threshold
        )
        gt_mask = binarize_mask(
            gt, mode=gt_binarization_mode, fixed_threshold=fixed_threshold
        )
        ious.append(compute_iou(pred_mask, gt_mask))

    if mask_alignment is None:
        mask_alignment = {"resized_pred_to_gt": False}

    meta = {
        "pred_binarization_mode": pred_binarization_mode,
        "gt_binarization_mode": gt_binarization_mode,
        "pred_threshold_mode": pred_binarization_mode,
        "gt_threshold_mode": gt_binarization_mode,
        "fixed_threshold": fixed_threshold,
        "mask_alignment": mask_alignment,
    }

    if not ious:
        return {
            "num_frames": 0,
            "JM": float("nan"),
            "JR": float("nan"),
            "mean_iou": float("nan"),
            "ious": [],
            **meta,
        }

    jm = float(np.mean(ious))
    jr = float(np.mean([1.0 if i >= iou_threshold else 0.0 for i in ious]))
    return {
        "num_frames": len(ious),
        "JM": jm,
        "JR": jr,
        "mean_iou": jm,
        "ious": ious,
        **meta,
    }


def _to_float01(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float64) / 255.0
    return rgb.astype(np.float64)


def compute_psnr_ssim(
    restored_dir: str | Path,
    gt_frame_dir: str | Path,
    mask_dir: str | Path | None = None,
    pred_mask_threshold_mode: str = "auto",
    pred_mask_fixed_threshold: int = 127,
) -> dict:
    """Full-frame PSNR/SSIM vs GT; optional masked PSNR (MSE inside foreground only)."""
    restored_dir = Path(restored_dir)
    gt_frame_dir = Path(gt_frame_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else None

    rest_files = _list_image_files(restored_dir)
    gt_files = _list_image_files(gt_frame_dir)
    if len(rest_files) != len(gt_files):
        raise ValueError(
            f"Frame count mismatch: {len(rest_files)} restored frames in {restored_dir} "
            f"vs {len(gt_files)} GT frames in {gt_frame_dir}."
        )

    mask_files: list[Path] | None = None
    if mask_dir is not None:
        mask_files = _list_image_files(mask_dir)
        if len(mask_files) != len(rest_files):
            raise ValueError(
                f"Frame count mismatch for masks: {len(mask_files)} in {mask_dir} "
                f"vs {len(rest_files)} restored frames."
            )

    psnr_per_frame: list[float] = []
    ssim_per_frame: list[float] = []
    masked_psnr_per_frame: list[float] = []
    valid_masked: list[float] = []

    for idx, (rest_path, gt_path) in enumerate(zip(rest_files, gt_files)):
        rest = load_rgb_image(rest_path)
        gt = load_rgb_image(gt_path)
        if gt.shape[:2] != rest.shape[:2]:
            gt = cv2.resize(
                gt,
                (rest.shape[1], rest.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        rest_f = _to_float01(rest)
        gt_f = _to_float01(gt)

        psnr_pf = float(peak_signal_noise_ratio(gt_f, rest_f, data_range=1.0))
        ssim_pf = float(
            structural_similarity(
                gt_f,
                rest_f,
                data_range=1.0,
                channel_axis=-1,
            )
        )
        psnr_per_frame.append(psnr_pf)
        ssim_per_frame.append(ssim_pf)

        if mask_files is None:
            masked_psnr_per_frame.append(float("nan"))
            continue

        mpath = mask_files[idx]
        fg = load_binary_mask(
            mpath,
            mode=pred_mask_threshold_mode,
            threshold=pred_mask_fixed_threshold,
        )
        if fg.shape[:2] != rest.shape[:2]:
            fg = cv2.resize(
                fg.astype(np.uint8),
                (rest.shape[1], rest.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if not np.any(fg):
            masked_psnr_per_frame.append(float("nan"))
            continue

        diff = (rest_f - gt_f) ** 2
        mse = float(diff[fg].mean())
        if mse <= 0.0:
            m_psnr = float("inf")
        else:
            m_psnr = float(10.0 * math.log10(1.0 / mse))
        masked_psnr_per_frame.append(m_psnr)
        if math.isfinite(m_psnr):
            valid_masked.append(m_psnr)

    num = len(psnr_per_frame)
    out: dict = {
        "num_frames": num,
        "PSNR": float(np.mean(psnr_per_frame)) if num else float("nan"),
        "SSIM": float(np.mean(ssim_per_frame)) if num else float("nan"),
        "psnr_per_frame": psnr_per_frame,
        "ssim_per_frame": ssim_per_frame,
        "masked_PSNR": float(np.mean(valid_masked)) if valid_masked else None,
        "masked_psnr_per_frame": masked_psnr_per_frame,
    }
    return out
