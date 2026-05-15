"""CLI: evaluate one method on one dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation.metrics import compute_jm_jr, compute_psnr_ssim


def _require_pred_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path.resolve()}. "
            "Fix the path or generate predictions before running evaluation."
        )
    if not path.is_dir():
        raise NotADirectoryError(
            f"{label} is not a directory: {path.resolve()}."
        )


def _gt_dir_usable(path: Path | None) -> bool:
    return path is not None and path.is_dir()


def _sanitize_for_json(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    return obj


def _print_summary(payload: dict) -> None:
    print()
    print(f"{'Method':<12} {payload.get('method', '')}")
    print(f"{'Dataset':<12} {payload.get('dataset', '')}")
    print("-" * 44)
    mm = payload.get("mask_metrics")
    vm = payload.get("video_metrics")
    if mm:
        print(f"{'JM':<12} {mm.get('JM'):.4f}")
        print(f"{'JR':<12} {mm.get('JR'):.4f}")
        print(f"{'Frames (mask)':<12} {mm.get('num_frames')}")
    else:
        print("Mask metrics: (skipped)")
    print("-" * 44)
    if vm:
        psnr = vm.get("PSNR")
        ssim = vm.get("SSIM")
        mpsnr = vm.get("masked_PSNR")
        psnr_s = f"{psnr:.4f}" if isinstance(psnr, (int, float)) and not math.isnan(psnr) else "—"
        ssim_s = f"{ssim:.4f}" if isinstance(ssim, (int, float)) and not math.isnan(ssim) else "—"
        print(f"{'PSNR':<12} {psnr_s}")
        print(f"{'SSIM':<12} {ssim_s}")
        if mpsnr is not None and isinstance(mpsnr, (int, float)) and not math.isnan(mpsnr):
            print(f"{'masked PSNR':<12} {mpsnr:.4f}")
        print(f"{'Frames (vid)':<12} {vm.get('num_frames')}")
    else:
        print("Video metrics: (skipped)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted masks and restored frames against optional GT."
    )
    parser.add_argument("--method-name", required=True, help="Method label, e.g. part1, part2.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name.")
    parser.add_argument("--pred-mask-dir", required=True, type=Path)
    parser.add_argument("--pred-frame-dir", required=True, type=Path)
    parser.add_argument("--gt-mask-dir", type=Path, default=None)
    parser.add_argument("--gt-frame-dir", type=Path, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--masked-psnr",
        action="store_true",
        help="If set, compute masked PSNR using masks from --pred-mask-dir.",
    )
    args = parser.parse_args()

    _require_pred_dir(args.pred_mask_dir, "Prediction mask directory")
    _require_pred_dir(args.pred_frame_dir, "Prediction frame directory")

    payload: dict = {
        "method": args.method_name,
        "dataset": args.dataset_name,
        "mask_metrics": None,
        "video_metrics": None,
    }

    if _gt_dir_usable(args.gt_mask_dir):
        mm = compute_jm_jr(
            args.pred_mask_dir,
            args.gt_mask_dir,
            iou_threshold=args.iou_threshold,
        )
        payload["mask_metrics"] = mm
    else:
        print("GT masks not found. Skipping JM/JR.")

    if _gt_dir_usable(args.gt_frame_dir):
        vm = compute_psnr_ssim(
            args.pred_frame_dir,
            args.gt_frame_dir,
            mask_dir=args.pred_mask_dir if args.masked_psnr else None,
        )
        payload["video_metrics"] = vm
    else:
        print("GT clean frames not found. Skipping PSNR/SSIM.")

    _print_summary(payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(_sanitize_for_json(payload), f, indent=2)
        print(f"Wrote metrics JSON to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
