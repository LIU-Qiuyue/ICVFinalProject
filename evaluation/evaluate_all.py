"""Batch evaluation from a JSON config file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from evaluation.metrics import compute_jm_jr, compute_psnr_ssim


DEFAULT_PRED_MASK_TEMPLATE = "{results_root}/{method}/{dataset}/masks"
DEFAULT_PRED_FRAME_TEMPLATE = "{results_root}/{method}/{dataset}/frames"
DEFAULT_GT_MASK_TEMPLATE = "{gt_mask_root}/{dataset}"
DEFAULT_GT_FRAME_TEMPLATE = "{gt_frame_root}/{dataset}"


def _format_template(tpl: str, **kwargs: str) -> Path:
    return Path(tpl.format(**kwargs))


def _gt_dir_usable(path: Path) -> bool:
    return path.is_dir()


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


def _one_row(
    dataset: str,
    method: str,
    mask_metrics: dict | None,
    video_metrics: dict | None,
) -> dict:
    def _m(key: str, src: dict | None) -> float | str | int | None:
        if not src:
            return ""
        val = src.get(key)
        if val is None:
            return ""
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return ""
        return val

    return {
        "dataset": dataset,
        "method": method,
        "JM": _m("JM", mask_metrics),
        "JR": _m("JR", mask_metrics),
        "PSNR": _m("PSNR", video_metrics),
        "SSIM": _m("SSIM", video_metrics),
        "masked_PSNR": _m("masked_PSNR", video_metrics),
        "num_mask_frames": mask_metrics["num_frames"] if mask_metrics else "",
        "num_video_frames": video_metrics["num_frames"] if video_metrics else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for all method/dataset pairs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to eval config JSON.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = json.load(f)

    datasets: list[str] = cfg["datasets"]
    methods: list[str] = cfg["methods"]
    results_root = Path(cfg["results_root"])
    gt_mask_root = Path(cfg["gt_mask_root"])
    gt_frame_root = Path(cfg["gt_frame_root"])
    output_dir = Path(cfg["output_dir"])
    iou_threshold = float(cfg.get("iou_threshold", 0.5))
    masked_psnr = bool(cfg.get("masked_psnr", False))

    pred_mask_tpl = cfg.get("pred_mask_template", DEFAULT_PRED_MASK_TEMPLATE)
    pred_frame_tpl = cfg.get("pred_frame_template", DEFAULT_PRED_FRAME_TEMPLATE)
    gt_mask_tpl = cfg.get("gt_mask_template", DEFAULT_GT_MASK_TEMPLATE)
    gt_frame_tpl = cfg.get("gt_frame_template", DEFAULT_GT_FRAME_TEMPLATE)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for method in methods:
        for dataset in datasets:
            pred_mask_dir = _format_template(
                pred_mask_tpl,
                results_root=str(results_root),
                method=method,
                dataset=dataset,
            )
            pred_frame_dir = _format_template(
                pred_frame_tpl,
                results_root=str(results_root),
                method=method,
                dataset=dataset,
            )
            gt_mask_dir = _format_template(
                gt_mask_tpl,
                gt_mask_root=str(gt_mask_root),
                dataset=dataset,
            )
            gt_frame_dir = _format_template(
                gt_frame_tpl,
                gt_frame_root=str(gt_frame_root),
                dataset=dataset,
            )

            payload: dict = {
                "method": method,
                "dataset": dataset,
                "mask_metrics": None,
                "video_metrics": None,
            }

            if not pred_mask_dir.is_dir() or not pred_frame_dir.is_dir():
                if not pred_mask_dir.is_dir():
                    print(
                        f"[skip {method}/{dataset}] Prediction mask directory missing: {pred_mask_dir}"
                    )
                if not pred_frame_dir.is_dir():
                    print(
                        f"[skip {method}/{dataset}] Prediction frame directory missing: {pred_frame_dir}"
                    )
                out_skip = output_dir / f"{method}_{dataset}_metrics.json"
                with out_skip.open("w", encoding="utf-8") as wf:
                    json.dump(_sanitize_for_json(payload), wf, indent=2)
                print(f"Wrote {out_skip} (skipped, no predictions)")
                rows.append(_one_row(dataset, method, None, None))
                continue

            if _gt_dir_usable(gt_mask_dir):
                try:
                    payload["mask_metrics"] = compute_jm_jr(
                        pred_mask_dir, gt_mask_dir, iou_threshold=iou_threshold
                    )
                except (FileNotFoundError, ValueError) as e:
                    print(f"[warn {method}/{dataset}] Mask metrics failed: {e}")
            else:
                print(f"GT masks not found. Skipping JM/JR. ({gt_mask_dir})")

            if _gt_dir_usable(gt_frame_dir):
                try:
                    payload["video_metrics"] = compute_psnr_ssim(
                        pred_frame_dir,
                        gt_frame_dir,
                        mask_dir=pred_mask_dir if masked_psnr else None,
                    )
                except (FileNotFoundError, ValueError) as e:
                    print(f"[warn {method}/{dataset}] Video metrics failed: {e}")
            else:
                print(f"GT clean frames not found. Skipping PSNR/SSIM. ({gt_frame_dir})")

            out_path = output_dir / f"{method}_{dataset}_metrics.json"
            with out_path.open("w", encoding="utf-8") as wf:
                json.dump(_sanitize_for_json(payload), wf, indent=2)
            print(f"Wrote {out_path}")

            rows.append(
                _one_row(
                    dataset,
                    method,
                    payload["mask_metrics"],
                    payload["video_metrics"],
                )
            )

    summary_path = output_dir / "summary_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
