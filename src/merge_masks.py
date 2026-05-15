#!/usr/bin/env python3
"""
Merge two directories of grayscale mask PNGs (same naming as elsewhere in this project).

Pairing uses the intersection of filenames (stems aligned). Default mode is logical OR (union).

Example:
  cd /data1/Qiuyue/projects/AIAA3201_project3
  python3 src/merge_masks.py \\
    --mask-dir-a outputs/masks/part3/bmx-trees \\
    --mask-dir-b outputs/masks/sam3/bmx-trees \\
    --output-dir outputs/masks/merged/bmx-trees \\
    --mode union
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from evaluation.metrics import load_binary_mask, natural_image_sort_key, sorted_image_paths


def _stems(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine two folders of binary mask images (paired by matching stems)."
    )
    parser.add_argument("--mask-dir-a", type=Path, required=True)
    parser.add_argument("--mask-dir-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("union", "intersection", "side_by_side"),
        default="union",
        help="union = foreground if either mask; intersection = both; "
        "side_by_side = horizontal grayscale concat for quick visual compare.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Foreground threshold when loading masks (same as evaluation.metrics.load_binary_mask).",
    )
    args = parser.parse_args()

    for label, d in ("--mask-dir-a", args.mask_dir_a), ("--mask-dir-b", args.mask_dir_b):
        if not d.is_dir():
            raise FileNotFoundError(f"{label} is not a directory: {d.resolve()}")

    pa = sorted_image_paths(args.mask_dir_a)
    pb = sorted_image_paths(args.mask_dir_b)
    ma, mb = _stems(pa), _stems(pb)
    common = sorted(ma.keys() & mb.keys(), key=lambda s: natural_image_sort_key(Path(s + ".png")))
    only_a = sorted(ma.keys() - mb.keys(), key=lambda s: natural_image_sort_key(Path(s + ".png")))
    only_b = sorted(mb.keys() - ma.keys(), key=lambda s: natural_image_sort_key(Path(s + ".png")))

    if not common:
        raise ValueError(
            "No overlapping mask filenames between the two directories. "
            f"A has {len(pa)} images, B has {len(pb)}."
        )
    if only_a or only_b:
        if only_a:
            print(f"Warning: {len(only_a)} files only in A (skipped): {only_a[:5]}{'...' if len(only_a) > 5 else ''}")
        if only_b:
            print(f"Warning: {len(only_b)} files only in B (skipped): {only_b[:5]}{'...' if len(only_b) > 5 else ''}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stem in common:
        path_a = ma[stem]
        path_b = mb[stem]
        suffix = path_a.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"

        if args.mode == "side_by_side":
            ga = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
            gb = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
            if ga is None or gb is None:
                raise FileNotFoundError(f"Failed to read grayscale: {path_a} or {path_b}")
            if ga.shape != gb.shape:
                raise ValueError(
                    f"Shape mismatch for stem {stem}: {ga.shape} vs {gb.shape}. "
                    "Resize or regenerate masks before side_by_side merge."
                )
            out = np.hstack([ga, gb])
        else:
            bina = load_binary_mask(path_a, threshold=args.threshold)
            binb = load_binary_mask(path_b, threshold=args.threshold)
            if bina.shape != binb.shape:
                raise ValueError(
                    f"Shape mismatch for stem {stem}: {bina.shape} vs {binb.shape}."
                )
            if args.mode == "union":
                merged = np.logical_or(bina, binb)
            else:
                merged = np.logical_and(bina, binb)
            out = (merged.astype(np.uint8) * 255)

        out_path = args.output_dir / f"{stem}{suffix}"
        if not cv2.imwrite(str(out_path), out):
            raise RuntimeError(f"cv2.imwrite failed: {out_path}")

    print(
        f"Wrote {len(common)} merged image(s) to {args.output_dir.resolve()} "
        f"(mode={args.mode})."
    )


if __name__ == "__main__":
    main()
