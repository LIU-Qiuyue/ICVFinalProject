"""Build qualitative comparison grids for the project report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import load_binary_mask, load_rgb_image, sorted_image_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a comparison figure: input, then per-method mask and result."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--input-frame-dir", type=Path, required=True)
    parser.add_argument("--mask-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--result-frame-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--method-names", type=str, nargs="+", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    n_methods = len(args.mask_dirs)
    if len(args.result_frame_dirs) != n_methods or len(args.method_names) != n_methods:
        raise ValueError(
            "--mask-dirs, --result-frame-dirs, and --method-names must have the same length."
        )

    if not args.input_frame_dir.is_dir():
        raise FileNotFoundError(f"Input frame directory not found: {args.input_frame_dir.resolve()}")
    for d in list(args.mask_dirs) + list(args.result_frame_dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d.resolve()}")

    input_files = sorted_image_paths(args.input_frame_dir)
    mask_file_lists = [sorted_image_paths(d) for d in args.mask_dirs]
    result_file_lists = [sorted_image_paths(d) for d in args.result_frame_dirs]

    def _empty_report(name: str, path: Path, n: int) -> str | None:
        if n == 0:
            return f"{name} has no .png/.jpg/.jpeg files: {path.resolve()}"
        return None

    empty_msgs = [
        _empty_report("Input", args.input_frame_dir, len(input_files)),
    ]
    for i, d in enumerate(args.mask_dirs):
        empty_msgs.append(_empty_report(f"Mask dir {i}", d, len(mask_file_lists[i])))
    for i, d in enumerate(args.result_frame_dirs):
        empty_msgs.append(_empty_report(f"Result dir {i}", d, len(result_file_lists[i])))
    empty_msgs = [m for m in empty_msgs if m]
    if empty_msgs:
        raise FileNotFoundError("\n".join(empty_msgs))

    max_idx = max(args.frame_indices)
    min_len = min(len(input_files), *[len(x) for x in mask_file_lists], *[len(x) for x in result_file_lists])
    if min(args.frame_indices) < 0 or max_idx >= min_len:
        raise IndexError(
            f"frame_indices out of range (max index {max_idx}, available frames {min_len})."
        )

    n_rows = len(args.frame_indices)
    n_cols = 1 + 2 * n_methods

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.6,
        }
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    col_titles = ["Input"]
    for name in args.method_names:
        col_titles.extend([f"{name}\nmask", f"{name}\nresult"])

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, pad=6)

    for r, fi in enumerate(args.frame_indices):
        inp = load_rgb_image(input_files[fi])
        axes[r, 0].imshow(inp)
        axes[r, 0].axis("off")

        for m in range(n_methods):
            mcol = 1 + 2 * m
            mask_path = mask_file_lists[m][fi]
            res_path = result_file_lists[m][fi]
            mask = load_binary_mask(mask_path)
            res = load_rgb_image(res_path)

            axes[r, mcol].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[r, mcol].axis("off")

            axes[r, mcol + 1].imshow(res)
            axes[r, mcol + 1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.06, hspace=0.15)
    fig.suptitle(f"{args.dataset_name} — qualitative comparison", fontsize=12, y=0.98)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.dataset_name}_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {out_path.resolve()}")


if __name__ == "__main__":
    main()
