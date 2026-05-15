#!/usr/bin/env python3
"""
Export per-frame masks from SAM 3 / SAM 3.1 video tracking (text / points / boxes),
compatible with ProPainter (sorted grayscale PNGs per frame).

Prerequisites (see third_party/sam3/README.md):
  - Python 3.12+, CUDA-capable GPU, PyTorch with CUDA
  - pip install -e third_party/sam3  (from project root)
  - Hugging Face access to facebook/sam3 or facebook/sam3.1 + `hf auth login`

Example:
  conda activate /data1/Qiuyue/conda_envs/sam3
  cd /data1/Qiuyue/projects/AIAA3201_project3
  python3 src/sam3_video_masks.py \
    --resource data/frames/bmx-trees \
    --text "bicycle" \
    --frame-index 0 \
    --mask-out-dir outputs/masks/sam3/bmx-trees \
    --version sam3.1
"""
from __future__ import annotations

import argparse
import ast
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = REPO_ROOT / "third_party" / "sam3"


def _ensure_sam3_import_path() -> None:
    """Allow running without a global pip install when repo contains third_party/sam3."""
    try:
        import sam3  # noqa: F401
    except ImportError:
        if SAM3_ROOT.is_dir():
            sys.path.insert(0, str(SAM3_ROOT))
        import sam3  # noqa: F401


def _count_frames(resource: Path) -> int:
    import cv2

    if resource.is_file():
        cap = cv2.VideoCapture(str(resource))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {resource}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n <= 0:
            raise RuntimeError(f"Invalid frame count for video: {resource}")
        return n

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    files = sorted(
        p for p in resource.iterdir() if p.is_file() and p.suffix in exts
    )
    if not files:
        raise FileNotFoundError(f"No image frames under {resource}")
    return len(files)


def _first_frame_size(resource: Path) -> tuple[int, int]:
    import cv2

    if resource.is_file():
        cap = cv2.VideoCapture(str(resource))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {resource}")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame from {resource}")
        h, w = frame.shape[:2]
        return h, w

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    files = sorted(
        p for p in resource.iterdir() if p.is_file() and p.suffix in exts
    )
    if not files:
        raise FileNotFoundError(f"No image frames under {resource}")
    img = np.array(Image.open(files[0]).convert("RGB"))
    return img.shape[0], img.shape[1]


def _collapse_masks(masks: np.ndarray) -> np.ndarray:
    """(N,H,W) bool -> (H,W) bool union; empty -> (0,0) shape used as sentinel."""
    if masks is None or masks.size == 0:
        return np.zeros((0, 0), dtype=bool)
    if masks.ndim == 2:
        return masks.astype(bool)
    return np.logical_or.reduce(masks.astype(bool), axis=0)


def _merge_same_frame_union(
    by_frame: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for fi, parts in by_frame.items():
        if not parts:
            continue
        h, w = parts[0].shape
        acc = np.zeros((h, w), dtype=bool)
        for p in parts:
            if p.size == h * w:
                acc |= p.astype(bool)
        out[fi] = acc
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM 3 video masks: session + propagate, write PNGs for ProPainter."
    )
    p.add_argument(
        "--resource",
        type=str,
        required=True,
        help="MP4/MOV path or directory of ordered JPEG/PNG frames.",
    )
    p.add_argument(
        "--mask-out-dir",
        type=str,
        required=True,
        help="Output directory (created); one zero-padded six-digit PNG per frame when merged.",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt (open-vocabulary), e.g. person",
    )
    p.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index for the initial prompt.",
    )
    p.add_argument(
        "--version",
        type=str,
        choices=("sam3", "sam3.1"),
        default="sam3.1",
        help="Model variant (sam3.1 recommended for speed).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint .pt; default downloads from Hugging Face.",
    )
    p.add_argument(
        "--propagation-direction",
        type=str,
        choices=("both", "forward", "backward"),
        default="both",
        help="Temporal propagation; 'both' may visit a frame twice (unioned).",
    )
    p.add_argument(
        "--output-prob-thresh",
        type=float,
        default=0.5,
        help="Detection/propagation probability threshold.",
    )
    p.set_defaults(merge=True)
    p.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="Write one PNG per tracked object (not ProPainter-friendly); default merges all objects per frame.",
    )
    p.add_argument(
        "--points",
        type=str,
        default=None,
        help='Optional points as Python literal, e.g. "[[0.5,0.3],[0.2,0.4]]" '
        "in relative 0–1 coords if --rel-coords (default), else pixels.",
    )
    p.add_argument(
        "--point-labels",
        type=str,
        default=None,
        help='Optional labels matching --points, e.g. "1,1,0" or "[1,1,0]"',
    )
    p.add_argument(
        "--boxes",
        type=str,
        default=None,
        help='Optional boxes as "[[x,y,w,h],...]" same coord system as points.',
    )
    p.add_argument(
        "--box-labels",
        type=str,
        default=None,
        help='Labels for --boxes, e.g. "1,1"',
    )
    p.add_argument(
        "--rel-coords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prompt coordinates are normalized 0–1 (default True).",
    )
    p.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Pass through to start_session (saves VRAM, slower).",
    )
    p.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Pass through to start_session (saves VRAM, slower).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile (supported mainly on SAM 3.1; may speed up).",
    )
    return p.parse_args()


def _parse_literal_list(s: str, name: str) -> Any:
    try:
        v = ast.literal_eval(s)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {e}") from e
    return v


def main() -> None:
    args = parse_args()
    if not args.text and not args.points and not args.boxes:
        raise SystemExit("Provide at least one of --text, --points, or --boxes.")

    _ensure_sam3_import_path()
    import torch
    from huggingface_hub import get_token
    from sam3.model_builder import build_sam3_predictor

    if not get_token():
        print(
            "[WARN] No Hugging Face token found; checkpoint download may fail. "
            "Run: hf auth login",
            file=sys.stderr,
        )
    if not torch.cuda.is_available():
        print(
            "[WARN] CUDA is not available; SAM3 video predictor requires a GPU.",
            file=sys.stderr,
        )

    resource = Path(args.resource).resolve()
    out_dir = Path(args.mask_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    num_frames = _count_frames(resource)
    height, width = _first_frame_size(resource)

    predictor = build_sam3_predictor(
        checkpoint_path=args.checkpoint,
        version=args.version,
        compile=args.compile,
        async_loading_frames=True,
    )

    session_id: str | None = None
    try:
        start_req: dict[str, Any] = {
            "type": "start_session",
            "resource_path": str(resource),
        }
#        if args.offload_video_to_cpu:
#            start_req["offload_video_to_cpu"] = True
#        if args.offload_state_to_cpu:
#            start_req["offload_state_to_cpu"] = True

        resp = predictor.handle_request(start_req)
        session_id = resp["session_id"]

        add_req: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.frame_index,
            "output_prob_thresh": args.output_prob_thresh,
            "rel_coordinates": args.rel_coords,
        }
        if args.text:
            add_req["text"] = args.text
        if args.points:
            pts = _parse_literal_list(args.points, "points")
            add_req["points"] = pts
        if args.point_labels:
            labels = _parse_literal_list(args.point_labels, "point_labels")
            if isinstance(labels, (list, tuple)):
                add_req["point_labels"] = list(labels)
            else:
                add_req["point_labels"] = labels
        if args.boxes:
            add_req["bounding_boxes"] = _parse_literal_list(args.boxes, "boxes")
        if args.box_labels:
            bl = _parse_literal_list(args.box_labels, "box_labels")
            if isinstance(bl, (list, tuple)):
                add_req["bounding_box_labels"] = list(bl)
            else:
                add_req["bounding_box_labels"] = bl

        predictor.handle_request(add_req)

        by_frame: dict[int, list[np.ndarray]] = {}
        by_frame_obj: dict[int, dict[int, list[np.ndarray]]] = {}
        stream_req: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": args.propagation_direction,
            "output_prob_thresh": args.output_prob_thresh,
        }
        for response in predictor.handle_stream_request(stream_req):
            fi = int(response["frame_index"])
            outputs = response["outputs"]
            masks = outputs.get("out_binary_masks")
            oids = outputs.get("out_obj_ids")
            if masks is None or np.asarray(masks).size == 0:
                single = np.zeros((height, width), dtype=bool)
                by_frame.setdefault(fi, []).append(single)
            else:
                m = np.asarray(masks).astype(bool)
                if m.ndim == 2:
                    if m.shape != (height, width):
                        m = np.zeros((height, width), dtype=bool)
                    by_frame.setdefault(fi, []).append(m)
                    oid = 0
                    if oids is not None and len(np.asarray(oids)) > 0:
                        oid = int(np.asarray(oids)[0])
                    by_frame_obj.setdefault(fi, {}).setdefault(oid, []).append(m.copy())
                elif m.ndim == 3:
                    by_frame.setdefault(fi, []).append(_collapse_masks(m))
                    oids_a = np.asarray(oids) if oids is not None else None
                    for k in range(m.shape[0]):
                        oid = int(oids_a[k]) if oids_a is not None and k < len(oids_a) else k
                        sl = m[k]
                        if sl.shape == (height, width):
                            by_frame_obj.setdefault(fi, {}).setdefault(oid, []).append(
                                sl.copy()
                            )
                else:
                    by_frame.setdefault(fi, []).append(
                        np.zeros((height, width), dtype=bool)
                    )

        merged = _merge_same_frame_union(by_frame)

        if args.merge:
            for i in range(num_frames):
                m = merged.get(i)
                if m is None:
                    u8 = np.zeros((height, width), dtype=np.uint8)
                else:
                    u8 = (m.astype(np.uint8)) * 255
                Image.fromarray(u8, mode="L").save(out_dir / f"{i:06d}.png")
        else:
            merged_obj: dict[tuple[int, int], np.ndarray] = {}
            for fi, objs in by_frame_obj.items():
                for oid, parts in objs.items():
                    if not parts:
                        continue
                    acc = np.zeros((height, width), dtype=bool)
                    for p in parts:
                        if p.shape == (height, width):
                            acc |= p
                    merged_obj[(fi, oid)] = acc
            for i in range(num_frames):
                row = [(oid, m) for (fi, oid), m in merged_obj.items() if fi == i]
                if not row:
                    Image.fromarray(
                        np.zeros((height, width), dtype=np.uint8), mode="L"
                    ).save(out_dir / f"{i:06d}.png")
                else:
                    for oid, m in sorted(row, key=lambda x: x[0]):
                        u8 = m.astype(np.uint8) * 255
                        Image.fromarray(u8, mode="L").save(
                            out_dir / f"{i:06d}_obj{oid:03d}.png"
                        )

        print(f"[INFO] wrote masks under {out_dir} for {num_frames} frames.")
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
