#!/usr/bin/env python3

"""
Extract every frame from a video into a sorted PNG sequence (`000000.png`, ...).

Example:

    python src/extract_frames.py \\
      --video data/videos/tennis.mp4 \\
      --out_dir data/frames/tennis

End-to-end commands for masks, inpainting, merging, and metrics live in README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames to sequential PNG files."
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to input video file, e.g. data/videos/tennis.mp4",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Directory to save extracted frames, e.g. data/frames/tennis",
    )
    return parser.parse_args()


def extract_all_frames(cap: cv2.VideoCapture, out_dir: Path) -> int:
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = out_dir / f"{saved:06d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))

    print(f"[INFO] video: {video_path}")
    print(f"[INFO] total_frames: {total_frames}")
    print(f"[INFO] source_fps: {src_fps:.4f}")
    print(f"[INFO] output_dir: {out_dir}")

    saved = extract_all_frames(cap, out_dir)

    cap.release()
    print(f"[INFO] saved_frames: {saved}")


if __name__ == "__main__":
    main()
