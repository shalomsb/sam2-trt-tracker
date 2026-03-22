#!/usr/bin/env python3
"""Overlay two mask sets on the original video for visual comparison.

Reads per-frame .npz masks from two directories and draws both on
each video frame with different colors. Outputs a single video.
"""

import argparse
import os

import cv2
import numpy as np


def load_mask(directory: str, frame_idx: int) -> np.ndarray | None:
    path = os.path.join(directory, f"{frame_idx:05d}.npz")
    if os.path.exists(path):
        return np.load(path)["mask"]
    return None


def overlay(frame, mask, color, alpha=0.35):
    """Overlay a binary mask on a frame with given color and transparency."""
    if mask is None:
        return frame
    roi = mask > 0
    out = frame.copy()
    out[roi] = (out[roi].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def main():
    ap = argparse.ArgumentParser(description="Overlay two mask sets on video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--masks-a", required=True, help="First mask directory")
    ap.add_argument("--masks-b", required=True, help="Second mask directory")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--output", default="/output/compare_overlay_video.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    color_a = (0, 0, 255)    # red (BGR)
    color_b = (255, 0, 0)    # blue (BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        mask_a = load_mask(args.masks_a, frame_idx)
        mask_b = load_mask(args.masks_b, frame_idx)

        vis = frame.copy()
        vis = overlay(vis, mask_a, color_a, alpha=0.3)
        vis = overlay(vis, mask_b, color_b, alpha=0.3)

        # IoU text
        if mask_a is not None and mask_b is not None:
            iou = compute_iou(mask_a, mask_b)
            iou_text = f"IoU={iou:.3f}"
        else:
            iou_text = "IoU=N/A"

        cv2.putText(vis, f"frame {frame_idx}  {iou_text}", (10, 30),
                    font, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, args.label_a, (10, 60), font, 0.7, color_a, 2)
        cv2.putText(vis, args.label_b, (10, 85), font, 0.7, color_b, 2)

        writer.write(vis)

        if frame_idx % 50 == 0:
            print(f"  [{frame_idx:5d}/{total}] {iou_text}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done — {frame_idx} frames written to {args.output}")


if __name__ == "__main__":
    main()
