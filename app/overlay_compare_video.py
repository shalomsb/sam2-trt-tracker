#!/usr/bin/env python3
"""Overlay two mask sets (or bbox CSVs) on the original video for visual comparison.

Reads per-frame .npz masks from two directories (mask mode) or per-frame
bounding boxes from two CSVs (bbox mode) and draws both on each video
frame with different colors. Outputs a single video.
"""

import argparse
import csv
import os
import subprocess

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


def load_bboxes_csv(path: str) -> dict[int, tuple[int, int, int, int]]:
    """Load bbox CSV (frame_idx,x1,y1,x2,y2,iou) into {frame_idx: (x1,y1,x2,y2)}."""
    bboxes = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            fidx = int(row[0])
            if row[1] == "":
                continue
            bboxes[fidx] = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
    return bboxes


def compute_bbox_iou(a: tuple[int, int, int, int],
                     b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def main():
    ap = argparse.ArgumentParser(description="Overlay two mask sets or bbox CSVs on video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", choices=["mask", "bbox"], default="mask",
                    help="Overlay mode: mask (default) or bbox")
    ap.add_argument("--masks-a", help="First mask directory (mask mode)")
    ap.add_argument("--masks-b", help="Second mask directory (mask mode)")
    ap.add_argument("--bboxes-a", help="First bbox CSV path (bbox mode)")
    ap.add_argument("--bboxes-b", help="Second bbox CSV path (bbox mode)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--output", default="/output/compare_overlay_video.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "mask" and (not args.masks_a or not args.masks_b):
        ap.error("--masks-a and --masks-b are required in mask mode")
    if args.mode == "bbox" and (not args.bboxes_a or not args.bboxes_b):
        ap.error("--bboxes-a and --bboxes-b are required in bbox mode")

    bbox_mode = args.mode == "bbox"
    bboxes_a = bboxes_b = None
    if bbox_mode:
        bboxes_a = load_bboxes_csv(args.bboxes_a)
        bboxes_b = load_bboxes_csv(args.bboxes_b)

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

        vis = frame.copy()

        if bbox_mode:
            box_a = bboxes_a.get(frame_idx)
            box_b = bboxes_b.get(frame_idx)
            if box_a:
                cv2.rectangle(vis, (box_a[0], box_a[1]), (box_a[2], box_a[3]),
                              color_a, 2)
            if box_b:
                cv2.rectangle(vis, (box_b[0], box_b[1]), (box_b[2], box_b[3]),
                              color_b, 2)
            if box_a and box_b:
                iou = compute_bbox_iou(box_a, box_b)
                iou_text = f"IoU={iou:.3f}"
            else:
                iou_text = "IoU=N/A"
        else:
            mask_a = load_mask(args.masks_a, frame_idx)
            mask_b = load_mask(args.masks_b, frame_idx)
            vis = overlay(vis, mask_a, color_a, alpha=0.3)
            vis = overlay(vis, mask_b, color_b, alpha=0.3)
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

    # Re-encode mp4v → H.264 so the video plays in VS Code / browsers.
    h264_path = args.output.replace(".mp4", "_h264.mp4")
    print(f"Re-encoding to H.264: {h264_path}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", args.output, "-c:v", "libx264", "-crf", "23", h264_path],
        capture_output=True,
    )
    if os.path.isfile(h264_path):
        os.replace(h264_path, args.output)
        print(f"H.264 output: {args.output}")


if __name__ == "__main__":
    main()
