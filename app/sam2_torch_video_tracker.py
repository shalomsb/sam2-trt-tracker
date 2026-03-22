#!/usr/bin/env python3
"""SAM2.1 Hiera Tiny — PyTorch reference tracker.

Uses Facebook's SAM2VideoPredictor for ground-truth mask generation.
Saves per-frame binary masks as .npz for comparison with TRT tracker.
"""

import argparse
import os
import subprocess
import tempfile
import time

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def extract_frames(video_path: str, frame_dir: str, max_frames: int = 0):
    """Extract JPEG frames via ffmpeg (required by SAM2VideoPredictor)."""
    os.makedirs(frame_dir, exist_ok=True)
    cmd = ["ffmpeg", "-i", video_path, "-q:v", "2", "-start_number", "0"]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd.append(os.path.join(frame_dir, "%05d.jpg"))
    subprocess.run(cmd, check=True, capture_output=True)
    count = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    print(f"Extracted {count} frames to {frame_dir}")
    return count


def main():
    ap = argparse.ArgumentParser(description="SAM2.1 PyTorch Reference Tracker")
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", help="x1,y1,x2,y2 (pixel coords)")
    ap.add_argument("--point", help="x,y (pixel coords)")
    ap.add_argument("--checkpoint", default="/models/sam2/tiny/sam2.1_hiera_tiny.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--save-masks", required=True, help="Directory to save per-frame binary masks")
    ap.add_argument("--save-bboxes", default=None,
                    help="Path to write per-frame bbox CSV (frame_idx,x1,y1,x2,y2)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    args = ap.parse_args()

    if not args.bbox and not args.point:
        ap.error("must provide --bbox or --point")
    if args.bbox and args.point:
        ap.error("provide either --bbox or --point, not both")

    os.makedirs(args.save_masks, exist_ok=True)

    bbox_csv_file = None
    bbox_csv_writer = None
    if args.save_bboxes:
        import csv
        os.makedirs(os.path.dirname(args.save_bboxes) or ".", exist_ok=True)
        bbox_csv_file = open(args.save_bboxes, "w", newline="")
        bbox_csv_writer = csv.writer(bbox_csv_file)
        bbox_csv_writer.writerow(["frame_idx", "x1", "y1", "x2", "y2", "iou"])

    # Extract frames to a temp directory
    frame_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    num_frames = extract_frames(args.video, frame_dir)

    # Build predictor
    print("Building SAM2 video predictor...")
    predictor = build_sam2_video_predictor(args.config, args.checkpoint)

    t0 = time.perf_counter()

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=frame_dir,
        )

        # Add prompt on frame 0
        # normalize_coords=True (default): SAM2 divides pixel coords by [W,H]
        # then scales by image_size (1024). Same mapping as the hybrid tracker.
        if args.bbox:
            bbox = [float(v) for v in args.bbox.split(",")]
            bbox_tensor = np.array(bbox, dtype=np.float32)
            _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1, box=bbox_tensor,
            )
        else:
            pt = [float(v) for v in args.point.split(",")]
            points = np.array([[pt[0], pt[1]]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=points, labels=labels,
            )

        print(f"Prompt added on frame 0, propagating {num_frames} frames...")

        # Propagate and save masks
        saved = 0
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
            np.savez_compressed(
                os.path.join(args.save_masks, f"{frame_idx:05d}.npz"),
                mask=mask,
            )
            if bbox_csv_writer is not None:
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    bbox_csv_writer.writerow([frame_idx, xs.min(), ys.min(),
                                              xs.max(), ys.max(), ""])
                else:
                    bbox_csv_writer.writerow([frame_idx, "", "", "", "", ""])
            saved += 1
            if frame_idx % 50 == 0:
                print(f"  [{frame_idx:5d}/{num_frames}]")
            if args.max_frames and saved >= args.max_frames:
                break

    if bbox_csv_file:
        bbox_csv_file.close()
        print(f"Bboxes written to {args.save_bboxes}")

    elapsed = time.perf_counter() - t0
    fps = saved / elapsed if elapsed > 0 else 0
    print(f"Done — {saved} masks saved to {args.save_masks} ({fps:.1f} fps, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
