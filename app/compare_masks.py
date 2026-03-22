#!/usr/bin/env python3
"""Compare two sets of per-frame binary masks via IoU.

Loads .npz mask files from two directories, computes per-frame
intersection-over-union, and prints a summary.
"""

import argparse
import csv
import os
import sys

import numpy as np


def load_masks(directory: str) -> dict[str, np.ndarray]:
    """Load all .npz masks from a directory, keyed by filename stem."""
    masks = {}
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(directory, fname))
            masks[fname] = data["mask"]
    return masks


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def save_diff_image(a: np.ndarray, b: np.ndarray, path: str):
    """Save a difference visualization: green=agree, red=A-only, blue=B-only."""
    h, w = a.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    agree = np.logical_and(a_bool, b_bool)
    a_only = np.logical_and(a_bool, ~b_bool)
    b_only = np.logical_and(~a_bool, b_bool)
    img[agree] = [0, 200, 0]       # green
    img[a_only] = [0, 0, 200]      # red (BGR for cv2)
    img[b_only] = [200, 0, 0]      # blue (BGR for cv2)
    try:
        import cv2
        cv2.imwrite(path, img)
    except ImportError:
        from PIL import Image
        # PIL uses RGB, swap channels
        img_rgb = img[:, :, ::-1]
        Image.fromarray(img_rgb).save(path)


def save_overlay_image(a: np.ndarray, b: np.ndarray, iou_val: float,
                       label_a: str, label_b: str, path: str):
    """Save side-by-side mask overlay: A in red, B in blue, overlap in purple."""
    import cv2
    h, w = a.shape
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)

    def mask_to_color(mask, color):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[mask] = color
        return img

    img_a = mask_to_color(a_bool, (0, 0, 200))    # red
    img_b = mask_to_color(b_bool, (200, 0, 0))     # blue
    # Combined: red + blue = purple where overlapping
    img_both = np.zeros((h, w, 3), dtype=np.uint8)
    img_both[a_bool] += np.array([0, 0, 200], dtype=np.uint8)
    img_both[b_bool] += np.array([200, 0, 0], dtype=np.uint8)

    # Draw contours on combined view
    contours_a, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_both, contours_a, -1, (0, 0, 255), 1)
    cv2.drawContours(img_both, contours_b, -1, (255, 0, 0), 1)

    # Arrange: [A] [B] [overlay]
    canvas = np.hstack([img_a, img_b, img_both])
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label_a, (10, 30), font, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, label_b, (w + 10, 30), font, 0.8, (255, 0, 0), 2)
    cv2.putText(canvas, f"IoU={iou_val:.3f}", (2 * w + 10, 30), font, 0.8, (255, 255, 255), 2)

    cv2.imwrite(path, canvas)


def load_bboxes(csv_path: str) -> dict[int, tuple]:
    """Load bbox CSV (frame_idx,x1,y1,x2,y2,iou) into {frame_idx: (x1,y1,x2,y2)}."""
    bboxes = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["frame_idx"])
            if row["x1"] == "":
                bboxes[idx] = None
            else:
                bboxes[idx] = (int(row["x1"]), int(row["y1"]),
                               int(row["x2"]), int(row["y2"]))
    return bboxes


def compute_bbox_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) bounding boxes."""
    if a is None or b is None:
        return 0.0
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
    return float(inter / union)


def main():
    ap = argparse.ArgumentParser(description="Compare two mask directories via IoU")
    ap.add_argument("dir_a", help="First mask directory or bbox CSV")
    ap.add_argument("dir_b", help="Second mask directory or bbox CSV")
    ap.add_argument("--mode", choices=["mask", "bbox"], default="mask",
                    help="Comparison mode: 'mask' for .npz dirs, 'bbox' for CSV files")
    ap.add_argument("--label-a", default="A", help="Label for first directory")
    ap.add_argument("--label-b", default="B", help="Label for second directory")
    ap.add_argument("--csv", default=None, help="Path to write per-frame CSV")
    ap.add_argument("--viz-dir", default=None, help="Directory for difference visualizations")
    ap.add_argument("--overlay-dir", default=None, help="Directory for side-by-side mask overlays")
    ap.add_argument("--skip-first", type=int, default=0, help="Skip first N frames from comparison")
    args = ap.parse_args()

    if args.mode == "bbox":
        bboxes_a = load_bboxes(args.dir_a)
        bboxes_b = load_bboxes(args.dir_b)
        common_idx = sorted(set(bboxes_a.keys()) & set(bboxes_b.keys()))
        if args.skip_first:
            common_idx = [i for i in common_idx if i >= args.skip_first]
        if not common_idx:
            print("ERROR: No common frames found in bbox CSVs!")
            sys.exit(1)
        print(f"Comparing {len(common_idx)} frames (bbox mode): "
              f"{args.label_a} ({args.dir_a}) vs {args.label_b} ({args.dir_b})")

        ious = []
        csv_rows = []
        for idx in common_idx:
            iou_val = compute_bbox_iou(bboxes_a[idx], bboxes_b[idx])
            ious.append(iou_val)
            csv_rows.append({"frame": str(idx), "iou": f"{iou_val:.6f}"})

        # Re-use common for summary (as string keys)
        common = [str(i) for i in common_idx]
    else:
        masks_a = load_masks(args.dir_a)
        masks_b = load_masks(args.dir_b)

        common = sorted(set(masks_a.keys()) & set(masks_b.keys()))
        if args.skip_first:
            common = common[args.skip_first:]
        only_a = set(masks_a.keys()) - set(masks_b.keys())
        only_b = set(masks_b.keys()) - set(masks_a.keys())

        if only_a:
            print(f"WARNING: {len(only_a)} masks only in {args.label_a}")
        if only_b:
            print(f"WARNING: {len(only_b)} masks only in {args.label_b}")

        if not common:
            print("ERROR: No common mask files found!")
            sys.exit(1)

        print(f"Comparing {len(common)} frames: {args.label_a} ({args.dir_a}) vs {args.label_b} ({args.dir_b})")

        if args.viz_dir:
            os.makedirs(args.viz_dir, exist_ok=True)
        if args.overlay_dir:
            os.makedirs(args.overlay_dir, exist_ok=True)

        ious = []
        csv_rows = []

        for fname in common:
            iou_val = compute_iou(masks_a[fname], masks_b[fname])
            ious.append(iou_val)
            csv_rows.append({"frame": fname, "iou": f"{iou_val:.6f}"})

            png_name = fname.replace(".npz", ".png")
            if args.viz_dir:
                save_diff_image(masks_a[fname], masks_b[fname],
                                os.path.join(args.viz_dir, png_name))
            if args.overlay_dir:
                save_overlay_image(masks_a[fname], masks_b[fname], iou_val,
                                   args.label_a, args.label_b,
                                   os.path.join(args.overlay_dir, png_name))

    # Write CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "iou"])
            w.writeheader()
            w.writerows(csv_rows)
        print(f"Per-frame CSV written to {args.csv}")

    # Summary
    ious_arr = np.array(ious)
    below_95 = int(np.sum(ious_arr < 0.95))
    below_90 = int(np.sum(ious_arr < 0.90))
    below_80 = int(np.sum(ious_arr < 0.80))

    print(f"\n{'='*60}")
    print(f"  IoU Summary: {args.label_a} vs {args.label_b}")
    print(f"{'='*60}")
    print(f"  Frames compared: {len(ious)}")
    print(f"  Mean IoU:   {ious_arr.mean():.4f}")
    print(f"  Std IoU:    {ious_arr.std():.4f}")
    print(f"  Median IoU: {np.median(ious_arr):.4f}")
    print(f"  Min IoU:    {ious_arr.min():.4f}  (frame: {common[np.argmin(ious_arr)]})")
    print(f"  Max IoU:    {ious_arr.max():.4f}  (frame: {common[np.argmax(ious_arr)]})")
    print(f"  Below 0.95: {below_95}/{len(ious)}")
    print(f"  Below 0.90: {below_90}/{len(ious)}")
    print(f"  Below 0.80: {below_80}/{len(ious)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
