#!/usr/bin/env python3
"""SAM2.1 Hiera Tiny — ONNX Runtime Video Tracker

Single-object tracking: give a bounding box or point on frame 0, track forever.
Uses ONNX Runtime + CUDA for inference.
"""

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# ─── Constants ───────────────────────────────────────────────────────────────

NUM_MASKMEM = 7        # 1 conditioning + 6 non-conditioning
IMG_SIZE = 1024        # SAM2 input resolution
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ─── Memory bank ─────────────────────────────────────────────────────────────

@dataclass
class MemoryFrame:
    maskmem_features: np.ndarray  # [1, 64, 64, 64]
    maskmem_pos_enc: np.ndarray   # [1, 4096, 64]
    obj_ptr: np.ndarray           # [1, 256]
    frame_idx: int
    is_cond: bool


class MemoryBank:
    def __init__(self):
        self.cond_frame: MemoryFrame | None = None
        self.non_cond_frames: list[MemoryFrame] = []
        self.temporal_code: np.ndarray | None = None  # [7, 1, 1, 64]

    def add(self, frame: MemoryFrame):
        if frame.is_cond:
            self.cond_frame = frame
        else:
            self.non_cond_frames.append(frame)
            if len(self.non_cond_frames) > NUM_MASKMEM - 1:
                self.non_cond_frames.pop(0)  # evict oldest

    def build_memory_inputs(self, current_idx: int):
        """Build the four inputs for the memory_attention ONNX model."""
        frames_tpos = [(self.cond_frame, 0)]
        for f in self.non_cond_frames:
            t_rel = current_idx - f.frame_idx
            t_pos = NUM_MASKMEM - t_rel
            if t_pos >= 1:
                frames_tpos.append((f, t_pos))
        frames_tpos.sort(key=lambda x: x[1])

        # memory_1 [1, num_masks, 64, 64, 64]
        memory_1 = np.stack(
            [f.maskmem_features[0] for f, _ in frames_tpos], axis=0
        )[np.newaxis].astype(np.float32)

        # memory_pos_embed (mask portion): maskmem_pos_enc + temporal_code
        pos_parts = []
        for f, t_pos in frames_tpos:
            tcode_idx = NUM_MASKMEM - 1 - t_pos
            tcode = self.temporal_code[tcode_idx]
            tcode = tcode.reshape(1, 1, 64)
            pos = f.maskmem_pos_enc + tcode
            pos_parts.append(pos[0])
        mask_pos = np.concatenate(pos_parts, axis=0)

        # memory_0 [1, num_obj_ptr, 256]
        obj_ptrs = [self.cond_frame.obj_ptr[0]]
        for f in reversed(self.non_cond_frames):
            t_rel = current_idx - f.frame_idx
            if 1 <= t_rel <= NUM_MASKMEM - 1:
                obj_ptrs.append(f.obj_ptr[0])
        memory_0 = np.stack(obj_ptrs, axis=0)[np.newaxis].astype(np.float32)

        num_obj_ptr = memory_0.shape[1]

        # memory_pos_embed [1, num_masks*4096 + num_obj_ptr*4, 64]
        obj_pos_zeros = np.zeros((num_obj_ptr * 4, 64), dtype=np.float32)
        memory_pos_embed = np.concatenate(
            [mask_pos, obj_pos_zeros], axis=0
        )[np.newaxis].astype(np.float32)

        # cond_frame_id_diff (scalar float32)
        cond_frame_id_diff = np.array(
            current_idx - self.cond_frame.frame_idx, dtype=np.float32
        )

        return memory_0, memory_1, memory_pos_embed, cond_frame_id_diff


# ─── Pre/post processing ────────────────────────────────────────────────────

def preprocess(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """BGR frame -> normalised NCHW float32 [1,3,1024,1024]."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = (resized.astype(np.float32) - MEAN) / STD
    return img.transpose(2, 0, 1)[np.newaxis], h, w


def scale_bbox(bbox, orig_w, orig_h):
    """Scale bbox coords from original resolution to 1024x1024."""
    x1, y1, x2, y2 = bbox
    return [x1 * IMG_SIZE / orig_w, y1 * IMG_SIZE / orig_h,
            x2 * IMG_SIZE / orig_w, y2 * IMG_SIZE / orig_h]


def postprocess_mask(pred_mask: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    """pred_mask [1,1,256,256] logits -> binary mask at original resolution."""
    logits = pred_mask[0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    resized = cv2.resize(prob.astype(np.float32), (orig_w, orig_h),
                         interpolation=cv2.INTER_LINEAR)
    return (resized > 0.5).astype(np.uint8)


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    """Draw semi-transparent coloured mask + contour on frame."""
    out = frame.copy()
    roi = mask > 0
    out[roi] = (out[roi].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ─── ONNX helpers ────────────────────────────────────────────────────────────

def make_session(path: str, providers) -> ort.InferenceSession:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(path, providers=providers)


def print_io(name: str, sess: ort.InferenceSession):
    print(f"  {name}:")
    for inp in sess.get_inputs():
        print(f"    IN  {inp.name:30s} {inp.shape}")
    for out in sess.get_outputs():
        print(f"    OUT {out.name:30s} {out.shape}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SAM2.1 ONNX Runtime Video Tracker")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--bbox", help="x1,y1,x2,y2 bounding box on frame 0")
    ap.add_argument("--point", help="x,y foreground point on frame 0")
    ap.add_argument("--model-dir", default="/models/sam2/tiny/onnx/",
                    help="Directory containing the 4 ONNX models")
    ap.add_argument("--output", default="/output/output_tracked_onnx.mp4",
                    help="Output video path")
    ap.add_argument("--show", action="store_true",
                    help="Display frames live (press q to quit)")
    args = ap.parse_args()

    if not args.bbox and not args.point:
        ap.error("must provide --bbox or --point")
    if args.bbox and args.point:
        ap.error("provide either --bbox or --point, not both")

    # ── Load ONNX sessions ──────────────────────────────────────────────
    providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    print("Loading ONNX models...")
    enc_sess  = make_session(os.path.join(args.model_dir, "image_encoder.onnx"), providers)
    dec_sess  = make_session(os.path.join(args.model_dir, "mask_decoder.onnx"), providers)
    menc_sess = make_session(os.path.join(args.model_dir, "memory_encoder.onnx"), providers)
    matt_sess = make_session(os.path.join(args.model_dir, "memory_attention.onnx"), providers)

    for name, s in [("image_encoder", enc_sess), ("mask_decoder", dec_sess),
                    ("memory_encoder", menc_sess), ("memory_attention", matt_sess)]:
        print_io(name, s)

    # ── Open video ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w0}x{h0} @ {fps:.1f} fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w0, h0))

    bank = MemoryBank()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()

        # ── Image encoder ───────────────────────────────────────────
        img, orig_h, orig_w = preprocess(frame)
        pix_feat, hr0, hr1, vision_feats, vision_pos = enc_sess.run(
            None, {"image": img}
        )

        if frame_idx == 0:
            # ── Frame 0: prompted with bbox or point ────────────────
            if args.bbox:
                bbox = [float(v) for v in args.bbox.split(",")]
                sb = scale_bbox(bbox, orig_w, orig_h)
                point_coords = np.array(
                    [[[sb[0], sb[1]], [sb[2], sb[3]]]], dtype=np.float32
                )
                point_labels = np.array([[2.0, 3.0]], dtype=np.float32)
            else:
                pt = [float(v) for v in args.point.split(",")]
                sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
                point_coords = np.array(
                    [[[pt[0] * sx, pt[1] * sy]]], dtype=np.float32
                )
                point_labels = np.array([[1.0]], dtype=np.float32)

            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec_sess.run(
                None, {
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "image_embed": vision_feats,
                    "high_res_feats_0": hr0,
                    "high_res_feats_1": hr1,
                }
            )

            maskmem_feat, maskmem_pos, temporal_code = menc_sess.run(
                None, {
                    "mask_for_mem": mask_for_mem,
                    "pix_feat": pix_feat,
                    "occ_logit": occ_logit,
                }
            )

            bank.temporal_code = temporal_code
            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat,
                maskmem_pos_enc=maskmem_pos,
                obj_ptr=obj_ptr,
                frame_idx=0,
                is_cond=True,
            ))

        else:
            # ── Frame 1+: propagation with memory ──────────────────
            mem0, mem1, mem_pos, cond_diff = bank.build_memory_inputs(frame_idx)

            image_embed = matt_sess.run(
                None, {
                    "current_vision_feat": vision_feats,
                    "current_vision_pos_embed": vision_pos,
                    "memory_0": mem0,
                    "memory_1": mem1,
                    "memory_pos_embed": mem_pos,
                    "cond_frame_id_diff": cond_diff,
                }
            )[0]

            point_coords = np.zeros((1, 1, 2), dtype=np.float32)
            point_labels = np.full((1, 1), -1.0, dtype=np.float32)

            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec_sess.run(
                None, {
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "image_embed": image_embed,
                    "high_res_feats_0": hr0,
                    "high_res_feats_1": hr1,
                }
            )

            maskmem_feat, maskmem_pos, _ = menc_sess.run(
                None, {
                    "mask_for_mem": mask_for_mem,
                    "pix_feat": pix_feat,
                    "occ_logit": occ_logit,
                }
            )

            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat,
                maskmem_pos_enc=maskmem_pos,
                obj_ptr=obj_ptr,
                frame_idx=frame_idx,
                is_cond=False,
            ))

        # ── Visualize ───────────────────────────────────────────────
        binary_mask = postprocess_mask(pred_mask, orig_h, orig_w)
        vis = overlay_mask(frame, binary_mask)

        dt = time.perf_counter() - t0
        fps_now = 1.0 / dt if dt > 0 else 0
        iou_val = float(iou.flat[0])
        cv2.putText(vis, f"frame {frame_idx}  iou={iou_val:.2f}  {fps_now:.1f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(vis)

        if args.show:
            cv2.imshow("SAM2 Tracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 50 == 0 or frame_idx == total - 1:
            print(f"  [{frame_idx:5d}/{total}]  iou={iou_val:.2f}  {fps_now:.1f} fps")

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done — {frame_idx} frames written to {args.output}")


if __name__ == "__main__":
    main()
