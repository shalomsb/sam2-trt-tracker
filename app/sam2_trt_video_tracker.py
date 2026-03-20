#!/usr/bin/env python3
"""SAM2.1 Hiera Tiny — TensorRT Video Tracker

Zero-copy GPU pipeline: all intermediate tensors stay on GPU.
Only the raw frame (in) and binary mask (out) cross PCIe.

Uses PyTorch CUDA tensors for memory management + TensorRT for inference.
Threaded video I/O overlaps disk access with GPU compute.
"""

import argparse
import os
import time
import threading
import queue
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

# ─── Constants ───────────────────────────────────────────────────────────────

NUM_MASKMEM = 3
IMG_SIZE = 1024
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_NP_TO_TORCH = {
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
}


# ─── TensorRT engine (zero-copy GPU) ────────────────────────────────────────

class TRTEngine:
    """TRT engine with pre-allocated GPU output buffers. No host copies."""

    def __init__(self, engine_path: str, stream: torch.cuda.Stream = None):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = stream

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.dynamic_inputs: set[str] = set()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                if -1 in self.engine.get_tensor_shape(name):
                    self.dynamic_inputs.add(name)
            else:
                self.output_names.append(name)

        # Pre-allocate output buffers for static-shape outputs
        self._static_bufs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if -1 not in shape:
                dt = _NP_TO_TORCH[np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))]
                self._static_bufs[name] = torch.empty(shape, dtype=dt, device="cuda")

    def __call__(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        ctx = self.context

        for name in self.input_names:
            t = inputs[name].contiguous()
            if name in self.dynamic_inputs:
                ctx.set_input_shape(name, tuple(t.shape))
            ctx.set_tensor_address(name, t.data_ptr())

        outputs: list[torch.Tensor] = []
        for name in self.output_names:
            if name in self._static_bufs:
                buf = self._static_bufs[name]
            else:
                shape = tuple(ctx.get_tensor_shape(name))
                dt = _NP_TO_TORCH[np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))]
                buf = torch.empty(shape, dtype=dt, device="cuda")
            ctx.set_tensor_address(name, buf.data_ptr())
            outputs.append(buf)

        ctx.execute_async_v3(self.stream.cuda_stream)
        return outputs

    def print_io(self, label: str):
        print(f"  {label}:")
        for name in self.input_names:
            s = self.engine.get_tensor_shape(name)
            tag = " *dynamic*" if name in self.dynamic_inputs else ""
            print(f"    IN  {name:30s} {tuple(s)}{tag}")
        for name in self.output_names:
            s = self.engine.get_tensor_shape(name)
            tag = " [pre-alloc]" if name in self._static_bufs else ""
            print(f"    OUT {name:30s} {tuple(s)}{tag}")


# ─── GPU Memory bank (pre-allocated buffers) ────────────────────────────────

@dataclass
class MemoryFrame:
    maskmem_features: torch.Tensor  # [1,64,64,64]  cuda
    maskmem_pos_enc: torch.Tensor   # [1,4096,64]   cuda
    obj_ptr: torch.Tensor           # [1,256]        cuda
    frame_idx: int
    is_cond: bool


class MemoryBank:
    def __init__(self):
        self.cond_frame: MemoryFrame | None = None
        self.non_cond_frames: list[MemoryFrame] = []
        self.temporal_code: torch.Tensor | None = None  # [7,1,1,64] cuda

        # Pre-allocated output buffers — written into every frame, never reallocated
        max_pos = NUM_MASKMEM * 4096 + NUM_MASKMEM * 4
        self._mem1 = torch.empty(1, NUM_MASKMEM, 64, 64, 64, device="cuda")
        self._mem0 = torch.empty(1, NUM_MASKMEM, 256, device="cuda")
        self._mempos = torch.zeros(1, max_pos, 64, device="cuda")
        self._cond_diff = torch.empty((), dtype=torch.float32, device="cuda")

    def add(self, frame: MemoryFrame):
        if frame.is_cond:
            self.cond_frame = frame
        else:
            self.non_cond_frames.append(frame)
            if len(self.non_cond_frames) > NUM_MASKMEM - 1:
                self.non_cond_frames.pop(0)

    def build_memory_inputs(self, current_idx: int):
        """Write into pre-allocated GPU buffers and return sliced views."""
        frames_tpos = [(self.cond_frame, 0)]
        for f in self.non_cond_frames:
            t_rel = current_idx - f.frame_idx
            t_pos = NUM_MASKMEM - t_rel
            if t_pos >= 1:
                frames_tpos.append((f, t_pos))
        frames_tpos.sort(key=lambda x: x[1])

        n = len(frames_tpos)

        # memory_1 — write maskmem_features into pre-allocated slots
        for i, (f, _) in enumerate(frames_tpos):
            self._mem1[0, i].copy_(f.maskmem_features[0])

        # memory_pos_embed — mask portion: pos_enc + temporal_code (in-place)
        for i, (f, t_pos) in enumerate(frames_tpos):
            tcode = self.temporal_code[NUM_MASKMEM - 1 - t_pos].view(1, 64)
            s = i * 4096
            self._mempos[0, s:s + 4096].copy_(f.maskmem_pos_enc[0])
            self._mempos[0, s:s + 4096].add_(tcode)

        # memory_0 — cond first, then newest->oldest non-cond
        self._mem0[0, 0].copy_(self.cond_frame.obj_ptr[0])
        ptr_n = 1
        for f in reversed(self.non_cond_frames):
            if 1 <= (current_idx - f.frame_idx) <= NUM_MASKMEM - 1:
                self._mem0[0, ptr_n].copy_(f.obj_ptr[0])
                ptr_n += 1

        # Zero the obj_ptr portion of mempos
        total_pos = n * 4096 + ptr_n * 4
        self._mempos[0, n * 4096:total_pos].zero_()

        self._cond_diff.fill_(float(current_idx - self.cond_frame.frame_idx))

        return (self._mem0[:, :ptr_n],
                self._mem1[:, :n],
                self._mempos[:, :total_pos],
                self._cond_diff)


# ─── GPU pre / post processing ──────────────────────────────────────────────

def preprocess_gpu(frame_bgr: np.ndarray, mean: torch.Tensor, std: torch.Tensor):
    """BGR uint8 numpy -> normalised [1,3,1024,1024] float32 CUDA tensor."""
    h, w = frame_bgr.shape[:2]
    t = torch.from_numpy(frame_bgr).cuda().float()          # [H,W,3]
    t = t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W] RGB
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                      mode="bilinear", align_corners=False)
    t.sub_(mean).div_(std)                                   # in-place normalize
    return t, h, w


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    out = frame.copy()
    roi = mask > 0
    out[roi] = (out[roi].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


# ─── Threaded video I/O ─────────────────────────────────────────────────────

class VideoReaderThread:
    def __init__(self, path: str, qsize: int = 3):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.q: queue.Queue = queue.Queue(maxsize=qsize)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ret, frame = self.cap.read()
            self.q.put(frame if ret else None)
            if not ret:
                break

    def read(self):
        return self.q.get()

    @property
    def fps(self): return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    @property
    def frame_count(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    @property
    def width(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self): self.cap.release()


class VideoWriterThread:
    def __init__(self, path: str, fourcc, fps, size, qsize: int = 5):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.q: queue.Queue = queue.Queue(maxsize=qsize)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            self.writer.write(frame)

    def write(self, frame): self.q.put(frame)

    def release(self):
        self.q.put(None)
        self._t.join()
        self.writer.release()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="SAM2.1 TRT Tracker (GPU-optimised)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", help="x1,y1,x2,y2")
    ap.add_argument("--point", help="x,y")
    ap.add_argument("--model-dir", default="/models/sam2/tiny/trt/")
    ap.add_argument("--output", default="/output/output_tracked_trt.mp4")
    ap.add_argument("--save-masks", default=None, help="Directory to save per-frame binary masks as .npz")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.save_masks:
        os.makedirs(args.save_masks, exist_ok=True)

    if not args.bbox and not args.point:
        ap.error("must provide --bbox or --point")
    if args.bbox and args.point:
        ap.error("provide either --bbox or --point, not both")

    # GPU constants (allocated once)
    mean_gpu = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1)
    std_gpu = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1)

    # No-prompt tensors (reused every frame 1+)
    no_prompt_coords = torch.zeros(1, 1, 2, dtype=torch.float32, device="cuda")
    no_prompt_labels = torch.full((1, 1), -1.0, dtype=torch.float32, device="cuda")

    # ── Load TRT engines (shared non-default stream) ───────────────────
    trt_stream = torch.cuda.Stream()
    print("Loading TensorRT engines...")
    enc  = TRTEngine(os.path.join(args.model_dir, "image_encoder.engine"), trt_stream)
    dec  = TRTEngine(os.path.join(args.model_dir, "mask_decoder.engine"), trt_stream)
    menc = TRTEngine(os.path.join(args.model_dir, "memory_encoder.engine"), trt_stream)
    matt = TRTEngine(os.path.join(args.model_dir, "memory_attention.engine"), trt_stream)

    for label, e in [("image_encoder", enc), ("mask_decoder", dec),
                     ("memory_encoder", menc), ("memory_attention", matt)]:
        e.print_io(label)

    # ── Video I/O (threaded) ────────────────────────────────────────────
    reader = VideoReaderThread(args.video)
    w0, h0, fps_in = reader.width, reader.height, reader.fps
    total = reader.frame_count
    print(f"Video: {w0}x{h0} @ {fps_in:.1f} fps, {total} frames")

    writer = VideoWriterThread(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w0, h0)
    )

    # ── Warmup all engines (first call has TRT context init cost) ───────
    print("Warmup...")
    dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device="cuda")
    pf, h0f, h1f, vf, vp = enc({"image": dummy_img})
    dec({"point_coords": torch.zeros(1, 1, 2, device="cuda"),
         "point_labels": torch.zeros(1, 1, device="cuda"),
         "image_embed": vf, "high_res_feats_0": h0f, "high_res_feats_1": h1f})
    menc({"mask_for_mem": torch.zeros(1, 1, 1024, 1024, device="cuda"),
          "pix_feat": pf, "occ_logit": torch.zeros(1, 1, device="cuda")})
    matt({"current_vision_feat": vf, "current_vision_pos_embed": vp,
          "memory_0": torch.zeros(1, 1, 256, device="cuda"),
          "memory_1": torch.zeros(1, 1, 64, 64, 64, device="cuda"),
          "memory_pos_embed": torch.zeros(1, 4100, 64, device="cuda"),
          "cond_frame_id_diff": torch.tensor(1.0, device="cuda")})
    trt_stream.synchronize()
    del dummy_img, pf, h0f, h1f, vf, vp
    print("Ready.")

    bank = MemoryBank()
    frame_idx = 0

    # ── CUDA event timing ──────────────────────────────────────────────
    def ev():
        return torch.cuda.Event(enable_timing=True)

    stats = defaultdict(list)  # stage_name -> [ms, ms, ...]

    while True:
        frame = reader.read()
        if frame is None:
            break
        t0 = time.perf_counter()

        # ── GPU preprocess ──────────────────────────────────────────
        e_pre_s, e_pre_e = ev(), ev()
        e_pre_s.record(trt_stream)
        with torch.cuda.stream(trt_stream):
            img, orig_h, orig_w = preprocess_gpu(frame, mean_gpu, std_gpu)
        e_pre_e.record(trt_stream)

        # ── Image encoder ───────────────────────────────────────────
        e_enc_s, e_enc_e = ev(), ev()
        e_enc_s.record(trt_stream)
        pix_feat, hr0, hr1, vision_feats, vision_pos = enc({"image": img})
        e_enc_e.record(trt_stream)

        if frame_idx == 0:
            # ── Frame 0: prompted ────────────────────────────────────
            sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
            if args.bbox:
                bbox = [float(v) for v in args.bbox.split(",")]
                point_coords = torch.tensor(
                    [[[bbox[0]*sx, bbox[1]*sy], [bbox[2]*sx, bbox[3]*sy]]],
                    dtype=torch.float32, device="cuda",
                )
                point_labels = torch.tensor(
                    [[2.0, 3.0]], dtype=torch.float32, device="cuda"
                )
            else:
                pt = [float(v) for v in args.point.split(",")]
                point_coords = torch.tensor(
                    [[[pt[0]*sx, pt[1]*sy]]],
                    dtype=torch.float32, device="cuda",
                )
                point_labels = torch.tensor(
                    [[1.0]], dtype=torch.float32, device="cuda"
                )

            e_dec_s, e_dec_e = ev(), ev()
            e_dec_s.record(trt_stream)
            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec({
                "point_coords": point_coords,
                "point_labels": point_labels,
                "image_embed": vision_feats,
                "high_res_feats_0": hr0,
                "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, temporal_code = menc({
                "mask_for_mem": mask_for_mem,
                "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            bank.temporal_code = temporal_code.clone()
            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(),
                frame_idx=0,
                is_cond=True,
            ))

        else:
            # ── Frame 1+: propagation ────────────────────────────────
            e_bank_s, e_bank_e = ev(), ev()
            e_bank_s.record(trt_stream)
            mem0, mem1, mem_pos, cond_diff = bank.build_memory_inputs(frame_idx)
            e_bank_e.record(trt_stream)

            e_matt_s, e_matt_e = ev(), ev()
            e_matt_s.record(trt_stream)
            (image_embed,) = matt({
                "current_vision_feat": vision_feats,
                "current_vision_pos_embed": vision_pos,
                "memory_0": mem0,
                "memory_1": mem1,
                "memory_pos_embed": mem_pos,
                "cond_frame_id_diff": cond_diff,
            })
            e_matt_e.record(trt_stream)

            e_dec_s, e_dec_e = ev(), ev()
            e_dec_s.record(trt_stream)
            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec({
                "point_coords": no_prompt_coords,
                "point_labels": no_prompt_labels,
                "image_embed": image_embed,
                "high_res_feats_0": hr0,
                "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, _ = menc({
                "mask_for_mem": mask_for_mem,
                "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(),
                frame_idx=frame_idx,
                is_cond=False,
            ))

        # ── Visualize (only pred_mask + iou cross to CPU) ───────────
        e_post_s, e_post_e = ev(), ev()
        e_post_s.record(trt_stream)
        with torch.cuda.stream(trt_stream):
            logits = F.interpolate(pred_mask, size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False)
            mask_gpu = (logits[0, 0] > 0.0).byte()
        e_post_e.record(trt_stream)
        trt_stream.synchronize()

        # ── Collect timings ─────────────────────────────────────────
        stats["preprocess"].append(e_pre_s.elapsed_time(e_pre_e))
        stats["image_encoder"].append(e_enc_s.elapsed_time(e_enc_e))
        stats["mask_decoder"].append(e_dec_s.elapsed_time(e_dec_e))
        stats["memory_encoder"].append(e_menc_s.elapsed_time(e_menc_e))
        stats["postprocess"].append(e_post_s.elapsed_time(e_post_e))
        if frame_idx > 0:
            stats["mem_bank_build"].append(e_bank_s.elapsed_time(e_bank_e))
            stats["memory_attention"].append(e_matt_s.elapsed_time(e_matt_e))

        binary_mask = mask_gpu.cpu().numpy()
        if args.save_masks:
            np.savez_compressed(os.path.join(args.save_masks, f"{frame_idx:05d}.npz"), mask=binary_mask)
        iou_val = iou.item()
        stats["iou"].append(iou_val)

        vis = overlay_mask(frame, binary_mask)
        dt = time.perf_counter() - t0
        fps_now = 1.0 / dt if dt > 0 else 0
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

    reader.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done — {frame_idx} frames written to {args.output}")

    # ── Timing summary ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Timing breakdown ({frame_idx} frames, CUDA events, ms)")
    print(f"{'─'*70}")
    print(f"  {'stage':<20s} {'mean':>8s} {'min':>8s} {'max':>8s} {'total':>10s}  {'%':>5s}")
    print(f"  {'─'*57}")
    grand_total = sum(sum(v) for v in stats.values())
    order = ["preprocess", "image_encoder", "mem_bank_build",
             "memory_attention", "mask_decoder", "memory_encoder", "postprocess"]
    for name in order:
        if name not in stats:
            continue
        vals = stats[name]
        s = sum(vals)
        pct = 100.0 * s / grand_total if grand_total > 0 else 0
        print(f"  {name:<20s} {s/len(vals):>8.2f} {min(vals):>8.2f} {max(vals):>8.2f}"
              f" {s:>10.1f}  {pct:>5.1f}%")
    print(f"  {'─'*57}")
    print(f"  {'TOTAL':<20s} {'':>8s} {'':>8s} {'':>8s} {grand_total:>10.1f}  100.0%")
    print(f"{'─'*70}")

    # ── IoU summary ────────────────────────────────────────────────────
    ious = stats["iou"]
    mean_iou = sum(ious) / len(ious)
    below_90 = sum(1 for v in ious if v < 0.90)
    below_80 = sum(1 for v in ious if v < 0.80)
    print(f"\n  IoU: mean={mean_iou:.3f}  min={min(ious):.3f}  max={max(ious):.3f}"
          f"  <0.9={below_90}  <0.8={below_80}")


if __name__ == "__main__":
    main()
