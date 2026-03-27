#!/usr/bin/env python3
"""SAM2.1 Hiera Tiny — TensorRT Video Tracker

Zero-copy GPU pipeline: all intermediate tensors stay on GPU.
Only the raw frame (in) and binary mask (out) cross PCIe.
"""

import argparse
import os
import subprocess
import time

import cv2
import torch
import torch.nn.functional as F

from tracker.engines import TRTEngine
from tracker.memory import MemoryBank, MemoryFrame
from tracker.video_io import VideoReaderThread, VideoWriterThread
from tracker.preprocess import preprocess_gpu, overlay_mask
from tracker.timing import TimingStats
from tracker.constants import IMG_SIZE


def main():
    ap = argparse.ArgumentParser(description="SAM2.1 TRT Tracker (GPU-optimised)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", help="x1,y1,x2,y2")
    ap.add_argument("--point", help="x,y")
    ap.add_argument("--model-dir", default="/models/sam2/tiny/trt/")
    ap.add_argument("--output", default="/output/output_tracked_trt.mp4")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.bbox and not args.point:
        ap.error("must provide --bbox or --point")
    if args.bbox and args.point:
        ap.error("provide either --bbox or --point, not both")

    # GPU constants
    mean_gpu = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1)
    std_gpu = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1)
    no_prompt_coords = torch.zeros(1, 1, 2, dtype=torch.float32, device="cuda")
    no_prompt_labels = torch.full((1, 1), -1.0, dtype=torch.float32, device="cuda")

    # ── Load TRT engines ──────────────────────────────────────────────
    trt_stream = torch.cuda.Stream()
    print("Loading TensorRT engines...")
    enc  = TRTEngine(os.path.join(args.model_dir, "image_encoder.engine"), trt_stream)
    dec  = TRTEngine(os.path.join(args.model_dir, "mask_decoder.engine"), trt_stream)
    menc = TRTEngine(os.path.join(args.model_dir, "memory_encoder.engine"), trt_stream)
    matt = TRTEngine(os.path.join(args.model_dir, "memory_attention.engine"), trt_stream)

    for label, e in [("image_encoder", enc), ("mask_decoder", dec),
                     ("memory_encoder", menc), ("memory_attention", matt)]:
        e.print_io(label)

    # ── Video I/O ─────────────────────────────────────────────────────
    reader = VideoReaderThread(args.video)
    w0, h0, fps_in = reader.width, reader.height, reader.fps
    total = reader.frame_count
    print(f"Video: {w0}x{h0} @ {fps_in:.1f} fps, {total} frames")

    writer = VideoWriterThread(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w0, h0)
    )

    # ── Warmup ────────────────────────────────────────────────────────
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
    ts = TimingStats()
    ev = ts.event

    # ── Tracking loop ─────────────────────────────────────────────────
    while True:
        frame = reader.read()
        if frame is None:
            break
        t0 = time.perf_counter()

        # Preprocess
        e_pre_s, e_pre_e = ev(), ev()
        e_pre_s.record(trt_stream)
        with torch.cuda.stream(trt_stream):
            img, orig_h, orig_w = preprocess_gpu(frame, mean_gpu, std_gpu)
        e_pre_e.record(trt_stream)

        # Image encoder
        e_enc_s, e_enc_e = ev(), ev()
        e_enc_s.record(trt_stream)
        pix_feat, hr0, hr1, vision_feats, vision_pos = enc({"image": img})
        e_enc_e.record(trt_stream)

        if frame_idx == 0:
            # ── Frame 0: prompted ─────────────────────────────────────
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
                "point_coords": point_coords, "point_labels": point_labels,
                "image_embed": vision_feats,
                "high_res_feats_0": hr0, "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, temporal_code = menc({
                "mask_for_mem": mask_for_mem, "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            bank.temporal_code = temporal_code.clone()
            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(), frame_idx=0, is_cond=True,
            ))

        else:
            # ── Frame 1+: propagation ─────────────────────────────────
            e_bank_s, e_bank_e = ev(), ev()
            e_bank_s.record(trt_stream)
            mem0, mem1, mem_pos, cond_diff = bank.build_memory_inputs(frame_idx)
            e_bank_e.record(trt_stream)

            e_matt_s, e_matt_e = ev(), ev()
            e_matt_s.record(trt_stream)
            (image_embed,) = matt({
                "current_vision_feat": vision_feats,
                "current_vision_pos_embed": vision_pos,
                "memory_0": mem0, "memory_1": mem1,
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
                "high_res_feats_0": hr0, "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, _ = menc({
                "mask_for_mem": mask_for_mem, "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(), frame_idx=frame_idx, is_cond=False,
            ))

        # ── Postprocess ───────────────────────────────────────────────
        e_post_s, e_post_e = ev(), ev()
        e_post_s.record(trt_stream)
        with torch.cuda.stream(trt_stream):
            logits = F.interpolate(pred_mask, size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False)
            mask_gpu = (logits[0, 0] > 0.0).byte()
        e_post_e.record(trt_stream)
        trt_stream.synchronize()

        # ── Timings ───────────────────────────────────────────────────
        ts.record("preprocess", e_pre_s, e_pre_e)
        ts.record("image_encoder", e_enc_s, e_enc_e)
        ts.record("mask_decoder", e_dec_s, e_dec_e)
        ts.record("memory_encoder", e_menc_s, e_menc_e)
        ts.record("postprocess", e_post_s, e_post_e)
        if frame_idx > 0:
            ts.record("mem_bank_build", e_bank_s, e_bank_e)
            ts.record("memory_attention", e_matt_s, e_matt_e)

        binary_mask = mask_gpu.cpu().numpy()
        iou_val = iou.item()
        ts.append("iou", iou_val)

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

    ts.print_summary(frame_idx)
    ts.print_iou_summary()


if __name__ == "__main__":
    main()
