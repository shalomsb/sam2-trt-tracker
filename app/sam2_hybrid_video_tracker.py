#!/usr/bin/env python3
"""SAM2.1 Hiera Tiny — Hybrid TRT + ONNX Runtime Video Tracker

3 TRT engines (image_encoder, memory_encoder, memory_attention)
+ ONNX Runtime CUDA EP for mask_decoder (TRT 10.3 Myelin bug workaround).
"""

import argparse
import os
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tracker.engines import TRTEngine, ORTEngine
from tracker.memory import MemoryBank, MemoryFrame
from tracker.video_io import VideoReaderThread, VideoWriterThread
from tracker.preprocess import preprocess_gpu, overlay_mask, bbox_from_mask_gpu, draw_bbox
from tracker.timing import TimingStats
from tracker.constants import IMG_SIZE


def main():
    ap = argparse.ArgumentParser(description="SAM2.1 Hybrid TRT+ORT Tracker")
    ap.add_argument("--video", required=True)
    ap.add_argument("--bbox", help="x1,y1,x2,y2")
    ap.add_argument("--point", help="x,y")
    ap.add_argument("--model-dir", default="/models/sam2/tiny/trt/")
    ap.add_argument("--onnx-dir", default="/models/sam2/tiny/onnx/")
    ap.add_argument("--output", default="/output/output_tracked_hybrid.mp4")
    ap.add_argument("--save-masks", default=None, help="Directory to save per-frame binary masks (.npz)")
    ap.add_argument("--bbox-only", action="store_true",
                    help="Output bounding boxes instead of full masks (lighter postprocessing)")
    ap.add_argument("--save-bboxes", default=None,
                    help="Path to write per-frame bbox CSV (frame_idx,x1,y1,x2,y2,iou)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.bbox and not args.point:
        ap.error("must provide --bbox or --point")
    if args.bbox and args.point:
        ap.error("provide either --bbox or --point, not both")

    save_masks = args.save_masks
    if save_masks:
        os.makedirs(save_masks, exist_ok=True)

    bbox_csv_file = None
    bbox_csv_writer = None
    if args.save_bboxes:
        import csv
        os.makedirs(os.path.dirname(args.save_bboxes) or ".", exist_ok=True)
        bbox_csv_file = open(args.save_bboxes, "w", newline="")
        bbox_csv_writer = csv.writer(bbox_csv_file)
        bbox_csv_writer.writerow(["frame_idx", "x1", "y1", "x2", "y2", "iou"])

    # ImageNet mean/std for normalising input frames, pre-loaded on GPU
    # to avoid CPU→GPU transfer every frame. Shape [1,3,1,1] for broadcasting.
    mean_gpu = torch.tensor([123.675, 116.28, 103.53], device="cuda").view(1, 3, 1, 1)
    std_gpu = torch.tensor([58.395, 57.12, 57.375], device="cuda").view(1, 3, 1, 1)
    # "No prompt" inputs for frames 1+. Label -1.0 tells the decoder "no point was clicked".
    no_prompt_coords = torch.zeros(1, 1, 2, dtype=torch.float32, device="cuda")
    no_prompt_labels = torch.full((1, 1), -1.0, dtype=torch.float32, device="cuda")

    # ── Load engines ──────────────────────────────────────────────────
    # All TRT engines share one stream so they run sequentially on the GPU.
    # dec is ORT because TRT 10.3 has a Myelin compiler bug on mask_decoder.
    trt_stream = torch.cuda.Stream()
    print("Loading TRT engines + ORT mask_decoder...")
    enc  = TRTEngine(os.path.join(args.model_dir, "image_encoder.engine"), trt_stream)
    dec  = ORTEngine(os.path.join(args.onnx_dir, "mask_decoder.onnx"))
    menc = TRTEngine(os.path.join(args.model_dir, "memory_encoder.engine"), trt_stream)
    matt = TRTEngine(os.path.join(args.model_dir, "memory_attention.engine"), trt_stream)

    # Print engine I/O info for debugging (names, shapes, dtypes)
    for label, e in [("image_encoder", enc), ("mask_decoder", dec),
                     ("memory_encoder", menc), ("memory_attention", matt)]:
        e.print_io(label)

    # ── Video I/O ─────────────────────────────────────────────────────
    # We use threaded reader that decodes video frames in a background thread, so the GPU can start processing the next
    # frame without waiting for the next one to decode.
    reader = VideoReaderThread(args.video)
    # Get video properties for setting up the writer and for scaling prompts. 
    w0, h0, fps_in = reader.width, reader.height, reader.fps
    total = reader.frame_count
    print(f"Video: {w0}x{h0} @ {fps_in:.1f} fps, {total} frames")

    # The writer thread encodes and saves output frames in the background, so the main loop can move on to the next frame
    # without waiting for disk I/O. We wirte the output at the same resolution and same fps as the input.
    writer = VideoWriterThread(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w0, h0)
    )

    # ── Warmup ────────────────────────────────────────────────────────
    # Run each engine once with dummy data to trigger TRT's internal setup
    # (kernel selection, scratch memory allocation). Without this, frame 0
    # would be artificially slow.
    print("Warmup...")
    dummy_img = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device="cuda")
    # These are references to enc's pre-allocated output buffers, not copies.
    pf, h0f, h1f, vf, vp = enc({"image": dummy_img})
    # Sync before ORT: TRT runs async, so we must ensure enc's outputs are
    # fully written before ORT tries to read them on its own CUDA stream.
    trt_stream.synchronize()
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
    # Wait for all warmup GPU work to finish before starting the real loop.
    trt_stream.synchronize()
    # Free dummy_img (the only real allocation here). pf/vf/etc. are just references
    # to enc's pre-allocated buffers — those stay alive inside enc._static_bufs.
    del dummy_img, pf, h0f, h1f, vf, vp
    print("Ready.")

    bank = MemoryBank()
    frame_idx = 0
    ts = TimingStats()
    ev = ts.event  # shorthand for creating CUDA timing events
    bbox_only = args.bbox_only

    # ── Tracking loop ─────────────────────────────────────────────────
    while True:
        frame = reader.read()  # blocks until the reader thread has a frame ready (None = video ended)
        if frame is None:
            break
        t0 = time.perf_counter()  # wall-clock timer for FPS display

        # Preprocess: BGR uint8 numpy (e.g. 1280×720) → RGB → resize 1024×1024 → normalize
        # → [1,3,1024,1024] float32 on GPU. Also returns orig_h, orig_w to scale mask back later.
        # ev() creates CUDA timing events — stamps on the GPU timeline for accurate timing.
        e_pre_s, e_pre_e = ev(), ev()
        e_pre_s.record(trt_stream)
        # "with torch.cuda.stream" makes PyTorch ops (resize, normalize) run on our TRT stream
        # instead of the default stream, so they stay in order with the TRT engine calls.
        with torch.cuda.stream(trt_stream):
            img, orig_h, orig_w = preprocess_gpu(frame, mean_gpu, std_gpu)
        e_pre_e.record(trt_stream)

        # Image encoder: img → 5 outputs (all references to enc's pre-allocated buffers):
        #   pix_feat     [1,256,64,64]  → memory encoder (pixel-level features)
        #   hr0          [1,32,256,256] → mask decoder (high-res features for fine detail)
        #   hr1          [1,64,128,128] → mask decoder (mid-res features)
        #   vision_feats [1,4096,256]   → mask decoder or memory attention (main vision features)
        #   vision_pos   [1,4096,256]   → memory attention (positional encoding of vision features)
        e_enc_s, e_enc_e = ev(), ev()
        e_enc_s.record(trt_stream)
        pix_feat, hr0, hr1, vision_feats, vision_pos = enc({"image": img})
        e_enc_e.record(trt_stream)

        if frame_idx == 0:
            # ── Frame 0: prompted ─────────────────────────────────────
            # Scale the user's bbox/point from original resolution to 1024x1024 model input.
            sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
            if args.bbox:
                bbox = [float(v) for v in args.bbox.split(",")]
                # Bbox is encoded as 2 points (top-left, bottom-right) with labels 2,3.
                point_coords = torch.tensor(
                    [[[bbox[0]*sx, bbox[1]*sy], [bbox[2]*sx, bbox[3]*sy]]],
                    dtype=torch.float32, device="cuda",
                )
                point_labels = torch.tensor(
                    [[2.0, 3.0]], dtype=torch.float32, device="cuda"
                )
            # If it's a point prompt, we encode it as a single point with label 1.0 (foreground).
            else:
                pt = [float(v) for v in args.point.split(",")]
                # we multiply by sx, sy to scale the point from original resolution to 1024x1024 model input size.
                point_coords = torch.tensor(
                    [[[pt[0]*sx, pt[1]*sy]]],
                    dtype=torch.float32, device="cuda",
                )
                # label 1.0 = foreground point, 0.0 = background point.
                point_labels = torch.tensor(
                    [[1.0]], dtype=torch.float32, device="cuda"
                )

            # Sync: enc (TRT) ran async → must finish before dec (ORT) reads its outputs.
            trt_stream.synchronize()

            # Mask decoder: takes the prompt + encoder features → produces mask + object pointer.
            e_dec_s, e_dec_e = ev(), ev()
            e_dec_s.record(trt_stream)
            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec({
                "point_coords": point_coords, "point_labels": point_labels,
                "image_embed": vision_feats,
                "high_res_feats_0": hr0, "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            # Memory encoder: takes the predicted mask + pixel features → produces memory
            # tensors that will be stored in the memory bank for future frames.
            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, temporal_code = menc({
                "mask_for_mem": mask_for_mem, "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            # Store frame 0 as the conditioning frame. .clone() because the tensors
            # are references to pre-allocated engine buffers that get overwritten each frame.
            bank.temporal_code = temporal_code.clone()
            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(), frame_idx=0, is_cond=True,
            ))

        else:
            # ── Frame 1+: propagation (no prompt, uses memory) ────────
            # Pack stored memory frames into the 4 tensors memory_attention expects.
            e_bank_s, e_bank_e = ev(), ev()
            e_bank_s.record(trt_stream)
            # mem0 -> [1,NUM_MASKMEM,256] object pointers,
            # mem1 -> [1,NUM_MASKMEM,64,64,64] mask memory features,
            # mem_pos -> [1,NUM_MASKMEM*4096+NUM_MASKMEM*4,64] positional encodings for both (4096 for mem1's spatial features, 4 for mem0's object pointers),
            # cond_diff -> scalar tensor with current frame_idx - cond_frame.frame_idx, to tell the model how many frames ago the conditioning frame was. 
            mem0, mem1, mem_pos, cond_diff = bank.build_memory_inputs(frame_idx)
            e_bank_e.record(trt_stream)

            # Memory attention: enriches current frame's vision features with past memory.
            # Output image_embed replaces vision_feats for the decoder.
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

            # Sync: matt (TRT) ran async → must finish before dec (ORT) reads image_embed.
            trt_stream.synchronize()

            # Mask decoder: no prompt this time (label=-1), relies on memory-enriched features.
            e_dec_s, e_dec_e = ev(), ev()
            e_dec_s.record(trt_stream)
            obj_ptr, mask_for_mem, pred_mask, iou, occ_logit = dec({
                "point_coords": no_prompt_coords,
                "point_labels": no_prompt_labels,
                "image_embed": image_embed,
                "high_res_feats_0": hr0, "high_res_feats_1": hr1,
            })
            e_dec_e.record(trt_stream)

            # Memory encoder: encode this frame's mask into memory for future frames.
            e_menc_s, e_menc_e = ev(), ev()
            e_menc_s.record(trt_stream)
            maskmem_feat, maskmem_pos, _ = menc({
                "mask_for_mem": mask_for_mem, "pix_feat": pix_feat,
                "occ_logit": occ_logit,
            })
            e_menc_e.record(trt_stream)

            # Store as non-conditioning frame (evicts the oldest if we already have 2).
            bank.add(MemoryFrame(
                maskmem_features=maskmem_feat.clone(),
                maskmem_pos_enc=maskmem_pos.clone(),
                obj_ptr=obj_ptr.clone(), frame_idx=frame_idx, is_cond=False,
            ))

        # ── Postprocess ───────────────────────────────────────────────
        # pred_mask is 256x256 logits. Either extract a bbox from it directly,
        # or upscale to original resolution and threshold into a binary mask.
        e_post_s, e_post_e = ev(), ev()
        e_post_s.record(trt_stream)

        if bbox_only:
            cur_bbox = bbox_from_mask_gpu(pred_mask, orig_w, orig_h)
        else:
            with torch.cuda.stream(trt_stream):
                logits = F.interpolate(pred_mask, size=(orig_h, orig_w),
                                       mode="bilinear", align_corners=False)
                mask_gpu = (logits[0, 0] > 0.0).byte()

        e_post_e.record(trt_stream)
        # Sync: we need the mask/bbox on CPU for visualization and file I/O.
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

        iou_val = iou.item()
        ts.append("iou", iou_val)

        # ── Visualize and save ────────────────────────────────────────
        if bbox_only:
            if bbox_csv_writer and cur_bbox:
                bbox_csv_writer.writerow([frame_idx, *cur_bbox, f"{iou_val:.4f}"])
            elif bbox_csv_writer:
                bbox_csv_writer.writerow([frame_idx, "", "", "", "", f"{iou_val:.4f}"])
            vis = draw_bbox(frame, cur_bbox)
        else:
            binary_mask = mask_gpu.cpu().numpy()
            if save_masks:
                np.savez_compressed(
                    os.path.join(save_masks, f"{frame_idx:05d}.npz"),
                    mask=binary_mask,
                )
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
        if args.max_frames and frame_idx >= args.max_frames:
            break

    reader.release()
    writer.release()
    if bbox_csv_file:
        bbox_csv_file.close()
        print(f"Bboxes written to {args.save_bboxes}")
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done — {frame_idx} frames written to {args.output}")

    # Re-encode mp4v → H.264 so the video plays in VS Code / browsers.
    # OpenCV on Jetson can't write H.264 directly, so we do it as a post-step.
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
