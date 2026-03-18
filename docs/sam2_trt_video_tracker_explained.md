# SAM2.1 TensorRT Video Tracker — Architecture & Code Explained

## What This Does

Tracks an object across every frame of a video using **SAM2.1 Hiera Tiny**.
You give it a video and a prompt (bounding box or point) on frame 0.
It segments the object in every subsequent frame automatically.

All heavy computation stays on the **GPU** — only the raw frame (in) and binary mask (out) cross PCIe.

---

## High-Level Pipeline

```
                         ┌─────────────────────────────────────────────────┐
                         │                   GPU (CUDA)                    │
                         │                                                 │
  ┌──────────┐  frame    │  ┌────────────┐    ┌──────────────────────┐     │   mask     ┌──────────┐
  │  Video   │──(BGR)───▶│  │ Preprocess │───▶│    Image Encoder     │     │──(binary)─▶│  Video   │
  │  Reader  │  uint8    │  │  (resize,  │    │   (TensorRT)         │     │   uint8    │  Writer  │
  │ (Thread) │  numpy    │  │ normalize) │    └──────────┬───────────┘     │   numpy    │ (Thread) │
  └──────────┘           │  └────────────┘               │                 │            └──────────┘
       │                 │                               ▼                 │                 │
       │                 │               ┌───────────────────────────┐     │                 │
       │                 │               │  Frame 0?                 │     │                 │
       │                 │               │  YES ──▶ Mask Decoder     │     │                 │
       │                 │               │          (with prompt)    │     │                 │
       │                 │               │                           │     │                 │
       │                 │               │  NO  ──▶ Memory Attention │     │                 │
       │                 │               │          ──▶ Mask Decoder │     │                 │
       │                 │               │          (no prompt)      │     │                 │
       │                 │               └─────────────┬─────────────┘     │                 │
       │                 │                             │                   │                 │
       │                 │                             ▼                   │                 │
       │                 │                    ┌─────────────────┐          │                 │
       │                 │                    │ Memory Encoder  │          │                 │
       │                 │                    │  (store frame   │          │                 │
       │                 │                    │   in bank)      │          │                 │
       │                 │                    └────────┬────────┘          │                 │
       │                 │                             │                   │                 │
       │                 │                             ▼                   │                 │
       │                 │                    ┌─────────────────┐          │                 │
       │                 │                    │  Postprocess    │          │                 │
       │                 │                    │  (resize mask,  │          │                 │
       │                 │                    │   threshold)    │          │                 │
       │                 │                    └─────────────────┘          │                 │
       │                 └─────────────────────────────────────────────────┘                 │
       │                                                                                    │
       │                            ┌──────────────────────┐                                │
       └───────────────────────────▶│  Overlay mask on     │◀───────────────────────────────┘
                  original frame    │  frame + write       │          output frame
                   (CPU numpy)      └──────────────────────┘          (CPU numpy)
```

---

## The 4 TensorRT Engines

Each box below is a separate `.engine` file loaded at startup:

```
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                        TensorRT Engines (all on GPU)                            │
  │                                                                                 │
  │  ┌──────────────────────┐   ┌──────────────────────┐                            │
  │  │   IMAGE ENCODER      │   │   MASK DECODER       │                            │
  │  │                      │   │                      │                            │
  │  │  IN:  image          │   │  IN:  point_coords   │                            │
  │  │       [1,3,1024,1024]│   │       point_labels   │                            │
  │  │                      │   │       image_embed     │                            │
  │  │  OUT: pix_feat       │   │       high_res_0     │                            │
  │  │       high_res_0     │   │       high_res_1     │                            │
  │  │       high_res_1     │   │                      │                            │
  │  │       vision_feats   │   │  OUT: obj_ptr        │                            │
  │  │       vision_pos     │   │       mask_for_mem   │                            │
  │  └──────────────────────┘   │       pred_mask      │                            │
  │                              │       iou            │                            │
  │  ┌──────────────────────┐   │       occ_logit      │                            │
  │  │  MEMORY ENCODER      │   └──────────────────────┘                            │
  │  │                      │                                                       │
  │  │  IN:  mask_for_mem   │   ┌──────────────────────┐                            │
  │  │       pix_feat       │   │  MEMORY ATTENTION    │                            │
  │  │       occ_logit      │   │                      │                            │
  │  │                      │   │  IN:  vision_feat    │                            │
  │  │  OUT: maskmem_feat   │   │       vision_pos     │                            │
  │  │       maskmem_pos    │   │       memory_0       │                            │
  │  │       temporal_code  │   │       memory_1       │                            │
  │  └──────────────────────┘   │       memory_pos     │                            │
  │                              │       cond_diff      │                            │
  │                              │                      │                            │
  │                              │  OUT: image_embed    │                            │
  │                              └──────────────────────┘                            │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Frame 0 — Prompted Segmentation

The user provides a **bounding box** or **point** to tell SAM2 what to track.

```
  USER PROMPT                           FRAME 0 PIPELINE
  ─────────                             ─────────────────

  --bbox x1,y1,x2,y2                   ┌─────────┐
        │                               │  Frame  │
        │  scale to 1024x1024           │  (BGR)  │
        ▼                               └────┬────┘
  ┌──────────────┐                           │
  │ point_coords │                           ▼
  │ [[x1,y1],    │                     ┌───────────┐
  │  [x2,y2]]    │                     │ Preprocess│──▶ [1,3,1024,1024]
  │              │                     └─────┬─────┘
  │ point_labels │                           │
  │ [2.0, 3.0]  │                           ▼
  │ (box prompt) │                     ┌───────────┐
  └──────┬───────┘                     │  Image    │──▶ pix_feat
         │                             │  Encoder  │──▶ high_res_0, high_res_1
         │                             │           │──▶ vision_feats
         │                             └─────┬─────┘──▶ vision_pos
         │                                   │
         │         ┌─────────────────────────┘
         │         │
         ▼         ▼
    ┌────────────────────┐
    │    Mask Decoder     │
    │                     │
    │  prompt + features  │──▶ pred_mask ──▶ DISPLAY
    │                     │──▶ obj_ptr
    │                     │──▶ mask_for_mem
    │                     │──▶ iou, occ_logit
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │   Memory Encoder   │
    │                     │──▶ maskmem_feat ─┐
    │  mask + pix_feat   │──▶ maskmem_pos  ──┤
    │                     │──▶ temporal_code ─┤
    └────────────────────┘                   │
                                             ▼
                                   ┌──────────────────┐
                                   │   MEMORY BANK    │
                                   │                  │
                                   │  cond_frame = {  │
                                   │    features,     │
                                   │    pos_enc,      │
                                   │    obj_ptr,      │
                                   │    frame_idx: 0, │
                                   │    is_cond: True │
                                   │  }               │
                                   └──────────────────┘
```

**Prompt types:**

| Argument | `point_coords` shape | `point_labels` | Meaning |
|----------|---------------------|----------------|---------|
| `--bbox x1,y1,x2,y2` | `[1, 2, 2]` | `[2.0, 3.0]` | Top-left & bottom-right corners |
| `--point x,y` | `[1, 1, 2]` | `[1.0]` | Single foreground point |

---

## Frame 1+ — Memory-Driven Propagation

No user prompt. The model uses **memory** from previous frames to find the object.

```
    ┌──────────┐
    │  Frame N │
    │  (BGR)   │
    └────┬─────┘
         │
         ▼
    ┌───────────┐
    │ Preprocess│──▶ [1,3,1024,1024]
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │   Image   │──▶ pix_feat
    │  Encoder  │──▶ high_res_0, high_res_1
    │           │──▶ vision_feats ──────────────────────────┐
    └───────────┘──▶ vision_pos ────────────────────────────┤
                                                            │
    ┌──────────────────┐                                    │
    │   MEMORY BANK    │                                    │
    │                  │                                    │
    │  build_memory    │──▶ memory_0 (obj pointers) ────────┤
    │  _inputs()       │──▶ memory_1 (mask features) ───────┤
    │                  │──▶ memory_pos_embed ────────────────┤
    │                  │──▶ cond_frame_id_diff ─────────────┤
    └──────────────────┘                                    │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │ Memory Attention │
                                                  │                  │
                                                  │ Fuses current    │
                                                  │ frame features   │
                                                  │ with memory      │
                                                  │                  │
                                                  │ OUT: image_embed │
                                                  └────────┬─────────┘
                                                           │
                      ┌────────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────┐
    │      Mask Decoder          │
    │                            │
    │  IN: image_embed (fused)   │
    │      high_res_0, 1         │
    │      NO-PROMPT coords/     │
    │      labels (zeros / -1)   │──▶ pred_mask ──▶ DISPLAY
    │                            │──▶ obj_ptr
    │                            │──▶ mask_for_mem
    └──────────┬─────────────────┘──▶ iou, occ_logit
               │
               ▼
    ┌────────────────────┐
    │   Memory Encoder   │──▶ maskmem_feat ──┐
    │                     │──▶ maskmem_pos ───┤
    └────────────────────┘                   │
                                             ▼
                                   ┌──────────────────┐
                                   │   MEMORY BANK    │
                                   │                  │
                                   │  non_cond_frames │
                                   │  .append({       │
                                   │    features,     │
                                   │    pos_enc,      │
                                   │    obj_ptr,      │
                                   │    frame_idx: N, │
                                   │    is_cond: False│
                                   │  })              │
                                   └──────────────────┘
```

---

## Memory Bank — How It Works

The memory bank is the key to tracking across frames. It stores a **sliding window** of recent frames plus the **conditioning frame** (frame 0).

```
  NUM_MASKMEM = 3  (max memory slots)

  ┌─────────────────────────────────────────────────────────────────────┐
  │                         MEMORY BANK                                │
  │                                                                    │
  │  ┌─────────────────────────────────────────────────┐               │
  │  │  cond_frame (always kept, never evicted)        │               │
  │  │                                                  │               │
  │  │  maskmem_features : [1, 64, 64, 64]  (spatial)  │               │
  │  │  maskmem_pos_enc  : [1, 4096, 64]    (position) │               │
  │  │  obj_ptr          : [1, 256]         (identity) │               │
  │  │  frame_idx        : 0                            │               │
  │  │  is_cond          : True                         │               │
  │  └─────────────────────────────────────────────────┘               │
  │                                                                    │
  │  ┌─────────────────────────────────────────────────┐               │
  │  │  non_cond_frames  (sliding window, max 2 slots) │               │
  │  │                                                  │               │
  │  │  slot 0: frame N-2  ◀── oldest (evicted first)  │               │
  │  │  slot 1: frame N-1  ◀── newest                   │               │
  │  │                                                  │               │
  │  │  When a 3rd arrives, slot 0 is popped (FIFO)    │               │
  │  └─────────────────────────────────────────────────┘               │
  │                                                                    │
  │  temporal_code : [7, 1, 1, 64]  (shared time encoding)            │
  │                                                                    │
  └─────────────────────────────────────────────────────────────────────┘
```

### `build_memory_inputs()` assembles 4 tensors for Memory Attention:

```
  Memory Bank State (example at frame 10):
  ──────────────────────────────────────────

    cond_frame    (frame 0)    t_pos = 0       ──┐
    non_cond[0]   (frame 8)    t_pos = 3-2 = 1  ──┤
    non_cond[1]   (frame 9)    t_pos = 3-1 = 2  ──┤
                                                   │
                         ┌─────────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  OUTPUT TENSORS (sliced views of pre-allocated GPU buffers)     │
  │                                                                  │
  │  memory_1   [1, 3, 64, 64, 64]                                  │
  │  ├── slot 0: cond_frame.maskmem_features                        │
  │  ├── slot 1: non_cond[0].maskmem_features                       │
  │  └── slot 2: non_cond[1].maskmem_features                       │
  │                                                                  │
  │  memory_pos_embed   [1, total_pos, 64]                           │
  │  ├── [0..4095]   : cond.pos_enc + temporal_code[t=0]            │
  │  ├── [4096..8191]: nc[0].pos_enc + temporal_code[t=1]           │
  │  ├── [8192..12287]: nc[1].pos_enc + temporal_code[t=2]          │
  │  └── [12288..12299]: obj_ptr zero-padding (4 per pointer)       │
  │                                                                  │
  │  memory_0   [1, ptr_count, 256]                                  │
  │  ├── slot 0: cond_frame.obj_ptr     (always first)              │
  │  ├── slot 1: non_cond[1].obj_ptr    (newest first)              │
  │  └── slot 2: non_cond[0].obj_ptr    (then older)                │
  │                                                                  │
  │  cond_frame_id_diff   scalar = 10.0   (current_idx - 0)         │
  └──────────────────────────────────────────────────────────────────┘
```

---

## GPU vs CPU — What Lives Where

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                          CPU (Host)                                │
  │                                                                    │
  │   ┌──────────────┐              ┌───────────────┐                  │
  │   │ VideoReader   │  frame_bgr   │  overlay_mask │  output frame   │
  │   │ Thread        │──(numpy)──▶  │  + putText    │──(numpy)──▶     │
  │   └──────────────┘              └───────────────┘                  │
  │                                        ▲                           │
  │         Only these cross PCIe:         │                           │
  │         ─────────────────────          │                           │
  │         1. frame_bgr (IN)        binary_mask                       │
  │         2. binary_mask (OUT)     (numpy uint8)                     │
  │         3. iou scalar                  │                           │
  └────────────────────────────────────────┼───────────────────────────┘
                                           │
  ═══════════════════ PCIe Bus ════════════╪════════════════════════════
                                           │
  ┌────────────────────────────────────────┼───────────────────────────┐
  │                          GPU (CUDA)    │                           │
  │                                        │                           │
  │   torch.from_numpy(bgr).cuda()    mask_gpu.cpu().numpy()          │
  │           │                                ▲                       │
  │           ▼                                │                       │
  │   ┌─────────────┐  All intermediate  ┌────┴──────┐                │
  │   │  preprocess │  tensors stay on   │ threshold │                │
  │   │  (GPU ops)  │  GPU — zero copy   │ + resize  │                │
  │   └──────┬──────┘  between engines   └─────▲─────┘                │
  │          │                                  │                      │
  │          ▼                                  │                      │
  │   ┌──────────┐  ┌──────────┐  ┌─────────┐ │  ┌──────────────┐    │
  │   │  Image   │─▶│  Mem     │─▶│  Mask   │─┘  │   Memory     │    │
  │   │  Encoder │  │  Attn    │  │  Decoder│────▶│   Encoder    │    │
  │   └──────────┘  └──────────┘  └─────────┘     └──────┬───────┘    │
  │                                                       │            │
  │                                                       ▼            │
  │                                               ┌──────────────┐    │
  │                                               │ Memory Bank  │    │
  │                                               │ (GPU tensors)│    │
  │                                               └──────────────┘    │
  └────────────────────────────────────────────────────────────────────┘
```

---

## Threading Model

Disk I/O runs in background threads so GPU never waits for the disk.

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                          TIMELINE                                       │
  │                                                                         │
  │  Reader Thread        Main Thread (GPU)           Writer Thread          │
  │  ──────────────       ─────────────────           ─────────────          │
  │                                                                         │
  │  read frame 3 ──┐                                                       │
  │                  │    ┌────────────────────┐                             │
  │  read frame 4   └──▶ │ process frame 2    │                             │
  │       │               │ (encode+decode+    │                             │
  │       │               │  memory)           │──vis──▶ write frame 1      │
  │       │               └────────────────────┘              │              │
  │  read frame 5 ──┐                                         │              │
  │                  │    ┌────────────────────┐               │              │
  │                  └──▶ │ process frame 3    │──vis──▶ write frame 2      │
  │                       └────────────────────┘                             │
  │                                                                         │
  │  Queue sizes:   reader.q (max 3)        writer.q (max 5)               │
  │                 frames buffered ahead    frames buffered for writing    │
  └──────────────────────────────────────────────────────────────────────────┘

  VideoReaderThread                         VideoWriterThread
  ┌──────────────────────┐                  ┌──────────────────────┐
  │ cv2.VideoCapture     │                  │ cv2.VideoWriter      │
  │                      │                  │                      │
  │ daemon thread reads  │                  │ daemon thread writes │
  │ frames into queue    │                  │ frames from queue    │
  │                      │                  │                      │
  │ Puts None at EOF     │                  │ None = stop signal   │
  │ (signals end)        │                  │ join() waits for     │
  │                      │                  │ all writes to finish │
  └──────────────────────┘                  └──────────────────────┘
```

---

## TRTEngine Class — Zero-Copy Design

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                     TRTEngine.__init__()                         │
  │                                                                  │
  │  1. Deserialize .engine file                                     │
  │  2. Create execution context                                     │
  │  3. Scan all I/O tensors:                                        │
  │     ┌──────────────────────────────────────────────┐             │
  │     │  for each tensor:                            │             │
  │     │    INPUT?  ──▶ input_names[]                 │             │
  │     │               has -1 in shape? ──▶ dynamic   │             │
  │     │    OUTPUT? ──▶ output_names[]                │             │
  │     │               static shape? ──▶ pre-allocate │             │
  │     └──────────────────────────────────────────────┘             │
  │                                                                  │
  │  Pre-allocated static output buffers live for the engine's       │
  │  lifetime — no malloc per call.                                  │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                     TRTEngine.__call__()                         │
  │                                                                  │
  │  inputs: dict[str, Tensor]  ──▶  outputs: list[Tensor]          │
  │                                                                  │
  │  ┌─────────────────────────────────────────────┐                 │
  │  │  For each INPUT:                            │                 │
  │  │    .contiguous()                            │                 │
  │  │    set_input_shape (if dynamic)             │                 │
  │  │    set_tensor_address ──▶ GPU pointer       │                 │
  │  └─────────────────────────────────────────────┘                 │
  │                         │                                        │
  │                         ▼                                        │
  │  ┌─────────────────────────────────────────────┐                 │
  │  │  For each OUTPUT:                           │                 │
  │  │    use pre-allocated buffer (static)        │                 │
  │  │    OR allocate new tensor (dynamic)         │                 │
  │  │    set_tensor_address ──▶ GPU pointer       │                 │
  │  └─────────────────────────────────────────────┘                 │
  │                         │                                        │
  │                         ▼                                        │
  │  ┌─────────────────────────────────────────────┐                 │
  │  │  execute_async_v3(cuda_stream)              │                 │
  │  │  (non-blocking, runs on dedicated stream)   │                 │
  │  └─────────────────────────────────────────────┘                 │
  │                                                                  │
  │  Key: ALL data stays on GPU. Addresses are raw CUDA pointers    │
  │  from torch tensors (.data_ptr()). No host↔device copies.       │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Preprocessing — CPU to GPU Boundary

```
  Input: frame_bgr (numpy uint8, H x W x 3, BGR color order)

  ┌─────────────────────────────────────────────────────────────────┐
  │                      preprocess_gpu()                           │
  │                                                                 │
  │  numpy BGR uint8                                                │
  │       │                                                         │
  │       ▼  torch.from_numpy().cuda().float()                      │
  │  [H, W, 3] float32 GPU                                         │
  │       │                                                         │
  │       ▼  [:,:,[2,1,0]]  (BGR ──▶ RGB)                           │
  │  [H, W, 3] RGB                                                  │
  │       │                                                         │
  │       ▼  .permute(2,0,1).unsqueeze(0)                           │
  │  [1, 3, H, W]                                                   │
  │       │                                                         │
  │       ▼  F.interpolate(bilinear) ──▶ 1024 x 1024                │
  │  [1, 3, 1024, 1024]                                             │
  │       │                                                         │
  │       ▼  .sub_(mean).div_(std)    (ImageNet normalization)      │
  │  [1, 3, 1024, 1024] normalized                                  │
  │       │                                                         │
  │       ▼                                                         │
  │  Ready for Image Encoder                                        │
  └─────────────────────────────────────────────────────────────────┘

  mean = [123.675, 116.28, 103.53]   ◀── ImageNet RGB means
  std  = [58.395,  57.12,  57.375]   ◀── ImageNet RGB stds

  These are pre-allocated as [1,3,1,1] GPU tensors (broadcast-ready).
```

---

## Postprocessing — GPU to CPU Boundary

```
  pred_mask: [1, 1, 256, 256]  (raw logits from decoder, on GPU)

  ┌─────────────────────────────────────────────────────────────────┐
  │                      Postprocessing                             │
  │                                                                 │
  │  [1, 1, 256, 256] logits                                       │
  │       │                                                         │
  │       ▼  F.interpolate(bilinear) ──▶ original H x W            │
  │  [1, 1, H, W] logits at original resolution                    │
  │       │                                                         │
  │       ▼  > 0.0  (threshold)                                     │
  │  [H, W] boolean                                                 │
  │       │                                                         │
  │       ▼  .byte()                                                │
  │  [H, W] uint8 GPU   (0 or 1)                                   │
  │       │                                                         │
  │  ═════╪═══════════ PCIe transfer ════════════════════           │
  │       │                                                         │
  │       ▼  .cpu().numpy()                                         │
  │  [H, W] uint8 numpy  (binary mask)                              │
  │       │                                                         │
  │       ▼  overlay_mask()                                         │
  │  Green semi-transparent overlay + contour on original frame     │
  └─────────────────────────────────────────────────────────────────┘
```

---

## CUDA Stream & Timing

All TRT engines share a single non-default CUDA stream for ordered execution:

```
  Default Stream          trt_stream (non-default)
  ──────────────          ────────────────────────

  (idle)                  preprocess ──▶ encode ──▶ decode ──▶ mem_enc ──▶ postprocess
                                │              │           │            │
                              event          event       event        event
                              start/end      s/e         s/e          s/e

  trt_stream.synchronize()  ◀── waits for all GPU work to finish
                                 before reading mask on CPU
```

**CUDA Events** measure GPU time for each stage independently:

```
  ┌────────────────────────────────────────────────────────────────┐
  │  Timing Stats (printed at end)                                │
  │                                                                │
  │  stage              mean     min     max      total      %    │
  │  ─────────────────────────────────────────────────────────    │
  │  preprocess         X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  image_encoder      X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  mem_bank_build     X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  memory_attention   X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  mask_decoder       X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  memory_encoder     X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  │  postprocess        X.XX    X.XX    X.XX    XXXX.X    XX.X%  │
  └────────────────────────────────────────────────────────────────┘
```

---

## Tensor Flow Through the Entire Pipeline

A complete data-flow showing every tensor and which engine produces/consumes it:

```
  frame_bgr
      │
      ▼
  ╔══════════════╗
  ║  PREPROCESS  ║──▶ img [1,3,1024,1024]
  ╚══════════════╝
         │
         ▼
  ╔══════════════╗     pix_feat ────────────────────────────────▶ MEMORY ENCODER
  ║    IMAGE     ║     high_res_0 ──────────────────────────┐
  ║   ENCODER    ║     high_res_1 ──────────────────────┐   │
  ║              ║     vision_feats ──┐                  │   │
  ╚══════════════╝     vision_pos ──┐ │                  │   │
                                    │ │                  │   │
                     (frame 1+ only)│ │                  │   │
                                    ▼ ▼                  │   │
  ┌──────────────┐    ╔══════════════════╗               │   │
  │ MEMORY BANK  │──▶ ║    MEMORY        ║               │   │
  │              │    ║   ATTENTION      ║──▶ image_embed│   │
  │ mem_0        │    ╚══════════════════╝       │        │   │
  │ mem_1        │                               │        │   │
  │ mem_pos      │           (frame 0: use       │        │   │
  │ cond_diff    │            vision_feats       │        │   │
  └──────────────┘            directly)          │        │   │
                                    │            │        │   │
                                    ▼            ▼        ▼   ▼
                              ╔══════════════════════════════════╗
  no_prompt / prompt ────────▶║         MASK DECODER             ║
                              ║                                  ║
                              ║  OUT:                            ║
                              ║    obj_ptr [1,256] ─────────┐   ║
                              ║    mask_for_mem [1,1,1024²]──┤   ║
                              ║    pred_mask [1,1,256,256]   │   ║
                              ║    iou [1,1]                 │   ║
                              ║    occ_logit [1,1] ──────────┤   ║
                              ╚══════════════════════════════╝   │
                                                     │           │
                                        pred_mask    │           │
                                           │         ▼           │
                                           │   ╔═════════════╗   │
                                           │   ║   MEMORY    ║   │
                                           │   ║   ENCODER   ║   │
                                           │   ║             ║   │
                                           │   ║ maskmem_feat║───┤
                                           │   ║ maskmem_pos ║───┤
                                           │   ╚═════════════╝   │
                                           │         │           │
                                           ▼         ▼           ▼
                                      ┌─────────┐  ┌──────────────┐
                                      │ DISPLAY │  │ MEMORY BANK  │
                                      │ (resize │  │  .add(frame) │
                                      │  + CPU) │  └──────────────┘
                                      └─────────┘
```

---

## Startup Sequence

```
  1. Parse args
     │
     ▼
  2. Allocate GPU constants (mean, std, no-prompt tensors)
     │
     ▼
  3. Load 4 TRT engines (shared CUDA stream)
     ├──▶ image_encoder.engine
     ├──▶ mask_decoder.engine
     ├──▶ memory_encoder.engine
     └──▶ memory_attention.engine
     │
     ▼
  4. Print I/O shapes for each engine
     │
     ▼
  5. Open video (threaded reader + writer)
     │
     ▼
  6. WARMUP: run each engine once with dummy zeros
     │        (TRT context initialization has one-time cost)
     │
     ▼
  7. Enter main loop
```

---

## Command-Line Usage

```bash
# Track with bounding box prompt
python sam2_trt_video_tracker.py \
    --video input.mp4 \
    --bbox "100,50,300,250" \
    --model-dir /models/sam2/tiny/trt/ \
    --output output_tracked.mp4

# Track with point prompt
python sam2_trt_video_tracker.py \
    --video input.mp4 \
    --point "200,150" \
    --model-dir /models/sam2/tiny/trt/ \
    --output output_tracked.mp4 \
    --show   # display live window
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Input video path |
| `--bbox` | — | Bounding box `x1,y1,x2,y2` on frame 0 |
| `--point` | — | Single point `x,y` on frame 0 |
| `--model-dir` | `/models/sam2/tiny/trt/` | Directory with 4 `.engine` files |
| `--output` | `/output/output_tracked_trt.mp4` | Output video path |
| `--show` | off | Show live `cv2.imshow` window |

---

## Key Design Decisions

```
  ┌──────────────────────────┬──────────────────────────────────────────────┐
  │  Decision                │  Why                                         │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  Non-default CUDA stream │  All TRT work runs on one stream so         │
  │                          │  kernels execute in order without            │
  │                          │  explicit synchronization between engines    │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  Pre-allocated buffers   │  Memory bank uses fixed GPU tensors and     │
  │  in MemoryBank           │  writes into them with .copy_() — avoids   │
  │                          │  cudaMalloc per frame                        │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  Pre-allocated static    │  TRTEngine pre-allocates output buffers     │
  │  output buffers          │  for engines with known output shapes —     │
  │                          │  only dynamic outputs allocate per call      │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  Threaded video I/O      │  Reader and writer run in daemon threads    │
  │                          │  with bounded queues — disk I/O overlaps    │
  │                          │  GPU compute                                 │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  .clone() for bank       │  TRT engines reuse output buffers, so       │
  │  entries                 │  data stored in the memory bank must be     │
  │                          │  cloned to avoid being overwritten           │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │  Single sync point       │  trt_stream.synchronize() is called once   │
  │                          │  per frame, right before CPU needs the      │
  │                          │  mask — minimizes GPU stalls                 │
  └──────────────────────────┴──────────────────────────────────────────────┘
```

---

## File Map

```
  sam2_trt_video_tracker.py
  │
  ├── Constants (lines 25-36)
  │   └── NUM_MASKMEM, IMG_SIZE, TRT_LOGGER, dtype map
  │
  ├── TRTEngine class (lines 41-104)
  │   ├── __init__    load engine, scan I/O, pre-alloc static outputs
  │   ├── __call__    bind inputs/outputs, execute async
  │   └── print_io    debug: show all tensor names & shapes
  │
  ├── MemoryFrame dataclass (lines 109-115)
  │   └── maskmem_features, maskmem_pos_enc, obj_ptr, frame_idx, is_cond
  │
  ├── MemoryBank class (lines 118-179)
  │   ├── __init__          pre-allocate GPU buffers
  │   ├── add()             store frame (cond or sliding window)
  │   └── build_memory_inputs()  assemble 4 tensors for mem attention
  │
  ├── preprocess_gpu() (lines 184-192)
  │   └── BGR numpy ──▶ normalized [1,3,1024,1024] GPU tensor
  │
  ├── overlay_mask() (lines 195-202)
  │   └── green overlay + contours on frame
  │
  ├── VideoReaderThread (lines 207-234)
  │   └── daemon thread, bounded queue, returns None at EOF
  │
  ├── VideoWriterThread (lines 237-256)
  │   └── daemon thread, bounded queue, None = stop signal
  │
  └── main() (lines 261-529)
      ├── arg parsing + validation
      ├── GPU constant allocation
      ├── engine loading
      ├── warmup pass
      ├── main tracking loop
      │   ├── frame 0: prompted decode
      │   └── frame 1+: memory attention ──▶ decode
      └── timing + IoU summary
```
