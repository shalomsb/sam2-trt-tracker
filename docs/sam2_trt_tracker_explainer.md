# SAM2 TRT Video Tracker — Visual Explainer

---

## 1. The Big Picture: What Does This Code Do?

```
VIDEO FILE
    │
    │  (one frame at a time)
    ▼
┌─────────────┐
│  PREPROCESS │  BGR numpy → float32 CUDA tensor [1,3,1024,1024]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   ENCODER   │  image_encoder.engine  →  vision features (stay on GPU)
└──────┬──────┘
       │
       │
       ├── frame 0? ─────────────────────────────────────────────────┐
       │   (user gave a point/bbox)                                   │
       │                                                              ▼
       │                                               ┌─────────────────────────┐
       │                                               │    MASK DECODER (dec)   │
       │                                               │  point/bbox → first mask│
       │                                               └────────────┬────────────┘
       │                                                            │
       │                                               ┌────────────▼────────────┐
       │                                               │   MEMORY ENCODER (menc) │
       │                                               │  mask → memory feature  │
       │                                               └────────────┬────────────┘
       │                                                            │
       │                                               ┌────────────▼────────────┐
       │                                               │       MEMORY BANK       │
       │                                               │  stores cond_frame here │
       │                                               └─────────────────────────┘
       │
       └── frame 1+? ──────────────────────────────────────────────┐
           (no user prompt, auto-propagate)                         │
                                                                    ▼
                                               ┌─────────────────────────────────┐
                                               │       MEMORY BANK               │
                                               │  build_memory_inputs(frame_idx) │
                                               │  (returns pre-allocated buffers)│
                                               └────────────┬────────────────────┘
                                                            │
                                               ┌────────────▼────────────────────┐
                                               │   MEMORY ATTENTION (matt)       │
                                               │  fuses current frame + past mem │
                                               └────────────┬────────────────────┘
                                                            │
                                               ┌────────────▼────────────────────┐
                                               │    MASK DECODER (dec)           │
                                               │  no prompt → propagated mask    │
                                               └────────────┬────────────────────┘
                                                            │
                                               ┌────────────▼────────────────────┐
                                               │   MEMORY ENCODER (menc)         │
                                               │  new mask → non_cond memory     │
                                               └────────────┬────────────────────┘
                                                            │
                                               ┌────────────▼────────────────────┐
                                               │       MEMORY BANK               │
                                               │  sliding window: keeps last 2   │
                                               └─────────────────────────────────┘

                                All paths converge ↓

                             ┌──────────────────────┐
                             │    POSTPROCESS       │
                             │ resize → binary mask │
                             │ mask_gpu → .cpu()    │  ← only CPU transfer!
                             └──────────┬───────────┘
                                        │
                                        ▼
                             overlay_mask() → write frame
```

---

## 2. The 4 TRT Engines — What Goes In / What Comes Out

```
enc  = image_encoder.engine
      IN:  image          [1, 3, 1024, 1024]  float32
      OUT: pix_feat       [1, 256, 64, 64]
           high_res_0     [1, 32, 256, 256]
           high_res_1     [1, 64, 128, 128]
           vision_feats   [1, 4096, 256]
           vision_pos     [1, 4096, 256]

dec  = mask_decoder.engine
      IN:  point_coords   [1, N, 2]
           point_labels   [1, N]
           image_embed    [1, 4096, 256]
           high_res_0     [1, 32, 256, 256]
           high_res_1     [1, 64, 128, 128]
      OUT: obj_ptr        [1, 256]
           mask_for_mem   [1, 1, 1024, 1024]
           pred_mask      [1, 1, 256, 256]
           iou            [1, 1]
           occ_logit      [1, 1]

menc = memory_encoder.engine
      IN:  mask_for_mem   [1, 1, 1024, 1024]
           pix_feat       [1, 256, 64, 64]
           occ_logit      [1, 1]
      OUT: maskmem_feat   [1, 64, 64, 64]
           maskmem_pos    [1, 4096, 64]
           temporal_code  [7, 1, 1, 64]  (frame 0 only — reused after)

matt = memory_attention.engine
      IN:  current_vision_feat     [1, 4096, 256]
           current_vision_pos_embed[1, 4096, 256]
           memory_0                [1, K, 256]    K = # obj_ptrs
           memory_1                [1, M, 64,64,64] M = # mask memories
           memory_pos_embed        [1, P, 64]
           cond_frame_id_diff      scalar
      OUT: image_embed             [1, 4096, 256]
```

---

## 3. Zero-Copy: What It Actually Means

**The Triton way you already know:**
```
PyTorch tensor (GPU)
       │
       │  serialize to protobuf / HTTP / shared mem
       ▼
  Triton Server
       │
       │  Triton manages its own GPU buffer
       ▼
  TRT engine runs
       │
       │  copy result back
       ▼
PyTorch tensor (GPU)
```

**The zero-copy way in this code:**
```
PyTorch tensor (GPU)
       │
       │  t.data_ptr()  ← just an integer: the GPU memory address
       │                   e.g. 0x7f3a40000000
       │
       ▼
ctx.set_tensor_address("image", 0x7f3a40000000)
       │
       │  TRT writes output DIRECTLY into pre-allocated PyTorch buffer
       │  No copy. No serialize. Same GPU memory.
       ▼
PyTorch tensor (GPU)  ← already has the result, TRT wrote into it
```

**The key line (TRTEngine.__call__, line ~79):**
```python
ctx.set_tensor_address(name, t.data_ptr())
#                            ^^^^^^^^^^
#                      "hey TRT, here is the GPU pointer,
#                       read your input FROM there,
#                       write your output TO there"
```

**Pre-allocated output buffers:**
```
At __init__ time (once):
    self._static_bufs["vision_feats"] = torch.empty(shape, device="cuda")
                                                             ^^^^^^^^^^^
                                                      lives on GPU forever

At __call__ time (every frame):
    ctx.set_tensor_address("vision_feats", self._static_bufs["vision_feats"].data_ptr())
    ctx.execute_async_v3(stream)
    # TRT writes INTO that same tensor — no allocation, no copy
```

---

## 4. CUDA Streams — Why One Stream for All 4 Engines

```
Default CUDA stream:
    [other PyTorch ops can mix in here unpredictably]

trt_stream (non-default, created once):
    ──────────────────────────────────────────────────────────────► time
    │ preprocess │ enc │ dec │ menc │ matt │ dec │ menc │ post │
                                                                  ▲
                                                    trt_stream.synchronize()
                                                    "wait here before .cpu()"
```

All 4 engines share the SAME stream → they run **sequentially**, never overlap.
That's intentional — each engine needs the previous engine's output.

```python
trt_stream = torch.cuda.Stream()          # created once in main()
enc  = TRTEngine("encoder.engine",  trt_stream)  # all share it
dec  = TRTEngine("decoder.engine",  trt_stream)
menc = TRTEngine("menc.engine",     trt_stream)
matt = TRTEngine("matt.engine",     trt_stream)
```

---

## 5. MemoryBank — The Sliding Window

SAM2 uses **3 memory slots** (`NUM_MASKMEM = 3`):

```
Slot 0 (cond_frame):  ALWAYS the frame where the user gave a point/bbox
                      Never evicted. This is the "anchor."

Slot 1 (non_cond):    The most recent past frame
Slot 2 (non_cond):    The second most recent past frame

Frame timeline:

  frame 0  frame 1  frame 2  frame 3  frame 4  frame 5  frame 6
  [cond]   [nc]     [nc]     [nc]     [nc]     [nc]     [nc]

At frame 4, build_memory_inputs looks like:
  ┌──────────────────────────────────────────────┐
  │ memory_0 (obj_ptrs):                         │
  │   slot 0 ← frame 0  (cond, t_pos=0)          │
  │   slot 1 ← frame 3  (most recent, t_pos=2)   │
  │   slot 2 ← frame 2  (second, t_pos=1 -- wait)│
  │                                              │
  │ memory_1 (maskmem_features):                 │
  │   same frames, spatial features [64,64,64]   │
  │                                              │
  │ memory_pos_embed:                            │
  │   pos_enc + temporal_code (distance-coded)   │
  └──────────────────────────────────────────────┘

  Why does non_cond_frames.pop(0) keep only 2?
  NUM_MASKMEM=3, cond takes 1 slot → 2 slots left for non-cond
```

**Pre-allocated buffers in MemoryBank:**
```python
# Allocated ONCE at __init__, reused every frame:
self._mem1   = torch.empty(1, NUM_MASKMEM, 64, 64, 64, device="cuda")
self._mem0   = torch.empty(1, NUM_MASKMEM, 256,        device="cuda")
self._mempos = torch.zeros(1, max_pos,     64,         device="cuda")

# build_memory_inputs() writes into them with .copy_() and .add_() in-place
# then returns VIEWS (slices) — still same GPU memory, no copy
return self._mem0[:, :ptr_n], self._mem1[:, :n], ...
#              ^^^^^^^^^^                          slice = view, not copy
```

---

## 6. CUDA Event Timing — Just a GPU Stopwatch

```python
# CPU stopwatch (what you normally use):
t0 = time.perf_counter()
do_gpu_work()
dt = time.perf_counter() - t0   # WRONG for GPU! CPU doesn't know when GPU finished

# GPU stopwatch (what this code uses):
e_start = torch.cuda.Event(enable_timing=True)
e_end   = torch.cuda.Event(enable_timing=True)

e_start.record(trt_stream)   # timestamp injected INTO the GPU stream
do_gpu_work()
e_end.record(trt_stream)     # another timestamp injected

trt_stream.synchronize()     # now safe to read
ms = e_start.elapsed_time(e_end)   # GPU-accurate milliseconds
```

---

## 7. Threaded Video I/O — Why?

```
Without threading (slow):
    read frame (disk) ──wait──► GPU work ──wait──► write frame (disk)
    read frame (disk) ──wait──► GPU work ──wait──► write frame ...

With threading (this code):
    Reader thread:  read read read read read  ...   (fills queue, size=3)
                         │
                    queue (buffer)
                         │
    Main thread:         ▼
                    GPU work ► write write write  ...
                                    │
                               write queue (size=5)
                                    │
    Writer thread:                  ▼
                               write write write ...

GPU is never waiting for disk.
```

---

## 8. Frame 0 vs Frame 1+ — Full Side-by-Side

```
FRAME 0                              FRAME 1+
────────────────────────────────     ────────────────────────────────
img = preprocess(frame)              img = preprocess(frame)
        │                                    │
        ▼                                    ▼
pix_feat, hr0, hr1,             pix_feat, hr0, hr1,
vision_feats, vision_pos        vision_feats, vision_pos
= enc(img)                      = enc(img)
        │                                    │
        │                         mem0, mem1, mempos, diff
        │                         = bank.build_memory_inputs(idx)
        │                                    │
        │                                    ▼
        │                         (image_embed,)
        │                         = matt(vision_feats, vision_pos,
        │                                mem0, mem1, mempos, diff)
        │                                    │
        ▼                                    ▼
obj_ptr, mask_for_mem,          obj_ptr, mask_for_mem,
pred_mask, iou, occ_logit       pred_mask, iou, occ_logit
= dec(                          = dec(
    point_coords=user_input,        point_coords=zeros,   ← no prompt!
    point_labels=user_labels,       point_labels=[-1],
    image_embed=vision_feats,       image_embed=image_embed, ← from matt
    hr0, hr1)                       hr0, hr1)
        │                                    │
        ▼                                    ▼
maskmem_feat, maskmem_pos,      maskmem_feat, maskmem_pos, _
temporal_code                   = menc(mask_for_mem, pix_feat, occ_logit)
= menc(mask_for_mem, pix_feat, occ)
        │                                    │
bank.add(cond_frame)             bank.add(non_cond_frame)
bank.temporal_code = ...                     │
        │                                    │
        └──────────────┬──────────────────────┘
                       ▼
        logits = F.interpolate(pred_mask, orig size)
        mask_gpu = (logits > 0).byte()
        binary_mask = mask_gpu.cpu().numpy()   ← ONLY CPU transfer
        overlay_mask(frame, binary_mask) → write
```

---

## 9. TRTEngine Class — Line by Line

```python
class TRTEngine:

    def __init__(self, engine_path, stream):

        # Load the .engine file (compiled TRT binary) into GPU
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(open(path,"rb").read())
        self.context = self.engine.create_execution_context()
        #              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #              "context" = one stateful inference session
        #              (like a Triton model instance)

        # Walk all tensor names, sort into inputs vs outputs
        # Mark dynamic-shape inputs (those with -1 in shape)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if mode == INPUT:  input_names.append(name)
            else:              output_names.append(name)

        # Pre-allocate output GPU buffers for STATIC-shape outputs
        # Dynamic shapes get allocated fresh each call
        for name in output_names:
            if no -1 in shape:
                self._static_bufs[name] = torch.empty(shape, device="cuda")


    def __call__(self, inputs: dict):

        # Step 1: set input addresses
        for name in input_names:
            t = inputs[name].contiguous()       # must be contiguous!
            if dynamic: ctx.set_input_shape(name, t.shape)
            ctx.set_tensor_address(name, t.data_ptr())  # ← the zero-copy key

        # Step 2: set output addresses (reuse pre-alloc or create new)
        for name in output_names:
            buf = self._static_bufs[name]  OR  torch.empty(dynamic_shape)
            ctx.set_tensor_address(name, buf.data_ptr())

        # Step 3: run asynchronously on the shared stream
        ctx.execute_async_v3(self.stream.cuda_stream)

        return outputs   # list of GPU tensors (already written by TRT)
```

---

## 10. Memory Flow Summary (Where Does Data Live?)

```
                    ┌─────────────────────────────────────────────────────┐
                    │                    GPU MEMORY                        │
                    │                                                       │
                    │  mean_gpu, std_gpu          (constants, never freed)  │
                    │  no_prompt_coords/labels    (constants, never freed)  │
                    │                                                       │
                    │  TRTEngine._static_bufs[]   (pre-alloc, never freed) │
                    │    vision_feats, obj_ptr, maskmem_feat, etc.         │
                    │                                                       │
                    │  MemoryBank._mem0/1/mempos  (pre-alloc, never freed) │
                    │                                                       │
                    │  img tensor                 (created each frame)      │
                    │  logits, mask_gpu           (created each frame)      │
                    └─────────────────────────────────────────────────────┘
                              │                              ▲
                    only TWO PCIe transfers per frame:
                              │                              │
                    frame_bgr (numpy) ──────────────────► .cuda()
                    binary_mask (numpy) ◄────────────── .cpu()
```

---

## Quick Reference: New Concepts vs What You Already Know

| This code | Your Triton equivalent |
|---|---|
| `trt.Runtime().deserialize_cuda_engine()` | `tritonclient` connecting to a loaded model |
| `engine.create_execution_context()` | One model instance in Triton |
| `ctx.set_tensor_address(name, ptr)` | Triton handles this internally |
| `ctx.execute_async_v3(stream)` | `tritonclient.async_stream_infer()` |
| `t.data_ptr()` | No equivalent — Triton hid this |
| `torch.cuda.Stream()` | Triton's internal stream management |
| `trt_stream.synchronize()` | `tritonclient` response callback |
| `torch.cuda.Event` timing | Triton model metrics |
