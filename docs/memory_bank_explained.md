# SAM2 Memory Bank — How Frame-to-Frame Tracking Works

## The Big Idea

SAM2 doesn't just look at the current frame. It **remembers** past frames to keep tracking accurate — like following a person through a crowd by remembering what they looked like a few moments ago.

The memory bank stores that history and packages it for the `memory_attention` engine every frame.

---

## What Gets Saved Per Frame

Every frame that goes through the pipeline produces three things from the **memory encoder**:

| Tensor | Shape | What it is |
|--------|-------|------------|
| `maskmem_features` | `[1, 64, 64, 64]` | A 64×64 spatial feature map (64 channels). Encodes what the scene looked like around the mask. Think of it as a compressed "snapshot" of the area near the tracked object. |
| `maskmem_pos_enc` | `[1, 4096, 64]` | Positional encoding — tells the attention model *where* things were in the image. 4096 = 64×64 (the flattened spatial grid), each position has a 64-dim vector. |
| `obj_ptr` | `[1, 256]` | A compact 256-dimensional fingerprint of the object. Summarizes "what the object looks like in this frame" without any spatial information. |

These three tensors are `.clone()`d (copied) and stored as a `MemoryFrame`.

---

## Two Types of Memory

### 1. Conditioning frame (frame 0) — always kept

This is the frame where you gave the initial prompt (bounding box or point click). It's your "ground truth" — the moment you told the model "this is the object I want to track."

- Stored once, **never evicted**
- Gets `t_pos = 0` (temporal position zero — the anchor)
- Its `obj_ptr` always goes **first** in the memory_0 input

### 2. Non-conditioning frames (frames 1, 2, 3, ...) — rolling window of last 2

These are frames where the model tracked the object on its own (no human prompt). Only the **most recent 2** are kept. When a 3rd arrives, the oldest is evicted.

Example at frame 50:
```
Memory contains:
  - Frame 0  (conditioning)  — always kept
  - Frame 48 (non-cond)      — will be evicted when frame 51 arrives
  - Frame 49 (non-cond)      — most recent
```

At frame 51:
```
  - Frame 0  (conditioning)  — still here
  - Frame 49 (non-cond)      — was newest, now second oldest
  - Frame 50 (non-cond)      — newest
  (Frame 48 is gone)
```

### Why keep frame 0 forever?

Frame 0 is the only frame where a human told the model what to track. Without it, the model would rely entirely on its own predictions — and small errors would compound over time (drift). Frame 0 acts as a permanent anchor.

### Why only 2 non-conditioning frames?

More memory = more GPU memory + slower attention computation. 2 recent frames give enough context for smooth tracking without overloading the Jetson's limited GPU memory.

---

## What the Memory Attention Engine Receives

Every frame (except frame 0), `build_memory_inputs()` packs the stored frames into 4 tensors:

### memory_1 — Scene features
```
Shape: [1, N, 64, 64, 64]    (N = number of frames in memory, up to 3)

[ cond_frame features | recent_frame_1 features | recent_frame_2 features ]
```
The `maskmem_features` from each stored frame, stacked side by side.

### memory_0 — Object pointers
```
Shape: [1, P, 256]    (P = number of pointers, up to 3)

[ cond_frame obj_ptr | newest_frame obj_ptr | older_frame obj_ptr ]
```
Conditioning frame's pointer always comes first. Non-conditioning frames are ordered newest-first.

### memory_pos_embed — Position encodings + temporal codes
```
Shape: [1, N×4096 + P×4, 64]

[ spatial positions for each frame (with temporal codes added) | zeros for obj_ptrs ]
```

Each frame's 4096 position vectors get a **temporal code** added — a learned embedding that tells the model "this memory is from N frames ago." The temporal code depends on `t_pos`:

| t_pos | Meaning |
|-------|---------|
| 0 | Conditioning frame (the anchor) |
| 1 | 2 frames ago |
| 2 | 1 frame ago (most recent) |

After the spatial positions, there are zero-filled slots for the object pointers (they have no spatial meaning).

### cond_frame_id_diff — Scalar
```
Shape: scalar float

Value: current_frame_idx - 0 = current_frame_idx
```
Simply how many frames have passed since the conditioning frame. At frame 100, this is `100.0`. Tells the model how stale the original prompt is.

---

## Pre-allocated Buffers

All four outputs are written into **pre-allocated GPU tensors** that are created once in `__init__` and reused every frame:

```python
self._mem1    = torch.empty(1, 3, 64, 64, 64, device="cuda")   # max 3 frames
self._mem0    = torch.empty(1, 3, 256, device="cuda")           # max 3 pointers
self._mempos  = torch.zeros(1, 12300, 64, device="cuda")        # max positions
self._cond_diff = torch.empty((), device="cuda")                # scalar
```

Since the actual number of frames varies (1 on frame 1, up to 3 later), the method returns **slices** of these buffers:

```python
return (self._mem0[:, :ptr_n],       # only the filled pointer slots
        self._mem1[:, :n],           # only the filled feature slots
        self._mempos[:, :total_pos], # only the filled position slots
        self._cond_diff)
```

No new GPU memory is allocated per frame. The slices are just views into the existing buffers.

---

## Timeline Example

```
Frame 0 (prompted with bbox):
  image_encoder  → pix_feat, vision_feats, ...
  mask_decoder   → obj_ptr, mask_for_mem, pred_mask  (using your bbox prompt)
  memory_encoder → maskmem_feat, maskmem_pos, temporal_code
  STORE: cond_frame = {maskmem_feat, maskmem_pos, obj_ptr, frame_idx=0}
  STORE: temporal_code (used for all future frames)

Frame 1:
  image_encoder  → pix_feat, vision_feats, ...
  BUILD MEMORY:  pack cond_frame → memory_attention inputs
  memory_attention → image_embed (vision_feats enriched with memory)
  mask_decoder   → obj_ptr, mask_for_mem, pred_mask  (no prompt, uses memory)
  memory_encoder → maskmem_feat, maskmem_pos
  STORE: non_cond[0] = {maskmem_feat, maskmem_pos, obj_ptr, frame_idx=1}

Frame 2:
  BUILD MEMORY:  pack cond_frame + non_cond[0] → memory_attention inputs
  ... same pipeline ...
  STORE: non_cond[1] = {..., frame_idx=2}

Frame 3:
  BUILD MEMORY:  pack cond_frame + non_cond[0] + non_cond[1] → 3 frames
  ... same pipeline ...
  STORE: non_cond[2] → evicts non_cond[0] (frame 1 is gone)
  Memory now: frame 0 (cond) + frame 2 + frame 3

Frame 100:
  Memory: frame 0 (cond) + frame 98 + frame 99
  cond_frame_id_diff = 100.0
```
