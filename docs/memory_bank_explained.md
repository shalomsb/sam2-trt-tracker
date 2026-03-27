# SAM2 Memory Bank — How Frame-to-Frame Tracking Works

## The Big Idea

SAM2 doesn't just look at the current frame. It **remembers** past frames to keep tracking accurate — like following a person through a crowd by remembering what they looked like a few moments ago.

The memory bank stores that history and packages it for the `memory_attention` engine every frame.

---

## What Gets Saved Per Frame

Each frame produces data from **two engines** that gets stored in the memory bank:

From the **memory encoder** (`menc`):

| Tensor | Shape | What it is |
|--------|-------|------------|
| `maskmem_features` | `[1, 64, 64, 64]` | A 64x64 spatial feature map (64 channels). Encodes what the scene looked like around the mask. |
| `maskmem_pos_enc` | `[1, 4096, 64]` | Positional encoding — tells the attention model *where* things were in the image. 4096 = 64x64 flattened grid. |

From the **mask decoder** (`dec`):

| Tensor | Shape | What it is |
|--------|-------|------------|
| `obj_ptr` | `[1, 256]` | A compact 256-dimensional fingerprint of the object. Summarizes "what the object looks like in this frame." |

Also stored **once** on frame 0, from the **memory encoder**:

| Tensor | Shape | What it is |
|--------|-------|------------|
| `temporal_code` | `[7, 1, 1, 64]` | Learned time embeddings. Used in `build_memory_inputs()` to tell the model "this memory is N frames old." |

These tensors are `.clone()`d (copied) from the engine's pre-allocated buffers before storing, because those buffers get overwritten on the next frame.

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

Every frame (except frame 0), `build_memory_inputs()` packs the stored frames into 4 tensors.

The memory_attention engine was built with **static shapes** — it always expects the full-size buffers. Unused slots are zeroed so they don't affect the attention output.

### memory_1 — Scene features
```
Shape: [1, 3, 64, 64, 64]    (always 3 — unused slots zeroed)

[ cond_frame features | recent_frame_1 features | recent_frame_2 features ]
```
The `maskmem_features` from each stored frame, stacked side by side.

### memory_0 — Object pointers
```
Shape: [1, 3, 256]    (always 3 — unused slots zeroed)

[ cond_frame obj_ptr | newest_frame obj_ptr | older_frame obj_ptr ]
```
Conditioning frame's pointer always comes first. Non-conditioning frames are ordered newest-first.

### memory_pos_embed — Position encodings + temporal codes
```
Shape: [1, 12300, 64]    (always 12300 — unused slots zeroed)

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
self._mem1     = torch.empty(1, 3, 64, 64, 64, device="cuda")  # maskmem_features per frame
self._mem0     = torch.empty(1, 3, 256, device="cuda")          # obj_ptr per frame
self._mempos   = torch.zeros(1, 12300, 64, device="cuda")       # maskmem_pos_enc + temporal_code
self._cond_diff = torch.empty((), device="cuda")                # scalar
```

Every call to `build_memory_inputs()` zeros all buffers first, fills the active slots, and returns the full buffers (not slices). The engine always sees the same fixed shapes.

```python
self._mem0.zero_()
self._mem1.zero_()
self._mempos.zero_()
# ... fill active slots ...
return self._mem0, self._mem1, self._mempos, self._cond_diff
```

No new GPU memory is allocated per frame.

---

## Timeline Example

```
Frame 0 (prompted with bbox):
  image_encoder  -> pix_feat, vision_feats, ...
  mask_decoder   -> obj_ptr, mask_for_mem, pred_mask  (using your bbox prompt)
  memory_encoder -> maskmem_feat, maskmem_pos, temporal_code
  STORE: cond_frame = {maskmem_feat, maskmem_pos, obj_ptr, frame_idx=0}
  STORE: temporal_code (used for all future frames)

Frame 1:
  image_encoder  -> pix_feat, vision_feats, ...
  BUILD MEMORY:  pack cond_frame -> memory_attention inputs (unused slots = zeros)
  memory_attention -> image_embed (vision_feats enriched with memory)
  mask_decoder   -> obj_ptr, mask_for_mem, pred_mask  (no prompt, uses memory)
  memory_encoder -> maskmem_feat, maskmem_pos
  STORE: non_cond[0] = {maskmem_feat, maskmem_pos, obj_ptr, frame_idx=1}

Frame 2:
  BUILD MEMORY:  pack cond_frame + non_cond[0] -> memory_attention inputs
  ... same pipeline ...
  STORE: non_cond[1] = {..., frame_idx=2}

Frame 3:
  BUILD MEMORY:  pack cond_frame + non_cond[0] + non_cond[1] -> all 3 slots filled
  ... same pipeline ...
  STORE: non_cond[2] -> evicts non_cond[0] (frame 1 is gone)
  Memory now: frame 0 (cond) + frame 2 + frame 3

Frame 100:
  Memory: frame 0 (cond) + frame 98 + frame 99
  cond_frame_id_diff = 100.0
```
