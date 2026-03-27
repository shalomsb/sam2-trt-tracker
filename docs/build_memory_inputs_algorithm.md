# build_memory_inputs() — Step-by-Step Algorithm

## Input

- `current_idx`: the frame number we're about to process (e.g. 5)
- `self.cond_frame`: MemoryFrame from frame 0 (always present)
- `self.non_cond_frames`: list of up to 2 recent MemoryFrames (e.g. frames 3, 4)
- `self.temporal_code`: `[7, 1, 1, 64]` — learned time embeddings (set once on frame 0)

## Output

4 tensors for the memory_attention engine (always **full fixed size**, unused slots zeroed):

```
mem0      [1, 3, 256]         — obj_ptr from each stored frame
mem1      [1, 3, 64, 64, 64]  — maskmem_features from each stored frame
mem_pos   [1, 12300, 64]      — maskmem_pos_enc + temporal_code + obj_ptr zeros
cond_diff scalar               — how many frames since frame 0
```

The engine was built with static shapes — it always expects these exact sizes.

---

## Algorithm

### Step 1: Assign temporal positions

Each frame gets a `t_pos` that encodes how recent it is:

```
t_pos = 0  ->  conditioning frame (always)
t_pos = 1  ->  2 frames ago
t_pos = 2  ->  1 frame ago (most recent)
```

Example at current_idx=5 with frames 0, 3, 4:

```
Frame 0:  cond frame               -> t_pos = 0  (always 0)
Frame 3:  t_pos = 3 - (5-3) = 1    -> included
Frame 4:  t_pos = 3 - (5-4) = 2    -> included
```

Result: [(frame_0, 0), (frame_3, 1), (frame_4, 2)]

### Step 2: Zero all buffers

```python
self._mem0.zero_()
self._mem1.zero_()
self._mempos.zero_()
```

This ensures unused slots are clean. On frames 1-2 when we have fewer than 3 memories, the empty slots are zeros instead of garbage.

### Step 3: Pack mem1 and mempos in one pass

For each frame, copy its scene features and position encodings (with temporal code added):

```
_mem1 buffer: [1, 3, 64, 64, 64]
                  |
                  +-- slot 0 <- frame_0.maskmem_features  (cond)
                  +-- slot 1 <- frame_3.maskmem_features
                  +-- slot 2 <- frame_4.maskmem_features

_mempos buffer: [1, 12300, 64]

+---------------------+---------------------+---------------------+--------------+
| frame_0 positions   | frame_3 positions   | frame_4 positions   | zeros        |
| 4096 vectors        | 4096 vectors        | 4096 vectors        | (obj_ptr     |
| + temporal_code[2]  | + temporal_code[1]  | + temporal_code[0]  |  positions)  |
+---------------------+---------------------+---------------------+--------------+
```

temporal_code index = NUM_MASKMEM - 1 - t_pos:
```
  t_pos=0 (cond)    -> temporal_code[2]
  t_pos=1 (older)   -> temporal_code[1]
  t_pos=2 (newest)  -> temporal_code[0]
```

The temporal code is **added** to the position vectors so the attention model
can distinguish "this position is from 1 frame ago" vs "this position is from the anchor frame."

### Step 4: Pack mem0 — object pointers

Conditioning frame first, then non-conditioning newest-first:

```
_mem0 buffer: [1, 3, 256]
                  |
                  +-- slot 0 <- frame_0.obj_ptr  (cond, always first)
                  +-- slot 1 <- frame_4.obj_ptr   (newest)
                  +-- slot 2 <- frame_3.obj_ptr   (older)
```

Note: mem0 ordering (newest-first) is different from mem1 ordering (by t_pos).

### Step 5: cond_diff

```
cond_diff = current_idx - cond_frame.frame_idx
          = 5 - 0
          = 5.0
```

Single scalar telling the model how stale the conditioning frame is.

---

## Full picture at frame 5

```
Memory bank contains:
  +----------------------------------------------------------+
  | cond_frame (frame 0):                                    |
  |   maskmem_features [1,64,64,64]   (from memory encoder)  |
  |   maskmem_pos_enc  [1,4096,64]    (from memory encoder)  |
  |   obj_ptr          [1,256]        (from mask decoder)     |
  +----------------------------------------------------------+
  | non_cond_frames[0] (frame 3):                            |
  |   maskmem_features [1,64,64,64]                          |
  |   maskmem_pos_enc  [1,4096,64]                           |
  |   obj_ptr          [1,256]                               |
  +----------------------------------------------------------+
  | non_cond_frames[1] (frame 4):                            |
  |   maskmem_features [1,64,64,64]                          |
  |   maskmem_pos_enc  [1,4096,64]                           |
  |   obj_ptr          [1,256]                               |
  +----------------------------------------------------------+

                    build_memory_inputs(5)
                            |
                            v

  +----------------------------------------------------------+
  | mem1 [1, 3, 64, 64, 64]                                  |
  |   maskmem_features stacked by t_pos order                |
  +----------------------------------------------------------+
  | mem0 [1, 3, 256]                                         |
  |   obj_ptrs: cond first, then newest-first                |
  +----------------------------------------------------------+
  | mem_pos [1, 12300, 64]                                   |
  |   maskmem_pos_enc + temporal_code for each frame         |
  |   + zeroed obj_ptr positions                             |
  +----------------------------------------------------------+
  | cond_diff = 5.0                                          |
  +----------------------------------------------------------+

                            |
                            v

                    memory_attention(
                      current_vision_feat,      <- from image encoder (current frame)
                      current_vision_pos_embed,
                      memory_0 = mem0,          <- obj_ptrs from past
                      memory_1 = mem1,          <- scene features from past
                      memory_pos_embed = mem_pos,
                      cond_frame_id_diff = cond_diff,
                    )
                            |
                            v

                    image_embed [1, 256, 64, 64]
                    (current frame's vision features enriched with past memory)
```

## On frames 1-2 (fewer than 3 memories)

The buffers are still full size `[1, 3, ...]` but unused slots are zeroed:

```
Frame 1 (only cond_frame in memory):
  mem1 = [ cond_features | zeros | zeros ]
  mem0 = [ cond_obj_ptr  | zeros | zeros ]

Frame 2 (cond_frame + 1 non-cond):
  mem1 = [ cond_features | frame_1_features | zeros ]
  mem0 = [ cond_obj_ptr  | frame_1_obj_ptr  | zeros ]

Frame 3+ (all 3 slots filled):
  mem1 = [ cond_features | older_features | newest_features ]
  mem0 = [ cond_obj_ptr  | newest_obj_ptr | older_obj_ptr   ]
```
