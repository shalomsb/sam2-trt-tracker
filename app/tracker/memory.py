"""GPU memory bank for SAM2 frame-to-frame propagation."""

from dataclasses import dataclass

import torch

from .constants import NUM_MASKMEM


# Per-frame data stored in the memory bank. Two tensors come from the memory encoder
# (maskmem_features, maskmem_pos_enc) and one from the mask decoder (obj_ptr).
# We .clone() them from the engine's pre-allocated buffers before storing,
# because those buffers get overwritten on the next frame.
@dataclass
class MemoryFrame:
    maskmem_features: torch.Tensor  # [1,64,64,64]  — spatial feature map of the scene near the mask
    maskmem_pos_enc: torch.Tensor   # [1,4096,64]   — positional encoding (4096 = 64×64 flattened grid)
    obj_ptr: torch.Tensor           # [1,256]        — compact fingerprint of the tracked object
    frame_idx: int                  # which frame number this was
    is_cond: bool                   # True = prompted frame (frame 0), False = propagated frame


class MemoryBank:
    """Pre-allocated GPU memory bank. Avoids per-frame allocations.

    Stores up to 3 MemoryFrames — NOT raw images, but outputs from two engines:
      From memory encoder (menc):
        - maskmem_features [1,64,64,64] — spatial feature map (what the scene looked like near the mask)
        - maskmem_pos_enc  [1,4096,64]  — positional encoding (where things were in the image)
      From mask decoder (dec):
        - obj_ptr          [1,256]      — compact fingerprint of the tracked object

    Also stores (once, on frame 0):
      From memory encoder (menc):
        - temporal_code    [7,1,1,64]   — learned time embeddings, used in build_memory_inputs()
                                          to tell the model "this memory is N frames old"

    These are what the memory_attention engine uses to recall past frames.

    Layout: 1 conditioning frame (frame 0, always kept) + up to 2 most recent
    non-conditioning frames (rolling window, oldest evicted when a 3rd arrives).
    """

    def __init__(self):
        self.cond_frame: MemoryFrame | None = None
        self.non_cond_frames: list[MemoryFrame] = []
        # Temporal code is a set of learned embeddings that tell the model "how far
        # in the past is this memory." Produced by the memory encoder on frame 0,
        # set once and never changes.
        self.temporal_code: torch.Tensor | None = None  # [7,1,1,64] cuda

        # Pre-allocate output buffers sized for the maximum (NUM_MASKMEM = 3 frames).
        # build_memory_inputs() writes into these and returns slices — no allocations per frame.
        max_pos = NUM_MASKMEM * 4096 + NUM_MASKMEM * 4  # maskmem_pos_enc positions + obj_ptr positions
        self._mem1 = torch.empty(1, NUM_MASKMEM, 64, 64, 64, device="cuda")  # maskmem_features per frame
        self._mem0 = torch.empty(1, NUM_MASKMEM, 256, device="cuda")          # obj_ptr per frame
        self._mempos = torch.zeros(1, max_pos, 64, device="cuda")             # maskmem_pos_enc + temporal_code
        self._cond_diff = torch.empty((), dtype=torch.float32, device="cuda") # current_idx - cond_frame.frame_idx

    def add(self, frame: MemoryFrame):
        if frame.is_cond:
            self.cond_frame = frame
        else:
            self.non_cond_frames.append(frame)
            if len(self.non_cond_frames) > NUM_MASKMEM - 1:
                self.non_cond_frames.pop(0)  # evict the oldest

    def build_memory_inputs(self, current_idx: int):
        """Pack stored frames into the 4 tensors that memory_attention expects.

        Writes into pre-allocated GPU buffers (full fixed size) and returns them.
        The memory_attention engine was built with static shapes:
          memory_0 [1, 3, 256], memory_1 [1, 3, 64, 64, 64], memory_pos_embed [1, 12300, 64].
        Unused slots are zeroed so they don't affect attention output.
        """

        # Collect frames with temporal positions.
        # t_pos encodes recency: 0 = conditioning, 2 = 1 frame ago, 1 = 2 frames ago.
        frames_tpos = [(self.cond_frame, 0)]
        for f in self.non_cond_frames:
            t_pos = NUM_MASKMEM - (current_idx - f.frame_idx)
            if t_pos >= 1:
                frames_tpos.append((f, t_pos))

        # Zero all buffers — unused slots must be clean.
        self._mem0.zero_()
        self._mem1.zero_()
        self._mempos.zero_()

        # Pack mem1 (scene features) and mempos (positions + temporal codes) in one pass.
        for i, (f, t_pos) in enumerate(frames_tpos):
            self._mem1[0, i].copy_(f.maskmem_features[0])
            s = i * 4096
            self._mempos[0, s:s + 4096].copy_(f.maskmem_pos_enc[0])
            self._mempos[0, s:s + 4096].add_(
                self.temporal_code[NUM_MASKMEM - 1 - t_pos].view(1, 64)
            )

        # Pack mem0 (object pointers): cond first, then newest-first.
        self._mem0[0, 0].copy_(self.cond_frame.obj_ptr[0])
        for i, f in enumerate(reversed(self.non_cond_frames)):
            if 1 <= (current_idx - f.frame_idx) <= NUM_MASKMEM - 1:
                self._mem0[0, 1 + i].copy_(f.obj_ptr[0])

        self._cond_diff.fill_(float(current_idx - self.cond_frame.frame_idx))

        # Return full-size buffers (engine expects fixed shapes, not slices).
        return self._mem0, self._mem1, self._mempos, self._cond_diff
