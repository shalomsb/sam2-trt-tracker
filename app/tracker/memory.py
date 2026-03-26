"""GPU memory bank for SAM2 frame-to-frame propagation."""

from dataclasses import dataclass

import torch

from .constants import NUM_MASKMEM


@dataclass
class MemoryFrame:
    maskmem_features: torch.Tensor  # [1,64,64,64]  cuda
    maskmem_pos_enc: torch.Tensor   # [1,4096,64]   cuda
    obj_ptr: torch.Tensor           # [1,256]        cuda
    frame_idx: int
    is_cond: bool


class MemoryBank:
    """Pre-allocated GPU memory bank. Avoids per-frame allocations."""

    def __init__(self):
        self.cond_frame: MemoryFrame | None = None
        self.non_cond_frames: list[MemoryFrame] = []
        self.temporal_code: torch.Tensor | None = None  # [7,1,1,64] cuda

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

        for i, (f, _) in enumerate(frames_tpos):
            self._mem1[0, i].copy_(f.maskmem_features[0])

        for i, (f, t_pos) in enumerate(frames_tpos):
            tcode = self.temporal_code[NUM_MASKMEM - 1 - t_pos].view(1, 64)
            s = i * 4096
            self._mempos[0, s:s + 4096].copy_(f.maskmem_pos_enc[0])
            self._mempos[0, s:s + 4096].add_(tcode)

        self._mem0[0, 0].copy_(self.cond_frame.obj_ptr[0])
        ptr_n = 1
        for f in reversed(self.non_cond_frames):
            if 1 <= (current_idx - f.frame_idx) <= NUM_MASKMEM - 1:
                self._mem0[0, ptr_n].copy_(f.obj_ptr[0])
                ptr_n += 1

        total_pos = n * 4096 + ptr_n * 4
        self._mempos[0, n * 4096:total_pos].zero_()

        self._cond_diff.fill_(float(current_idx - self.cond_frame.frame_idx))

        return (self._mem0[:, :ptr_n],
                self._mem1[:, :n],
                self._mempos[:, :total_pos],
                self._cond_diff)
