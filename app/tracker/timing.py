"""CUDA event timing stats collector and summary printer."""

from collections import defaultdict

import torch


class TimingStats:
    """Collects per-stage CUDA event timings and prints a summary table."""

    def __init__(self):
        self._stats: dict[str, list[float]] = defaultdict(list)

    @staticmethod
    def event():
        return torch.cuda.Event(enable_timing=True)

    def record(self, name: str, start_event, end_event):
        self._stats[name].append(start_event.elapsed_time(end_event))

    def append(self, name: str, value: float):
        self._stats[name].append(value)

    def get(self, name: str) -> list[float]:
        return self._stats[name]

    def print_summary(self, frame_count: int):
        order = ["preprocess", "image_encoder", "mem_bank_build",
                 "memory_attention", "mask_decoder", "memory_encoder", "postprocess"]

        print(f"\n{'─'*70}")
        print(f"  Timing breakdown ({frame_count} frames, CUDA events, ms)")
        print(f"{'─'*70}")
        print(f"  {'stage':<20s} {'mean':>8s} {'min':>8s} {'max':>8s} {'total':>10s}  {'%':>5s}")
        print(f"  {'─'*57}")

        grand_total = sum(sum(v) for k, v in self._stats.items() if k != "iou")
        for name in order:
            if name not in self._stats:
                continue
            vals = self._stats[name]
            s = sum(vals)
            pct = 100.0 * s / grand_total if grand_total > 0 else 0
            print(f"  {name:<20s} {s/len(vals):>8.2f} {min(vals):>8.2f} {max(vals):>8.2f}"
                  f" {s:>10.1f}  {pct:>5.1f}%")

        print(f"  {'─'*57}")
        print(f"  {'TOTAL':<20s} {'':>8s} {'':>8s} {'':>8s} {grand_total:>10.1f}  100.0%")
        print(f"{'─'*70}")

    def print_iou_summary(self):
        ious = self._stats["iou"]
        if not ious:
            return
        mean_iou = sum(ious) / len(ious)
        below_90 = sum(1 for v in ious if v < 0.90)
        below_80 = sum(1 for v in ious if v < 0.80)
        print(f"\n  IoU: mean={mean_iou:.3f}  min={min(ious):.3f}  max={max(ious):.3f}"
              f"  <0.9={below_90}  <0.8={below_80}")
