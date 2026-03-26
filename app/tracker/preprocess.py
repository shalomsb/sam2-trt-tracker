"""GPU preprocessing and visualisation helpers."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .constants import IMG_SIZE


def preprocess_gpu(frame_bgr: np.ndarray, mean: torch.Tensor, std: torch.Tensor):
    """BGR uint8 numpy -> normalised [1,3,1024,1024] float32 CUDA tensor."""
    h, w = frame_bgr.shape[:2]
    t = torch.from_numpy(frame_bgr).cuda().float()          # [H,W,3]
    t = t[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W] RGB
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                      mode="bilinear", align_corners=False)
    t.sub_(mean).div_(std)
    return t, h, w


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    out = frame.copy()
    roi = mask > 0
    out[roi] = (out[roi].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def bbox_from_mask_gpu(pred_mask, orig_w, orig_h):
    """Extract bbox from 256x256 pred_mask logits, scale to original coords."""
    mask = (pred_mask[0, 0] > 0.0)
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        return None
    sx, sy = orig_w / pred_mask.shape[3], orig_h / pred_mask.shape[2]
    return (int(xs.min().item() * sx), int(ys.min().item() * sy),
            int(xs.max().item() * sx), int(ys.max().item() * sy))


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """Draw bounding box rectangle on frame."""
    out = frame.copy()
    if bbox:
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return out
