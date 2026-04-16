"""
Export EdgeTAM to ONNX format.

Exports 4 ONNX models:
1. image_encoder.onnx   - RepViT backbone + FPN
2. mask_decoder.onnx    - SAM prompt encoder + mask decoder
3. memory_encoder.onnx  - Memory encoder + spatial perceiver
4. memory_attention.onnx - Memory attention (cross-attention with memory bank)

Adapted from NVIDIA's SAM2 ONNX export for EdgeTAM architecture.

Usage:
    python export_edgetam_onnx.py [--output_dir OUTPUT_DIR]
"""

import os
import argparse

import torch
import onnx

from onnx_export.modules import ImageEncoder, MaskDecoder, MemEncoder, MemAttention
from sam2.build_sam import build_sam2


def export_image_encoder(model, onnx_path):
    print(">>> Exporting Image Encoder...")
    input_img = torch.randn(1, 3, 1024, 1024).cpu()
    model(input_img)  # warmup

    output_names = [
        "pix_feat",
        "high_res_feat0",
        "high_res_feat1",
        "vision_feats",
        "vision_pos_embed",
    ]
    torch.onnx.export(
        model,
        input_img,
        os.path.join(onnx_path, "image_encoder.onnx"),
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
    )
    onnx_model = onnx.load(os.path.join(onnx_path, "image_encoder.onnx"))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Image Encoder exported successfully!")


def export_mask_decoder(model, onnx_path):
    print(">>> Exporting Mask Decoder...")
    batch_size = 1
    point_coords = torch.randn(batch_size, 1, 2).cpu()
    point_labels = torch.randn(batch_size, 1).cpu()
    image_embed = torch.randn(batch_size, 256, 64, 64).cpu()
    high_res_feats_0 = torch.randn(1, 32, 256, 256).cpu()
    high_res_feats_1 = torch.randn(1, 64, 128, 128).cpu()

    model(
        point_coords=point_coords,
        point_labels=point_labels,
        image_embed=image_embed,
        high_res_feats_0=high_res_feats_0,
        high_res_feats_1=high_res_feats_1,
    )  # warmup

    input_names = [
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
    ]
    output_names = ["obj_ptr", "mask_for_mem", "pred_mask", "iou", "occ_logit"]
    dynamic_axes = {
        "point_coords": {0: "batch_size", 1: "num_points"},
        "point_labels": {0: "batch_size", 1: "num_points"},
        "image_embed": {0: "batch_size"},
    }
    torch.onnx.export(
        model,
        (point_coords, point_labels, image_embed, high_res_feats_0, high_res_feats_1),
        os.path.join(onnx_path, "mask_decoder.onnx"),
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(os.path.join(onnx_path, "mask_decoder.onnx"))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Mask Decoder exported successfully!")


def export_memory_encoder(model, onnx_path):
    print(">>> Exporting Memory Encoder (with Spatial Perceiver)...")
    batch_size = 1
    mask_for_mem = torch.randn(batch_size, 1, 1024, 1024).cpu()
    pix_feat = torch.randn(1, 256, 64, 64).cpu()
    occ_logit = torch.randn(batch_size, 1).cpu()

    model(
        mask_for_mem=mask_for_mem, pix_feat=pix_feat, occ_logit=occ_logit
    )  # warmup

    dynamic_axes = {
        "mask_for_mem": {0: "batch_size"},
        "occ_logit": {0: "batch_size"},
    }
    input_names = ["mask_for_mem", "pix_feat", "occ_logit"]
    output_names = ["maskmem_features", "maskmem_pos_enc", "temporal_code"]
    torch.onnx.export(
        model,
        (mask_for_mem, pix_feat, occ_logit),
        os.path.join(onnx_path, "memory_encoder.onnx"),
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(os.path.join(onnx_path, "memory_encoder.onnx"))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Memory Encoder exported successfully!")


def export_memory_attention(model, onnx_path):
    print(">>> Exporting Memory Attention...")
    batch_size = 1
    num_obj_ptr = 16
    num_masks = 7
    mem_tokens = 512  # perceiver output tokens per mask

    current_vision_feat = torch.randn(1, 256, 64, 64).cpu()
    current_vision_pos_embed = torch.randn(4096, 1, 256).cpu()
    # Object pointer memory: [batch, num_obj_ptr, 256]
    memory_0 = torch.randn(batch_size, num_obj_ptr, 256).cpu()
    # Spatial memory from perceiver: [batch, num_masks, 512, 64]
    memory_1 = torch.randn(batch_size, num_masks, mem_tokens, 64).cpu()
    # Position embeddings: [batch, num_masks*512 + 4*num_obj_ptr, 64]
    total_pos_tokens = num_masks * mem_tokens + 4 * num_obj_ptr
    memory_pos_embed = torch.randn(batch_size, total_pos_tokens, 64).cpu()
    num_obj_ptr_tensor = torch.tensor(num_obj_ptr, dtype=torch.int64).cpu()

    model(
        current_vision_feat=current_vision_feat,
        current_vision_pos_embed=current_vision_pos_embed,
        memory_0=memory_0,
        memory_1=memory_1,
        memory_pos_embed=memory_pos_embed,
        num_obj_ptr=num_obj_ptr_tensor,
    )  # warmup

    input_names = [
        "current_vision_feat",
        "current_vision_pos_embed",
        "memory_0",
        "memory_1",
        "memory_pos_embed",
        "num_obj_ptr",
    ]
    dynamic_axes = {
        "memory_0": {0: "batch_size", 1: "num_obj_ptr"},
        "memory_1": {0: "batch_size", 1: "num_masks"},
        "memory_pos_embed": {0: "batch_size", 1: "total_tokens"},
    }
    torch.onnx.export(
        model,
        (
            current_vision_feat,
            current_vision_pos_embed,
            memory_0,
            memory_1,
            memory_pos_embed,
            num_obj_ptr_tensor,
        ),
        os.path.join(onnx_path, "memory_attention.onnx"),
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["image_embed"],
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(os.path.join(onnx_path, "memory_attention.onnx"))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] Memory Attention exported successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EdgeTAM to ONNX")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx_models",
        help="Output directory for ONNX models",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = "edgetam.yaml"
    checkpoint = "checkpoints/edgetam.pt"

    print(f"Loading EdgeTAM model...")
    sam2_model = build_sam2(config, checkpoint, device="cpu")
    print(f"Model loaded. Exporting to {args.output_dir}/\n")

    image_encoder = ImageEncoder(sam2_model).cpu().eval()
    export_image_encoder(image_encoder, args.output_dir)

    mask_decoder = MaskDecoder(sam2_model).cpu().eval()
    export_mask_decoder(mask_decoder, args.output_dir)

    mem_encoder = MemEncoder(sam2_model).cpu().eval()
    export_memory_encoder(mem_encoder, args.output_dir)

    mem_attention = MemAttention(sam2_model).cpu().eval()
    export_memory_attention(mem_attention, args.output_dir)

    print(f"\nAll models exported to {args.output_dir}/")
    print("  - image_encoder.onnx")
    print("  - mask_decoder.onnx")
    print("  - memory_encoder.onnx")
    print("  - memory_attention.onnx")
