"""
ONNX export wrapper modules for EdgeTAM.

Adapted from NVIDIA's SAM2 ONNX export:
https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/sam2-onnx-tensorrt

Key EdgeTAM differences from SAM2:
- RepViT-m1 backbone (lighter than Hiera)
- Spatial Perceiver compresses memory from [B, 64, H, W] to [B, 512, 64]
- RoPEAttentionv2 with asymmetric q/k sizes (64x64 / 16x16)
- add_tpos_enc_to_obj_ptrs=False (no temporal pos enc on object pointers)
- mem_dim=64 (object pointers split into 256/64=4 tokens each)
"""

import torch
from torch import nn
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.misc import fill_holes_in_mask_scores


class ImageEncoder(nn.Module):
    """Exports the image encoder (RepViT backbone + FPN neck).

    Input:
        image: [1, 3, 1024, 1024]

    Outputs:
        pix_feat:           [1, 256, 64, 64]   - pixel features for memory encoder
        high_res_feat0:     [1, 32, 256, 256]   - high-res features level 0
        high_res_feat1:     [1, 64, 128, 128]   - high-res features level 1
        vision_feats:       [1, 256, 64, 64]    - vision features (with no_mem_embed for init frame)
        vision_pos_embed:   [4096, 1, 256]      - positional embeddings
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed  # [1, 1, 256]
        self.image_encoder = sam_model.image_encoder
        self.num_feature_levels = sam_model.num_feature_levels
        self.prepare_backbone_features = sam_model._prepare_backbone_features

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder(image)
        # Apply conv projections for high-res features
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        pix_feat = backbone_out["vision_features"]  # [1, 256, 64, 64]

        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"],
            "vision_pos_enc": backbone_out["vision_pos_enc"],
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(1, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(1, -1, -1, -1)

        (_, current_vision_feats, current_vision_pos_embeds, _) = (
            self.prepare_backbone_features(expanded_backbone_out)
        )

        # For init frame: add no_mem_embed (EdgeTAM uses directly_add_no_mem_embed=True)
        current_vision_feat = current_vision_feats[-1] + self.no_mem_embed
        # Reshape (HW)xNxC -> NxCxHxW: [4096, 1, 256] -> [1, 256, 64, 64]
        current_vision_feat2 = current_vision_feat.reshape(64, 64, 1, 256).permute(
            2, 3, 0, 1
        )

        # High-res features: (HW)xNxC -> NxCxHxW
        high_res_feat0 = current_vision_feats[0].reshape(256, 256, 1, 32).permute(
            2, 3, 0, 1
        )  # [1, 32, 256, 256]
        high_res_feat1 = current_vision_feats[1].reshape(128, 128, 1, 64).permute(
            2, 3, 0, 1
        )  # [1, 64, 128, 128]

        return (
            pix_feat,
            high_res_feat0,
            high_res_feat1,
            current_vision_feat2,
            current_vision_pos_embeds[-1],
        )


class MaskDecoder(nn.Module):
    """Exports the SAM mask decoder (prompt encoder + mask decoder).

    Inputs:
        point_coords:    [B, num_points, 2]   - point prompt coordinates
        point_labels:    [B, num_points]       - point labels (1=pos, 0=neg)
        image_embed:     [B, 256, 64, 64]     - image embeddings
        high_res_feats_0: [1, 32, 256, 256]   - high-res features level 0
        high_res_feats_1: [1, 64, 128, 128]   - high-res features level 1

    Outputs:
        obj_ptr:      [B, 256]             - object pointer for memory
        mask_for_mem: [B, 1, 1024, 1024]   - sigmoid-scaled mask for memory encoder
        pred_mask:    [B, 1, 256, 256]     - prediction mask (hole-filled, resized)
        iou:          [B, 1]               - best IoU score
        occ_logit:    [B, 1]               - object presence logit
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.sigmoid_scale_for_mem_enc = sam_model.sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sam_model.sigmoid_bias_for_mem_enc

    @torch.no_grad()
    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
    ):
        frame_size = [256, 256]
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}

        batch_size = point_coords.size(0)
        high_res_feats_0 = high_res_feats_0.repeat(batch_size, 1, 1, 1)
        high_res_feats_1 = high_res_feats_1.repeat(batch_size, 1, 1, 1)
        high_res_feats = [high_res_feats_0, high_res_feats_1]

        sam_outputs = self.model._forward_sam_heads(
            backbone_features=image_embed,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_feats,
            multimask_output=True,
        )
        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            occ_logit,
        ) = sam_outputs

        # Sigmoid-scaled mask for memory encoder
        mask_for_mem = torch.sigmoid(high_res_masks)
        mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # Fill holes in low-res masks
        low_res_masks = fill_holes_in_mask_scores(low_res_masks, 8)

        # Resize to frame size
        pred_mask = torch.nn.functional.interpolate(
            low_res_masks,
            size=(frame_size[0], frame_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        # Best IoU
        iou = torch.max(ious, dim=-1, keepdim=True)[0]

        return obj_ptr, mask_for_mem, pred_mask, iou, occ_logit


class MemEncoder(nn.Module):
    """Exports the memory encoder + spatial perceiver.

    EdgeTAM uses a spatial perceiver that compresses memory features from
    [B, 64, 64, 64] (4096 spatial tokens) to [B, 512, 64] (512 latent tokens).

    Inputs:
        mask_for_mem: [B, 1, 1024, 1024]  - sigmoid-scaled mask
        pix_feat:     [1, 256, 64, 64]    - pixel features from image encoder
        occ_logit:    [B, 1]              - object presence logit

    Outputs:
        maskmem_features: [B, 512, 64]    - compressed memory features
        maskmem_pos_enc:  [B, 512, 64]    - compressed position encodings
        temporal_code:    [7, 1, 1, 64]   - temporal position encoding
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.memory_encoder = sam_model.memory_encoder
        self.spatial_perceiver = sam_model.spatial_perceiver
        self.maskmem_tpos_enc = sam_model.maskmem_tpos_enc
        self.no_obj_embed_spatial = sam_model.no_obj_embed_spatial

    @torch.no_grad()
    def forward(
        self,
        mask_for_mem: torch.Tensor,
        pix_feat: torch.Tensor,
        occ_logit: torch.Tensor,
    ):
        batch_size = mask_for_mem.shape[0]
        pix_feat = pix_feat.repeat(batch_size, 1, 1, 1)

        # Run the memory encoder (mask is already sigmoid-scaled)
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        maskmem_features = maskmem_out["vision_features"]  # [B, 64, 64, 64]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]  # list of [B, 64, 64, 64]

        # Add no-object spatial embedding if configured
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (occ_logit > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        # Apply spatial perceiver (EdgeTAM-specific)
        # Compresses [B, 64, 64, 64] -> [B, 512, 64]
        if self.spatial_perceiver is not None:
            maskmem_features, maskmem_pos_enc_out = self.spatial_perceiver(
                maskmem_features, maskmem_pos_enc[0]
            )
        else:
            # Fallback for non-perceiver models
            maskmem_pos_enc_out = maskmem_pos_enc[0].view(
                batch_size, 64, 64 * 64
            ).permute(0, 2, 1)

        return maskmem_features, maskmem_pos_enc_out, self.maskmem_tpos_enc


class MemAttention(nn.Module):
    """Exports the memory attention module.

    EdgeTAM differences from SAM2:
    - Memory features from perceiver are [B, num_masks, 512, 64] (not [B, num_masks, 64, 64, 64])
    - add_tpos_enc_to_obj_ptrs=False: object pointer positions are zeros
    - mem_dim=64, so obj_ptrs [B, num_obj_ptr, 256] are split into 4 tokens each
    - num_spatial_mem is passed for RoPEAttentionv2's rope_k_repeat

    Inputs:
        current_vision_feat:      [1, 256, 64, 64]
        current_vision_pos_embed: [4096, 1, 256]
        memory_0:                 [B, num_obj_ptr, 256]     - object pointer memory
        memory_1:                 [B, num_masks, 512, 64]   - spatial memory (perceiver output)
        memory_pos_embed:         [B, num_masks*512+4*num_obj_ptr, 64] - memory pos encodings
        num_obj_ptr:              scalar int

    Output:
        image_embed: [B, 256, 64, 64]  - memory-conditioned image embeddings
    """

    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.no_mem_embed = sam_model.no_mem_embed
        self.memory_attention = sam_model.memory_attention
        self.mem_dim = sam_model.mem_dim  # 64

    @torch.no_grad()
    def forward(
        self,
        current_vision_feat: torch.Tensor,
        current_vision_pos_embed: torch.Tensor,
        memory_0: torch.Tensor,
        memory_1: torch.Tensor,
        memory_pos_embed: torch.Tensor,
        num_obj_ptr: torch.Tensor,
    ):
        batch_size = memory_0.size(0)
        n_obj_ptr = memory_0.size(1)
        num_masks = memory_1.size(1)
        C = 256  # hidden_dim

        # Reshape current vision feat: [1, 256, 64, 64] -> [4096, 1, 256]
        current_vision_feat = current_vision_feat.permute(2, 3, 0, 1).reshape(
            4096, 1, 256
        )
        # Subtract no_mem_embed (it was added by ImageEncoder for init frame)
        current_vision_feat = current_vision_feat - self.no_mem_embed

        # Expand for batch
        current_vision_feat = current_vision_feat.repeat(1, batch_size, 1)
        current_vision_pos_embed = current_vision_pos_embed.repeat(1, batch_size, 1)

        # Process spatial memory from perceiver: [B, num_masks, 512, 64]
        # -> [num_masks*512, B, 64]
        memory_1 = memory_1.permute(1, 2, 0, 3).reshape(-1, batch_size, 64)

        # Process object pointers: [B, num_obj_ptr, 256]
        # Split into 4 tokens each (since mem_dim=64, C=256, ratio=4)
        # -> [4*num_obj_ptr, B, 64]
        memory_0 = memory_0.reshape(batch_size, -1, C // self.mem_dim, self.mem_dim)
        memory_0 = memory_0.permute(1, 2, 0, 3).flatten(0, 1)

        num_obj_ptr_tokens = n_obj_ptr * (C // self.mem_dim)  # 4 * num_obj_ptr

        # Memory pos embed: [B, total_tokens, 64] -> [total_tokens, B, 64]
        memory_pos_embed = memory_pos_embed.permute(1, 0, 2)

        # Concatenate memories: spatial first, then object pointers
        memory = torch.cat((memory_1, memory_0), dim=0)

        # Run memory attention
        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feat,
            curr_pos=current_vision_pos_embed,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            num_spatial_mem=num_masks,
        )

        # Reshape output: (HW)xBxC -> BxCxHxW
        image_embed = pix_feat_with_mem.permute(1, 2, 0).view(
            batch_size, 256, 64, 64
        )

        return image_embed
