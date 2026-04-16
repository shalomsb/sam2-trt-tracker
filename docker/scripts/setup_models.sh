#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/opt/cache"
EDGETAM_DIR="$CACHE_DIR/EdgeTAM"
ONNX_DEST="/models/edgetam/onnx"

mkdir -p "$CACHE_DIR" "$ONNX_DEST"

echo "============================================"
echo "  EdgeTAM — Model Setup"
echo "============================================"

# ── Step 1: Clone EdgeTAM ──
echo ""
echo ">>> [1/5] Clone EdgeTAM"
if [[ ! -d "$EDGETAM_DIR" ]]; then
    git clone https://github.com/facebookresearch/EdgeTAM.git "$EDGETAM_DIR"
else
    echo "    Already present"
fi

# ── Step 2: Download EdgeTAM checkpoint ──
echo ""
echo ">>> [2/5] Download EdgeTAM checkpoint"
CKPT_DIR="$EDGETAM_DIR/checkpoints"
mkdir -p "$CKPT_DIR"
if [[ ! -f "$CKPT_DIR/edgetam.pt" ]]; then
    cd "$CKPT_DIR"
    bash download_ckpts.sh
    echo "    Downloaded edgetam.pt"
else
    echo "    Already present"
fi
# Copy checkpoint to host-mounted models volume
mkdir -p /models/edgetam
cp "$CKPT_DIR/edgetam.pt" /models/edgetam/edgetam.pt
echo "    Copied checkpoint to /models/edgetam/"

# ── Step 3: Patch EdgeTAM for ONNX export ──
echo ""
echo ">>> [3/5] Patch EdgeTAM for ONNX compatibility"

# 3a: Patch mask_decoder.py (repeat_interleave -> tile)
MASK_DEC="$EDGETAM_DIR/sam2/modeling/sam/mask_decoder.py"
if [[ -f "$MASK_DEC" ]] && grep -q "torch.repeat_interleave" "$MASK_DEC"; then
    sed -i \
        's/src = torch.repeat_interleave(image_embeddings, tokens.shape\[0\], dim=0)/src = torch.tile(image_embeddings, (tokens.shape[0], 1, 1, 1))/g' \
        "$MASK_DEC"
    sed -i \
        's/pos_src = torch.repeat_interleave(image_pe, tokens.shape\[0\], dim=0)/pos_src = torch.tile(image_pe, (tokens.shape[0], 1, 1, 1))/g' \
        "$MASK_DEC"
    echo "    Patched mask_decoder.py (repeat_interleave -> tile)"
else
    echo "    mask_decoder.py already patched or not found"
fi

# 3b: Patch prompt_encoder.py (masked indexing -> torch.where for ONNX)
PROMPT_ENC="$EDGETAM_DIR/sam2/modeling/sam/prompt_encoder.py"
if [[ -f "$PROMPT_ENC" ]] && grep -q 'point_embedding\[labels ==' "$PROMPT_ENC"; then
    python3 -c "
import re
with open('$PROMPT_ENC', 'r') as f:
    code = f.read()
old = '''        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding'''
new = '''        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        # Use torch.where for ONNX compatibility (masked indexing traces incorrectly)
        labels_expanded = labels.unsqueeze(-1)  # [B, N, 1] for broadcasting with [B, N, C]
        is_neg1 = (labels_expanded == -1)
        is_0 = (labels_expanded == 0)
        is_1 = (labels_expanded == 1)
        is_2 = (labels_expanded == 2)
        is_3 = (labels_expanded == 3)
        point_embedding = torch.where(is_neg1, torch.zeros_like(point_embedding), point_embedding)
        point_embedding = point_embedding + is_neg1.float() * self.not_a_point_embed.weight
        point_embedding = point_embedding + is_0.float() * self.point_embeddings[0].weight
        point_embedding = point_embedding + is_1.float() * self.point_embeddings[1].weight
        point_embedding = point_embedding + is_2.float() * self.point_embeddings[2].weight
        point_embedding = point_embedding + is_3.float() * self.point_embeddings[3].weight
        return point_embedding'''
code = code.replace(old, new)
with open('$PROMPT_ENC', 'w') as f:
    f.write(code)
"
    echo "    Patched prompt_encoder.py (masked indexing -> torch.where)"
else
    echo "    prompt_encoder.py already patched or not found"
fi

# 3c: Patch position_encoding.py (complex numbers -> real-valued for ONNX)
POS_ENC="$EDGETAM_DIR/sam2/modeling/position_encoding.py"
if [[ -f "$POS_ENC" ]] && grep -q "torch.polar\|view_as_complex" "$POS_ENC"; then
    # Copy the pre-patched version from the export package
    cp /opt/patches/position_encoding_onnx.py "$POS_ENC"
    echo "    Patched position_encoding.py (complex -> real-valued RoPE)"
else
    echo "    position_encoding.py already patched or not found"
fi

# ── Step 4: Install EdgeTAM + copy export code ──
echo ""
echo ">>> [4/5] Install EdgeTAM and prepare export"
cd "$EDGETAM_DIR"
pip install timm 2>/dev/null || true
pip install -e . 2>/dev/null || true

# Copy ONNX export modules
cp -r /opt/edgetam_export/onnx_export "$EDGETAM_DIR/"
cp /opt/edgetam_export/export_edgetam_onnx.py "$EDGETAM_DIR/"

# ── Step 5: Export ONNX models ──
echo ""
echo ">>> [5/5] Export EdgeTAM ONNX models"
if [[ ! -f "$ONNX_DEST/image_encoder.onnx" ]]; then
    cd "$EDGETAM_DIR"
    python3 export_edgetam_onnx.py --output_dir "$ONNX_DEST"
else
    echo "    ONNX files already present"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  ONNX models: $ONNX_DEST/"
ls -lh "$ONNX_DEST/"
echo "============================================"
