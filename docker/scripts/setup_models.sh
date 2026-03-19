#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/opt/cache"
DEEPSTREAM_TOOLS="$CACHE_DIR/deepstream_tools"
SAM2_DIR="$DEEPSTREAM_TOOLS/sam2-onnx-tensorrt"
ONNX_DEST="/models/sam2/tiny/onnx"

mkdir -p "$CACHE_DIR" "$ONNX_DEST"

# Skip everything if all ONNX files already exist (e.g. copied from PC)
if [[ -f "$ONNX_DEST/image_encoder.onnx" ]] && \
   [[ -f "$ONNX_DEST/mask_decoder.onnx" ]] && \
   [[ -f "$ONNX_DEST/memory_encoder.onnx" ]] && \
   [[ -f "$ONNX_DEST/memory_attention.onnx" ]]; then
    echo "All ONNX models already present — skipping export"
    ls -lh "$ONNX_DEST/"*.onnx
    exit 0
fi

echo "============================================"
echo "  SAM2.1 Hiera Tiny — Model Setup"
echo "============================================"

# ── Step 1: Clone deepstream_tools ──
echo ""
echo ">>> [1/4] Clone sam2-onnx-tensorrt"
if [[ ! -d "$DEEPSTREAM_TOOLS" ]]; then
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_tools.git "$DEEPSTREAM_TOOLS"
else
    echo "    Already present"
fi

# ── Step 2: Patch mask_decoder.py ──
echo ""
echo ">>> [2/4] Patch mask_decoder.py (remove repeat_interleave, fixed to 1 mask)"
MASK_DEC="$SAM2_DIR/sam2/modeling/sam/mask_decoder.py"
if [[ -f "$MASK_DEC" ]] && grep -q "torch.repeat_interleave" "$MASK_DEC"; then
    sed -i \
        's/src = torch.repeat_interleave(image_embeddings, tokens.shape\[0\], dim=0)/src = image_embeddings/g' \
        "$MASK_DEC"
    sed -i \
        's/pos_src = torch.repeat_interleave(image_pe, tokens.shape\[0\], dim=0)/pos_src = image_pe/g' \
        "$MASK_DEC"
    echo "    Patched"
else
    echo "    Already patched or not found"
fi

# ── Step 2b: Patch export to use static shapes for mask_decoder ──
echo ""
echo ">>> [2b/4] Patch export_sam2_onnx.py (fixed batch=1, num_points=1, no dynamic_axes)"
EXPORT_SCRIPT="$SAM2_DIR/export_sam2_onnx.py"
if [[ -f "$EXPORT_SCRIPT" ]] && grep -q "batch_size = 20" "$EXPORT_SCRIPT"; then
    sed -i '/def export_mask_decoder/,/SUCCESS.*Mask Decoder/{
        s/batch_size = 20/batch_size = 1/
        s/torch.randn(batch_size,2,2)/torch.randn(1,1,2)/
        s/torch.randn(batch_size,2)/torch.randn(1,1)/
        s/torch.randn(batch_size,256,64,64)/torch.randn(1,256,64,64)/
        s/"point_coords":{0: "batch_size",1:"num_points"},/# dynamic_axes removed for static export/
        s/"point_labels": {0: "batch_size",1:"num_points"},//
        s/"image_embed": {0: "batch_size"},//
    }' "$EXPORT_SCRIPT"
    # Replace dynamic_axes dict with empty dict
    sed -i '/def export_mask_decoder/,/SUCCESS.*Mask Decoder/{
        s/dynamic_axes = dynamic_axes/dynamic_axes = {}/
    }' "$EXPORT_SCRIPT"
    echo "    Patched"
else
    echo "    Already patched or not found"
fi

# # ── Step 2b: Patch export to use opset 20 (cleaner graph for TRT 10.3) ──
# echo ""
# echo ">>> [2b/4] Patch export opset 17 -> 20"
# EXPORT_SCRIPT="$SAM2_DIR/export_sam2_onnx.py"
# if [[ -f "$EXPORT_SCRIPT" ]] && grep -q "opset_version=17" "$EXPORT_SCRIPT"; then
#     sed -i 's/opset_version=17/opset_version=20/g' "$EXPORT_SCRIPT"
#     echo "    Patched"
# else
#     echo "    Already patched or not found"
# fi

# ── Step 3: Download SAM2.1 Hiera Tiny checkpoint ──
echo ""
echo ">>> [3/4] Download SAM2.1 Hiera Tiny checkpoint"
CKPT_DIR="$SAM2_DIR/checkpoints"
mkdir -p "$CKPT_DIR"
cd "$CKPT_DIR"
if [[ ! -f "sam2.1_hiera_tiny.pt" ]]; then
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    echo "    Downloaded sam2.1_hiera_tiny.pt"
else
    echo "    Already present"
fi

# ── Step 4: Export ONNX (tiny only) ──
echo ""
echo ">>> [4/4] Export SAM2 ONNX models (tiny)"
cd "$SAM2_DIR"

# Install sam2 package (needed for export script)
pip install -e . 2>/dev/null || true

ONNX_SRC="$SAM2_DIR/checkpoints/tiny"
mkdir -p "$ONNX_SRC"
if [[ ! -f "$ONNX_SRC/image_encoder.onnx" ]]; then
    python3 export_sam2_onnx.py --model tiny
else
    echo "    ONNX files already present"
fi

# Copy to destination
echo "    Copying ONNX files to $ONNX_DEST/"
for f in image_encoder.onnx mask_decoder.onnx memory_encoder.onnx memory_attention.onnx; do
    if [[ -f "$ONNX_SRC/$f" ]]; then
        cp "$ONNX_SRC/$f" "$ONNX_DEST/"
    else
        echo "    WARNING: $f not found in $ONNX_SRC"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  ONNX models: $ONNX_DEST/"
ls -lh "$ONNX_DEST/"
echo "============================================"
