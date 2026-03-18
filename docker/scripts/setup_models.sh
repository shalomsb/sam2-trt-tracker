#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/opt/cache"
DEEPSTREAM_TOOLS="$CACHE_DIR/deepstream_tools"
SAM2_DIR="$DEEPSTREAM_TOOLS/sam2-onnx-tensorrt"
ONNX_DEST="/models/sam2/tiny/onnx"

mkdir -p "$CACHE_DIR" "$ONNX_DEST"

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
echo ">>> [2/4] Patch mask_decoder.py (repeat_interleave -> tile)"
MASK_DEC="$SAM2_DIR/sam2/modeling/sam/mask_decoder.py"
if [[ -f "$MASK_DEC" ]] && grep -q "torch.repeat_interleave" "$MASK_DEC"; then
    sed -i \
        's/src = torch.repeat_interleave(image_embeddings, tokens.shape\[0\], dim=0)/src = torch.tile(image_embeddings, (tokens.shape[0], 1, 1, 1))/g' \
        "$MASK_DEC"
    sed -i \
        's/pos_src = torch.repeat_interleave(image_pe, tokens.shape\[0\], dim=0)/pos_src = torch.tile(image_pe, (tokens.shape[0], 1, 1, 1))/g' \
        "$MASK_DEC"
    echo "    Patched"
else
    echo "    Already patched or not found"
fi

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
