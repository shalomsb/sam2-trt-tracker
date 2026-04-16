ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:26.02-py3
FROM ${BASE_IMAGE}

# Minimal OS deps
RUN apt-get update && apt-get install -y \
    curl git wget build-essential python3-venv ffmpeg cmake \
    && rm -rf /var/lib/apt/lists/*

# Python venv with system-site-packages (for tensorrt bindings from base image)
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

# Fast pip via uv
RUN pip install uv

# Python requirements
COPY docker/requirements.txt /opt/requirements.txt
RUN uv pip install --python /opt/venv/bin/python \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --no-cache-dir -r /opt/requirements.txt

# EdgeTAM package (for PyTorch reference tracker)
RUN SAM2_BUILD_CUDA=0 uv pip install --python /opt/venv/bin/python --no-cache-dir \
    "EdgeTAM @ git+https://github.com/facebookresearch/EdgeTAM.git"

# Expose pip-installed CUDA 12 libs for onnxruntime-gpu
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

# App code
COPY app/ /app/

# EdgeTAM ONNX export code (used by setup_models.sh)
COPY edgetam_export/onnx_export/ /opt/edgetam_export/onnx_export/
COPY edgetam_export/export_edgetam_onnx.py /opt/edgetam_export/

# Pre-patched position_encoding.py for ONNX (complex -> real-valued RoPE)
COPY edgetam_export/patches/ /opt/patches/

# Scripts
COPY docker/scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Create mount point directories
RUN mkdir -p /models/edgetam/onnx /models/edgetam/trt /streams /output

WORKDIR /app
CMD ["/opt/entrypoint.sh", "-r"]
