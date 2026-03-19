ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

# Minimal OS deps
RUN apt-get update && apt-get install -y \
    curl git wget build-essential python3-venv ffmpeg cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Python venv with system-site-packages (for tensorrt bindings from base image)
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/usr/src/tensorrt/bin:/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH}"

# Fast pip via uv
RUN pip install uv

# Jetson AI Lab wheels (aarch64, CUDA 12.6, Python 3.10)
RUN python3 -m pip install --ignore-installed --no-cache-dir \
    https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc \
    https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1 \
    https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b

# Remaining Python requirements
COPY docker/requirements.txt /opt/requirements.txt
RUN uv pip install --python /opt/venv/bin/python \
    --no-cache-dir -r /opt/requirements.txt

# App code
COPY app/ /app/

# Scripts
COPY docker/scripts/ /opt/scripts/
RUN chmod +x /opt/scripts/*.sh

# Entrypoint
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

# Create mount point directories
RUN mkdir -p /models/sam2/tiny/onnx /models/sam2/tiny/trt /streams /output

# Install cuDSS (needed by Jetson torch 2.10.0)
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y cudss \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
CMD ["/opt/entrypoint.sh", "-r"]
