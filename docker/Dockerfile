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

WORKDIR /app
CMD ["/opt/entrypoint.sh", "-r"]
