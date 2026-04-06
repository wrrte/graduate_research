# docker build -f Dockerfile -t drama .
# 24.05 baseline: CUDA 12.4.1 + PyTorch 2.4.x preinstalled in the image.
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace

# Runtime libs used by gym/render/video stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Keep build tooling available for CUDA extension wheels.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

COPY requirements.txt /tmp/requirements.txt

# Build flash-attn against container-provided torch/cuda.
RUN pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

EXPOSE 6006