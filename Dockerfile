# docker build -f Dockerfile -t drama .
# 24.05 baseline: CUDA 12.4.1 + PyTorch 2.4.x preinstalled in the image.
FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /workspace

# Runtime libs used by gym/render/video stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Keep build tooling available for CUDA extension wheels.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# NGC base image can include an "opencv" package that conflicts with
# opencv-python-headless. Remove all OpenCV variants first.
RUN pip uninstall -y opencv opencv-python opencv-contrib-python opencv-contrib-python-headless || true

COPY requirements.txt /tmp/requirements.txt

# Build flash-attn from source against container-provided torch/cuda to
# avoid ABI mismatch with NGC torch builds.
RUN FLASH_ATTN_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation \
    --no-binary=flash-attn flash-attn==2.4.2
# Install torch-dependent CUDA extensions before requirements to avoid
# PEP517 build-isolation missing torch errors.
RUN pip install --no-cache-dir causal-conv1d==1.6.1 --no-build-isolation
# Install mamba-ssm from source tarball (not PyPI sdist) and keep
# container-provided torch/triton stack by skipping dependency resolution.
RUN pip install --no-cache-dir --no-build-isolation --no-deps \
    "mamba-ssm @ https://github.com/state-spaces/mamba/archive/refs/tags/v1.2.0.post1.tar.gz"
# Prevent pip resolver from reinstalling these CUDA extensions or pulling a different triton.
RUN grep -Ev '^(mamba-ssm==|causal-conv1d==)' /tmp/requirements.txt > /tmp/requirements.base.txt \
    && pip install --no-cache-dir -r /tmp/requirements.base.txt

COPY . /workspace

EXPOSE 6006