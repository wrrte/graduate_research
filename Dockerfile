# docker build -f Dockerfile -t drama-b200 .
# B200/TMA focus: default to a Blackwell-capable (non-latest) NGC PyTorch image.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}

ARG APP_DIR=/home/jovyan/gpus-4-nodes-volume/minjun/graduate_research
ARG TORCH_CUDA_ARCH_LIST=10.0
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG CAUSAL_CONV1D_VERSION=1.6.1
ARG MAMBA_SSM_TAG=v1.2.0.post1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=16 \
    FLASH_ATTN_FORCE_BUILD=TRUE \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    MAMBA_FORCE_BUILD=TRUE

WORKDIR ${APP_DIR}

# Runtime + build deps for gym/render and CUDA extension source builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging ninja cmake

# Avoid conflicts with opencv-python-headless from requirements.
RUN pip uninstall -y opencv opencv-python opencv-contrib-python opencv-contrib-python-headless || true

COPY requirements.txt /tmp/requirements.txt

# Build architecture-sensitive CUDA extensions from source (no cached wheels).
RUN pip install --no-build-isolation --no-binary=flash-attn \
    flash-attn==${FLASH_ATTN_VERSION}
RUN pip install --no-build-isolation --no-binary=causal-conv1d \
    causal-conv1d==${CAUSAL_CONV1D_VERSION}
RUN pip install --no-build-isolation --no-deps \
    "mamba-ssm @ https://github.com/state-spaces/mamba/archive/refs/tags/${MAMBA_SSM_TAG}.tar.gz"

# Keep NGC torch/triton stack and skip known CUDA-sensitive packages here.
RUN grep -Ev '^(mamba-ssm==|causal-conv1d==|causal_conv1d==|flash-attn==|flash_attn==|torch==|torchvision==|torchaudio==|xformers==|triton==)' /tmp/requirements.txt \
    > /tmp/requirements.base.txt \
    && pip install -r /tmp/requirements.base.txt

COPY . ${APP_DIR}

EXPOSE 6006