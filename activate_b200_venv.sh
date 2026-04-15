#!/usr/bin/env bash
source "/home/jovyan/gpus-4-nodes-volume/minjun/graduate_research/.venv-b200/bin/activate"
export APP_DIR="/home/jovyan/gpus-4-nodes-volume/minjun/graduate_research"
export PYTHONPATH="/home/jovyan/gpus-4-nodes-volume/minjun/graduate_research:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0a;10.0"
export CMAKE_CUDA_ARCHITECTURES="90a;100"
export CUDA_HOME="/usr/local/cuda"
export MAX_JOBS="8"
