#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/jovyan/gpus-4-nodes-volume/minjun/graduate_research"
VENV_DIR="${APP_DIR}/.venv-b200"
REQ_FILE="${APP_DIR}/requirements.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# CUDA-sensitive packages (source build only)
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.1}"
CUDA_PYTHON_VERSION="${CUDA_PYTHON_VERSION:-13.2.0}"
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.4.2}"
QUACK_KERNELS_VERSION="${QUACK_KERNELS_VERSION:-0.3.10}"

# OOM-safe default for large cloud instances.
MAX_JOBS="${MAX_JOBS:-8}"
if [[ "${MAX_JOBS}" -gt 8 ]]; then
  MAX_JOBS=8
fi

cd "${APP_DIR}"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] requirements.txt not found: ${REQ_FILE}" >&2
  exit 1
fi

echo "[1/13] create venv (reuse container torch/cuda stack)"
"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[2/13] verify build toolchain"
for c in gcc g++ make nvcc; do
  if ! command -v "${c}" >/dev/null 2>&1; then
    echo "[ERROR] missing command: ${c}" >&2
    exit 1
  fi
done

echo "[3/13] set B200/Hopper-compatible TMA build flags"
export APP_DIR
export PIP_NO_CACHE_DIR=1
export CUDA_HOME=/usr/local/cuda
# Include 9.0a explicitly for TMA path compatibility and 10.0 for Blackwell.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a;10.0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90a;100}"
export FORCE_CUDA=1
export MAX_JOBS
export FLASH_ATTN_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
# Always prioritize local Mamba-3 Python code in this workspace.
export PYTHONPATH="${APP_DIR}:${PYTHONPATH:-}"

echo "[4/13] verify container torch/cuda baseline"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit("ERROR: CUDA-enabled torch is required.")
major = int(torch.version.cuda.split(".")[0])
if major < 12:
    raise SystemExit(f"ERROR: CUDA>=12 required, found {torch.version.cuda}")
PY

echo "[5/13] upgrade build helpers"
python -m pip install --upgrade pip setuptools wheel packaging ninja cmake

echo "[6/13] remove conflicting OpenCV variants"
python -m pip uninstall -y opencv opencv-python opencv-contrib-python opencv-contrib-python-headless || true

echo "[7/13] source-build flash-attn"
python -m pip install --no-build-isolation --no-binary=flash-attn "flash-attn==${FLASH_ATTN_VERSION}"

echo "[8/13] source-build causal-conv1d"
python -m pip install --no-build-isolation --no-binary=causal-conv1d "causal-conv1d==${CAUSAL_CONV1D_VERSION}"

echo "[9/13] build local selective_scan_cuda extension from workspace csrc"
python - <<'PY'
import os
import glob
import shutil
from pathlib import Path
from torch.utils.cpp_extension import load

app_dir = Path(os.environ["APP_DIR"])
src_dir = app_dir / "csrc" / "selective_scan"
if not src_dir.is_dir():
  raise SystemExit(f"ERROR: local selective_scan source dir not found: {src_dir}")

sources = [str(src_dir / "selective_scan.cpp")]
sources += sorted(glob.glob(str(src_dir / "selective_scan_*.cu")))
if len(sources) < 2:
  raise SystemExit("ERROR: local selective_scan sources are incomplete")

build_dir = app_dir / ".build" / "selective_scan_cuda"
build_dir.mkdir(parents=True, exist_ok=True)

# Remove stale in-repo extension files to prevent accidental stale imports.
for stale in app_dir.glob("selective_scan_cuda*.so"):
  stale.unlink()

module = load(
  name="selective_scan_cuda",
  sources=sources,
  extra_include_paths=[str(src_dir)],
  extra_cflags=["-O3", "-std=c++17"],
  extra_cuda_cflags=[
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-lineinfo",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
  ],
  with_cuda=True,
  build_directory=str(build_dir),
  verbose=True,
)

module_path = Path(module.__file__).resolve()
dst = app_dir / module_path.name
shutil.copy2(str(module_path), str(dst))
print("built:", module_path)
print("copied:", module_path, "->", dst)
PY

echo "[10/13] install remaining requirements (skip CUDA core stack + rebuilt libs)"
grep -Ev '^(mamba-ssm==|mamba_ssm\s*@|causal-conv1d==|causal_conv1d==|flash-attn==|flash_attn==|torch==|torchvision==|torchaudio==|xformers==|triton==)' \
  "${REQ_FILE}" > /tmp/requirements.base.b200.txt
python -m pip install -r /tmp/requirements.base.b200.txt

echo "[11/13] install Mamba3 Cute step-kernel dependencies"
python -m pip install --upgrade \
  "cuda-python==${CUDA_PYTHON_VERSION}" \
  "nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}" \
  "quack-kernels==${QUACK_KERNELS_VERSION}"

echo "[12/13] persist activation helper"
cat > "${APP_DIR}/activate_b200_venv.sh" <<EOF
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
export APP_DIR="${APP_DIR}"
export PYTHONPATH="${APP_DIR}:\${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
export CUDA_HOME="${CUDA_HOME}"
export MAX_JOBS="${MAX_JOBS}"
EOF
chmod +x "${APP_DIR}/activate_b200_venv.sh"

echo "[13/13] sanity checks (local Mamba-3 path + CUDA extensions)"
python - <<'PY'
import os
import mamba_ssm
from mamba_ssm.modules import mamba3 as mamba3_module
import selective_scan_cuda
from causal_conv1d import causal_conv1d_fn
import torch

app_dir = os.environ["APP_DIR"]
mamba_path = os.path.abspath(mamba_ssm.__file__)
print("mamba_ssm path:", mamba_path)
print("selective_scan_cuda path:", selective_scan_cuda.__file__)
print("causal_conv1d_fn:", callable(causal_conv1d_fn))
step_ok = getattr(mamba3_module, "mamba3_step_fn", None) is not None
print("mamba3_step_fn available:", step_ok)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
if not mamba_path.startswith(app_dir):
    raise SystemExit("ERROR: local mamba_ssm was not prioritized")
if not step_ok:
  import_error = getattr(mamba3_module, "mamba3_step_import_error", None)
  raise SystemExit(f"ERROR: Mamba3 step kernel unavailable. import_error={import_error!r}")
PY

echo
echo "Done. Activate with:"
echo "source ${APP_DIR}/activate_b200_venv.sh"
