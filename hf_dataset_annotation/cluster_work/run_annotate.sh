#!/bin/bash
#SBATCH --job-name=annotate_aac
#SBATCH --partition=l40
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/annotate_aac_%j.out
#SBATCH --error=logs/annotate_aac_%j.err

# ── 0. Paths — edit these ────────────────────────────────────────────────────
VENV_DIR="../python_venvs/BDATM"
NOTEBOOK="annotate_dataset.ipynb"
OUTPUT_NOTEBOOK="annotate_dataset_executed_${SLURM_JOB_ID}.ipynb"

# Sentinel: created after a successful install + sanity check.
# As long as it exists the entire setup block is skipped on subsequent runs.
SENTINEL="${VENV_DIR}/.setup_ok"

mkdir -p logs

# ── 1. Fast path: venv already set up and verified ───────────────────────────
if [ -f "${SENTINEL}" ]; then
    echo "[setup] Sentinel found — skipping install and sanity check."
    source "${VENV_DIR}/bin/activate"
    echo "[setup] Python : $(which python3)"
else
    # ── 2. Load CUDA module ───────────────────────────────────────────────────
    echo "[setup] Available CUDA modules:"
    module avail cuda 2>&1 | grep -i cuda || echo "  (none found via module avail)"

    LOADED=0
    for VER in 12.4 12.3 12.2 12.1 12.0 11.8 11.7; do
        if module load cuda/${VER} 2>/dev/null; then
            echo "[setup] Loaded cuda/${VER}"
            LOADED=1
            break
        fi
    done
    [ ${LOADED} -eq 0 ] && echo "[setup] No cuda module found — relying on system CUDA"

    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    echo "[setup] nvcc CUDA version: ${CUDA_VER:-not found}"

    if [ -z "${CUDA_VER}" ]; then
        echo "[ERROR] nvcc not found — cannot continue without CUDA." >&2
        exit 1
    fi

    # Map CUDA version → pinned PyTorch wheel
    case "${CUDA_VER}" in
        12.4*) TORCH_INDEX="cu124"; TORCH_VER="2.7.1+cu124"; TV_VER="0.22.1+cu124"; TA_VER="2.7.1+cu124" ;;
        12.3*|12.2*|12.1*) TORCH_INDEX="cu121"; TORCH_VER="2.7.1+cu121"; TV_VER="0.22.1+cu121"; TA_VER="2.7.1+cu121" ;;
        12.0*|11.8*) TORCH_INDEX="cu118"; TORCH_VER="2.7.1+cu118"; TV_VER="0.22.1+cu118"; TA_VER="2.7.1+cu118" ;;
        11.7*) TORCH_INDEX="cu117"; TORCH_VER="2.0.1+cu117"; TV_VER="0.15.2+cu117"; TA_VER="2.0.2+cu117" ;;
        *)
            echo "[WARN] Unrecognised CUDA ${CUDA_VER} — defaulting to cu118" >&2
            TORCH_INDEX="cu118"; TORCH_VER="2.7.1+cu118"; TV_VER="0.22.1+cu118"; TA_VER="2.7.1+cu118"
            ;;
    esac
    echo "[setup] torch==${TORCH_VER}  torchvision==${TV_VER}  torchaudio==${TA_VER}"

    # ── 3. Create venv if missing, then install ───────────────────────────────
    if [ ! -d "${VENV_DIR}" ]; then
        echo "[setup] Creating virtual environment at ${VENV_DIR} ..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    echo "[setup] Python : $(which python3)"

    echo "[setup] Installing pip packages ..."
    pip install -q --upgrade pip

    pip install -q \
        "torch==${TORCH_VER}" \
        "torchvision==${TV_VER}" \
        "torchaudio==${TA_VER}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

    pip install -q --upgrade \
        "transformers>=4.40,<5.0" \
        "accelerate>=0.29" \
        "bitsandbytes>=0.43" \
        "datasets>=2.19" \
        "pandas>=2.0" \
        "pyarrow>=14" \
        "huggingface_hub>=0.22" \
        "tqdm>=4.66" \
        "jupyter" \
        "nbconvert>=7.0" \
        "nbformat" \
        "ipykernel" \
        "sentencepiece>=0.1.99" \
        "protobuf>=3.20"

    echo "[setup] Package installation complete."

    # ── 4. Sanity check (only on first install) ───────────────────────────────
    python3 - << 'PYCHECK'
import importlib, sys
from packaging.version import Version

required = [
    ("torch",            "2.0.0"),
    ("transformers",     "4.40.0"),
    ("accelerate",       "0.29.0"),
    ("bitsandbytes",     "0.43.0"),
    ("datasets",         "2.19.0"),
    ("pandas",           "2.0.0"),
    ("pyarrow",          "14.0.0"),
    ("huggingface_hub",  "0.22.0"),
    ("tqdm",             "4.66.0"),
    ("nbconvert",        "7.0.0"),
]

missing = []
for pkg, min_ver in required:
    try:
        mod = importlib.import_module(pkg)
        installed = getattr(mod, "__version__", "0.0.0")
        if Version(installed) < Version(min_ver):
            print(f"  [WARN] {pkg} {installed} < required {min_ver}", file=sys.stderr)
        else:
            print(f"  [OK]   {pkg} {installed}")
    except ImportError:
        missing.append(pkg)
        print(f"  [FAIL] {pkg} not importable!", file=sys.stderr)

if missing:
    print(f"\nAborting: missing packages: {missing}", file=sys.stderr)
    sys.exit(1)

import torch
print(f"\n  torch          : {torch.__version__}  (CUDA built: {torch.version.cuda})")
if not torch.cuda.is_available():
    print("[ERROR] GPU not visible — check CUDA installation and torch wheel.", file=sys.stderr)
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}           : {p.name}  ({p.total_memory/1e9:.1f} GB)")

import transformers
if Version(transformers.__version__) >= Version("5.0.0"):
    print(f"  [ERROR] transformers {transformers.__version__} >= 5.0 — pin to <5.0.", file=sys.stderr)
    sys.exit(1)
print(f"  transformers   : {transformers.__version__}  (< 5.0 ✓)")

from transformers import AutoModelForCausalLM
print("  AutoModelForCausalLM import : OK")
PYCHECK

    if [ $? -ne 0 ]; then
        echo "[ERROR] Sanity-check failed. Aborting." >&2
        exit 1
    fi

    # ── 5. Register Jupyter kernel (once) ─────────────────────────────────────
    python3 -m ipykernel install --user --name annotate_aac --display-name "annotate_aac" 2>/dev/null || true

    # Mark setup as complete so future runs skip straight to execution
    touch "${SENTINEL}"
    echo "[setup] Sentinel written — future runs will skip setup entirely."
fi

# ── 6. Execute the notebook ───────────────────────────────────────────────────
echo "[run] Starting notebook execution: ${NOTEBOOK}"
echo "[run] Output → ${OUTPUT_NOTEBOOK}"
echo "[run] Job started at: $(date)"

jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --output "${OUTPUT_NOTEBOOK}" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=annotate_aac \
    "${NOTEBOOK}"

EXIT_CODE=$?
echo "[run] Job finished at: $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[run] Notebook completed successfully → ${OUTPUT_NOTEBOOK}"
else
    echo "[ERROR] Notebook execution failed (exit ${EXIT_CODE})" >&2
    echo "[ERROR] Inspect: ${OUTPUT_NOTEBOOK}" >&2
fi

deactivate
exit ${EXIT_CODE}
