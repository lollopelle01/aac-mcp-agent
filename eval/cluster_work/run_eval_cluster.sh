#!/bin/bash
#SBATCH --job-name=eval_aac_hf
#SBATCH --partition=l40
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Resolve script directory early (works with sbatch — BASH_SOURCE[0] is the real path)
SCRIPT_DIR="/scratch.hpc/lorenzo.pellegrino2/aac-mcp-agent/eval/cluster_work"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")/.."   # <project_root>/
NOTEBOOK="${PROJECT_ROOT}/eval/eval.ipynb"
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results" "${PROJECT_ROOT}/eval/results"

# Keep ALL caches out of home — redirect everything to scratch inside the project
export PIP_CACHE_DIR="/scratch.hpc/${USER}/pip_cache"
export TORCH_HOME="${PROJECT_ROOT}/.torch_cache"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.xdg_cache"
export HF_HOME="/scratch.hpc/${USER}/hf_cache"
mkdir -p "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}"

# Redirect output to logs in the script directory
exec > >(tee -a "${SCRIPT_DIR}/logs/eval_aac_hf_${SLURM_JOB_ID}.out") \
     2> >(tee -a "${SCRIPT_DIR}/logs/eval_aac_hf_${SLURM_JOB_ID}.err" >&2)

# ── 0. Configuration — edit these before submitting ──────────────────────────────
VENV_DIR="/scratch.hpc/lorenzo.pellegrino2/python_venvs/BDATM"

# Notebook parameters injected as env vars (read by the Config cell via os.environ).
# Overridable on the sbatch command line:  MODELS="..." sbatch run_eval_cluster.sh
export NB_IS_COLAB="False"
export NB_MODELS="${NB_MODELS:-Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct ibm-granite/granite-3.1-2b-instruct}"
export NB_SPLIT="${NB_SPLIT:-both}"
export NB_N_ROWS="${NB_N_ROWS:-0}"          # 0 = full dataset on cluster
export NB_SEED="${NB_SEED:-42}"
export NB_LOAD_8BIT="${NB_LOAD_8BIT:-False}"
export NB_MAX_NEW_TOKENS="${NB_MAX_NEW_TOKENS:-512}"
export NB_LANG="${NB_LANG:-en_eval}"
export NB_OUTPUT_CSV="eval/results/eval_cluster_${SLURM_JOB_ID}.csv"

SENTINEL="${VENV_DIR}/.setup_ok"

# ── 1. Fast path: venv already ready ──────────────────────────────────────────────
if [ -f "${SENTINEL}" ]; then
    echo "[setup] Sentinel found — skipping install."
    source "${VENV_DIR}/bin/activate"
    echo "[setup] Python: $(which python3)"
else
    # ── 2. CUDA module ────────────────────────────────────────────────────────────────
    echo "[setup] Probing CUDA modules …"
    LOADED=0
    for VER in 12.4 12.3 12.2 12.1 12.0 11.8 11.7; do
        if module load cuda/${VER} 2>/dev/null; then
            echo "[setup] Loaded cuda/${VER}"
            LOADED=1; break
        fi
    done
    [ ${LOADED} -eq 0 ] && echo "[setup] No CUDA module found — relying on system CUDA."

    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" \
               | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    echo "[setup] nvcc CUDA version: ${CUDA_VER:-not found}"
    if [ -z "${CUDA_VER}" ]; then
        echo "[ERROR] nvcc not found." >&2; exit 1
    fi

    case "${CUDA_VER}" in
        12.4*)
            TORCH_INDEX="cu124"
            TORCH_VER="2.7.1+cu124"; TV_VER="0.22.1+cu124"; TA_VER="2.7.1+cu124" ;;
        12.3*|12.2*|12.1*)
            TORCH_INDEX="cu121"
            TORCH_VER="2.7.1+cu121"; TV_VER="0.22.1+cu121"; TA_VER="2.7.1+cu121" ;;
        12.0*|11.8*)
            TORCH_INDEX="cu118"
            TORCH_VER="2.7.1+cu118"; TV_VER="0.22.1+cu118"; TA_VER="2.7.1+cu118" ;;
        *)
            echo "[WARN] Unrecognised CUDA ${CUDA_VER} — defaulting to cu118." >&2
            TORCH_INDEX="cu118"
            TORCH_VER="2.7.1+cu118"; TV_VER="0.22.1+cu118"; TA_VER="2.7.1+cu118" ;;
    esac
    echo "[setup] torch==${TORCH_VER}"

    # ── 3. Create / activate venv ─────────────────────────────────────────────────────
    if [ ! -d "${VENV_DIR}" ]; then
        echo "[setup] Creating venv at ${VENV_DIR} …"
        python3 -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"

    # ── 4. Install packages ─────────────────────────────────────────────────────────────
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
        "sentencepiece>=0.1.99" \
        "protobuf>=3.20" \
        "spacy>=3.7" \
        "nbconvert>=7.0" \
        "nbformat>=5.9" \
        "ipykernel>=6.0" \
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

    # Project-specific deps (no ollama needed on cluster)
    pip install -q --upgrade \
        "fastmcp" \
        "pydantic>=2.0" \
        "httpx>=0.24"

    # ── 5. Sanity check ───────────────────────────────────────────────────────────────
    python3 - << 'PYCHECK'
import importlib, sys, torch
from packaging.version import Version

required = [
    ("torch",        "2.0.0"),
    ("transformers", "4.40.0"),
    ("accelerate",   "0.29.0"),
    ("pandas",       "2.0.0"),
    ("pydantic",     "2.0.0"),
    ("nbconvert",    "7.0.0"),
]
ok = True
for pkg, min_ver in required:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "0.0.0")
        if Version(ver) < Version(min_ver):
            print(f"  [WARN] {pkg} {ver} < {min_ver}", file=sys.stderr); ok = False
        else:
            print(f"  [OK]   {pkg} {ver}")
    except ImportError:
        print(f"  [FAIL] {pkg} missing", file=sys.stderr); ok = False

print(f"\n  CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  ({p.total_memory/1e9:.1f} GB)")

if not ok or not torch.cuda.is_available():
    sys.exit(1)
PYCHECK

    [ $? -ne 0 ] && { echo "[ERROR] Sanity check failed." >&2; exit 1; }

    touch "${SENTINEL}"
    echo "[setup] Sentinel written."
fi

# ── 6. Run the notebook via nbconvert ────────────────────────────────────────────────
echo "[run] Notebook : ${NOTEBOOK}"
echo "[run] Models   : ${NB_MODELS}"
echo "[run] Split    : ${NB_SPLIT}  N_rows: ${NB_N_ROWS}"
echo "[run] Output   : ${PROJECT_ROOT}/${NB_OUTPUT_CSV}"
echo "[run] Started  : $(date)"

# nbconvert executes the notebook in-place and writes a timestamped executed copy
# to logs/ for debugging. The notebook itself reads config from NB_* env vars.
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "${SCRIPT_DIR}/logs/eval_executed_${SLURM_JOB_ID}.ipynb" \
    "${NOTEBOOK}"

EXIT_CODE=$?
echo "[run] Finished : $(date)  exit=${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[run] CSV saved to: ${PROJECT_ROOT}/${NB_OUTPUT_CSV}"
else
    echo "[ERROR] Notebook execution failed (exit ${EXIT_CODE})" >&2
fi

deactivate
exit ${EXIT_CODE}
