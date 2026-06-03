#!/bin/bash

#SBATCH --job-name=aac_annotate
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch.hpc/lorenzo.pellegrino2/aac-mcp-agent/annotation/cluster_work/logs/annotate_%j.out
#SBATCH --error=/scratch.hpc/lorenzo.pellegrino2/aac-mcp-agent/annotation/cluster_work/logs/annotate_%j.out

#### Paths ######################################################################
SCRATCH_ROOT="/scratch.hpc/lorenzo.pellegrino2"
PROJECT_ROOT="${SCRATCH_ROOT}/aac-mcp-agent"
VENV_DIR="${SCRATCH_ROOT}/python_venvs/aac_annotate"
NB_IN="${PROJECT_ROOT}/annotation/cluster_work/annotate_eval.ipynb"
NB_OUT="${PROJECT_ROOT}/annotation/cluster_work/annotate_eval_out.ipynb"
LOGS_DIR="${PROJECT_ROOT}/logs"

#### Configuration ##############################################################
export NB_PROJECT_ROOT="${PROJECT_ROOT}"
export NB_MODEL_ID="${NB_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
export NB_LOAD_IN_4BIT="${NB_LOAD_IN_4BIT:-1}"
export NB_MAX_NEW_TOKENS="${NB_MAX_NEW_TOKENS:-256}"
export NB_MAX_PROMPT_LENGTH="${NB_MAX_PROMPT_LENGTH:-2048}"
export NB_BATCH_SIZE="${NB_BATCH_SIZE:-16}"
export NB_MAX_ROW_RETRIES="${NB_MAX_ROW_RETRIES:-3}"
export NB_MAX_ANNOTATION_RETRIES="${NB_MAX_ANNOTATION_RETRIES:-2}"
export NB_BACKUP_EVERY_N="${NB_BACKUP_EVERY_N:-10}"
export NB_N_ROWS="${NB_N_ROWS:-0}"

if [[ -z "${HF_TOKEN}" ]] && [[ -f "${PROJECT_ROOT}/app/.env" ]]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "${PROJECT_ROOT}/app/.env" \
                | head -1 | cut -d= -f2- | tr -d '\"' )"
    export HF_TOKEN
fi

#### Redirect ALL caches into scratch (avoid filling home quota) ###################
export HF_HOME="${SCRATCH_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Triton compiles kernel C extensions into ~/.triton/cache by default — redirect to scratch
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/.triton/cache"

#### Sanity check: ensure PROJECT_ROOT is on scratch (not home) ####################
case "${PROJECT_ROOT}" in
    /scratch.hpc/lorenzo.pellegrino2/*) ;;
    *) echo "ERROR: PROJECT_ROOT is outside /scratch.hpc/lorenzo.pellegrino2"; exit 1 ;;
esac

mkdir -p "${LOGS_DIR}" "${HF_HOME}"

echo "======================================================="
echo "  JOB ID   : ${SLURM_JOB_ID}"
echo "  NODE     : $(hostname)"
echo "  GPU      : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  MODEL    : ${NB_MODEL_ID}"
echo "  N_ROWS   : ${NB_N_ROWS} (0=all)"
echo "  VENV     : ${VENV_DIR}"
echo "  NOTEBOOK : ${NB_IN}"
echo "======================================================="

#### Create venv (only if missing) #####################################################
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup] Creating venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip --quiet

    # Notebook dependencies (not re-installed inside the .ipynb)
    pip install --quiet \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124

    pip install --quiet \
        transformers \
        accelerate \
        bitsandbytes \
        sentencepiece \
        protobuf \
        pandas \
        pyarrow \
        python-dotenv \
        papermill \
        jupyter \
        ipykernel

    # Register kernel inside the venv (--sys-prefix avoids ~/.local conflicts)
    python -m ipykernel install \
        --sys-prefix \
        --name aac_annotate \
        --display-name "aac_annotate"

    echo "[setup] Venv ready."
else
    echo "[setup] Venv already exists — skipping installation."
    source "${VENV_DIR}/bin/activate"
fi

# Verify GPU is visible from Python (runs after venv activation + torch install)
python - <<'PYCHECK'
import torch, sys
if not torch.cuda.is_available():
    print("WARNING: CUDA not available — job will be extremely slow on CPU.")
else:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU OK: {props.name}  {props.total_memory/1e9:.1f} GB VRAM")
PYCHECK

#### Launch notebook via papermill #################################################
echo "[run] Starting papermill ..."
papermill \
    "${NB_IN}" \
    "${NB_OUT}" \
    --kernel aac_annotate \
    --log-output \
    --execution-timeout 28800

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[done] Notebook completed successfully."
    echo "  Output parquet : ${PROJECT_ROOT}/annotation/eval_annotated.parquet"
    echo "  Annotation log : ${PROJECT_ROOT}/annotation/annotation_log.jsonl"
    echo "  Notebook output: ${NB_OUT}"
else
    echo "[ERROR] papermill exited with code ${EXIT_CODE}."
    echo "  Check: ${LOGS_DIR}/annotate_${SLURM_JOB_ID}.out"
fi

exit ${EXIT_CODE}
