#!/bin/bash
set -euo pipefail

# ============================================================
# OpenFactScore Entity-wise Evaluation (SciFactCheck)
# ============================================================

# -------------------- SLURM CONFIG --------------------
export JOB_NAME=${JOB_NAME:-ofs_eval}
export ANY_PARTITION=${ANY_PARTITION:-RTXA6000,V100-32GB,H100,A100-40GB,A100-80GB,A100-PCI,RTXA6000-SLT}
export SLURM_CPUS=${SLURM_CPUS:-2}
export SLURM_MEM=${SLURM_MEM:-128G}
export SLURM_GPUS=${SLURM_GPUS:-1}

# -------------------- CONTAINER -----------------------
export ENROOT_IMAGE_DIR=/netscratch/$USER/images
export IMAGE=${IMAGE:-$ENROOT_IMAGE_DIR/OpenFS_upgrade_hf_3_login.sqsh}

# -------------------- PATHS ---------------------------
export BASE_DIR=/netscratch/$USER/SciFactCheck
export CODE_DIR=$BASE_DIR/OpenFActScore
export CLAIMS_DIR=$BASE_DIR/VeriScore/data/claims_input_files
export LOGS_DIR=$CODE_DIR/logs
export RESULTS_DIR=$CODE_DIR/results

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# -------------------- INPUTS --------------------------
export EXTRACTED_CLAIMS_PATH=$CLAIMS_DIR/QwenQwen3-8B_veriscore_input.jsonl
export KNOWLEDGE_SOURCE_DIR=$BASE_DIR/data/OFS_KB_files/jsonl_clean_files

# -------------------- MODEL CONFIG --------------------
export AFV_MODEL=${AFV_MODEL:-google/gemma-7b-it}

# -------------------- LOGGING -------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_FILE=$LOGS_DIR/${JOB_NAME}_${TIMESTAMP}.out
ERR_FILE=$LOGS_DIR/${JOB_NAME}_${TIMESTAMP}.err

# -------------------- SRUN TEMPLATE -------------------
export SRUN="srun -K \
    --job-name=${JOB_NAME} \
    --partition=${ANY_PARTITION} \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=${SLURM_CPUS} \
    --gpus-per-task=${SLURM_GPUS} \
    --mem=${SLURM_MEM} \
    --container-image=${IMAGE} \
    --container-workdir=${CODE_DIR} \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,${CODE_DIR}:${CODE_DIR} \
    --export=ALL"

# -------------------- COMMAND -------------------------
CMD=(
  python -m scifactcheck.ofs_eval
    --extracted_claims_path "${EXTRACTED_CLAIMS_PATH}"
    --knowledge_source_dir "${KNOWLEDGE_SOURCE_DIR}"
    --afv_model "${AFV_MODEL}"
    --data_dir ".cache/factscore"
    --model_dir ".cache/factscore"
    --cache_dir ".cache/factscore"
    --output_path "${RESULTS_DIR}"
    --debug_logger
)

# ============================================================
# RUN
# ============================================================

echo "============================================================="
echo " OpenFactScore Evaluation Job"
echo "-------------------------------------------------------------"
echo "Job name:          ${JOB_NAME}"
echo "AFV model:         ${AFV_MODEL}"
echo "Claims input:      ${EXTRACTED_CLAIMS_PATH}"
echo "Knowledge sources: ${KNOWLEDGE_SOURCE_DIR}"
echo "Results dir:       ${RESULTS_DIR}"
echo "Image:             ${IMAGE}"
echo "Partitions:        ${ANY_PARTITION}"
echo "GPUs:              ${SLURM_GPUS}"
echo "Memory:            ${SLURM_MEM}"
echo "Start time:        $(date)"
echo "Logs:"
echo "  STDOUT → ${OUT_FILE}"
echo "  STDERR → ${ERR_FILE}"
echo "============================================================="

# -------------------- EXECUTION -----------------------
${SRUN} "${CMD[@]}" \
  2> >(tee "${ERR_FILE}" >&2) | tee "${OUT_FILE}"

# -------------------- CLEANUP -------------------------
kill $HEARTBEAT_PID >/dev/null 2>&1 || true

echo "============================================================="
echo " Job finished at $(date)"
echo " Results written to: ${RESULTS_DIR}"
echo " Logs:"
echo "   ${OUT_FILE}"
echo "   ${ERR_FILE}"
echo "============================================================="

