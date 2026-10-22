#!/usr/bin/env bash
# Train the muscleVLTA AR3 force + visuo-tactile reactive policy (pi05 + LIFT reactive expert + multi-rate MoT
# force / tactile memory) on GPUs 6,7 of the H100 server.
#
# Quick start (from the repository root):
#   bash scripts/train_ar3_force_visuotactile_gpu67.sh
#
# Overrides (environment variables):
#   OPENPI_CONFIG_NAME     pi05_ar3_force_visuotactile_reactive (default) | pi05_ar3_force_reactive | pi05_ar3_base
#   OPENPI_EXP_NAME        experiment name (default: ar3_mot_<date>), checkpoints/<config>/<exp>/
#   OPENPI_AR3_REPO_ID     LeRobot v3.0 dataset dir (default: /data/jiang/utilData/ar3_force_computeChargingPullOut_force_aware)
#   OPENPI_AR3_WEIGHT_PATH pi05_base params (default: /data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params)
#   OPENPI_CUDA_VISIBLE_DEVICES  GPUs (default 6,7); fsdp devices = number of GPUs
#   OPENPI_BATCH_SIZE (global, default 32), OPENPI_NUM_TRAIN_STEPS (default 30000), OPENPI_NUM_WORKERS (default 8)
#   OPENPI_RESUME=1 resumes the experiment, OPENPI_OVERWRITE=1 overwrites it
#   OPENPI_RECOMPUTE_NORM_STATS=1 recomputes assets/<config>/<asset_id>/norm_stats.json
#   WANDB_MODE  defaults to offline; use online / disabled as needed
#   OPENPI_DRY_RUN=1 prints the commands without running them
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_force_visuotactile_reactive}"
EXP_NAME="${OPENPI_EXP_NAME:-ar3_mot_$(date +%Y%m%d_%H%M)}"
GPUS="${OPENPI_CUDA_VISIBLE_DEVICES:-6,7}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"
BATCH_SIZE="${OPENPI_BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-300000}"
NUM_WORKERS="${OPENPI_NUM_WORKERS:-8}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"

export OPENPI_AR3_REPO_ID="${OPENPI_AR3_REPO_ID:-/data/jiang/ar3_force_computeChargingPullOut_force_aware}"
export OPENPI_AR3_ASSET_ID="${OPENPI_AR3_ASSET_ID:-$(basename "${OPENPI_AR3_REPO_ID}")}"
export OPENPI_AR3_WEIGHT_PATH="${OPENPI_AR3_WEIGHT_PATH:-/data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params}"
# paligemma_tokenizer.model is read from ${OPENPI_DATA_HOME}/big_vision/.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
# Aligned + downsampled tactile depth images are cached here on first use.
export OPENPI_TACTILE_CACHE_DIR="${OPENPI_TACTILE_CACHE_DIR:-${OPENPI_DATA_HOME}/lerobot_v3_tactile}"
# Always import this repository's openpi (the venv may point at another checkout).
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export WANDB_MODE="${WANDB_MODE:-offline}"

run() {
    echo "+ $*"
    if [[ "${OPENPI_DRY_RUN:-0}" != "1" ]]; then
        "$@"
    fi
}

for path in "${OPENPI_AR3_REPO_ID}/meta/info.json" "${OPENPI_AR3_WEIGHT_PATH}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}" >&2
        exit 1
    fi
done

if command -v nvidia-smi >/dev/null; then
    echo "GPU memory in use before training (GPUs ${GPUS}):"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "${GPUS}" || true
fi

# 1) Normalization statistics (state / actions / 19D measured force), computed once per dataset on the CPU.
NORM_STATS="${REPO_ROOT}/assets/${CONFIG_NAME}/${OPENPI_AR3_ASSET_ID}/norm_stats.json"
if [[ ! -f "${NORM_STATS}" || "${OPENPI_RECOMPUTE_NORM_STATS:-0}" == "1" ]]; then
    run env JAX_PLATFORMS=cpu "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
else
    echo "Using existing ${NORM_STATS}"
fi

# 2) Training.
TRAIN_ARGS=(
    "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --fsdp-devices "${NUM_GPUS}"
    --batch-size "${BATCH_SIZE}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
)
if [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
elif [[ "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--overwrite)
fi
if [[ "${WANDB_MODE}" == "disabled" ]]; then
    TRAIN_ARGS+=(--no-wandb-enabled)
fi

mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/${CONFIG_NAME}_${EXP_NAME}.log"
echo "Logging to ${LOG_FILE}"
if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}"
else
    run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
