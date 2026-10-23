#!/usr/bin/env bash
# Train the muscleVLTA AR3 force + visuo-tactile reactive policy (pi05 + LIFT reactive expert + multi-rate MoT
# force / tactile memory) on GPUs 6,7 of the shared H100 server, using only a small, capped amount of free memory.
#
# Quick start (from the repository root):
#   bash scripts/train_ar3_force_visuotactile_gpu67_lowmem.sh
#
# Normal training on dedicated GPUs (full fine-tuning): scripts/train_ar3_force_visuotactile_gpu67.sh
#
# Sharing the GPUs with other users:
#   * Waits until every selected GPU has at least OPENPI_MIN_FREE_GB free.
#   * No preallocation; JAX is hard-capped at OPENPI_GPU_MEM_GB per GPU (default: currently free memory minus
#     OPENPI_GPU_MEM_RESERVE_GB, so other users keep headroom). NCCL NVLS buffers are disabled, NCCL buffers are small.
#   * If the job dies (e.g. OOM because another user's usage grew), it waits for memory again and resumes from the
#     last checkpoint (saved every 500 steps by the lowmem config), up to OPENPI_MAX_RETRIES times.
#
# Overrides (environment variables):
#   OPENPI_CONFIG_NAME     pi05_ar3_force_visuotactile_reactive_lowmem (default: LoRA, ~8 GB/GPU)
#                          | pi05_ar3_force_visuotactile_reactive (full fine-tuning, needs dedicated GPUs)
#                          | pi05_ar3_force_reactive | pi05_ar3_base
#   OPENPI_EXP_NAME        experiment name (default: ar3_mot_<date>), checkpoints/<config>/<exp>/
#   OPENPI_AR3_REPO_ID     LeRobot v3.0 dataset dir (default: /data/jiang/utilData/ar3_force_computeChargingPullOut_force_aware)
#   OPENPI_AR3_WEIGHT_PATH pi05_base params (default: /data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params)
#   OPENPI_CUDA_VISIBLE_DEVICES  GPUs (default 6,7); fsdp devices = number of GPUs
#   OPENPI_MIN_FREE_GB (default 8), OPENPI_GPU_MEM_GB (default: auto), OPENPI_GPU_MEM_RESERVE_GB (default 2)
#   OPENPI_WAIT_INTERVAL seconds between memory checks (default 60), OPENPI_MAX_RETRIES (default 50)
#   OPENPI_BATCH_SIZE (global, default from config), OPENPI_NUM_TRAIN_STEPS (default 30000), OPENPI_NUM_WORKERS
#   OPENPI_RESUME=1 resumes the experiment, OPENPI_OVERWRITE=1 overwrites it
#   OPENPI_RECOMPUTE_NORM_STATS=1 recomputes assets/<config>/<asset_id>/norm_stats.json
#   WANDB_MODE  defaults to offline; use online / disabled as needed
#   OPENPI_DRY_RUN=1 prints the commands without running them
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_force_visuotactile_reactive_lowmem}"
EXP_NAME="${OPENPI_EXP_NAME:-ar3_mot_$(date +%Y%m%d_%H%M)}"
GPUS="${OPENPI_CUDA_VISIBLE_DEVICES:-6,7}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"
NUM_TRAIN_STEPS="${OPENPI_NUM_TRAIN_STEPS:-30000}"
MIN_FREE_GB="${OPENPI_MIN_FREE_GB:-8}"
RESERVE_GB="${OPENPI_GPU_MEM_RESERVE_GB:-2}"
WAIT_INTERVAL="${OPENPI_WAIT_INTERVAL:-60}"
MAX_RETRIES="${OPENPI_MAX_RETRIES:-50}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DRY_RUN="${OPENPI_DRY_RUN:-0}"

export OPENPI_AR3_REPO_ID="${OPENPI_AR3_REPO_ID:-/data/jiang/utilData/ar3_force_computeChargingPullOut_force_aware}"
export OPENPI_AR3_ASSET_ID="${OPENPI_AR3_ASSET_ID:-$(basename "${OPENPI_AR3_REPO_ID}")}"
export OPENPI_AR3_WEIGHT_PATH="${OPENPI_AR3_WEIGHT_PATH:-/data/jiang/checkpoints/openpi-assets/checkpoints/pi05_base/params}"
# paligemma_tokenizer.model is read from ${OPENPI_DATA_HOME}/big_vision/.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
# Aligned + downsampled tactile depth images are cached here on first use.
export OPENPI_TACTILE_CACHE_DIR="${OPENPI_TACTILE_CACHE_DIR:-${OPENPI_DATA_HOME}/lerobot_v3_tactile}"
# Always import this repository's openpi (the venv may point at another checkout).
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Shared-GPU friendly memory behaviour: allocate on demand, never beyond the cap set below.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
# CUDA graphs (command buffers) allocate driver memory outside the JAX cap; disable them on shared GPUs.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

run() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

for path in "${OPENPI_AR3_REPO_ID}/meta/info.json" "${OPENPI_AR3_WEIGHT_PATH}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing ${path}" >&2
        exit 1
    fi
done

# Prints "<min free MiB> <total MiB>" over the selected GPUs.
gpu_free() {
    nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i "${GPUS}" |
        awk -F', *' 'NR == 1 || $1 < min {min = $1} {total = $2} END {print min, total}'
}

# Waits until every selected GPU has MIN_FREE_GB free, then caps JAX at (free - reserve) or OPENPI_GPU_MEM_GB.
wait_and_cap_memory() {
    local free total cap_mib
    while true; do
        read -r free total < <(gpu_free)
        if (( free >= MIN_FREE_GB * 1024 )) || [[ "${DRY_RUN}" == "1" ]]; then
            break
        fi
        echo "$(date '+%F %T') GPUs ${GPUS}: min free ${free} MiB < ${MIN_FREE_GB} GiB, waiting ${WAIT_INTERVAL}s..."
        sleep "${WAIT_INTERVAL}"
    done
    if [[ -n "${OPENPI_GPU_MEM_GB:-}" ]]; then
        cap_mib=$(( OPENPI_GPU_MEM_GB * 1024 ))
    else
        cap_mib=$(( free - RESERVE_GB * 1024 ))
    fi
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$(awk -v c="${cap_mib}" -v t="${total}" 'BEGIN {printf "%.4f", (c > 0 ? c : 0) / t}')"
    echo "$(date '+%F %T') GPUs ${GPUS}: min free ${free} MiB, JAX memory cap ${cap_mib} MiB per GPU" \
        "(XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION})"
}

# 1) Normalization statistics (state / actions / 19D measured force), computed once per dataset on the CPU.
NORM_STATS="${REPO_ROOT}/assets/${CONFIG_NAME}/${OPENPI_AR3_ASSET_ID}/norm_stats.json"
if [[ ! -f "${NORM_STATS}" || "${OPENPI_RECOMPUTE_NORM_STATS:-0}" == "1" ]]; then
    # All AR3 configs share the same statistics; reuse them if another AR3 config already computed them.
    EXISTING="$(ls "${REPO_ROOT}"/assets/pi05_ar3_*/"${OPENPI_AR3_ASSET_ID}"/norm_stats.json 2>/dev/null | head -n 1 || true)"
    if [[ -n "${EXISTING}" && "${OPENPI_RECOMPUTE_NORM_STATS:-0}" != "1" ]]; then
        run mkdir -p "$(dirname "${NORM_STATS}")"
        run cp "${EXISTING}" "${NORM_STATS}"
    else
        run env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
    fi
else
    echo "Using existing ${NORM_STATS}"
fi

# 2) Training (with wait-for-memory and automatic resume).
TRAIN_ARGS=(
    "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --fsdp-devices "${NUM_GPUS}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
)
[[ -n "${OPENPI_BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--batch-size "${OPENPI_BATCH_SIZE}")
[[ -n "${OPENPI_NUM_WORKERS:-}" ]] && TRAIN_ARGS+=(--num-workers "${OPENPI_NUM_WORKERS}")
[[ "${WANDB_MODE}" == "disabled" ]] && TRAIN_ARGS+=(--no-wandb-enabled)

FIRST_FLAG=()
if [[ "${OPENPI_OVERWRITE:-0}" == "1" ]]; then
    FIRST_FLAG=(--overwrite)
elif [[ "${OPENPI_RESUME:-0}" == "1" ]]; then
    FIRST_FLAG=(--resume)
fi

mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/${CONFIG_NAME}_${EXP_NAME}.log"
echo "Logging to ${LOG_FILE}"

attempt=0
while true; do
    wait_and_cap_memory
    if (( attempt == 0 )); then
        flags=("${FIRST_FLAG[@]}")
    else
        # Resuming is safe even if no checkpoint was written yet (training then starts from scratch).
        flags=(--resume)
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" "${flags[@]}"
        exit 0
    fi
    # Own process group: after a crash, leftover data loader workers are killed so they cannot hold memory or block.
    echo "+ ${PYTHON} scripts/train.py ${TRAIN_ARGS[*]} ${flags[*]}" | tee -a "${LOG_FILE}"
    setsid "${PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" "${flags[@]}" >>"${LOG_FILE}" 2>&1 &
    train_pid=$!
    trap 'kill -- -"${train_pid}" 2>/dev/null; exit 130' INT TERM
    tail -n 0 -f --pid="${train_pid}" "${LOG_FILE}" &
    set +e
    wait "${train_pid}"
    status=$?
    set -e
    trap - INT TERM
    kill -- -"${train_pid}" 2>/dev/null || true
    if (( status == 0 )); then
        echo "Training finished."
        exit 0
    fi
    attempt=$(( attempt + 1 ))
    if (( attempt > MAX_RETRIES )); then
        echo "Training failed ${attempt} times (last exit code ${status}), giving up. See ${LOG_FILE}" >&2
        exit "${status}"
    fi
    echo "$(date '+%F %T') Training exited with ${status} (attempt ${attempt}/${MAX_RETRIES}); resuming after ${WAIT_INTERVAL}s."
    sleep "${WAIT_INTERVAL}"
done
