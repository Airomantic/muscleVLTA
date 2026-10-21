#!/usr/bin/env bash
# Serve a trained muscleVLTA AR3 force + visuo-tactile policy (pi05 + LIFT reactive expert + multi-rate MoT force /
# tactile memory) over WebSocket. The robot side is scripts/ar3_force_visuotactile_client_example.py.
#
# Usage (from any directory):
#   bash scripts/serve_ar3_force_visuotactile.sh                         # newest checkpoint of the newest run
#   bash scripts/serve_ar3_force_visuotactile.sh CHECKPOINT_STEP_DIR     # e.g. checkpoints/<config>/<exp>/30000
#   bash scripts/serve_ar3_force_visuotactile.sh [CHECKPOINT_STEP_DIR] [serve_policy.py arguments, e.g. --record]
#
# CHECKPOINT_STEP_DIR is a step directory written by training (it contains params/ and assets/). Without it, the
# highest step of the most recently modified run under checkpoints/<config>/ is served.
#
# The service is unauthenticated and binds to 127.0.0.1 by default. Reach it from the robot PC through an SSH tunnel
# (on the robot PC: ssh -N -L 8000:127.0.0.1:8000 <user>@<server>) or set OPENPI_HOST=0.0.0.0 on a trusted network.
#
# Overrides (environment variables):
#   OPENPI_CONFIG_NAME     training config of the checkpoint (default: pi05_ar3_force_visuotactile_reactive_lowmem)
#                          | pi05_ar3_force_visuotactile_reactive | pi05_ar3_force_reactive | pi05_ar3_base
#   OPENPI_EXP_NAME        run to serve when no checkpoint is given (default: newest run with a checkpoint)
#   OPENPI_CHECKPOINT_BASE_DIR  checkpoint root (default: <repo>/checkpoints)
#   OPENPI_SERVE_GPU       GPU index (default 6) or "cpu". GPU memory is allocated on demand (no preallocation).
#   OPENPI_HOST, OPENPI_PORT    bind address (default 127.0.0.1:8000)
#   OPENPI_PROMPT          task prompt used when the client sends none (default: the charging-plug task)
#   OPENPI_RECORD=1        save every request / response to policy_records/
#   OPENPI_AR3_ASSET_ID    norm stats id (default: detected from <checkpoint>/assets/)
#   OPENPI_PYTHON          python interpreter (default: <repo>/.venv/bin/python)
#   OPENPI_DRY_RUN=1       print the command without starting the server
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 0
fi

CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_force_visuotactile_reactive_lowmem}"
CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-${REPO_ROOT}/checkpoints}"
GPU="${OPENPI_SERVE_GPU:-6}"
HOST="${OPENPI_HOST:-127.0.0.1}"
PORT="${OPENPI_PORT:-8000}"
PROMPT="${OPENPI_PROMPT-Unplug the white charging plug from the white socket and place it on the table}"
PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"

# Prints the highest numeric step directory (with params/) of a run directory.
latest_step_dir() {
    local run_dir="${1%/}" best=-1 best_name="" dir name
    for dir in "${run_dir}"/*/; do
        name="$(basename "${dir}")"
        # Skip unfinished saves (<step>.orbax-checkpoint-tmp-*) and other files.
        [[ "${name}" =~ ^[0-9]+$ && -d "${dir}params" ]] || continue
        if (( 10#${name} > best )); then
            best=$((10#${name}))
            best_name="${name}"
        fi
    done
    [[ -n "${best_name}" ]] && printf '%s\n' "${run_dir}/${best_name}"
}

checkpoint=""
if [[ $# -gt 0 && "${1}" != -* ]]; then
    checkpoint="$1"
    shift
fi

if [[ -z "${checkpoint}" ]]; then
    config_dir="${CHECKPOINT_BASE_DIR}/${CONFIG_NAME}"
    if [[ -n "${OPENPI_EXP_NAME:-}" ]]; then
        checkpoint="$(latest_step_dir "${config_dir}/${OPENPI_EXP_NAME}" || true)"
    else
        while IFS= read -r run_dir; do
            checkpoint="$(latest_step_dir "${run_dir}" || true)"
            [[ -n "${checkpoint}" ]] && break
        done < <(ls -dt "${config_dir}"/*/ 2>/dev/null || true)
    fi
    if [[ -z "${checkpoint}" ]]; then
        echo "No checkpoint found under ${config_dir}${OPENPI_EXP_NAME:+/${OPENPI_EXP_NAME}}; pass CHECKPOINT_STEP_DIR." >&2
        exit 1
    fi
fi

if [[ ! -d "${checkpoint}/params" || ! -d "${checkpoint}/assets" ]]; then
    echo "${checkpoint} is not a checkpoint step directory (expected params/ and assets/ inside)." >&2
    exit 1
fi
checkpoint="$(cd "${checkpoint}" && pwd)"

# The norm stats are read from <checkpoint>/assets/<asset_id>/; the AR3 configs take the id from OPENPI_AR3_ASSET_ID.
if [[ -z "${OPENPI_AR3_ASSET_ID:-}" ]]; then
    stats=("${checkpoint}"/assets/*/norm_stats.json)
    if [[ ${#stats[@]} -eq 1 && -f "${stats[0]}" ]]; then
        OPENPI_AR3_ASSET_ID="$(basename "$(dirname "${stats[0]}")")"
    fi
fi

if [[ ! -f "${DATA_HOME}/big_vision/paligemma_tokenizer.model" ]]; then
    echo "Missing ${DATA_HOME}/big_vision/paligemma_tokenizer.model (set OPENPI_DATA_HOME)." >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]] && ! command -v "${PYTHON}" >/dev/null; then
    echo "Python interpreter ${PYTHON} not found (set OPENPI_PYTHON)." >&2
    exit 1
fi

environment=(
    "OPENPI_DATA_HOME=${DATA_HOME}"
    # Always import this repository's openpi (the venv may point at another checkout).
    "PYTHONPATH=${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
)
if [[ -n "${OPENPI_AR3_ASSET_ID:-}" ]]; then
    environment+=("OPENPI_AR3_ASSET_ID=${OPENPI_AR3_ASSET_ID}")
fi
if [[ "${GPU}" == "cpu" ]]; then
    environment+=("JAX_PLATFORMS=cpu" "CUDA_VISIBLE_DEVICES=")
else
    # Shared GPUs: allocate on demand instead of reserving 75% of the GPU memory.
    environment+=(
        "CUDA_VISIBLE_DEVICES=${GPU}"
        "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    )
fi

command=(
    env "${environment[@]}"
    "${PYTHON}" "${REPO_ROOT}/scripts/serve_policy.py"
    --config-name "${CONFIG_NAME}"
    --checkpoint-dir "${checkpoint}"
    --host "${HOST}"
    --port "${PORT}"
)
if [[ -n "${PROMPT}" ]]; then
    command+=(--default-prompt "${PROMPT}")
fi
if [[ "${OPENPI_RECORD:-0}" == "1" ]]; then
    command+=(--record)
fi
command+=("$@")

if [[ "${OPENPI_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

echo "Serving ${checkpoint} (${CONFIG_NAME}) on ${HOST}:${PORT}, device: ${GPU}" >&2
cd "${REPO_ROOT}"
exec "${command[@]}"
