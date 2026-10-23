#!/usr/bin/env bash
set -euo pipefail

cd /home/jiang/code/muscleVLTA

# 每个包最多尝试 3 次。
MAX_TRIES="${MAX_TRIES:-3}"

# 每一次尝试的总时限，不是“无进度超时”。
# 下载特别慢时可以设置为 60m。
PACKAGE_TIMEOUT="${PACKAGE_TIMEOUT:-30m}"

export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

export UV_CONCURRENT_DOWNLOADS=1
export UV_HTTP_TIMEOUT=300

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1
export GIT_LFS_SKIP_SMUDGE=1

for cmd in uv awk sort timeout flock tee; do
    command -v "$cmd" >/dev/null || {
        echo "缺少命令：$cmd"
        exit 1
    }
done

[[ -f uv.lock ]] || {
    echo "错误：项目目录中没有 uv.lock"
    exit 1
}

[[ "$MAX_TRIES" =~ ^[1-9][0-9]*$ ]] || {
    echo "错误：MAX_TRIES 必须是正整数"
    exit 1
}

# 防止本脚本被同时启动两份。
exec 9>.uv-sync-serial.lock
flock -n 9 || {
    echo "已有一个串行安装脚本正在运行。"
    exit 1
}

trap 'echo; echo "已中断；下次可重新运行本脚本。"; exit 130' INT
trap 'exit 143' TERM

mkdir -p .uv-sync-logs

# 只读取 [[package]] 条目中的顶层包名。
# 版本、平台适用性、下载来源仍交给 uv --locked 判断。
mapfile -t packages < <(
    awk -F '"' '
        /^\[\[package\]\]$/ { in_package = 1; next }
        /^\[/ { in_package = 0 }
        in_package && /^name = "/ {
            print $2
            in_package = 0
        }
    ' uv.lock | sort -u
)

((${#packages[@]} > 0)) || {
    echo "错误：无法从 uv.lock 中读取包名。"
    exit 1
}

# 一个阶段成功才返回；失败始终重试同一个阶段。
run_stage() {
    local label="$1"
    shift
    local log=".uv-sync-logs/${label}.log"
    local attempt status

    for ((attempt=1; attempt<=MAX_TRIES; attempt++)); do
        echo
        echo "===== $label：第 $attempt/$MAX_TRIES 次尝试 ====="

        if timeout --kill-after=15s "$PACKAGE_TIMEOUT" \
            uv --no-progress "$@" 2>&1 | tee -a "$log"; then
            echo "===== $label：完成 ====="
            return 0
        else
            status=$?
        fi

        echo "本次失败，退出码：$status"
        echo "日志：$log"

        if ((attempt < MAX_TRIES)); then
            echo "等待 5 秒，仍重试当前阶段，不进入下一个包。"
            sleep 5
        fi
    done

    echo "连续失败，停止在：$label"
    echo "修复网络或切换节点后，重新运行本脚本。"
    return 1
}

for package in "${packages[@]}"; do
    # 保留原命令的要求：不安装 torch 本体。
    [[ "$package" == "torch" ]] && continue

    args=(
        sync
        --locked
        --python 3.11
        --inexact
        --no-install-workspace
        --no-install-package torch
    )

    # 当前阶段只允许处理当前包。
    for other in "${packages[@]}"; do
        if [[ "$other" != "$package" && "$other" != "torch" ]]; then
            args+=(--no-install-package "$other")
        fi
    done

    run_stage "$package" "${args[@]}"
done

# 依赖逐包处理完后，再安装项目/工作区并核对最终状态。
run_stage final-sync \
    sync --locked --python 3.11 --no-install-package torch

echo
echo "全部同步成功。"