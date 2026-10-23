#!/usr/bin/env bash
# Prepare inference services; never start teleoperation, recording or policy action execution.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/prepare_ar3_inference.sh"
UPPERLIMB="$(dirname -- "$ROOT")"
AR3_WS="${INFER_AR3_WS:-$UPPERLIMB/ar3_ws}"
CAPTURE_DIR="${INFER_CAPTURE_DIR:-$UPPERLIMB/lerobot_89236e_muscleTWAM}"
GRIPPER_DIR="${INFER_GRIPPER_DIR:-$UPPERLIMB/V3_Gripper}"
TILT_DIR="${INFER_TILT_DIR:-$UPPERLIMB/../../huanEr}"
SERVO_PYTHON="${INFER_SERVO_PYTHON:-$HOME/anaconda3/envs/qwen3VL/bin/python}"
ROS_SETUP="${INFER_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
CAMERA_LAUNCH_DIR="${INFER_CAMERA_LAUNCH_DIR:-/opt/ros/jazzy/share/orbbec_camera/launch}"
TACTILE_CONFIG="${INFER_TACTILE_CONFIG:-$CAPTURE_DIR/examples/rokae_ar3/tactile_bridge.json}"
ARM_INTERFACE="${INFER_ARM_INTERFACE:-enp132s0}"
ARM_CIDR="${INFER_ARM_CIDR:-192.168.0.100/24}"
ARM_IP="${INFER_ARM_IP:-192.168.0.160}"
ARM_LOCAL_IP="${ARM_CIDR%/*}"
GRIPPER_HOST="${INFER_GRIPPER_HOST:-192.168.127.10}"
GRIPPER_USER="${INFER_GRIPPER_USER:-cftc}"
FRONT_SERIAL="${INFER_FRONT_SERIAL:-CPC8563000HX}"
WRIST_SERIAL="${INFER_WRIST_SERIAL:-CV2856D000BV}"
TILT_ANGLE="${INFER_TILT_ANGLE:-40.0}"
WAIT_TIMEOUT="${INFER_TIMEOUT:-180}"
CHECKPOINT="${INFER_CHECKPOINT:-$ROOT/infer_checkpoint/29999}"
export ROS_DOMAIN_ID="${INFER_ROS_DOMAIN_ID:-111}"
export OPENPI_SERVE_GPU="${OPENPI_SERVE_GPU:-0}"
export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_ar3_force_visuotactile_reactive_lowmem}"
export OPENPI_HOST="${OPENPI_HOST:-127.0.0.1}"
export OPENPI_PORT="${OPENPI_PORT:-8000}"

WITH_ARM=1 WITH_SSH=1 WITH_TACTILE=1 WITH_CAMERAS=1 WITH_TILT=1 WITH_POLICY=0
WINDOWS=0 MODE=start TASK=""
ORIGINAL_ARGS=("$@")

info() { printf '[infer_oneclick] %s\n' "$*"; }
die() { printf '[infer_oneclick] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./infer_oneclick.sh [OPTIONS]
  --dry-run          Print commands/dependencies; no device or network access.
  --check            Check local files/interpreters only; no device/network access.
  --checkpoint DIR   Checkpoint step directory (default: infer_checkpoint/29999).
  --no-ssh           Reuse remote services; still require both tactile ports and gRPC.
  --no-arm           Do not launch or check the arm.
  --no-tactile       Do not launch or check the local tactile ROS bridge.
  --no-cameras       Do not open/check RGB cameras (e.g. an SDK reader owns them).
  --no-tilt          Do not start or move the camera tilt servo.
  --with-policy      Also launch the WebSocket model server (not for local F5 debug).
  --no-policy        Do not launch/check the model server (default).
  --windows          Separate GNOME windows instead of tabs in one new window.
  -h, --help         Show this help.

Default: AR3 MoveIt, two SSH services, tactile bridge, front/wrist Orbbec RGB,
40-degree tilt, and a separate readiness tab, ROS domain 111. Start local model
inference with VS Code's infer_muscleVLTA_debug after preparation. --with-policy
is the alternative WebSocket workflow; do not load a second model with F5.
Arm bringup POWERS THE ARM; the tilt servo MOVES. No arm trajectory, gripper
initialization, teleoperation, recording, or model action-execution client is run.
Stop existing teleoperation/recording before starting. One owner per camera.

Paths: INFER_AR3_WS, INFER_CAPTURE_DIR, INFER_GRIPPER_DIR, INFER_TILT_DIR,
INFER_SERVO_PYTHON, INFER_ROS_SETUP, INFER_CAMERA_LAUNCH_DIR, INFER_TACTILE_CONFIG.
Devices: INFER_ARM_INTERFACE, INFER_ARM_CIDR, INFER_ARM_IP, INFER_GRIPPER_HOST,
INFER_GRIPPER_USER, INFER_FRONT_SERIAL, INFER_WRIST_SERIAL, INFER_TILT_ANGLE.
Runtime: INFER_ROS_DOMAIN_ID, INFER_TIMEOUT, INFER_CHECKPOINT, OPENPI_SERVE_GPU,
OPENPI_CONFIG_NAME, OPENPI_PYTHON, OPENPI_DATA_HOME, OPENPI_HOST, OPENPI_PORT.

SSH uses keys/agent or interactive authentication and VERIFIED known_hosts entries.
If a host is new, verify its fingerprint with a separate ssh login first.
Ctrl+C stops a tab's foreground process; press Enter after exit to close the tab.
Remote deploy.sh may leave services running. Closing tabs is NOT an emergency
stop and does not guarantee arm power-off or a physical stop.

This prepares services only. Readiness requires fresh, connected, initialized
gripper feedback, not just a responsive gRPC server. Read-only debug never
connects/initializes the motor or commands robot motion. Prepare its connection
through your approved driver; no teleop state owner is started here.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run|--check)
            [[ "$MODE" == start ]] || die "Choose only one of --check and --dry-run."
            MODE="${1#--}" ;;
        --checkpoint)
            [[ $# -ge 2 && -n "$2" ]] || die "--checkpoint requires a step directory."
            CHECKPOINT="$2"; shift ;;
        --no-ssh) WITH_SSH=0 ;;
        --no-arm) WITH_ARM=0 ;;
        --no-tactile) WITH_TACTILE=0 ;;
        --no-cameras) WITH_CAMERAS=0 ;;
        --no-tilt) WITH_TILT=0 ;;
        --with-policy) WITH_POLICY=1 ;;
        --no-policy) WITH_POLICY=0 ;;
        --windows) WINDOWS=1 ;;
        --task)
            [[ $# -ge 2 ]] || die "--task requires a name."
            TASK="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)." ;;
    esac
    shift
done
[[ "$CHECKPOINT" == /* ]] || CHECKPOINT="$PWD/$CHECKPOINT"
[[ "$WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "INFER_TIMEOUT must be a positive integer."
[[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ && ${#ROS_DOMAIN_ID} -le 3 ]] \
    && ((10#$ROS_DOMAIN_ID <= 232)) || die "INFER_ROS_DOMAIN_ID must be 0..232."
[[ "$OPENPI_PORT" =~ ^[0-9]+$ && ${#OPENPI_PORT} -le 5 ]] \
    && ((10#$OPENPI_PORT >= 1 && 10#$OPENPI_PORT <= 65535)) || die "OPENPI_PORT must be 1..65535."
[[ "$TILT_ANGLE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v angle="$TILT_ANGLE" 'BEGIN {exit !(angle >= 0 && angle <= 60)}' \
    || die "INFER_TILT_ANGLE must be 0..60 degrees."

# ROS Jazzy uses system Python 3.12, never the model's Python 3.11 or inherited overlays.
ROS_CMD=(env -i "HOME=$HOME" "PATH=/usr/bin:/bin:/usr/local/bin"
    "LANG=${LANG:-C.UTF-8}" "TERM=${TERM:-xterm-256color}" "DISPLAY=${DISPLAY:-}"
    "XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}" "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    bash --noprofile --norc -c 'set -e; source "$1"; shift; exec "$@"' _ "$ROS_SETUP")
TASKS=()
((WITH_ARM)) && TASKS+=(arm)
((WITH_SSH)) && TASKS+=(sensor gripper)
((WITH_TACTILE)) && TASKS+=(tactile)
((WITH_CAMERAS)) && TASKS+=(front wrist)
((WITH_TILT)) && TASKS+=(tilt)
((WITH_POLICY)) && TASKS+=(policy)
TASKS+=(ready)

quote_remote() {
    local arg
    for arg in "$@"; do printf "'%s' " "${arg//\'/\'\\\'\'}"; done
}

command_for() {
    TASK_CWD="$ROOT"
    case "$1" in
        arm)
            TASK_CWD="$AR3_WS"
            COMMAND=(bash "$AR3_WS/jazzy.sh" launch rokae_hardware rokae_moveit_launch.py
                robot_type:=AR3R use_fake_hardware:=false "robot_ip:=$ARM_IP"
                "local_ip:=$ARM_LOCAL_IP" enable_movej:=false) ;;
        sensor|gripper)
            TASK_CWD="$GRIPPER_DIR"
            local remote='cd "$HOME/dmrobotics_Flux_pkg_V1.2.13.1/dmrobotics_Flux_pkg/sensor_server" && exec python -u grpc_server.py'
            [[ "$1" == sensor ]] || remote='cd "$HOME/gripper_sdk" && exec bash deploy.sh'
            COMMAND=(ssh -tt -o StrictHostKeyChecking=yes -o ConnectTimeout=10
                -o ServerAliveInterval=15 -o ServerAliveCountMax=3
                "$GRIPPER_USER@$GRIPPER_HOST" "$(quote_remote bash -lic "$remote")") ;;
        tactile)
            TASK_CWD="$CAPTURE_DIR"
            COMMAND=("${ROS_CMD[@]}" bash --noprofile --norc -c
                'export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"; shift; exec "$@"'
                _ "$CAPTURE_DIR/src" "$CAPTURE_DIR/.venv-ar3/bin/python"
                "$CAPTURE_DIR/scripts/ar3_tactile_bridge.py" --config "$TACTILE_CONFIG") ;;
        front|wrist)
            local launch=gemini_330_series.launch.py serial="$FRONT_SERIAL" format=MJPG
            # Gemini 305 exposes YUYV, not the front camera's MJPG profile.
            [[ "$1" == front ]] || { launch=gemini305.launch.py; serial="$WRIST_SERIAL"; format=YUYV; }
            COMMAND=("${ROS_CMD[@]}" ros2 launch "$CAMERA_LAUNCH_DIR/$launch"
                "camera_name:=$1" "serial_number:=$serial" device_num:=2 enable_color:=true
                color_width:=640 color_height:=480 color_fps:=30 "color_format:=$format"
                enable_depth:=false enable_point_cloud:=false enable_colored_point_cloud:=false
                enable_frame_sync:=false)
            [[ "$1" == front ]] || COMMAND+=(uvc_backend:=v4l2 enable_left_color:=false enable_right_color:=false) ;;
        tilt)
            TASK_CWD="$TILT_DIR"
            COMMAND=("${ROS_CMD[@]}" /usr/bin/python3 "$TILT_DIR/camera_tilt_ros2.py" --ros-args
                -p "startup_angle_deg:=$TILT_ANGLE" -p "servo_python:=$SERVO_PYTHON") ;;
        policy)
            COMMAND=(env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u AMENT_PREFIX_PATH
                -u COLCON_PREFIX_PATH -u CMAKE_PREFIX_PATH -u ROS_PACKAGE_PATH
                bash "$ROOT/scripts/serve_ar3_force_visuotactile.sh" "$CHECKPOINT") ;;
        ready) COMMAND=() ;;
        *) die "Unknown task: $1" ;;
    esac
}

grpc_ready() {
    # Reuse only the collection project's read-only GetAllState probe, not its teleop entrypoint.
    env -u PYTHONHOME -u LD_LIBRARY_PATH "PYTHONPATH=$CAPTURE_DIR/scripts" \
        "$GRIPPER_DIR/.venv-ros/bin/python" -c \
        'import sys; from ar3_gripper_connection import wait_for_gripper_server; wait_for_gripper_server(sys.argv[1], float(sys.argv[2]))' \
        "$GRIPPER_HOST:50008" "$WAIT_TIMEOUT"
}

gripper_feedback_ready() {
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
        "$GRIPPER_DIR/.venv-ros/bin/python" "$ROOT/scripts/ar3_ros_observation_worker.py" \
        --check-gripper --gripper-address "$GRIPPER_HOST:50008" --timeout "$WAIT_TIMEOUT" \
        || die "Gripper feedback is not ready. Prepare the motor through your approved driver; no motor commands were sent."
}

preflight() {
    local cmd path
    for cmd in bash env timeout awk ssh flock mktemp jq; do
        command -v "$cmd" >/dev/null || die "Missing command: $cmd"
    done
    local files=("$ROS_SETUP" "$CAPTURE_DIR/scripts/ar3_gripper_connection.py"
        "$ROOT/scripts/ar3_ros_observation_worker.py")
    local executables=("$GRIPPER_DIR/.venv-ros/bin/python")
    [[ -d "$GRIPPER_DIR" ]] || die "Missing gripper project: $GRIPPER_DIR"
    if ((WITH_ARM)); then
        for cmd in ip ping sudo; do command -v "$cmd" >/dev/null || die "Missing command: $cmd"; done
        files+=("$AR3_WS/install/setup.bash" "$AR3_WS/jazzy.sh")
        [[ -d "/sys/class/net/$ARM_INTERFACE" ]] || die "Missing arm interface: $ARM_INTERFACE"
    fi
    if ((WITH_TACTILE)); then
        command -v ip >/dev/null || die "Missing command: ip"
        files+=("$TACTILE_CONFIG"
            "$CAPTURE_DIR/scripts/ar3_tactile_bridge.py" "$CAPTURE_DIR/scripts/ar3_tactile_worker.py"
            "$CAPTURE_DIR/scripts/ar3_tactile_io.py")
        executables+=("$CAPTURE_DIR/.venv-ar3/bin/python")
    fi
    if ((WITH_CAMERAS)); then
        files+=("$CAMERA_LAUNCH_DIR/gemini_330_series.launch.py" "$CAMERA_LAUNCH_DIR/gemini305.launch.py")
        [[ -n "$FRONT_SERIAL" && -n "$WRIST_SERIAL" && "$FRONT_SERIAL" != "$WRIST_SERIAL" ]] \
            || die "Set distinct nonempty INFER_FRONT_SERIAL and INFER_WRIST_SERIAL."
    fi
    if ((WITH_TILT)); then
        files+=("$TILT_DIR/camera_tilt_ros2.py" "$TILT_DIR/usb_servo.py")
        executables+=("$SERVO_PYTHON")
    fi
    if ((WITH_POLICY)); then
        command -v curl >/dev/null || die "Missing command: curl"
        files+=("${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/big_vision/paligemma_tokenizer.model")
        executables+=("${OPENPI_PYTHON:-$ROOT/.venv/bin/python}")
        [[ -d "$CHECKPOINT/params" && -d "$CHECKPOINT/assets" ]] \
            || die "Checkpoint needs params/ and assets/: $CHECKPOINT"
        local stats=("$CHECKPOINT"/assets/*/norm_stats.json)
        if [[ -n "${OPENPI_AR3_ASSET_ID:-}" ]]; then
            files+=("$CHECKPOINT/assets/$OPENPI_AR3_ASSET_ID/norm_stats.json")
        else
            [[ ${#stats[@]} -eq 1 && -f "${stats[0]}" ]] \
                || die "Expected one normalization asset; set OPENPI_AR3_ASSET_ID."
        fi
    fi
    for path in "${files[@]}"; do [[ -f "$path" ]] || die "Missing file: $path"; done
    for path in "${executables[@]}"; do [[ -x "$path" ]] || die "Missing interpreter: $path"; done
    if ((WITH_TACTILE)); then
        jq -e --arg host "$GRIPPER_HOST" '
            (.sensors | length == 2) and
            any(.sensors[]; .name == "sensor_0" and .remote_addr == ($host + ":50051") and .enable_force == true) and
            any(.sensors[]; .name == "sensor_2" and .remote_addr == ($host + ":50052") and .enable_force == true)
        ' "$TACTILE_CONFIG" >/dev/null || die "Tactile config must match both force-enabled sensors at $GRIPPER_HOST."
        local sdk_python sdk_dir
        sdk_python=$(jq -r --arg fallback "$(dirname -- "$CAPTURE_DIR")/V3_Gripper/.venv/bin/python" \
            '.sdk_python // $fallback' "$TACTILE_CONFIG")
        sdk_dir=$(jq -r --arg fallback "$(dirname -- "$CAPTURE_DIR")/V3_Gripper/SDK_Publish_V1.2.13.1" \
            '.sdk_dir // $fallback' "$TACTILE_CONFIG")
        [[ -x "$sdk_python" && -d "$sdk_dir" ]] \
            || die "Missing tactile SDK: $sdk_python / $sdk_dir; set sdk_python/sdk_dir in INFER_TACTILE_CONFIG."
    fi
    info "Local prerequisites exist. No hardware, network or ROS commands were run."
}

print_plan() {
    info "Domain=$ROS_DOMAIN_ID; checkpoint=$CHECKPOINT; GPU=$OPENPI_SERVE_GPU"
    info "Arm bringup powers the arm; tilt moves to $TILT_ANGLE degrees. NO model actions are executed."
    local task
    for task in "${TASKS[@]}"; do
        command_for "$task"
        printf '\n[%s] cwd=%s\n' "$task" "$TASK_CWD"
        if [[ "$task" == ready ]]; then
            printf '  Require GetAllState on %s:50008 AND fresh, connected, initialized motor feedback (no Connect/Initialize/Move RPC).\n' "$GRIPPER_HOST"
            ((WITH_ARM)) && printf '  Wait for /joint_states and /position_joint_trajectory_controller/controller_state.\n'
            ((WITH_TACTILE)) && printf '  Wait for sensor_0/sensor_2 depth and wrench samples.\n'
            ((WITH_CAMERAS)) && printf '  Wait for /front/color/image_raw and /wrist/color/image_raw samples.\n'
            ((WITH_TILT)) && printf '  Wait for measured /camera_tilt/state_deg within %s +/- 2 degrees.\n' "$TILT_ANGLE"
            ((WITH_POLICY)) && printf '  Wait for http://%s:%s/healthz.\n' "$OPENPI_HOST" "$OPENPI_PORT"
        else
            printf '  '; printf '%q ' "${COMMAND[@]}"; printf '\n'
        fi
        case "$task" in
            arm) printf '  Add %s to %s only if absent; verify route and ping before bringup.\n' "$ARM_CIDR" "$ARM_INTERFACE" ;;
            sensor) printf '  Reuse only if both 50051 and 50052 listen; partial startup is an error.\n' ;;
            gripper|tactile) printf '  First wait for both tactile ports 50051/50052.\n' ;;
        esac
    done
    ((WITH_SSH)) || info "Remote startup skipped; both tactile endpoints and gripper gRPC must already be available."
    info "Readiness is service preparation only. F5 debug reads observations and predicts; it never executes actions."
}

tcp_ready() {
    timeout 2 bash --noprofile --norc -c 'exec 3<>/dev/tcp/"$1"/"$2"' _ "$1" "$2" >/dev/null 2>&1
}

wait_tcp() {
    local port="$1" deadline=$((SECONDS + WAIT_TIMEOUT))
    until tcp_ready "$GRIPPER_HOST" "$port"; do
        ((SECONDS < deadline)) || die "Timed out waiting for $GRIPPER_HOST:$port."
        sleep 2
    done
}

wait_topic() {
    local topic="$1" type="$2"; shift 2
    info "Waiting for data on $topic"
    timeout --signal=TERM --kill-after=3 "$WAIT_TIMEOUT" "${ROS_CMD[@]}" ros2 topic echo \
        "$topic" "$type" --once --no-daemon --qos-reliability best_effort "$@" \
        || die "No qualifying sample on $topic; inspect its task tab and ROS domain."
}

check_conflicts() {
    local nodes
    nodes=$(timeout --kill-after=3 15 "${ROS_CMD[@]}" ros2 node list --no-daemon) \
        || die "Cannot inspect ROS nodes; no tasks started."
    [[ "$nodes" != *ar3r_cartesian_retargeting* && "$nodes" != *gripper_teleop* ]] \
        || die "Teleoperation is running. Stop it before inference preparation."
    if ((WITH_ARM)) && [[ "$nodes" == *controller_manager* ]]; then
        die "A controller manager already exists; inspect it and use --no-arm to preserve it."
    fi
    if ((WITH_CAMERAS)) && [[ "$nodes" == *"/front/"* || "$nodes" == *"/wrist/"* ]]; then
        die "Camera nodes already exist; use --no-cameras to preserve their ownership."
    fi
    if ((WITH_TACTILE)) && [[ "$nodes" == *ar3_tactile_bridge* ]]; then
        die "The tactile bridge already exists; use --no-tactile."
    fi
    if ((WITH_TILT)) && [[ "$nodes" == *camera_tilt_servo* ]]; then
        die "The tilt node already exists; use --no-tilt (no servo command will be sent)."
    fi
    if ((WITH_POLICY)) && tcp_ready "$OPENPI_HOST" "$OPENPI_PORT"; then
        die "Policy port is occupied; use --no-policy or choose another OPENPI_PORT."
    fi
}

run_task() {
    local addresses route pc_host deadline
    case "$TASK" in
        arm)
            addresses=$(ip -o -4 addr show dev "$ARM_INTERFACE")
            if ! awk -v cidr="$ARM_CIDR" '$4 == cidr {found=1} END {exit !found}' <<<"$addresses"; then
                sudo ip addr add "$ARM_CIDR" dev "$ARM_INTERFACE"
            fi
            route=$(ip route get "$ARM_IP")
            [[ " $route " == *" dev $ARM_INTERFACE "* && " $route " == *" src $ARM_LOCAL_IP "* ]] \
                || die "Unexpected arm route: $route"
            ping -I "$ARM_LOCAL_IP" -c 2 -W 2 "$ARM_IP" ;;
        sensor)
            if tcp_ready "$GRIPPER_HOST" 50051 || tcp_ready "$GRIPPER_HOST" 50052; then
                tcp_ready "$GRIPPER_HOST" 50051 && tcp_ready "$GRIPPER_HOST" 50052 \
                    || die "Only one tactile port listens; inspect the remote service instead of starting a duplicate."
                info "Reusing remote tactile listeners; the readiness tab still requires sensor samples."
                return
            fi ;;
        gripper|tactile)
            wait_tcp 50051; wait_tcp 50052
            if [[ "$TASK" == gripper ]] && tcp_ready "$GRIPPER_HOST" 50008; then
                grpc_ready
                info "Reusing responsive gripper service."
                return
            fi
            if [[ "$TASK" == tactile ]]; then
                addresses=$(ip -o -4 addr show)
                while IFS= read -r pc_host; do
                    awk -v host="$pc_host" '{split($4,a,"/"); if(a[1]==host) found=1} END {exit !found}' <<<"$addresses" \
                        || die "Tactile return address $pc_host is not assigned locally; configure the device NIC first."
                done < <(jq -r '.sensors[].pc_host' "$TACTILE_CONFIG")
            fi ;;
        ready)
            wait_tcp 50051; wait_tcp 50052; grpc_ready
            if ((WITH_ARM)); then
                wait_topic /joint_states sensor_msgs/msg/JointState --field header
                wait_topic /position_joint_trajectory_controller/controller_state control_msgs/msg/JointTrajectoryControllerState --field header
            fi
            if ((WITH_TACTILE)); then
                local sensor
                for sensor in sensor_0 sensor_2; do
                    wait_topic "/v3_gripper/tactile/$sensor/depth" sensor_msgs/msg/Image --field header
                    wait_topic "/v3_gripper/tactile/$sensor/wrench" geometry_msgs/msg/WrenchStamped --field header
                done
            fi
            if ((WITH_CAMERAS)); then
                wait_topic /front/color/image_raw sensor_msgs/msg/Image --field header
                wait_topic /wrist/color/image_raw sensor_msgs/msg/Image --field header
            fi
            if ((WITH_TILT)); then
                wait_topic /camera_tilt/state_deg std_msgs/msg/Float64 --field data \
                    --qos-reliability reliable --qos-durability transient_local \
                    --filter "abs(m - $TILT_ANGLE) <= 2.0"
            fi
            gripper_feedback_ready
            if ((WITH_POLICY)); then
                deadline=$((SECONDS + WAIT_TIMEOUT))
                until curl --fail --silent --show-error --noproxy '*' --max-time 2 \
                    "http://$OPENPI_HOST:$OPENPI_PORT/healthz"; do
                    ((SECONDS < deadline)) || die "Policy readiness timed out; inspect its tab."
                    sleep 2
                done
            fi
            info "PREPARATION READY for the selected services. No policy actions were sent."
            info "Gripper feedback is valid and fresh; the motor was NOT connected/initialized by this launcher."
            info "Start infer_muscleVLTA_debug for real-observation prediction only; no hardware execution."
            return ;;
    esac
    command_for "$TASK"
    cd -- "$TASK_CWD"
    "${COMMAND[@]}"
}

open_terminal() {
    local option="$1" task="$2" output key value service="" screen=""
    local terminal_env=(env -u GNOME_TERMINAL_SERVICE -u GNOME_TERMINAL_SCREEN)
    if [[ "$option" == --tab ]]; then
        terminal_env+=("GNOME_TERMINAL_SERVICE=$TERMINAL_SERVICE" "GNOME_TERMINAL_SCREEN=$TERMINAL_SCREEN")
    fi
    output=$("${terminal_env[@]}" gnome-terminal "$option" --print-environment \
        "--title=AR3 inference: $task" "--working-directory=$ROOT" -- \
        bash --noprofile --norc "$SELF" "${ORIGINAL_ARGS[@]}" --checkpoint "$CHECKPOINT" --task "$task") \
        || die "Could not open task $task; inspect already opened tabs before retrying."
    while IFS='=' read -r key value; do
        case "$key" in
            GNOME_TERMINAL_SERVICE) service="$value" ;;
            GNOME_TERMINAL_SCREEN) screen="$value" ;;
        esac
    done <<<"$output"
    [[ -n "$service" && -n "$screen" ]] || die "GNOME did not return identifiers for $task."
    TERMINAL_SERVICE="$service" TERMINAL_SCREEN="$screen"
}

if [[ "$MODE" == dry-run ]]; then print_plan; exit 0; fi
if [[ "$MODE" == check ]]; then preflight; exit 0; fi

if [[ -n "$TASK" ]]; then
    command_for "$TASK"
    state_root="${XDG_RUNTIME_DIR:-/tmp}/musclevlta-infer-$UID-$ROS_DOMAIN_ID"
    umask 077
    mkdir -p -- "$state_root"
    [[ -O "$state_root" && ! -L "$state_root" ]] || die "Unsafe runtime directory: $state_root"
    exec 9>"$state_root/$TASK.lock"
    flock -n 9 || die "Task $TASK already has an owner in ROS domain $ROS_DOMAIN_ID."
    set +e
    (set -e; run_task)
    status=$?
    set -e
    flock -u 9
    info "Task $TASK exited with status $status. Closing a tab is not a physical stop."
    if [[ -t 0 ]]; then read -r -p "Press Enter to close this tab. " _; fi
    exit "$status"
fi

preflight
command -v gnome-terminal >/dev/null || die "Missing gnome-terminal; use --dry-run to launch tasks manually."
[[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] || die "No graphical session; use --dry-run."
check_conflicts
print_plan
option=--window
for task in "${TASKS[@]}"; do
    open_terminal "$option" "$task"
    ((WINDOWS)) || option=--tab
done
info "Preparation tasks opened. Wait for PREPARATION READY in the ready tab."
