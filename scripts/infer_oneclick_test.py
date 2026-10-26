import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infer_oneclick.sh"


def _executable(path, contents="#!/bin/bash\nexit 0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)
    return path


@pytest.fixture
def environment(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("INFER_", "OPENPI_"))}
    base = tmp_path / "devices with spaces"
    capture = base / "capture"
    gripper = base / "V3_Gripper"
    arm = base / "arm"
    tilt = base / "tilt"
    cameras = base / "camera launch"
    data_home = base / "openpi"
    checkpoint = base / "checkpoint"
    files = [
        arm / "jazzy.sh",
        arm / "install/setup.bash",
        capture / "scripts/ar3_gripper_connection.py",
        capture / "scripts/ar3_tactile_bridge.py",
        capture / "scripts/ar3_tactile_worker.py",
        capture / "scripts/ar3_tactile_io.py",
        tilt / "camera_tilt_ros2.py",
        tilt / "usb_servo.py",
        cameras / "gemini_330_series.launch.py",
        cameras / "gemini305.launch.py",
        data_home / "big_vision/paligemma_tokenizer.model",
        checkpoint / "assets/ar3/norm_stats.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (checkpoint / "params").mkdir()
    sdk_dir = gripper / "SDK_Publish_V1.2.13.1"
    sdk_dir.mkdir(parents=True)
    sdk_python = _executable(gripper / ".venv/bin/python")
    _executable(capture / ".venv-ar3/bin/python")
    _executable(
        gripper / ".venv-ros/bin/python",
        """#!/bin/bash
printf 'grpc %s\\n' "$*" >> "$INFER_TEST_CALLS"
if [[ "$1" == */prepare_ar3_gripper.py ]]; then
    exit "${INFER_TEST_GRIPPER_PREPARE_EXIT:-0}"
elif [[ " $* " == *" --check-gripper "* ]]; then
    exit "${INFER_TEST_GRIPPER_FEEDBACK_EXIT:-0}"
fi
""",
    )
    servo_python = _executable(tilt / "servo-python")
    config = capture / "tactile.json"
    config.write_text(
        json.dumps(
            {
                "sdk_python": str(sdk_python),
                "sdk_dir": str(sdk_dir),
                "sensors": [
                    {
                        "name": sensor,
                        "remote_addr": f"192.168.127.10:{port}",
                        "pc_host": "192.168.127.100",
                        "enable_force": True,
                    }
                    for sensor, port in (("sensor_0", 50051), ("sensor_2", 50052))
                ],
            }
        )
    )
    setup = base / "setup.bash"
    setup.write_text("")
    mock_bin = base / "bin"
    for name in ("ssh", "ping", "curl"):
        _executable(mock_bin / name, '#!/bin/bash\nprintf "%s\\n" "$0 $*" >> "$INFER_TEST_CALLS"\n')
    _executable(
        mock_bin / "ip",
        """#!/bin/bash
printf 'ip %s\\n' "$*" >> "$INFER_TEST_CALLS"
[[ "${1:-}" != -4 ]] || shift
if [[ "$*" == *"addr show"* ]]; then
    if [[ "${INFER_TEST_GRIPPER_ADDRESS_PRESENT:-1}" == 1 || -f "$INFER_TEST_IP_ADDED" ]]; then
        printf '1: %s inet %s scope global\\n' "${INFER_GRIPPER_INTERFACE:-lo}" "${INFER_GRIPPER_CIDR:-192.168.127.100/24}"
    fi
elif [[ "$1 $2" == "route get" ]]; then
    printf '%s dev %s src %s\\n' "$3" "${INFER_TEST_ROUTE_INTERFACE:-lo}" "${INFER_TEST_ROUTE_SOURCE:-192.168.127.100}"
elif [[ "$1 $2" == "addr add" ]]; then
    touch "$INFER_TEST_IP_ADDED"
fi
""",
    )
    _executable(
        mock_bin / "sudo",
        """#!/bin/bash
printf 'sudo %s\\n' "$*" >> "$INFER_TEST_CALLS"
[[ "$1 $2 $3" == "ip addr add" ]] || exit 1
exec "$@"
""",
    )
    _executable(
        mock_bin / "timeout",
        """#!/bin/bash
printf '%s\\n' "$*" >> "$INFER_TEST_CALLS"
if [[ "$*" == *"node list"* ]]; then
    printf '%s\\n' "${INFER_TEST_NODES:-}"
elif [[ "$*" == *"/dev/tcp/"* ]]; then
    exit "${INFER_TEST_TCP_EXIT:-0}"
elif [[ "$*" == *"${INFER_TEST_FAIL_TOPIC:-__none__}"* ]]; then
    exit 1
elif [[ "$*" == *"/camera_tilt/state_deg"* ]]; then
    exec "$INFER_TEST_PYTHON" - "$@" <<'PY'
import os
import sys
from types import SimpleNamespace

message = SimpleNamespace(data=float(os.environ.get("INFER_TEST_TILT_MEASURED", "40")))
if "--field" in sys.argv:
    for field in sys.argv[sys.argv.index("--field") + 1].split("."):
        message = getattr(message, field)
expression = sys.argv[sys.argv.index("--filter") + 1]
matched = eval(expression, {"__builtins__": {}, "abs": abs}, {"m": message})
raise SystemExit(0 if matched else 1)
PY
fi
""",
    )
    _executable(
        mock_bin / "gnome-terminal",
        """#!/bin/bash
printf '%s\\n' "GNOME_TERMINAL_SERVICE=:mock" "GNOME_TERMINAL_SCREEN=/mock"
printf 'terminal ' >> "$INFER_TEST_CALLS"
printf '%q ' "$@" >> "$INFER_TEST_CALLS"
printf '\\n' >> "$INFER_TEST_CALLS"
""",
    )
    env.update(
        {
            "INFER_AR3_WS": str(arm),
            "INFER_CAPTURE_DIR": str(capture),
            "INFER_GRIPPER_DIR": str(gripper),
            "INFER_TILT_DIR": str(tilt),
            "INFER_SERVO_PYTHON": str(servo_python),
            "INFER_CAMERA_LAUNCH_DIR": str(cameras),
            "INFER_ROS_SETUP": str(setup),
            "INFER_TACTILE_CONFIG": str(config),
            "INFER_ARM_INTERFACE": "lo",
            "INFER_CHECKPOINT": str(checkpoint),
            "INFER_TEST_CALLS": str(tmp_path / "calls"),
            "INFER_TEST_IP_ADDED": str(tmp_path / "gripper-address-added"),
            "INFER_TEST_PYTHON": sys.executable,
            "OPENPI_DATA_HOME": str(data_home),
            "OPENPI_PYTHON": sys.executable,
            "PATH": f"{mock_bin}:{env['PATH']}",
            "DISPLAY": ":mock",
            "XDG_RUNTIME_DIR": str(tmp_path),
        }
    )
    return env


def run(env, *args, cwd="/tmp", script=LAUNCHER):
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env=env, text=True, capture_output=True, check=False)


def test_default_plan_is_hardware_prep_not_teleoperation_or_model_execution(environment):
    result = run(environment, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "enable_movej:=false" in result.stdout
    assert "startup_angle_deg:=40.0" in result.stdout
    assert "StrictHostKeyChecking=yes" in result.stdout
    assert "gemini_330_series.launch.py" in result.stdout
    assert "gemini305.launch.py" in result.stdout
    front_command = next(line for line in result.stdout.splitlines() if "camera_name:=front" in line)
    wrist_command = next(line for line in result.stdout.splitlines() if "camera_name:=wrist" in line)
    assert "color_format:=MJPG" in front_command
    assert "color_format:=YUYV" in wrist_command
    assert "color_format:=MJPG" not in wrist_command
    assert "uvc_backend:=v4l2" in wrist_command
    assert "uvc_backend:=" not in front_command
    assert "[policy]" not in result.stdout
    assert "MessageBridge" not in result.stdout
    assert "Puppetry" not in result.stdout
    assert "--execute" not in result.stdout
    assert "--initialize" not in result.stdout
    assert "ar3_gripper_teleop" not in result.stdout
    assert "192.168.127.100/24" in result.stdout
    assert "192.168.127.10" in result.stdout
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


def test_check_is_local_only_and_preserves_paths_with_spaces(environment):
    result = run(environment, "--check", "--with-policy")
    assert result.returncode == 0, result.stderr
    assert "No hardware, network or ROS commands were run" in result.stdout
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


def test_check_rejects_missing_checkpoint_asset(environment):
    Path(environment["INFER_CHECKPOINT"], "assets/ar3/norm_stats.json").unlink()
    result = run(environment, "--check", "--with-policy")
    assert result.returncode != 0
    assert "normalization" in result.stderr
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


def test_check_rejects_tactile_host_mismatch(environment):
    environment["INFER_GRIPPER_HOST"] = "192.168.127.20"
    result = run(environment, "--check")
    assert result.returncode != 0
    assert "Tactile config must match" in result.stderr


def test_check_rejects_tactile_return_address_mismatch(environment):
    config_path = Path(environment["INFER_TACTILE_CONFIG"])
    config = json.loads(config_path.read_text())
    config["sensors"][1]["pc_host"] = "192.168.2.100"
    config_path.write_text(json.dumps(config))
    result = run(environment, "--check")
    assert result.returncode != 0
    assert "Tactile" in result.stderr
    assert "192.168.127.100" in result.stderr
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


def test_check_requires_gripper_interface_even_without_arm(environment):
    environment["INFER_GRIPPER_INTERFACE"] = "infer-no-such-nic"
    result = run(environment, "--check", "--no-arm")
    assert result.returncode != 0
    assert "gripper interface" in result.stderr.lower()
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


@pytest.mark.parametrize("value", ["-1", "61", "nan", "40;touch /tmp/unwanted"])
def test_rejects_unsafe_tilt_angles_even_in_dry_run(environment, value):
    environment["INFER_TILT_ANGLE"] = value
    result = run(environment, "--dry-run")
    assert result.returncode != 0
    assert "0..60" in result.stderr


def test_public_entrypoint_delegates_to_named_helper(environment):
    args = ("--dry-run", "--no-arm", "--no-ssh", "--no-tactile", "--no-cameras", "--no-tilt")
    result = run(environment, *args)
    helper = run(environment, *args, script=ROOT / "scripts/prepare_ar3_inference.sh")
    assert result.returncode == helper.returncode == 0
    assert result.stdout == helper.stdout
    assert not (ROOT / "infer_onecilck.sh").exists()
    assert "[arm]" not in result.stdout
    assert "[sensor]" not in result.stdout
    assert "[front]" not in result.stdout
    assert "[tilt]" not in result.stdout


def test_terminal_children_receive_absolute_checkpoint(environment, tmp_path):
    environment["INFER_TEST_TCP_EXIT"] = "1"
    result = run(environment, "--checkpoint", "devices with spaces/checkpoint", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    terminals = [shlex.split(line.removeprefix("terminal ")) for line in calls if line.startswith("terminal ")]
    assert len(terminals) == 8
    for args in terminals:
        assert str(ROOT / "scripts/prepare_ar3_inference.sh") in args
        final_checkpoint = max(i for i, arg in enumerate(args) if arg == "--checkpoint")
        assert args[final_checkpoint + 1] == environment["INFER_CHECKPOINT"]
    assert "PREPARATION READY" in result.stdout
    assert not any("GetAllState" in line for line in calls)


@pytest.mark.parametrize("args", [(), ("--no-arm",)])
def test_missing_gripper_address_is_added_before_any_terminal(environment, args):
    environment["INFER_TEST_GRIPPER_ADDRESS_PRESENT"] = "0"
    result = run(environment, *args)
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    add_index = calls.index("sudo ip addr add 192.168.127.100/24 dev lo")
    route_index = calls.index("ip -4 route get 192.168.127.10")
    terminal_indices = [i for i, line in enumerate(calls) if line.startswith("terminal ")]
    assert terminal_indices
    assert add_index < route_index < min(terminal_indices)
    assert Path(environment["INFER_TEST_IP_ADDED"]).exists()
    assert sum(line.startswith("sudo ") for line in calls) == 1
    if "--no-arm" in args:
        for index in terminal_indices:
            terminal_args = shlex.split(calls[index].removeprefix("terminal "))
            assert terminal_args[terminal_args.index("--task") + 1] != "arm"


def test_existing_gripper_address_needs_no_sudo(environment):
    result = run(environment)
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    assert "ip -4 route get 192.168.127.10" in calls
    assert any(line.startswith("terminal ") for line in calls)
    assert not any(line.startswith("sudo ") for line in calls)
    assert not Path(environment["INFER_TEST_IP_ADDED"]).exists()


@pytest.mark.parametrize(
    "args", [(), ("--task", "sensor"), ("--task", "gripper"), ("--task", "tactile"), ("--task", "ready")]
)
@pytest.mark.parametrize(
    ("variable", "value"),
    [("INFER_TEST_ROUTE_INTERFACE", "wlan0"), ("INFER_TEST_ROUTE_SOURCE", "192.168.2.100")],
)
def test_wrong_gripper_route_fails_before_remote_access_or_terminals(environment, args, variable, value):
    environment[variable] = value
    result = run(environment, *args)
    assert result.returncode != 0
    assert "route" in result.stderr.lower()
    assert "192.168.127.10" in result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "ip -4 route get 192.168.127.10" in calls
    assert "/dev/tcp/" not in calls
    assert "/ssh " not in calls
    assert "grpc " not in calls
    assert "terminal " not in calls
    assert "sudo " not in calls


def test_existing_teleop_blocks_startup_before_any_terminal(environment):
    environment["INFER_TEST_NODES"] = "/ar3r_cartesian_retargeting"
    result = run(environment)
    assert result.returncode != 0
    assert "Teleoperation is running" in result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "terminal " not in calls
    assert "ip " not in calls
    assert "sudo " not in calls


def test_readiness_requires_measured_tilt_and_both_wrenches(environment):
    result = run(environment, "--task", "ready")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "/sensor_0/wrench" in calls
    assert "/sensor_2/wrench" in calls
    assert "--filter abs(m - 40.0) <= 2.0" in calls
    assert "grpc" in calls
    assert "--check-gripper --gripper-address 192.168.127.10:50008" in calls
    assert "PREPARATION READY" in result.stdout


def test_readiness_connects_motor_before_ros_inputs_and_keeps_final_readonly_check(environment):
    result = run(environment, "--task", "ready")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    server_index = next(i for i, line in enumerate(calls) if "wait_for_gripper_server" in line)
    prepare_index = next(i for i, line in enumerate(calls) if "prepare_ar3_gripper.py" in line)
    topic_indices = [i for i, line in enumerate(calls) if "topic echo" in line]
    feedback_index = next(i for i, line in enumerate(calls) if "--check-gripper" in line)
    assert topic_indices
    assert server_index < prepare_index < min(topic_indices)
    assert max(topic_indices) < feedback_index
    assert "--connect" in calls[prepare_index]
    assert "--gripper-address 192.168.127.10:50008" in calls[prepare_index]
    assert "--channel can0" in calls[prepare_index]
    assert "--motor-id 1" in calls[prepare_index]
    assert "--timeout 180" in calls[prepare_index]
    assert "--initialize" not in "\n".join(calls)
    assert "--execute" not in "\n".join(calls)


def test_explicit_gripper_initialization_only_reaches_preparation_helper(environment):
    result = run(environment, "--ready-only", "--initialize-gripper")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    prepare = next(line for line in calls if "prepare_ar3_gripper.py" in line)
    feedback = next(line for line in calls if "--check-gripper" in line)
    assert "--connect" in prepare
    assert "--initialize" in shlex.split(prepare)
    assert "--initialize" not in feedback
    assert "--execute" not in "\n".join(calls)
    assert "terminal " not in "\n".join(calls)


def test_failed_gripper_preparation_stops_before_ros_inputs_or_final_readiness(environment):
    environment["INFER_TEST_GRIPPER_PREPARE_EXIT"] = "1"
    result = run(environment, "--ready-only")
    assert result.returncode != 0
    assert "PREPARATION READY" not in result.stdout
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "wait_for_gripper_server" in calls
    assert "prepare_ar3_gripper.py" in calls
    assert "topic echo" not in calls
    assert "--check-gripper" not in calls
    assert "terminal " not in calls


def test_ready_only_reuses_running_controller_and_cameras_without_opening_terminals(environment):
    environment["INFER_TEST_NODES"] = (
        "/controller_manager\n/position_joint_trajectory_controller\n/front/camera\n/wrist/camera"
    )
    result = run(environment, "--ready-only")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "node list" in calls
    assert "prepare_ar3_gripper.py" in calls
    assert "--check-gripper" in calls
    assert "terminal " not in calls
    assert "/ssh " not in calls
    assert "PREPARATION READY" in result.stdout


def test_ready_only_dry_run_describes_only_readiness_without_device_access(environment):
    result = run(environment, "--ready-only", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "[ready] cwd=" in result.stdout
    for task in ("arm", "sensor", "gripper", "tactile", "front", "wrist", "tilt", "policy"):
        assert f"[{task}] cwd=" not in result.stdout
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


@pytest.mark.parametrize("args", [("--ready-only",), ("--task", "ready")])
def test_readiness_refuses_active_teleoperation_before_preparing_motor(environment, args):
    environment["INFER_TEST_NODES"] = "/controller_manager\n/ar3r_cartesian_retargeting"
    result = run(environment, *args)
    assert result.returncode != 0
    assert "Teleoperation is running" in result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "node list" in calls
    assert "prepare_ar3_gripper.py" not in calls
    assert "--check-gripper" not in calls
    assert "terminal " not in calls


@pytest.mark.parametrize("args", [(), ("--ready-only",), ("--task", "ready")])
def test_active_live_control_blocks_gripper_preparation(environment, args):
    environment["INFER_TEST_NODES"] = "/controller_manager\n/openpi_ar3_control_1234"
    result = run(environment, *args)
    assert result.returncode != 0
    assert "Live inference is running" in result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "node list" in calls
    assert "prepare_ar3_gripper.py" not in calls
    assert "--check-gripper" not in calls
    assert "terminal " not in calls


def test_gripper_preparation_receives_selected_motor_and_timeout(environment):
    environment.update({"INFER_GRIPPER_CHANNEL": "can1", "INFER_GRIPPER_MOTOR_ID": "17", "INFER_TIMEOUT": "23"})
    result = run(environment, "--ready-only")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    prepare = next(line for line in calls if "prepare_ar3_gripper.py" in line)
    assert "--channel can1" in prepare
    assert "--motor-id 17" in prepare
    assert "--timeout 23" in prepare


@pytest.mark.parametrize("value", ["0", "-1", "256", "nan", "1;touch /tmp/unwanted", "9999999999999999999999"])
def test_invalid_gripper_motor_id_fails_before_any_device_access(environment, value):
    environment["INFER_GRIPPER_MOTOR_ID"] = value
    result = run(environment, "--dry-run")
    assert result.returncode != 0
    assert "INFER_GRIPPER_MOTOR_ID" in result.stderr
    assert "1..255" in result.stderr
    assert not Path(environment["INFER_TEST_CALLS"]).exists()


@pytest.mark.parametrize("value", ["1", "255"])
def test_gripper_motor_id_includes_both_valid_endpoints(environment, value):
    environment["INFER_GRIPPER_MOTOR_ID"] = value
    result = run(environment, "--ready-only")
    assert result.returncode == 0, result.stderr
    calls = Path(environment["INFER_TEST_CALLS"]).read_text().splitlines()
    prepare = next(line for line in calls if "prepare_ar3_gripper.py" in line)
    assert f"--motor-id {value}" in prepare


@pytest.mark.parametrize(
    ("target", "measured", "ready"),
    [(40, 38, True), (40, 42, True), (40, 42.1, False), (15, 15, True), (15, 40, False)],
)
def test_readiness_filters_selected_tilt_value_within_target_tolerance(environment, target, measured, ready):
    environment["INFER_TILT_ANGLE"] = str(target)
    environment["INFER_TEST_TILT_MEASURED"] = str(measured)
    result = run(environment, "--task", "ready")
    assert result.returncode == (0 if ready else 1), result.stderr
    assert ("PREPARATION READY" in result.stdout) is ready
    if not ready:
        assert "No qualifying sample on /camera_tilt/state_deg" in result.stderr


def test_responsive_gripper_service_with_invalid_feedback_never_reports_ready(environment):
    environment["INFER_TEST_GRIPPER_FEEDBACK_EXIT"] = "1"
    result = run(environment, "--task", "ready")
    assert result.returncode != 0
    assert "Gripper feedback is not ready" in result.stderr
    assert "PREPARATION READY" not in result.stdout
    calls = Path(environment["INFER_TEST_CALLS"]).read_text()
    assert "wait_for_gripper_server" in calls
    assert "--check-gripper" in calls


def test_missing_wrench_never_reports_ready(environment):
    environment["INFER_TEST_FAIL_TOPIC"] = "/sensor_2/wrench"
    result = run(environment, "--task", "ready")
    assert result.returncode != 0
    assert "No qualifying sample" in result.stderr
    assert "PREPARATION READY" not in result.stdout


def test_vscode_debug_is_local_model_and_readonly():
    launch = json.loads((ROOT / ".vscode/launch.json").read_text())
    debugs = [item for item in launch["configurations"] if item["name"].startswith("infer_muscleVLTA_debug")]
    assert "infer_muscleVLTA_debug" in [item["name"] for item in debugs]
    for debug in debugs:
        assert debug["python"] == "${workspaceFolder}/.venv/bin/python"
        assert debug["program"].endswith("/scripts/infer_musclevlta_debug.py")
        assert debug["args"][debug["args"].index("--source") + 1] == "ros-readonly"
        assert debug["justMyCode"] is False
        assert "--execute" not in debug["args"]
        assert "--prompt" in debug["args"]
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/infer_musclevlta_debug.py"), "--source", "fake"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires --source ros-readonly" in result.stderr


@pytest.fixture
def debug_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setitem(sys.modules, "serve_policy", None)
    return importlib.import_module("infer_musclevlta_debug")


def test_debug_check_only_needs_no_checkpoint_model_or_viewer(debug_module, monkeypatch, tmp_path):
    events = []
    observation = {
        "observation/front_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation/state": np.array([0] * 7 + [432], dtype=np.float32),
    }
    robot = SimpleNamespace(
        observe=lambda: (events.append("observe"), observation)[1],
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(debug_module.client_example, "make_robot", lambda args, metadata: robot)
    assert debug_module.main(["--check-observations", "--checkpoint-dir", str(tmp_path / "missing")]) is None
    assert events == ["observe", "close"]


@pytest.mark.parametrize("check_only", [False, True])
def test_debug_bad_observations_fail_before_model_loading_and_close_worker(debug_module, monkeypatch, check_only):
    events = []

    def observe():
        events.append("observe")
        raise RuntimeError("Missing ROS inputs: wrist; Gripper feedback unavailable")

    robot = SimpleNamespace(observe=observe, close=lambda: events.append("close"))
    monkeypatch.setattr(debug_module.client_example, "make_robot", lambda args, metadata: robot)
    with pytest.raises(RuntimeError, match="Missing ROS inputs: wrist"):
        debug_module.main(["--check-observations"] if check_only else [])
    assert events == ["observe", "close"]


def test_debug_check_only_rejects_live_execution_before_creating_worker(debug_module, monkeypatch):
    monkeypatch.setattr(
        debug_module.client_example, "make_robot", lambda *args: pytest.fail("Must not create a control worker")
    )
    with pytest.raises(SystemExit) as exc:
        debug_module.main(["--check-observations", "--source", "ros", "--execute"])
    assert exc.value.code == 2


def test_live_preflight_forces_readonly_without_mutating_execution_args(debug_module, monkeypatch):
    args = debug_module.client_example.parse_args(
        ["--source", "ros", "--execute", "--initialize-gripper", "--fps", "15"]
    )
    events = []
    observation = {
        "observation/front_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((2, 3, 3), dtype=np.uint8),
        "observation/state": np.array([0] * 7 + [432], dtype=np.float32),
    }

    def create_worker(preflight_args, metadata):
        assert preflight_args.source == "ros-readonly"
        assert not preflight_args.execute
        assert not preflight_args.initialize_gripper
        return SimpleNamespace(
            observe=lambda: (events.append("observe"), observation)[1],
            close=lambda: events.append("close"),
        )

    monkeypatch.setattr(debug_module.client_example, "make_robot", create_worker)
    debug_module.check_readonly_observations(args)
    assert events == ["observe", "close"]
    assert args.source == "ros"
    assert args.execute
    assert args.initialize_gripper


def test_live_missing_tactile_fails_before_model_loading(debug_module, monkeypatch, tmp_path):
    setup = tmp_path / "setup.bash"
    setup.write_text("")
    events = []

    def observe():
        events.append("observe")
        raise RuntimeError("Missing ROS inputs: depth_0, depth_2, wrench_0, wrench_2")

    def create_worker(args, metadata):
        assert args.source == "ros-readonly"
        assert not args.execute
        return SimpleNamespace(observe=observe, close=lambda: events.append("close"))

    monkeypatch.setattr(debug_module.client_example, "make_robot", create_worker)
    with pytest.raises(RuntimeError, match="Missing ROS inputs: depth_0"):
        debug_module.main(["--source", "ros", "--execute", "--ros-python", sys.executable, "--ros-setup", str(setup)])
    assert events == ["observe", "close"]


def test_explicit_motor_initialization_defers_full_preflight(debug_module, monkeypatch, tmp_path, capsys):
    setup = tmp_path / "setup.bash"
    setup.write_text("")
    monkeypatch.delenv("OPENPI_AR3_ASSET_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setattr(
        debug_module,
        "check_readonly_observations",
        lambda args: pytest.fail("Uninitialized motor readiness must remain in the explicitly opted-in LIVE worker"),
    )
    with pytest.raises(SystemExit) as exc:
        debug_module.main(
            [
                "--source",
                "ros",
                "--execute",
                "--initialize-gripper",
                "--ros-python",
                sys.executable,
                "--ros-setup",
                str(setup),
                "--checkpoint-dir",
                str(tmp_path / "missing-checkpoint"),
                "--gpu",
                "cpu",
            ]
        )
    assert exc.value.code == 2
    assert "Expected one checkpoint normalization asset" in capsys.readouterr().err


def test_live_debug_is_separate_explicit_bounded_and_keeps_worker_outside_debugger(debug_module):
    launch = json.loads((ROOT / ".vscode/launch.json").read_text())
    assert launch["configurations"][0]["name"] == "infer_muscleVLTA_debug"
    lives = [item for item in launch["configurations"] if "LIVE" in item["name"]]
    assert lives
    for live in lives:
        args, _ = debug_module.client_example.build_parser().parse_known_args(live["args"])
        debug_module.client_example.validate_execution_args(args)
        assert args.source == "ros"
        assert args.execute
        assert args.max_steps > 0
        assert args.max_joint_velocity <= 0.1
        assert "--prompt" in live["args"]
        assert args.prompt
        assert not args.initialize_gripper
        assert live["subProcess"] is False


@pytest.mark.parametrize("lora", [False, True])
def test_debug_rejects_lora_mismatch_between_checkpoint_and_config(debug_module, tmp_path, lora):
    keys = ["('params', 'PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'w', 'value')"]
    if lora:
        keys.append("('params', 'PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'lora_a', 'value')")
    (tmp_path / "params").mkdir()
    (tmp_path / "params/_METADATA").write_text(json.dumps({"tree_metadata": {key: {} for key in keys}}))
    full = SimpleNamespace(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    lowmem = SimpleNamespace(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    assert debug_module.lora_mismatch(tmp_path, "matching", lowmem if lora else full) is None
    error = debug_module.lora_mismatch(tmp_path, "mismatched", full if lora else lowmem)
    expected = debug_module.LOWMEM_CONFIG if lora else debug_module.FINE_TUNING_CONFIG
    assert error.endswith(f"use --config-name {expected}.")
    # An unreadable checkpoint layout never blocks loading; the model load verifies the structure itself.
    assert debug_module.lora_mismatch(tmp_path / "missing", "any", lowmem) is None


def test_live_debug_fails_before_loading_model_when_prerequisite_is_missing(debug_module, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        debug_module.main(["--source", "ros", "--execute", "--ros-python", str(tmp_path / "missing-python")])
    assert exc.value.code == 2
    assert "Missing live control prerequisite" in capsys.readouterr().err
