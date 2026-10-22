"""Robot client for the muscleVLTA AR3 force + visuo-tactile policy (server: scripts/serve_ar3_force_visuotactile.sh).

Observation sources (--source):
  dataset       Replays a recorded episode of the LeRobot v3.0 dataset through the policy server (open loop) and
                compares the predicted actions with the recorded ones: an offline check of a checkpoint over the exact
                deployment path (WebSocket, transforms, reactive streaming). Needs the openpi environment (uv run).
  fake          FakeAR3Robot: random sensor data, actions are only stored (connection / latency test).
  ros-readonly  Real ROS/camera/gripper observations with prediction previews only; never sends hardware actions.
  ros           Real observations and hardware execution, requiring explicit --execute.
  MODULE:CLASS  Your robot driver: a class with the AR3Robot interface below, constructed without arguments, e.g. on
                top of the ROS 2 topics listed in meta/ar3_capture.json of the dataset.

    # GPU server: serve a checkpoint, then replay episode 16 with metrics and a rerun recording
    bash scripts/serve_ar3_force_visuotactile.sh checkpoints/<config>/<run>/<step>
    uv run scripts/ar3_force_visuotactile_client_example.py --source dataset \\
        --dataset /data/jiang/utilData/ar3_force_computeChargingPullOut_force_aware --episode 16 \\
        --output-dir replay/episode_16 --rerun replay/episode_16.rrd
    # robot PC (needs numpy + packages/openpi-client, rerun-sdk only for --rerun)
    python scripts/ar3_force_visuotactile_client_example.py --host 127.0.0.1 --port 8000 --source my_ar3:AR3Robot

Observation sent to the server (layout of the training data, see src/openpi/policies/ar3_force_visuotactile_policy.py):

    key                           shape / dtype                meaning
    observation/front_image       (480, 640, 3) uint8          front RGB camera
    observation/wrist_image       (480, 640, 3) uint8          wrist RGB camera
    observation/state             (8,) float32                 7 joint positions [rad] + gripper position (SDK units
                                                               0..1000, 0 = closed); the 27D dataset state also works
    observation/arm_effort        (7,) or (t, 7) float32       measured joint torques [Nm]
    observation/finger_wrench_0   (6,) or (t, 6) float32       sensor_0 finger wrench Fx Fy Fz Mx My Mz (SDK units)
    observation/finger_wrench_2   (6,) or (t, 6) float32       sensor_2 finger wrench (its own, mirrored SDK frame)
    observation/tactile_depth     (2, h, w) or (t, 2, h, w)    tactile depth of sensor_0 and sensor_2 (raw SDK depth),
                                  float16                      area-downsampled by the client to the model resolution
                                                               (server metadata "tactile_shape", e.g. 72x96)
    observation/force_level       scalar or (t,) float32       current gripper force level command (20..100)
    prompt                        str                          task instruction (default: the episode task / PROMPT)

A single reading is used for the whole chunk; (t, ...) arrays are readings of consecutive steps since the chunk start.
Every response contains actions of shape (n, 9) at the 15 Hz dataset rate:

    actions[:, :7]   absolute joint position targets [rad]
    actions[:, 7]    gripper position target (0..1000, 0 = closed)
    actions[:, 8]    gripper force level command (20..100)

Before execution the client clips the gripper position and force level to their ranges and the joint targets to the
measured joints +- --max-joint-step [rad]; joint limits, workspace and velocity limits stay the job of the driver.
Joint torques and finger wrenches are inputs only: the recordings contain no torque commands, so the policy does not
predict joint forces. Contact forces result from tracking the joint / gripper targets with the commanded force level.

Modes (--mode, default: reactive if the server model is force-reactive):
  chunk     one inference per chunk; executes the first --execute-steps actions, the force / tactile readings at the
            chunk start are used for the whole chunk.
  reactive  LIFT late reactive force injection: the server encodes images, prompt and state once per chunk (replan)
            and returns the first k actions; every k steps the client sends the force / tactile readings recorded
            since the chunk start and the server re-runs only the reactive action expert, so the actions react to
            contact within the chunk. With k = cross_attention_latency + 1 (default) every action uses all readings
            it was trained to see.

Outputs: a summary of the inference latency and, for replayed episodes, of the action errors w.r.t. the recording
(per-joint MAE next to the "hold position" baseline, gripper / force level MAE, joint MAE per chunk step);
--output-dir writes rollout.npz (per-step state, policy / executed / recorded actions) and summary.json; --rerun
(spawn | FILE.rrd | rerun+http://HOST:9876/proxy) logs every step with the panels of the dataset viewer (cameras,
gripper target / actual, force level, joint actions / state, arm effort, finger wrenches, tactile depth) plus the
policy actions, a per-joint "policy vs recorded" tab and the inference latency.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Callable, Sequence
import dataclasses
import importlib
import json
import logging
import os
import pathlib
import time
from typing import Any, Protocol

import numpy as np
from openpi_client import websocket_client_policy

PROMPT = "Unplug the white charging plug from the white socket and place it on the table"
FRONT_IMAGE_KEY = "observation/front_image"
WRIST_IMAGE_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
ARM_EFFORT_KEY = "observation/arm_effort"
FINGER_WRENCH_KEYS = ("observation/finger_wrench_0", "observation/finger_wrench_2")
TACTILE_KEY = "observation/tactile_depth"
FORCE_LEVEL_KEY = "observation/force_level"
# Force / tactile readings: sent with every chunk and, in reactive mode, streamed every k steps.
READING_KEYS = (ARM_EFFORT_KEY, *FINGER_WRENCH_KEYS, TACTILE_KEY, FORCE_LEVEL_KEY)

NUM_JOINTS = 7
STATE_DIM = 8
ACTION_DIM = 9
GRIPPER_INDEX = 7
FORCE_LEVEL_INDEX = 8
JOINT_NAMES = tuple(f"joint{i}.pos" for i in range(1, NUM_JOINTS + 1))
TORQUE_NAMES = tuple(f"joint{i}.torque_Nm" for i in range(1, NUM_JOINTS + 1))
WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
TACTILE_SENSORS = ("sensor_0", "sensor_2")
GRIPPER_POSITION_RANGE = (0.0, 1000.0)
FORCE_LEVEL_RANGE = (20.0, 100.0)
# Video features of the LeRobot v3.0 dataset (dataset source, rerun entity paths).
FRONT_VIDEO_FEATURE = "observation.images.front"
WRIST_VIDEO_FEATURE = "observation.images.wrist"


class AR3Robot(Protocol):
    """Interface between the control loop and the robot (implemented by FakeAR3Robot and DatasetReplayRobot).

    Only observe() and execute() are required. Optional members: `done` (True ends the run), close(), `task` (default
    prompt) and `recorded` (recorded frame {"frame_index", "timestamp", "action", "valid"} for the evaluation).
    """

    def observe(self) -> dict:
        """Latest readings at the start of a control step: the observation keys above without prompt. The tactile
        depth can be sent at full resolution (2, 288, 384); the client downsamples it."""
        ...

    def execute(self, action: np.ndarray) -> np.ndarray | None:
        """Sends one (9,) action: 7 joint position targets [rad], gripper position (0..1000), force level (20..100).
        A driver may return its acknowledged, limited target for logging."""
        ...


class FakeAR3Robot:
    """Placeholder robot: random sensor data in the recorded layout, executed actions move the simulated joints."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.joints = np.zeros(NUM_JOINTS, np.float32)  # rad
        self.gripper_position = 1000.0  # SDK units 0..1000, 0 = closed
        self.force_level = 80.0  # gripper force level command 20..100

    def observe(self) -> dict:
        rng = self._rng
        return {
            FRONT_IMAGE_KEY: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            WRIST_IMAGE_KEY: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            STATE_KEY: np.append(self.joints, self.gripper_position).astype(np.float32),
            ARM_EFFORT_KEY: rng.normal(0.0, 1.0, NUM_JOINTS).astype(np.float32),  # joint torques [Nm]
            FINGER_WRENCH_KEYS[0]: rng.normal(0.0, 0.5, 6).astype(np.float32),  # sensor_0 Fx..Mz
            FINGER_WRENCH_KEYS[1]: rng.normal(0.0, 0.5, 6).astype(np.float32),  # sensor_2 Fx..Mz
            TACTILE_KEY: np.abs(rng.normal(0.0, 0.05, (2, 288, 384))).astype(np.float32),  # sensor_0, sensor_2
            FORCE_LEVEL_KEY: np.float32(self.force_level),  # last commanded force level
        }

    def execute(self, action: np.ndarray) -> None:
        self.joints = np.asarray(action[:NUM_JOINTS], np.float32)
        self.gripper_position = float(action[GRIPPER_INDEX])
        self.force_level = float(action[FORCE_LEVEL_INDEX])


class DatasetReplayRobot:
    """Replays one episode of a LeRobot v3.0 AR3 dataset (open loop: the actions do not change the recording).

    observe() returns the recorded frame: video frames, joint + gripper state, arm effort, finger wrenches, the tactile
    depth of both fingers (aligned and downsampled exactly as for training, through the tactile cache) and the
    recorded force level command, which training feeds as force level input. `recorded` holds the recorded action of
    the frame for the evaluation; execute() advances to the next frame.
    """

    def __init__(
        self,
        dataset: str,
        episode: int | None = None,
        *,
        tactile_shape: Sequence[int] | None = None,
        start_frame: int = 0,
        missing_video: str = "error",
        tactile_cache_dir: str | None = None,
    ) -> None:
        from openpi.data import lerobot_v3  # Only available in the openpi environment.

        meta = lerobot_v3.LeRobotV3Metadata(dataset)
        self.episode = int(next(iter(meta.episodes)) if episode is None else episode)
        tactile = None
        if tactile_shape is not None:
            tactile = lerobot_v3.TactileOptions(
                sensors=TACTILE_SENSORS,
                resolution=(int(tactile_shape[0]), int(tactile_shape[1])),
                cache_dir=tactile_cache_dir,
            )
        self._tactile_key = tactile.key if tactile is not None else None
        self._dataset = lerobot_v3.LeRobotV3Dataset(
            dataset,
            options=lerobot_v3.LeRobotV3Options(
                video_keys=(FRONT_VIDEO_FEATURE, WRIST_VIDEO_FEATURE),
                missing_video=missing_video,
                # Every frame in order; invalid frames are only excluded from the evaluation.
                require_valid=False,
                tactile=tactile,
                episodes=(self.episode,),
            ),
        )
        self.fps = self._dataset.meta.fps
        self.num_frames = len(self._dataset)
        if not 0 <= start_frame < self.num_frames:
            raise ValueError(f"start_frame {start_frame} is outside episode {self.episode} ({self.num_frames} frames)")
        tasks = self._dataset.meta.episodes[self.episode].get("tasks")
        self.task = str(tasks[0]) if tasks is not None and len(tasks) else None
        self.frame = start_frame
        self.recorded: dict | None = None

    @property
    def done(self) -> bool:
        return self.frame >= self.num_frames

    def observe(self) -> dict:
        item = self._dataset[self.frame]
        state = np.asarray(item["observation.state"], np.float32)
        action = np.asarray(item["action"], np.float32)
        valid = all(
            np.all(np.asarray(item[key]) >= 0.5) for key in ("observation.state_valid", "action_valid") if key in item
        )
        self.recorded = {
            "frame_index": int(item["frame_index"]),
            "timestamp": float(item["timestamp"]),
            "action": action,
            "valid": bool(valid),
        }
        observation = {
            FRONT_IMAGE_KEY: item[FRONT_VIDEO_FEATURE],
            WRIST_IMAGE_KEY: item[WRIST_VIDEO_FEATURE],
            STATE_KEY: state[:STATE_DIM],
            ARM_EFFORT_KEY: np.asarray(item["observation.arm_effort"], np.float32),
            FINGER_WRENCH_KEYS[0]: np.asarray(item["observation.gripper_wrench.sensor_0"], np.float32),
            FINGER_WRENCH_KEYS[1]: np.asarray(item["observation.gripper_wrench.sensor_2"], np.float32),
            FORCE_LEVEL_KEY: np.float32(action[FORCE_LEVEL_INDEX]),
        }
        if self._tactile_key is not None:
            # (2, h, w) float16 at the model resolution, identical to the training inputs.
            observation[TACTILE_KEY] = np.asarray(item[self._tactile_key][0])
        return observation

    def execute(self, action: np.ndarray) -> None:
        del action  # Open loop: the next recorded frame does not depend on the action.
        self.frame += 1


class Rate:
    """Keeps a fixed control rate (no limit if hz <= 0)."""

    def __init__(self, hz: float) -> None:
        self._period = 1.0 / hz if hz > 0 else 0.0
        self._next = time.monotonic()

    def sleep(self) -> None:
        if not self._period:
            return
        self._next += self._period
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            # Fell behind (e.g. while waiting for the server): restart the schedule from now.
            self._next = time.monotonic()


def downsample_tactile(depth: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Area-downsamples tactile depth images (..., H, W) to `shape` (288x384 -> 72x96: 16x less data).

    Equals the cv2.INTER_AREA resize of the training cache and of the server for integer factors. Images that are
    already at `shape` are returned unchanged; other sizes are returned as float32 and resized by the server.
    """
    depth = np.asarray(depth)
    height, width = int(shape[0]), int(shape[1])
    in_height, in_width = depth.shape[-2:]
    if (in_height, in_width) == (height, width):
        return depth
    depth = depth.astype(np.float32)
    if in_height % height or in_width % width:
        return depth
    blocks = depth.reshape(*depth.shape[:-2], height, in_height // height, width, in_width // width)
    return blocks.mean(axis=(-3, -1), dtype=np.float32)


def prepare_observation(observation: dict, tactile_shape: Sequence[int] | None) -> dict:
    """Converts a robot observation to the transmitted layout (float32 vectors, tactile at the model resolution)."""
    observation = dict(observation)
    for key in (STATE_KEY, ARM_EFFORT_KEY, *FINGER_WRENCH_KEYS, FORCE_LEVEL_KEY):
        if key in observation:
            observation[key] = np.asarray(observation[key], np.float32)
    if TACTILE_KEY in observation:
        if tactile_shape is None:
            # The served model has no tactile input.
            del observation[TACTILE_KEY]
        else:
            tactile = downsample_tactile(observation[TACTILE_KEY], tactile_shape)
            if tactile.shape[-2:] == (int(tactile_shape[0]), int(tactile_shape[1])):
                # float16 is the precision of the training cache and halves the bandwidth.
                tactile = tactile.astype(np.float16)
            observation[TACTILE_KEY] = tactile
    return observation


def response_actions(result: dict) -> np.ndarray:
    actions = np.asarray(result["actions"], np.float32)
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM or not len(actions):
        raise ValueError(f"Expected (n, {ACTION_DIM}) actions, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("The server returned non-finite actions")
    return actions


def limit_action(action: np.ndarray, state: np.ndarray, max_joint_step: float = 0.0) -> np.ndarray:
    """Safety limits before execution: gripper position and force level are clipped to their ranges and, if
    max_joint_step > 0, the joint targets to the measured joints +- max_joint_step [rad]."""
    action = np.array(action, dtype=np.float32)
    if max_joint_step > 0:
        joints = np.asarray(state, np.float32)[:NUM_JOINTS]
        action[:NUM_JOINTS] = np.clip(action[:NUM_JOINTS], joints - max_joint_step, joints + max_joint_step)
    action[GRIPPER_INDEX] = np.clip(action[GRIPPER_INDEX], *GRIPPER_POSITION_RANGE)
    action[FORCE_LEVEL_INDEX] = np.clip(action[FORCE_LEVEL_INDEX], *FORCE_LEVEL_RANGE)
    return action


@dataclasses.dataclass
class Step:
    """One executed control step, passed to the step callbacks."""

    index: int  # control step since the start
    chunk: int
    chunk_step: int  # step within the action chunk
    time_s: float  # wall-clock time since the start
    observation: dict  # observation of the step as transmitted (without prompt)
    policy_action: np.ndarray  # (9,) action predicted by the policy
    action: np.ndarray  # (9,) executed action (after limit_action)
    recorded: dict | None = None  # recorded frame (dataset replay)
    request: str | None = None  # server call made at this step: "infer" | "replan" | "step"
    round_trip_ms: float | None = None
    server_ms: float | None = None
    num_actions: int | None = None  # number of actions returned by that call


def run_policy(
    client,
    robot: AR3Robot,
    *,
    mode: str,
    action_horizon: int,
    prompt: str,
    tactile_shape: Sequence[int] | None = None,
    reactive_k: int = 1,
    execute_steps: int = 0,
    num_chunks: int = 0,
    max_steps: int = 0,
    fps: float = 0.0,
    max_joint_step: float = 0.0,
    callbacks: Sequence[Callable[[Step], None]] = (),
) -> int:
    """Runs the policy until the robot is done or the chunk / step budget is used. Returns the number of steps.

    Every step observes the robot first (readings of step i are taken before action i is executed, as in the
    recordings), then queries the server if needed, executes one action and calls the callbacks.
    """
    if mode not in ("chunk", "reactive"):
        raise ValueError(f"Unknown mode {mode!r}")
    steps_per_chunk = action_horizon if mode == "reactive" else min(execute_steps or action_horizon, action_horizon)
    rate = Rate(fps)
    start_time = time.monotonic()
    index = 0
    chunk = 0

    def budget_left() -> bool:
        return not getattr(robot, "done", False) and (max_steps <= 0 or index < max_steps)

    while budget_left() and (num_chunks <= 0 or chunk < num_chunks):
        pending: collections.deque[np.ndarray] = collections.deque()
        history: dict[str, list[np.ndarray]] = {}
        for chunk_step in range(steps_per_chunk):
            if not budget_left():
                break
            observation = prepare_observation(robot.observe(), tactile_shape)
            readings = {key: observation[key] for key in READING_KEYS if key in observation}
            request = result = None
            call_start = time.monotonic()
            if chunk_step == 0:
                # Chunk start: full observation, the VLM runs once for this chunk.
                if mode == "reactive":
                    history = {key: [value] for key, value in readings.items()}
                    request = "replan"
                    result = client.reactive_replan({**observation, "prompt": prompt}, k=reactive_k)
                else:
                    request = "infer"
                    result = client.infer({**observation, "prompt": prompt})
            elif mode == "reactive":
                if readings.keys() != history.keys():
                    raise RuntimeError(f"Readings changed within a chunk: {sorted(readings)} vs {sorted(history)}")
                for key, value in readings.items():
                    history[key].append(value)
                if not pending:
                    # All readings since the chunk start (images / state / prompt stay the replan ones).
                    request = "step"
                    result = client.reactive_step(
                        {key: np.stack(values) for key, values in history.items()}, k=reactive_k
                    )
                    if result.get("start_step") != chunk_step:
                        raise RuntimeError(f"Server is at step {result.get('start_step')}, client at {chunk_step}")
            round_trip_ms = (time.monotonic() - call_start) * 1000 if request else None
            num_actions = None
            if result is not None:
                actions = response_actions(result)
                num_actions = len(actions)
                pending.extend(actions[: steps_per_chunk - chunk_step])
            if not pending:
                raise RuntimeError(f"No action left for step {chunk_step} of chunk {chunk}")
            policy_action = np.asarray(pending.popleft(), np.float32)
            action = limit_action(
                policy_action,
                observation[STATE_KEY],
                max_joint_step=0.0 if getattr(robot, "handles_action_limits", False) else max_joint_step,
            )
            if np.any(action[GRIPPER_INDEX:] != policy_action[GRIPPER_INDEX:]):
                logging.warning(
                    "Control step %d: clipped gripper targets to SDK ranges: position %.6g -> %.6g (0..1000), "
                    "force level %.6g -> %.6g (20..100).",
                    index,
                    policy_action[GRIPPER_INDEX],
                    action[GRIPPER_INDEX],
                    policy_action[FORCE_LEVEL_INDEX],
                    action[FORCE_LEVEL_INDEX],
                )
            step = Step(
                index=index,
                chunk=chunk,
                chunk_step=chunk_step,
                time_s=time.monotonic() - start_time,
                observation=observation,
                policy_action=policy_action,
                action=action,
                recorded=getattr(robot, "recorded", None),
                request=request,
                round_trip_ms=round_trip_ms,
                server_ms=(result or {}).get("server_timing", {}).get("infer_ms"),
                num_actions=num_actions,
            )
            acknowledged = robot.execute(action)
            if acknowledged is not None:
                acknowledged = np.asarray(acknowledged, dtype=np.float32)
                if acknowledged.shape != (ACTION_DIM,) or not np.isfinite(acknowledged).all():
                    raise RuntimeError("Robot driver returned an invalid acknowledged action.")
                step.action = acknowledged
            for callback in callbacks:
                callback(step)
            index += 1
            rate.sleep()
        chunk += 1
    return index


def log_request(step: Step) -> None:
    """Logs every server call (chunk starts at INFO, reactive steps at DEBUG)."""
    if step.request is None:
        return
    frame = f" (frame {step.recorded['frame_index']})" if step.recorded else ""
    logging.log(
        logging.DEBUG if step.request == "step" else logging.INFO,
        "chunk %d %s at step %d%s: %d actions, round trip %.1f ms (server %.1f ms) | joints [rad] %s | gripper %.0f "
        "| force level %.0f",
        step.chunk,
        step.request,
        step.chunk_step,
        frame,
        step.num_actions,
        step.round_trip_ms,
        float("nan") if step.server_ms is None else step.server_ms,
        np.array2string(step.policy_action[:NUM_JOINTS], precision=3, suppress_small=True),
        step.policy_action[GRIPPER_INDEX],
        step.policy_action[FORCE_LEVEL_INDEX],
    )


def _latency_stats(values: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def evaluate_actions(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Errors of the policy actions w.r.t. the recorded actions over the valid frames.

    The "hold position" baseline (recorded action vs. the current state) gives the scale: the recorded joint targets
    stay close to the measured joints, so a useful policy has to be clearly below it.
    """
    valid = np.asarray(data["valid"], bool)
    if not valid.any():
        return {"frames": 0}
    policy = data["policy_action"][valid].astype(np.float64)
    recorded = data["recorded_action"][valid].astype(np.float64)
    state = data["state"][valid].astype(np.float64)
    chunk_step = data["chunk_step"][valid]
    joint_error = np.degrees(np.abs(policy[:, :NUM_JOINTS] - recorded[:, :NUM_JOINTS]))
    hold_error = np.degrees(np.abs(state[:, :NUM_JOINTS] - recorded[:, :NUM_JOINTS]))
    return {
        "frames": int(valid.sum()),
        "joint_mae_deg": {name: float(joint_error[:, i].mean()) for i, name in enumerate(JOINT_NAMES)},
        "joint_mae_deg_mean": float(joint_error.mean()),
        "joint_max_error_deg": float(joint_error.max()),
        "gripper_mae_sdk": float(np.abs(policy[:, GRIPPER_INDEX] - recorded[:, GRIPPER_INDEX]).mean()),
        "force_level_mae": float(np.abs(policy[:, FORCE_LEVEL_INDEX] - recorded[:, FORCE_LEVEL_INDEX]).mean()),
        "hold_baseline": {
            "joint_mae_deg_mean": float(hold_error.mean()),
            "gripper_mae_sdk": float(np.abs(state[:, GRIPPER_INDEX] - recorded[:, GRIPPER_INDEX]).mean()),
        },
        "joint_mae_deg_by_chunk_step": [
            float(joint_error[chunk_step == s].mean()) if np.any(chunk_step == s) else None
            for s in range(int(chunk_step.max()) + 1)
        ],
    }


class EpisodeRecorder:
    """Collects the executed steps for the summary (latency, errors w.r.t. the recording) and --output-dir."""

    def __init__(self) -> None:
        self._columns: dict[str, list] = collections.defaultdict(list)
        self._requests: dict[str, list[float]] = collections.defaultdict(list)
        self._server_ms: dict[str, list[float]] = collections.defaultdict(list)

    def __call__(self, step: Step) -> None:
        columns = self._columns
        columns["step"].append(step.index)
        columns["chunk"].append(step.chunk)
        columns["chunk_step"].append(step.chunk_step)
        columns["time_s"].append(step.time_s)
        columns["state"].append(np.asarray(step.observation[STATE_KEY], np.float32)[:STATE_DIM])
        columns["policy_action"].append(step.policy_action)
        columns["action"].append(step.action)
        if step.recorded is not None:
            columns["frame_index"].append(step.recorded["frame_index"])
            columns["timestamp"].append(step.recorded["timestamp"])
            columns["recorded_action"].append(step.recorded["action"])
            columns["valid"].append(step.recorded["valid"])
        if step.request is not None:
            self._requests[step.request].append(step.round_trip_ms)
            if step.server_ms is not None:
                self._server_ms[step.request].append(step.server_ms)

    def arrays(self) -> dict[str, np.ndarray]:
        steps = len(self._columns["step"])
        # Recorded columns are only kept if every step has them.
        return {key: np.asarray(values) for key, values in self._columns.items() if len(values) == steps}

    def summary(self) -> dict[str, Any]:
        data = self.arrays()
        summary: dict[str, Any] = {"steps": len(data["step"])}
        if not summary["steps"]:
            return summary
        summary["chunks"] = int(data["chunk"].max()) + 1
        summary["latency_ms"] = {kind: _latency_stats(values) for kind, values in self._requests.items()}
        for kind, values in self._server_ms.items():
            summary["latency_ms"][kind]["server_mean"] = float(np.mean(values))
        joints = slice(0, NUM_JOINTS)
        summary["limited_steps"] = int(np.any(data["policy_action"][:, joints] != data["action"][:, joints], -1).sum())
        if "recorded_action" in data:
            summary["evaluation"] = evaluate_actions(data)
        return summary

    def save(self, output_dir: str | pathlib.Path, summary: dict[str, Any]) -> None:
        output_dir = pathlib.Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "rollout.npz", **self.arrays())
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"{summary['steps']} steps in {summary.get('chunks', 0)} chunks"]
    for kind, stats in summary.get("latency_ms", {}).items():
        server = f", server mean {stats['server_mean']:.1f} ms" if "server_mean" in stats else ""
        lines.append(
            f"  {kind:<6} {stats['count']:4d} calls, round trip mean {stats['mean']:.1f} ms, "
            f"p95 {stats['p95']:.1f} ms, max {stats['max']:.1f} ms{server}"
        )
    if summary.get("limited_steps"):
        lines.append(f"  joint targets adjusted before execution in {summary['limited_steps']} steps (not stop events)")
    evaluation = summary.get("evaluation")
    if evaluation and evaluation["frames"]:
        per_joint = " ".join(f"{value:.2f}" for value in evaluation["joint_mae_deg"].values())
        by_step = " ".join(
            "-" if value is None else f"{value:.2f}" for value in evaluation["joint_mae_deg_by_chunk_step"]
        )
        baseline = evaluation["hold_baseline"]
        lines += [
            f"  policy vs recorded actions over {evaluation['frames']} valid frames:",
            f"    joint MAE [deg]   {per_joint} | mean {evaluation['joint_mae_deg_mean']:.2f} "
            f"(hold-position baseline {baseline['joint_mae_deg_mean']:.2f}), "
            f"max {evaluation['joint_max_error_deg']:.2f}",
            f"    gripper MAE {evaluation['gripper_mae_sdk']:.1f} SDK units "
            f"(baseline {baseline['gripper_mae_sdk']:.1f}), force level MAE {evaluation['force_level_mae']:.2f}",
            f"    joint MAE [deg] by chunk step: {by_step}",
        ]
    return "\n".join(lines)


RECORDED_COLOR = (31, 119, 180)
POLICY_COLOR = (255, 127, 14)
STATE_COLOR = (127, 127, 127)


class RerunLogger:
    """Logs every control step to rerun (pip install rerun-sdk; the openpi environment has it).

    The first tab mirrors the dataset viewer (cameras, tac G2 gripper target / actual, force level, recorded joint
    actions, state, arm effort, finger wrenches, tactile depth as sent to the model) and adds the policy actions; the
    second tab overlays policy and recorded targets per joint, with the inference latency and a log of the server
    calls. Timelines: frame_index (recorded frame index for dataset replay) and timestamp [s].
    """

    def __init__(
        self,
        target: str,
        *,
        application_id: str = "ar3_force_visuotactile_inference",
        jpeg_quality: int = 85,
        read_only: bool = False,
    ):
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self._jpeg_quality = jpeg_quality
        self._recording = rr.RecordingStream(application_id)
        blueprint = self.blueprint(rrb, read_only=read_only)
        if target == "spawn":
            self._recording.spawn(default_blueprint=blueprint)
        elif "://" in target:
            self._recording.connect_grpc(target, default_blueprint=blueprint)
        else:
            path = pathlib.Path(target).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._recording.save(path, default_blueprint=blueprint)
        self._log_styles()
        if read_only:
            self._recording.log(
                "inference_log", rr.TextLog("READ-ONLY: real observations; predictions are NOT sent to hardware.")
            )

    @staticmethod
    def blueprint(rrb, *, read_only: bool = False):
        def series(origin: str, name: str):
            return rrb.TimeSeriesView(origin=origin, name=name)

        episode = rrb.Grid(
            rrb.Spatial2DView(origin=FRONT_VIDEO_FEATURE, name=FRONT_VIDEO_FEATURE),
            rrb.Spatial2DView(origin=WRIST_VIDEO_FEATURE, name=WRIST_VIDEO_FEATURE),
            series("gripper", "tac G2 gripper (SDK position: 0 closed / 1000 open)"),
            series("gripper_force_level", "gripper_force_level (command)"),
            series("action", "action (recorded)"),
            series("policy/action", "action (policy)"),
            series("state", "state"),
            series("observation.arm_effort", "observation.arm_effort"),
            *(series(f"observation.gripper_wrench.{s}", f"observation.gripper_wrench.{s}") for s in TACTILE_SENSORS),
            *(
                rrb.TensorView(origin=f"observation.tactile_depth.{s}", name=f"observation.tactile_depth.{s}")
                for s in TACTILE_SENSORS
            ),
            grid_columns=4,
            name="live observations (read-only)" if read_only else "episode",
        )
        compare = rrb.Grid(
            *(series(f"compare/{name}", f"{name} [rad]") for name in JOINT_NAMES),
            series("inference", "inference round trip / server [ms]"),
            rrb.TextLogView(origin="inference_log", name="server calls"),
            grid_columns=3,
            name="predictions vs measured state" if read_only else "policy vs recorded",
        )
        return rrb.Blueprint(rrb.Tabs(episode, compare))

    def _log_styles(self) -> None:
        rr = self._rr
        styles = {
            "state": rr.SeriesLines(names=JOINT_NAMES),
            "action": rr.SeriesLines(names=JOINT_NAMES),
            "policy/action": rr.SeriesLines(names=JOINT_NAMES),
            "observation.arm_effort": rr.SeriesLines(names=TORQUE_NAMES),
            "gripper/target": rr.SeriesLines(names="target (recorded)", colors=RECORDED_COLOR),
            "gripper/actual": rr.SeriesLines(names="actual", colors=STATE_COLOR),
            "gripper/policy": rr.SeriesLines(names="policy", colors=POLICY_COLOR),
            "gripper_force_level/command": rr.SeriesLines(names="command (recorded)", colors=RECORDED_COLOR),
            "gripper_force_level/input_setting": rr.SeriesLines(names="input force setting", colors=STATE_COLOR),
            "gripper_force_level/policy": rr.SeriesLines(names="policy", colors=POLICY_COLOR),
            "inference/round_trip_ms": rr.SeriesLines(names="round trip"),
            "inference/server_ms": rr.SeriesLines(names="server"),
        }
        for sensor in TACTILE_SENSORS:
            styles[f"observation.gripper_wrench.{sensor}"] = rr.SeriesLines(names=WRENCH_COMPONENTS)
        for name in JOINT_NAMES:
            styles[f"compare/{name}/recorded"] = rr.SeriesLines(names="recorded", colors=RECORDED_COLOR)
            styles[f"compare/{name}/policy"] = rr.SeriesLines(names="policy", colors=POLICY_COLOR)
            styles[f"compare/{name}/state"] = rr.SeriesLines(names="state", colors=STATE_COLOR)
        for path, style in styles.items():
            self._recording.log(path, style, static=True)

    def __call__(self, step: Step) -> None:
        rr, recording = self._rr, self._recording
        recorded = step.recorded
        recording.set_time("frame_index", sequence=int(recorded["frame_index"]) if recorded else step.index)
        recording.set_time("timestamp", duration=float(recorded["timestamp"]) if recorded else step.time_s)

        def scalars(path: str, values) -> None:
            recording.log(path, rr.Scalars(np.asarray(values, np.float64).reshape(-1)))

        observation = step.observation
        for key, path in ((FRONT_IMAGE_KEY, FRONT_VIDEO_FEATURE), (WRIST_IMAGE_KEY, WRIST_VIDEO_FEATURE)):
            if key in observation:
                image = rr.Image(np.asarray(observation[key]), color_model="RGB")
                recording.log(path, image.compress(jpeg_quality=self._jpeg_quality))
        state = np.asarray(observation[STATE_KEY], np.float32)
        policy = step.policy_action
        scalars("state", state[:NUM_JOINTS])
        scalars("policy/action", policy[:NUM_JOINTS])
        scalars("gripper/actual", state[GRIPPER_INDEX])
        scalars("gripper/policy", policy[GRIPPER_INDEX])
        scalars("gripper_force_level/policy", policy[FORCE_LEVEL_INDEX])
        if FORCE_LEVEL_KEY in observation:
            scalars("gripper_force_level/input_setting", observation[FORCE_LEVEL_KEY])
        for index, name in enumerate(JOINT_NAMES):
            scalars(f"compare/{name}/policy", policy[index])
            scalars(f"compare/{name}/state", state[index])
        if recorded is not None:
            action = recorded["action"]
            scalars("action", action[:NUM_JOINTS])
            scalars("gripper/target", action[GRIPPER_INDEX])
            scalars("gripper_force_level/command", action[FORCE_LEVEL_INDEX])
            for index, name in enumerate(JOINT_NAMES):
                scalars(f"compare/{name}/recorded", action[index])
        if ARM_EFFORT_KEY in observation:
            scalars("observation.arm_effort", observation[ARM_EFFORT_KEY])
        for index, sensor in enumerate(TACTILE_SENSORS):
            if FINGER_WRENCH_KEYS[index] in observation:
                scalars(f"observation.gripper_wrench.{sensor}", observation[FINGER_WRENCH_KEYS[index]])
        if TACTILE_KEY in observation:
            tactile = np.asarray(observation[TACTILE_KEY], np.float32)
            for index, sensor in enumerate(TACTILE_SENSORS):
                recording.log(
                    f"observation.tactile_depth.{sensor}", rr.Tensor(tactile[index], dim_names=("height", "width"))
                )
        if step.request is not None:
            scalars("inference/round_trip_ms", step.round_trip_ms)
            if step.server_ms is not None:
                scalars("inference/server_ms", step.server_ms)
            recording.log(
                "inference_log",
                rr.TextLog(
                    f"chunk {step.chunk} {step.request} at step {step.chunk_step}: {step.num_actions} actions, "
                    f"round trip {step.round_trip_ms:.1f} ms"
                ),
            )

    def close(self) -> None:
        self._recording.flush()  # Blocks until sent (the flush arguments differ between rerun versions).
        self._recording.disconnect()


def make_robot(args: argparse.Namespace, metadata: dict) -> AR3Robot:
    if args.source == "fake":
        return FakeAR3Robot()
    if args.source in ("ros-readonly", "ros"):
        live = args.source == "ros"
        validate_execution_args(args)
        worker_args = [
            "--front-topic",
            args.front_topic,
            "--wrist-topic",
            args.wrist_topic,
            "--joint-topic",
            args.joint_topic,
            "--tactile-prefix",
            args.tactile_prefix,
            "--gripper-address",
            args.gripper_address,
            "--force-level",
            str(args.force_level),
            "--max-age",
            str(args.observation_max_age),
            "--timeout",
            str(args.observation_timeout),
        ]
        if live:
            worker_args += [
                "--execute",
                "--joint-command-topic",
                args.joint_command_topic,
                "--controller-name",
                args.controller_name,
                "--command-period",
                str(1.0 / args.fps),
                "--max-joint-step",
                str(args.max_joint_step),
                "--max-joint-velocity",
                str(args.max_joint_velocity),
                "--max-target-deviation",
                str(args.max_target_deviation),
                "--command-timeout",
                str(args.command_timeout),
                "--gripper-speed",
                str(args.gripper_speed),
            ]
            if args.initialize_gripper:
                worker_args.append("--initialize-gripper")
        if live:
            from openpi_client.ar3_ros_control import LiveAR3Robot

            robot_class = LiveAR3Robot
        else:
            from openpi_client.ar3_ros_readonly import ReadOnlyAR3Robot

            robot_class = ReadOnlyAR3Robot
        return robot_class(
            worker_script=pathlib.Path(__file__).with_name(
                "ar3_ros_control_worker.py" if live else "ar3_ros_observation_worker.py"
            ),
            ros_python=args.ros_python,
            ros_setup=args.ros_setup,
            domain_id=args.ros_domain_id,
            timeout=args.observation_timeout + 5,
            startup_timeout=max(60, 3 * args.observation_timeout + 20) if live else None,
            worker_args=worker_args,
        )
    if args.source == "dataset":
        if not args.dataset:
            raise ValueError("--source dataset needs --dataset (LeRobot v3.0 dataset directory)")
        return DatasetReplayRobot(
            args.dataset,
            args.episode,
            tactile_shape=metadata.get("tactile_shape"),
            start_frame=args.start_frame,
            missing_video="black" if args.allow_missing_videos else "error",
            tactile_cache_dir=args.tactile_cache_dir,
        )
    module_name, _, class_name = args.source.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Unknown --source {args.source!r}: use fake, dataset, ros-readonly, ros or MODULE:CLASS")
    return getattr(importlib.import_module(module_name), class_name)()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", default="fake", help="fake | dataset | ros-readonly | ros | MODULE:CLASS")
    parser.add_argument("--mode", choices=("auto", "chunk", "reactive"), default="auto")
    parser.add_argument("--prompt", default=None, help="task prompt (default: episode task for dataset replay)")
    parser.add_argument(
        "--fps", type=float, default=None, help="control rate (default: 15, dataset replay: unlimited; <= 0: no limit)"
    )
    parser.add_argument("--num-chunks", type=int, default=0, help="number of action chunks to run (0: no limit)")
    parser.add_argument("--max-steps", type=int, default=0, help="number of control steps to run (0: no limit)")
    parser.add_argument("--execute-steps", type=int, default=0, help="chunk mode: actions executed per chunk (0: all)")
    parser.add_argument("--reactive-k", type=int, default=0, help="reactive mode: steps per update (0: latency + 1)")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.1,
        help="safety limit of |joint target - measured joint| per step [rad] (<= 0: off)",
    )
    replay = parser.add_argument_group("dataset replay (--source dataset)")
    replay.add_argument("--dataset", default=None, help="LeRobot v3.0 dataset directory")
    replay.add_argument("--episode", type=int, default=None, help="episode index (default: first episode)")
    replay.add_argument("--start-frame", type=int, default=0)
    replay.add_argument("--allow-missing-videos", action="store_true", help="black frames for missing videos")
    replay.add_argument("--tactile-cache-dir", default=None, help="tactile cache (default: same as training)")
    ros = parser.add_argument_group("real ROS observations (--source ros-readonly or ros)")
    ros.add_argument(
        "--ros-python",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "V3_Gripper/.venv-ros/bin/python"),
        help="separate Python 3.12 interpreter with ROS Jazzy and the gripper gRPC client",
    )
    ros.add_argument("--ros-setup", default="/opt/ros/jazzy/setup.bash")
    ros.add_argument("--ros-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "111")))
    ros.add_argument("--front-topic", default="/front/color/image_raw")
    ros.add_argument("--wrist-topic", default="/wrist/color/image_raw")
    ros.add_argument("--joint-topic", default="/joint_states")
    ros.add_argument("--tactile-prefix", default="/v3_gripper/tactile")
    ros.add_argument("--gripper-address", default="192.168.127.10:50008")
    ros.add_argument("--force-level", type=float, default=80.0, help="known commanded SDK setting, NOT measured force")
    ros.add_argument("--observation-max-age", type=float, default=0.5, help="reject stale sensor data [s]")
    ros.add_argument("--observation-timeout", type=float, default=10.0, help="wait for fresh observations [s]")
    control = parser.add_argument_group("hardware actions (--source ros --execute only)")
    control.add_argument("--execute", action="store_true", help="explicitly allow physical arm/gripper actions")
    control.add_argument(
        "--initialize-gripper", action="store_true", help="allow motor initialization if needed; may move"
    )
    control.add_argument("--joint-command-topic", default="/position_joint_trajectory_controller/joint_trajectory")
    control.add_argument("--controller-name", default="position_joint_trajectory_controller")
    control.add_argument("--max-joint-velocity", type=float, default=0.2, help="conservative joint speed cap [rad/s]")
    control.add_argument(
        "--max-target-deviation", type=float, default=0.25, help="reject larger model/state gaps [rad]"
    )
    control.add_argument("--command-timeout", type=float, default=0.5, help="independent worker watchdog deadline [s]")
    control.add_argument("--gripper-speed", type=int, default=20, help="SDK speed setting [1..100]")
    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--output-dir", default=None, help="write rollout.npz and summary.json")
    outputs.add_argument("--rerun", default=None, help="spawn | FILE.rrd | rerun+http://HOST:9876/proxy")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


def validate_execution_args(args: argparse.Namespace) -> None:
    live = args.source == "ros"
    if live != args.execute:
        raise ValueError(
            "Hardware execution requires both --source ros and --execute; readonly/fake/replay cannot execute."
        )
    if args.initialize_gripper and not live:
        raise ValueError("--initialize-gripper requires --source ros --execute.")
    if live:
        for name in ("fps", "max_joint_step", "max_joint_velocity", "max_target_deviation", "command_timeout"):
            value = getattr(args, name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Live execution requires positive finite --{name.replace('_', '-')}.")
        if not 5 <= args.fps <= 50:
            raise ValueError("Live execution requires --fps in 5..50 for short bounded trajectories.")
        if not 1 / args.fps < args.command_timeout <= 2:
            raise ValueError("--command-timeout must exceed one control period and be <= 2 seconds.")
        if not 1 <= args.gripper_speed <= 100:
            raise ValueError("--gripper-speed must be in 1..100.")


def run(args: argparse.Namespace, *, policy_client=None) -> dict[str, Any]:
    validate_execution_args(args)
    client = policy_client or websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    if "action_horizon" not in metadata:
        raise RuntimeError("The server does not report its action horizon; serve with scripts/serve_policy.py.")
    action_horizon = int(metadata["action_horizon"])
    mode = args.mode
    if mode == "auto":
        mode = "reactive" if metadata.get("reactive") else "chunk"
    if mode == "reactive" and not metadata.get("reactive"):
        raise RuntimeError(f"{metadata.get('config_name')} is not a force-reactive model; use --mode chunk.")
    reactive_k = args.reactive_k or int(metadata.get("cross_attention_latency", 0)) + 1
    if mode == "reactive" and not 1 <= reactive_k <= action_horizon:
        raise ValueError(f"--reactive-k must be in [1, {action_horizon}], got {reactive_k}")
    if not 0 <= args.execute_steps <= action_horizon:
        raise ValueError(f"--execute-steps must be in [0, {action_horizon}], got {args.execute_steps}")

    prompt = args.prompt or PROMPT
    if args.source == "ros":
        logging.warning("Warming the policy on synthetic observations BEFORE creating the hardware control worker.")
        run_policy(
            client,
            FakeAR3Robot(),
            mode=mode,
            action_horizon=action_horizon,
            prompt=prompt,
            tactile_shape=metadata.get("tactile_shape"),
            reactive_k=reactive_k,
            execute_steps=args.execute_steps,
            num_chunks=1,
            fps=0,
        )
    robot = make_robot(args, metadata)
    read_only = bool(getattr(robot, "read_only", False))
    if read_only:
        logging.warning("READ-ONLY preview: predicted actions will not be sent to the arm or gripper.")
    prompt = args.prompt or getattr(robot, "task", None) or PROMPT
    fps = args.fps if args.fps is not None else (0.0 if args.source == "dataset" else 15.0)
    episode = getattr(robot, "episode", None)
    logging.info(
        "Running %s mode%s with %s%s, prompt %r",
        mode,
        f" (k={reactive_k})" if mode == "reactive" else "",
        args.source,
        f" episode {episode}" if episode is not None else "",
        prompt,
    )
    recorder = EpisodeRecorder()
    callbacks: list[Callable[[Step], None]] = [log_request, recorder]
    viewer = None
    try:
        if args.rerun:
            application_id = "ar3_force_visuotactile_inference" + (f"/episode_{episode}" if episode is not None else "")
            if read_only:
                application_id += "/read_only"
            viewer = RerunLogger(args.rerun, application_id=application_id, read_only=read_only)
            callbacks.append(viewer)
        steps = run_policy(
            client,
            robot,
            mode=mode,
            action_horizon=action_horizon,
            prompt=prompt,
            tactile_shape=metadata.get("tactile_shape"),
            reactive_k=reactive_k,
            execute_steps=args.execute_steps,
            num_chunks=args.num_chunks,
            max_steps=args.max_steps,
            fps=fps,
            max_joint_step=args.max_joint_step,
            callbacks=callbacks,
        )
        if args.max_steps > 0 and steps >= args.max_steps:
            logging.info("Control step budget reached (--max-steps=%d); requesting normal shutdown.", args.max_steps)
    except KeyboardInterrupt:
        logging.info("Stopped.")
    finally:
        try:
            if callable(close := getattr(robot, "close", None)):
                close()
        finally:
            if viewer is not None:
                viewer.close()

    summary = {
        "config_name": metadata.get("config_name"),
        "mode": mode,
        "reactive_k": reactive_k if mode == "reactive" else None,
        "source": args.source,
        "read_only": read_only,
        "dataset": args.dataset if args.source == "dataset" else None,
        "episode": episode,
        "prompt": prompt,
        **recorder.summary(),
    }
    logging.info("Summary: %s", format_summary(summary))
    if args.output_dir:
        recorder.save(args.output_dir, summary)
        logging.info("Wrote %s", pathlib.Path(args.output_dir) / "summary.json")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
