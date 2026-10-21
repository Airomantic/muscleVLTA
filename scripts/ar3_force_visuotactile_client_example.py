"""Example robot client for the muscleVLTA AR3 force + visuo-tactile policy (server: serve_ar3_force_visuotactile.sh).

Run the server on the GPU machine, then on the robot PC (needs only numpy and packages/openpi-client):

    pip install -e packages/openpi-client
    python scripts/ar3_force_visuotactile_client_example.py --host 127.0.0.1 --port 8000

The script drives FakeAR3Robot (random sensor data, actions are only stored). Replace its methods with the Rokae AR3,
Daimon tac G2 gripper and camera drivers. Observation sent to the server (layout of the training data, see
src/openpi/policies/ar3_force_visuotactile_policy.py):

    key                           shape / dtype                meaning
    observation/front_image       (480, 640, 3) uint8          front RGB camera
    observation/wrist_image       (480, 640, 3) uint8          wrist RGB camera
    observation/state             (8,) float32                 7 joint positions [rad] + gripper position (SDK units
                                                               0..1000, 0 = closed); the 27D dataset state also works
    observation/arm_effort        (7,) or (t, 7) float32       measured joint torques [Nm]
    observation/finger_wrench_0   (6,) or (t, 6) float32       sensor_0 finger wrench Fx Fy Fz [N] Mx My Mz [Nm]
    observation/finger_wrench_2   (6,) or (t, 6) float32       sensor_2 finger wrench (its own, mirrored SDK frame)
    observation/tactile_depth     (2, H, W) or (t, 2, H, W)    tactile depth of sensor_0 and sensor_2 (raw SDK depth)
    observation/force_level       scalar or (t,) float32       current gripper force level command (20..100)
    prompt                        str                          task instruction (the server has a default)

A single reading is used for the whole chunk; (t, ...) arrays are readings of consecutive steps since the chunk start.
Every response contains actions of shape (n, 9) at the 15 Hz dataset rate:

    actions[:, :7]   absolute joint position targets [rad]
    actions[:, 7]    gripper position target (0..1000, 0 = closed)
    actions[:, 8]    gripper force level command (20..100)

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
"""

from __future__ import annotations

import argparse
import logging
import time

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
NUM_JOINTS = 7
ACTION_DIM = 9
GRIPPER_POSITION_RANGE = (0.0, 1000.0)
FORCE_LEVEL_RANGE = (20.0, 100.0)


class FakeAR3Robot:
    """Placeholder for the AR3 arm, the tac G2 gripper and the cameras. Replace each method by the real drivers."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.joints = np.zeros(NUM_JOINTS, np.float32)  # rad
        self.gripper_position = 1000.0  # SDK units 0..1000, 0 = closed
        self.force_level = 80.0  # gripper force level command 20..100

    def read_cameras(self) -> dict:
        """Front and wrist RGB images, (480, 640, 3) uint8."""
        return {
            FRONT_IMAGE_KEY: self._rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
            WRIST_IMAGE_KEY: self._rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        }

    def read_state(self) -> np.ndarray:
        """7 joint positions [rad] + gripper position (0..1000)."""
        return np.append(self.joints, self.gripper_position).astype(np.float32)

    def read_force_tactile(self) -> dict:
        """Contact sensors, read once per control step (same frames / units as the recordings)."""
        return {
            ARM_EFFORT_KEY: self._rng.normal(0.0, 1.0, NUM_JOINTS).astype(np.float32),  # joint torques [Nm]
            FINGER_WRENCH_KEYS[0]: self._rng.normal(0.0, 0.5, 6).astype(np.float32),  # sensor_0 Fx..Mz
            FINGER_WRENCH_KEYS[1]: self._rng.normal(0.0, 0.5, 6).astype(np.float32),  # sensor_2 Fx..Mz
            TACTILE_KEY: np.abs(self._rng.normal(0.0, 0.05, (2, 288, 384))).astype(np.float32),  # sensor_0, sensor_2
        }

    def execute(self, action: np.ndarray) -> None:
        """Commands one action: 7 joint position targets [rad], gripper position (0..1000), force level (20..100).

        On the real robot, clip the targets to the joint limits and limit the change per step before sending them.
        """
        self.joints = np.asarray(action[:NUM_JOINTS], np.float32)
        self.gripper_position = float(action[NUM_JOINTS])
        self.force_level = float(action[NUM_JOINTS + 1])


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


def downsample_tactile(depth: np.ndarray, shape: tuple[int, int] | None) -> np.ndarray:
    """Area-downsamples tactile depth images to the model resolution before sending (288x384 -> 72x96: 16x less data).

    Equals the server-side cv2.INTER_AREA resize for integer factors; other sizes are sent unchanged (resized by the
    server).
    """
    depth = np.asarray(depth, np.float32)
    if shape is None:
        return depth
    height, width = shape
    in_height, in_width = depth.shape[-2:]
    if (in_height, in_width) == (height, width) or in_height % height or in_width % width:
        return depth
    blocks = depth.reshape(*depth.shape[:-2], height, in_height // height, width, in_width // width)
    return blocks.mean(axis=(-3, -1), dtype=np.float32)


def read_readings(robot: FakeAR3Robot, tactile_shape: tuple[int, int] | None) -> dict:
    """Force / tactile readings of the current step (+ the current gripper force level command)."""
    readings = dict(robot.read_force_tactile())
    if TACTILE_KEY in readings:
        readings[TACTILE_KEY] = downsample_tactile(readings[TACTILE_KEY], tactile_shape)
    readings[FORCE_LEVEL_KEY] = np.float32(robot.force_level)
    return readings


def make_observation(robot: FakeAR3Robot, readings: dict, prompt: str) -> dict:
    return {**robot.read_cameras(), STATE_KEY: robot.read_state(), **readings, "prompt": prompt}


def clip_action(action: np.ndarray) -> np.ndarray:
    """Clips the gripper position and force level to the ranges the gripper accepts."""
    action = np.array(action, dtype=np.float32)
    action[NUM_JOINTS] = np.clip(action[NUM_JOINTS], *GRIPPER_POSITION_RANGE)
    action[NUM_JOINTS + 1] = np.clip(action[NUM_JOINTS + 1], *FORCE_LEVEL_RANGE)
    return action


def log_actions(name: str, result: dict, round_trip_s: float) -> None:
    actions = np.asarray(result["actions"])
    if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected (n, {ACTION_DIM}) actions, got {actions.shape}")
    server_ms = result.get("server_timing", {}).get("infer_ms", float("nan"))
    logging.info(
        "%s: %d actions, round trip %.1f ms (server %.1f ms) | first joints [rad] %s | gripper %.1f | force level %.1f",
        name,
        len(actions),
        round_trip_s * 1000,
        server_ms,
        np.array2string(actions[0, :NUM_JOINTS], precision=3, suppress_small=True),
        actions[0, NUM_JOINTS],
        actions[0, NUM_JOINTS + 1],
    )


def run_chunk_mode(client, robot: FakeAR3Robot, args: argparse.Namespace, metadata: dict) -> None:
    rate = Rate(args.fps)
    execute_steps = args.execute_steps or metadata["action_horizon"]
    chunk = 0
    while not args.num_chunks or chunk < args.num_chunks:
        start = time.monotonic()
        result = client.infer(make_observation(robot, read_readings(robot, metadata.get("tactile_shape")), args.prompt))
        log_actions(f"chunk {chunk}", result, time.monotonic() - start)
        for action in np.asarray(result["actions"])[:execute_steps]:
            robot.execute(clip_action(action))
            rate.sleep()
        chunk += 1


def run_reactive_mode(client, robot: FakeAR3Robot, args: argparse.Namespace, metadata: dict) -> None:
    rate = Rate(args.fps)
    horizon = metadata["action_horizon"]
    tactile_shape = metadata.get("tactile_shape")
    k = args.reactive_k or metadata["cross_attention_latency"] + 1
    chunk = 0
    while not args.num_chunks or chunk < args.num_chunks:
        # Chunk start: full observation, the VLM runs once for this chunk.
        readings = read_readings(robot, tactile_shape)
        history = {key: [value] for key, value in readings.items()}
        start = time.monotonic()
        result = client.reactive_replan(make_observation(robot, readings, args.prompt), k=k)
        log_actions(f"chunk {chunk} replan", result, time.monotonic() - start)
        step = 0
        while True:
            for action in np.asarray(result["actions"]):
                robot.execute(clip_action(action))
                rate.sleep()
                step += 1
                if step < horizon:
                    # One reading per step, taken at the start of the step (as in the recordings).
                    for key, value in read_readings(robot, tactile_shape).items():
                        history[key].append(value)
            if step >= horizon:
                break
            # Re-run the reactive expert with all readings since the chunk start (images / state stay the replan ones).
            start = time.monotonic()
            result = client.reactive_step({key: np.stack(values) for key, values in history.items()}, k=k)
            if result["start_step"] != step:
                raise RuntimeError(f"Server is at step {result['start_step']}, client at step {step}")
            log_actions(f"chunk {chunk} step {step}", result, time.monotonic() - start)
        chunk += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("auto", "chunk", "reactive"), default="auto")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--fps", type=float, default=15.0, help="control rate of the recordings (<= 0: no limit)")
    parser.add_argument("--num-chunks", type=int, default=0, help="number of action chunks to run (0: until Ctrl-C)")
    parser.add_argument("--execute-steps", type=int, default=0, help="chunk mode: actions executed per chunk (0: all)")
    parser.add_argument("--reactive-k", type=int, default=0, help="reactive mode: steps per update (0: latency + 1)")
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    if "action_horizon" not in metadata:
        raise RuntimeError("The server does not report its action horizon; serve with scripts/serve_policy.py.")
    mode = args.mode
    if mode == "auto":
        mode = "reactive" if metadata.get("reactive") else "chunk"
    if mode == "reactive" and not metadata.get("reactive"):
        raise RuntimeError(f"{metadata.get('config_name')} is not a force-reactive model; use --mode chunk.")

    robot = FakeAR3Robot()
    try:
        if mode == "reactive":
            run_reactive_mode(client, robot, args, metadata)
        else:
            run_chunk_mode(client, robot, args, metadata)
    except KeyboardInterrupt:
        logging.info("Stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
