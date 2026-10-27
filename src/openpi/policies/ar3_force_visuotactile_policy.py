"""Transforms for the Rokae AR3 force-controlled arm + Daimon tac G2 visuo-tactile gripper (muscleVLTA).

Dataset (LeRobot v3.0, 15 fps) features used for training:

* ``observation.images.front`` / ``observation.images.wrist``: 480x640 RGB videos.
* ``observation.state`` (27D): 7 joint positions [rad], gripper position (SDK units 0..1000, 0 = closed),
  7 joint torques [Nm], sensor_0 wrench, sensor_2 wrench. Only the first 8 entries are used as proprioception, the
  force channels are fed to the reactive force branch instead (late force injection, see LIFT).
* ``action`` (9D): 7 joint position targets [rad], gripper position target (0..1000), gripper force level (20..100).
* ``observation.arm_effort`` (7D) and ``observation.gripper_wrench.sensor_{0,2}`` (6D each, Fx Fy Fz Mx My Mz in each
  finger sensor's own SDK frame) form the 19D measured force sequence of the reactive branch. Optionally
  (``use_force_level``), the commanded gripper force level (``action[..., 8]`` during training,
  ``observation/force_level`` at inference) is appended after normalization with its fixed 20..100 range, giving a 20D
  force input (MODEL_FORCE_DIM). It is constant (80) in the recordings, so the v2 configs drop it (19D force input).
* ``observation.tactile_depth`` (optional, native sidecars): visuo-tactile depth images of the two fingers.

With ``history_frames`` > 0 the force / tactile sequences also hold that many readings before the current frame
(oldest first), so the first actions of a chunk see recent contact forces and the client carries the end of the
previous chunk into the next one.

The two finger sensors face each other, so their local frames are mirrored. The raw local frames are kept by default:
every force channel is normalized independently (quantile normalization) and mixed by the learned input layer of the
force encoder. Set ``finger_wrench_2_rotation`` (e.g. FINGER_WRENCH_2_TO_0_ROTATION, used by the v2 configs) to
express the sensor_2 wrench in the sensor_0 frame, so that both fingers' readings share one frame (a drag or bump
along the fingers has the same sign on both, a squeeze has opposite normal forces); the same rotation is then applied
during training and inference.
"""

import dataclasses

import cv2
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

NUM_ARM_JOINTS = 7
WRENCH_COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
ARM_JOINT_NAMES = tuple(f"joint{i}.pos" for i in range(1, NUM_ARM_JOINTS + 1))
# Proprioceptive state used by the model: the first 8 entries of ``observation.state``.
STATE_NAMES = (*ARM_JOINT_NAMES, "gripper.position_sdk")
ACTION_NAMES = (*ARM_JOINT_NAMES, "gripper.position_sdk", "gripper.force_level")
FORCE_NAMES = (
    *(f"joint{i}.torque_Nm" for i in range(1, NUM_ARM_JOINTS + 1)),
    *(f"sensor_0.{name}" for name in WRENCH_COMPONENTS),
    *(f"sensor_2.{name}" for name in WRENCH_COMPONENTS),
)
TACTILE_SENSORS = ("sensor_0", "sensor_2")

FORCE_LEVEL_NAME = "gripper.force_level"
# Range of the gripper force level command (SDK units), mapped linearly to [-1, 1].
FORCE_LEVEL_RANGE = (20.0, 100.0)
FORCE_LEVEL_ACTION_INDEX = ACTION_NAMES.index(FORCE_LEVEL_NAME)
GRIPPER_ACTION_INDEX = ACTION_NAMES.index("gripper.position_sdk")
# Closed / open gripper position targets (SDK units) of the recordings.
GRIPPER_RANGE = (0.0, 1000.0)
# Row-major rotation (180 degrees about y) mapping sensor_2 force / moment vectors into the sensor_0 frame: the finger
# sensors face each other, so their normals (z) and one shear axis (x) are anti-parallel. Estimated from closed-grip
# recordings: raw corr(Fz0, Fz2) = +0.58 (same squeeze seen in mirrored frames), corr(Fx0, Fx2) = -0.52.
FINGER_WRENCH_2_TO_0_ROTATION = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

STATE_DIM = len(STATE_NAMES)
ACTION_DIM = len(ACTION_NAMES)
# Measured force channels (normalized with the dataset statistics under the "wrench" key).
FORCE_DIM = len(FORCE_NAMES)
# Force input of the model: measured force channels + commanded gripper force level.
MODEL_FORCE_DIM = FORCE_DIM + 1

# Keys produced by the dataset repack transform (training) and expected from the robot client (inference).
FRONT_IMAGE_KEY = "observation/front_image"
WRIST_IMAGE_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
ARM_EFFORT_KEY = "observation/arm_effort"
FINGER_WRENCH_KEYS = ("observation/finger_wrench_0", "observation/finger_wrench_2")
TACTILE_KEY = "observation/tactile_depth"
# Commanded gripper force level (scalar or [horizon] sequence) sent by the robot client at inference.
FORCE_LEVEL_KEY = "observation/force_level"


def make_ar3_force_visuotactile_example(action_horizon: int = 15, tactile_shape: tuple[int, int] = (288, 384)) -> dict:
    """Creates a random inference example as sent by the robot client (single force / tactile readings)."""
    return {
        FRONT_IMAGE_KEY: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        WRIST_IMAGE_KEY: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        STATE_KEY: np.random.rand(STATE_DIM).astype(np.float32),
        ARM_EFFORT_KEY: np.random.randn(NUM_ARM_JOINTS).astype(np.float32),
        FINGER_WRENCH_KEYS[0]: np.random.randn(6).astype(np.float32),
        FINGER_WRENCH_KEYS[1]: np.random.randn(6).astype(np.float32),
        TACTILE_KEY: np.random.rand(len(TACTILE_SENSORS), *tactile_shape).astype(np.float32),
        FORCE_LEVEL_KEY: np.float32(80.0),
        "prompt": "Unplug the white charging plug from the white socket and place it on the table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _to_sequence(value, horizon: int, item_ndim: int, history: int = 0) -> np.ndarray:
    """Returns a [history + horizon, ...] float32 sequence, oldest first (the current frame is at index history).

    A single reading (inference) is repeated for every frame. Sequences hold ``history`` readings before the current
    one followed by at least the current reading; shorter sequences are padded with their last reading, which matches
    the clamping of sequences at the end of an episode during training.
    """
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == item_ndim:
        value = np.repeat(value[None], history + 1, axis=0)
    if value.ndim != item_ndim + 1 or value.shape[0] <= history:
        raise ValueError(
            f"Expected a reading with {item_ndim} dims or a sequence of {history} history readings followed by the "
            f"current reading(s), got {value.shape}"
        )
    length = history + horizon
    if value.shape[0] < length:
        value = np.concatenate([value, np.repeat(value[-1:], length - value.shape[0], axis=0)], axis=0)
    return value[:length]


@dataclasses.dataclass(frozen=True)
class AR3ForceVisuotactileInputs(transforms.DataTransformFn):
    """Converts AR3 dataset items / robot observations to the model input format (training and inference)."""

    # Determines which model will be used.
    model_type: _model.ModelType
    # Length of the force / tactile sequences after the current frame (the action horizon of the model).
    action_horizon: int = 15
    # Readings before the current frame in the force / tactile sequences (Pi0Config.force_history_frames): the
    # sequences hold history_frames + action_horizon frames, oldest first.
    history_frames: int = 0
    # Feed the 19D force sequence (arm effort + two finger wrenches) to the reactive force branch.
    use_force: bool = True
    # Feed the tactile depth images to the model (requires ``Pi0Config.tactile_in_use``).
    use_tactile: bool = False
    # Tactile image resolution expected by the model. Full resolution images (inference) are area-downsampled.
    tactile_resolution: tuple[int, int] = (72, 96)
    # Tactile depth images are mapped to log1p(gain * max(depth, 0)), which keeps the contact patches of both
    # sensors (raw SDK depth, roughly 0..0.3 and 0..1.4 in the example episode) in a similar O(1) range.
    tactile_gain: float = 10.0
    # Optional row-major 3x3 rotation that maps sensor_2 force / moment vectors into the sensor_0 frame.
    finger_wrench_2_rotation: tuple[float, ...] | None = None
    # Provide the commanded gripper force level ("force_level", appended to the force input by AR3AppendForceLevel).
    use_force_level: bool = True
    # Force level used at inference when the client does not send observation/force_level.
    default_force_level: float = 80.0

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data[FRONT_IMAGE_KEY])
        wrist_image = _parse_image(data[WRIST_IMAGE_KEY])
        # The dataset state is 27D (joints, gripper, torques, wrenches); the robot client may send only the first 8.
        state = np.asarray(data[STATE_KEY], dtype=np.float32)[..., :STATE_DIM]
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected at least {STATE_DIM} state dims ({STATE_NAMES}), got {state.shape}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            # Copy: the delta action transform modifies the actions in place.
            inputs["actions"] = np.array(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.use_force and (wrench := self._force(data)) is not None:
            inputs["wrench"] = wrench
            if self.use_force_level:
                inputs["force_level"] = self._force_level(data)
            if "wrench_mask" in data:
                inputs["wrench_mask"] = np.asarray(data["wrench_mask"], dtype=bool)

        if self.use_tactile and TACTILE_KEY in data:
            inputs["tactile"] = self._tactile(data[TACTILE_KEY])

        return inputs

    def _force(self, data: dict) -> np.ndarray | None:
        sequence = {"horizon": self.action_horizon, "item_ndim": 1, "history": self.history_frames}
        if "wrench" in data:
            # Pre-assembled 19D force readings (e.g. sent by the robot client).
            wrench = _to_sequence(data["wrench"], **sequence)
            if wrench.shape[-1] != FORCE_DIM:
                raise ValueError(f"Expected {FORCE_DIM}D force readings {FORCE_NAMES}, got {wrench.shape}")
            return wrench
        keys = (ARM_EFFORT_KEY, *FINGER_WRENCH_KEYS)
        if not all(key in data for key in keys):
            if any(key in data for key in keys):
                raise ValueError(f"Force inputs require all of {keys}, got {[key for key in keys if key in data]}")
            return None
        effort, wrench_0, wrench_2 = (_to_sequence(data[key], **sequence) for key in keys)
        if effort.shape[-1] != NUM_ARM_JOINTS or wrench_0.shape[-1] != 6 or wrench_2.shape[-1] != 6:
            raise ValueError(f"Unexpected force shapes: {effort.shape}, {wrench_0.shape}, {wrench_2.shape}")
        if self.finger_wrench_2_rotation is not None:
            rotation = np.asarray(self.finger_wrench_2_rotation, dtype=np.float32).reshape(3, 3)
            wrench_2 = np.concatenate([wrench_2[:, :3] @ rotation.T, wrench_2[:, 3:] @ rotation.T], axis=-1)
        return np.concatenate([effort, wrench_0, wrench_2], axis=-1)

    def _force_level(self, data: dict) -> np.ndarray:
        if FORCE_LEVEL_KEY in data:
            level = np.asarray(data[FORCE_LEVEL_KEY], dtype=np.float32).reshape(-1, 1)
            if level.shape[0] == 1:
                level = level[0]  # A single reading.
        elif "actions" in data:
            # Training: the commanded force level of every step of the chunk (absolute, unaffected by delta actions),
            # with the first one repeated for the history frames.
            level = np.asarray(data["actions"], dtype=np.float32)[
                ..., FORCE_LEVEL_ACTION_INDEX : FORCE_LEVEL_ACTION_INDEX + 1
            ]
            level = np.concatenate([np.repeat(level[:1], self.history_frames, axis=0), level], axis=0)
        else:
            level = np.full((1,), self.default_force_level, dtype=np.float32)
        return _to_sequence(level, self.action_horizon, item_ndim=1, history=self.history_frames)

    def _tactile(self, value) -> np.ndarray:
        tactile = _to_sequence(value, self.action_horizon, item_ndim=3, history=self.history_frames)
        height, width = self.tactile_resolution
        if tactile.shape[-2:] != (height, width):
            tactile = np.stack(
                [
                    [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in step]
                    for step in tactile
                ]
            )
        return np.log1p(self.tactile_gain * np.maximum(tactile, 0.0)).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class AR3AppendForceLevel(transforms.DataTransformFn):
    """Appends the commanded gripper force level, scaled to [-1, 1], to the normalized force input.

    Runs after normalization (model transforms): wrench [horizon, FORCE_DIM] -> [horizon, MODEL_FORCE_DIM].
    """

    def __call__(self, data: dict) -> dict:
        if "force_level" not in data:
            return data
        data = dict(data)
        low, high = FORCE_LEVEL_RANGE
        level = (np.asarray(data.pop("force_level"), dtype=np.float32) - low) / (high - low) * 2.0 - 1.0
        if "wrench" in data:
            data["wrench"] = np.concatenate([np.asarray(data["wrench"], dtype=np.float32), level], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class AR3ForceVisuotactileOutputs(transforms.DataTransformFn):
    """Converts model outputs back to AR3 actions (inference only)."""

    # Snap the gripper target to open / closed at this SDK position (e.g. 500 for a gripper classification head,
    # whose unnormalized output is p(open) * 1000); None keeps the predicted position.
    gripper_threshold: float | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.array(data["actions"][:, :ACTION_DIM])
        if self.gripper_threshold is not None:
            closed, opened = GRIPPER_RANGE
            gripper = actions[:, GRIPPER_ACTION_INDEX]
            actions[:, GRIPPER_ACTION_INDEX] = np.where(gripper >= self.gripper_threshold, opened, closed)
        return {"actions": actions}
