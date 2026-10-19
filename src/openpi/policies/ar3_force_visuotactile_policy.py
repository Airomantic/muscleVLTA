"""Transforms for the Rokae AR3 force-controlled arm + Daimon tac G2 visuo-tactile gripper (muscleVLTA).

Dataset (LeRobot v3.0, 15 fps) features used for training:

* ``observation.images.front`` / ``observation.images.wrist``: 480x640 RGB videos.
* ``observation.state`` (27D): 7 joint positions [rad], gripper position (SDK units 0..1000, 0 = closed),
  7 joint torques [Nm], sensor_0 wrench, sensor_2 wrench. Only the first 8 entries are used as proprioception, the
  force channels are fed to the reactive force branch instead (late force injection, see LIFT).
* ``action`` (9D): 7 joint position targets [rad], gripper position target (0..1000), gripper force level (20..100).
* ``observation.arm_effort`` (7D) and ``observation.gripper_wrench.sensor_{0,2}`` (6D each, Fx Fy Fz Mx My Mz in each
  finger sensor's own SDK frame) form the 19D force sequence of the reactive branch.
* ``observation.tactile_depth`` (optional, native sidecars): visuo-tactile depth images of the two fingers.

The two finger sensors face each other, so their local frames are mirrored. The raw local frames are kept by default:
every force channel is normalized independently (quantile normalization) and mixed by the learned input layer of the
force encoder, so a fixed sign / axis permutation between the frames does not change what the model can learn. Set
``finger_wrench_2_rotation`` to express the sensor_2 wrench in the sensor_0 frame if a physically aligned frame is
preferred (the same rotation is then applied during training and inference).
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

STATE_DIM = len(STATE_NAMES)
ACTION_DIM = len(ACTION_NAMES)
FORCE_DIM = len(FORCE_NAMES)

# Keys produced by the dataset repack transform (training) and expected from the robot client (inference).
FRONT_IMAGE_KEY = "observation/front_image"
WRIST_IMAGE_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
ARM_EFFORT_KEY = "observation/arm_effort"
FINGER_WRENCH_KEYS = ("observation/finger_wrench_0", "observation/finger_wrench_2")
TACTILE_KEY = "observation/tactile_depth"


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
        "prompt": "Unplug the white charging plug from the white socket and place it on the table",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _to_sequence(value, horizon: int, item_ndim: int) -> np.ndarray:
    """Returns a [horizon, ...] float32 sequence.

    Single readings (inference) are repeated and shorter sequences are padded with their last reading, which matches
    the clamping of sequences at the end of an episode during training.
    """
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == item_ndim:
        value = value[None]
    if value.ndim != item_ndim + 1 or value.shape[0] == 0:
        raise ValueError(f"Expected a reading with {item_ndim} dims or a sequence of readings, got {value.shape}")
    if value.shape[0] < horizon:
        value = np.concatenate([value, np.repeat(value[-1:], horizon - value.shape[0], axis=0)], axis=0)
    return value[:horizon]


@dataclasses.dataclass(frozen=True)
class AR3ForceVisuotactileInputs(transforms.DataTransformFn):
    """Converts AR3 dataset items / robot observations to the model input format (training and inference)."""

    # Determines which model will be used.
    model_type: _model.ModelType
    # Length of the force / tactile sequences (the action horizon of the model).
    action_horizon: int = 15
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
            if "wrench_mask" in data:
                inputs["wrench_mask"] = np.asarray(data["wrench_mask"], dtype=bool)

        if self.use_tactile and TACTILE_KEY in data:
            inputs["tactile"] = self._tactile(data[TACTILE_KEY])

        return inputs

    def _force(self, data: dict) -> np.ndarray | None:
        if "wrench" in data:
            # Pre-assembled 19D force readings (e.g. sent by the robot client).
            wrench = _to_sequence(data["wrench"], self.action_horizon, item_ndim=1)
            if wrench.shape[-1] != FORCE_DIM:
                raise ValueError(f"Expected {FORCE_DIM}D force readings {FORCE_NAMES}, got {wrench.shape}")
            return wrench
        keys = (ARM_EFFORT_KEY, *FINGER_WRENCH_KEYS)
        if not all(key in data for key in keys):
            if any(key in data for key in keys):
                raise ValueError(f"Force inputs require all of {keys}, got {[key for key in keys if key in data]}")
            return None
        effort, wrench_0, wrench_2 = (_to_sequence(data[key], self.action_horizon, item_ndim=1) for key in keys)
        if effort.shape[-1] != NUM_ARM_JOINTS or wrench_0.shape[-1] != 6 or wrench_2.shape[-1] != 6:
            raise ValueError(f"Unexpected force shapes: {effort.shape}, {wrench_0.shape}, {wrench_2.shape}")
        if self.finger_wrench_2_rotation is not None:
            rotation = np.asarray(self.finger_wrench_2_rotation, dtype=np.float32).reshape(3, 3)
            wrench_2 = np.concatenate([wrench_2[:, :3] @ rotation.T, wrench_2[:, 3:] @ rotation.T], axis=-1)
        return np.concatenate([effort, wrench_0, wrench_2], axis=-1)

    def _tactile(self, value) -> np.ndarray:
        tactile = _to_sequence(value, self.action_horizon, item_ndim=3)
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
class AR3ForceVisuotactileOutputs(transforms.DataTransformFn):
    """Converts model outputs back to AR3 actions (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
