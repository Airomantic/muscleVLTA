import numpy as np
import pytest

from scripts.ar3_action_limits import JointLimits


def _urdf():
    return (
        "<robot>"
        + "".join(
            f'<joint name="joint{i}" type="revolute"><limit lower="-2" upper="2" velocity="1"/></joint>'
            for i in range(1, 8)
        )
        + "</robot>"
    )


def _limits():
    return JointLimits.from_urdf(_urdf(), [f"joint{i}" for i in range(1, 8)])


def _bound(action, **kwargs):
    return _limits().bound_action(
        action, np.zeros(7), period=0.1, max_step=0.03, max_velocity=0.2, max_deviation=0.25, **kwargs
    )


def test_bounds_joint_speed_and_rounds_gripper_commands():
    actual = _bound([0.1] * 7 + [500.1, 80.4])
    np.testing.assert_allclose(actual[:7], 0.02)
    np.testing.assert_array_equal(actual[7:], [500, 80])


def test_fast_command_bursts_cannot_exceed_speed_budget():
    actual = _bound([0.1] * 7 + [500, 80], previous_target=np.zeros(7), elapsed=0.01)
    np.testing.assert_allclose(actual[:7], 0.002)


@pytest.mark.parametrize(
    "action",
    [[3.0] * 7 + [500, 80], [0.3] * 7 + [500, 80], [float("nan")] * 9, [0.0] * 7 + [1001, 80]],
)
def test_rejects_invalid_or_implausible_targets(action):
    with pytest.raises(ValueError, match="limits|too far|finite|Gripper"):
        _bound(action)


def test_rejects_missing_or_wrong_robot_limits():
    with pytest.raises(ValueError, match="bounded revolute"):
        JointLimits.from_urdf(_urdf().replace('name="joint7"', 'name="other"'), [f"joint{i}" for i in range(1, 8)])
    with pytest.raises(ValueError, match="Invalid limits"):
        JointLimits.from_urdf(_urdf().replace('velocity="1"', 'velocity="0"'), [f"joint{i}" for i in range(1, 8)])


def test_external_joint_motion_cannot_be_hidden_by_clipping():
    with pytest.raises(ValueError, match="inconsistent"):
        _bound([0.1] * 7 + [500, 80], previous_target=np.full(7, 1.0), elapsed=0.1)
