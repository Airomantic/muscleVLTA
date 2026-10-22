"""Numerical limits for live AR3 targets. No device or ROS dependencies."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_urdf(cls, description: str, joint_names: Sequence[str]) -> "JointLimits":
        if len(joint_names) != 7 or len(set(joint_names)) != 7:
            raise ValueError("AR3 execution requires seven unique joint names.")
        robot = ET.fromstring(description)
        if robot.tag != "robot":
            raise ValueError("Expected a robot_description URDF.")
        joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
        values = []
        for name in joint_names:
            joint = joints.get(name)
            if joint is None or joint.get("type") != "revolute":
                raise ValueError(f"Missing bounded revolute joint in robot_description: {name}")
            limit = joint.find("limit")
            if limit is None or not all(key in limit.attrib for key in ("lower", "upper", "velocity")):
                raise ValueError(f"Missing position/velocity limits for {name}.")
            row = tuple(float(limit.attrib[key]) for key in ("lower", "upper", "velocity"))
            if not all(math.isfinite(value) for value in row) or row[0] >= row[1] or row[2] <= 0:
                raise ValueError(f"Invalid limits for {name}.")
            values.append(row)
        lower, upper, velocity = np.asarray(values, dtype=np.float64).T
        return cls(lower, upper, velocity)

    def bound_action(
        self,
        action,
        measured_joints,
        *,
        period: float,
        max_step: float,
        max_velocity: float,
        max_deviation: float,
        previous_target: np.ndarray | None = None,
        elapsed: float | None = None,
    ) -> np.ndarray:
        """Reject malformed/implausible targets, then cap changes relative to feedback and prior commands."""
        for value in (period, max_step, max_velocity, max_deviation):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Execution limits and period must be finite and positive.")
        target = np.asarray(action, dtype=np.float64)
        measured = np.asarray(measured_joints, dtype=np.float64)
        if (
            target.shape != (9,)
            or measured.shape != (7,)
            or not np.isfinite(target).all()
            or not np.isfinite(measured).all()
        ):
            raise ValueError("Expected a finite 9D action and seven measured joints.")
        if np.any(measured < self.lower) or np.any(measured > self.upper):
            raise ValueError("Measured joints are outside the robot's position limits.")
        if np.any(target[:7] < self.lower) or np.any(target[:7] > self.upper):
            raise ValueError("Model target is outside the robot's position limits.")
        if np.any(np.abs(target[:7] - measured) > max_deviation):
            raise ValueError("Model target is too far from measured joints; check state, units and checkpoint.")
        if not 0 <= target[7] <= 1000 or not 20 <= target[8] <= 100:
            raise ValueError("Gripper target must be position 0..1000 and force level 20..100.")
        velocity = np.minimum(self.velocity, max_velocity)
        delta = np.minimum(max_step, velocity * period)
        low, high = measured - delta, measured + delta
        if previous_target is not None:
            previous = np.asarray(previous_target, dtype=np.float64)
            if previous.shape != (7,) or not np.isfinite(previous).all():
                raise ValueError("Invalid previous joint target.")
            if elapsed is None or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Command timestamps must advance.")
            budget = velocity * min(elapsed, period)
            low = np.maximum(low, previous - budget)
            high = np.minimum(high, previous + budget)
            if np.any(low > high):
                raise ValueError("Measured motion is inconsistent with bounded prior targets; stopping.")
        bounded = target.copy()
        bounded[:7] = np.clip(target[:7], np.maximum(low, self.lower), np.minimum(high, self.upper))
        bounded[7:] = np.rint(bounded[7:])
        return bounded.astype(np.float32)
