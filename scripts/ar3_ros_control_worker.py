#!/usr/bin/env python3
"""Explicit AR3 execution worker; never used by ros-readonly.

Requires --execute, an already connected gripper, and the existing active ROS
position controller. Initialization additionally requires --initialize-gripper.
Each fresh observation authorizes ONE bounded target. No complete trajectory is
queued. A separate watchdog stops on missed commands, stale feedback or conflicts,
even when the model/debugger is paused. Stop means a short controller hold plus
an acknowledged gripper Stop, NOT a safety-rated emergency stop or arm power-off.
"""

import argparse
from collections.abc import Sequence
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO
import uuid

import numpy as np

if __package__:
    from scripts import ar3_ros_observation_worker as observations
    from scripts.ar3_action_limits import JointLimits
else:
    from ar3_action_limits import JointLimits
    import ar3_ros_observation_worker as observations

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    observations.add_observation_arguments(parser)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--initialize-gripper", action="store_true")
    parser.add_argument("--joint-command-topic", default="/position_joint_trajectory_controller/joint_trajectory")
    parser.add_argument("--controller-name", default="position_joint_trajectory_controller")
    parser.add_argument("--command-period", type=float, default=1 / 15)
    parser.add_argument("--max-joint-step", type=float, default=0.1)
    parser.add_argument("--max-joint-velocity", type=float, default=0.2)
    parser.add_argument("--max-target-deviation", type=float, default=0.25)
    parser.add_argument("--command-timeout", type=float, default=0.5)
    parser.add_argument("--gripper-speed", type=int, default=20)
    args = parser.parse_args(argv)
    observations.validate_observation_arguments(parser, args)
    if not args.execute:
        parser.error("Live control requires explicit --execute; no hardware was opened.")
    for name in ("command_period", "max_joint_step", "max_joint_velocity", "max_target_deviation", "command_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0.02 <= args.command_period <= 0.2:
        parser.error("--command-period must be 0.02..0.2 seconds (5..50 Hz)")
    if not args.command_period < args.command_timeout <= 2:
        parser.error("--command-timeout must exceed the command period and be <= 2 seconds")
    if not 1 <= args.gripper_speed <= 100:
        parser.error("--gripper-speed must be in 1..100")
    if not args.controller_name or "/" in args.controller_name:
        parser.error("--controller-name must be a nonempty controller name without '/'")
    if args.joint_command_topic != f"/{args.controller_name}/joint_trajectory":
        parser.error("--joint-command-topic must match the selected controller's joint_trajectory topic")
    return args


class GripperCommands:
    """Only this explicit worker binds write RPCs; it never disconnects another driver's CAN connection."""

    def __init__(self, args: argparse.Namespace):
        from DM_V3Gripper import gripper_pb2
        import grpc

        self.args = args
        self.pb = gripper_pb2
        self.rpc_error = grpc.RpcError
        self.channel = grpc.insecure_channel(args.gripper_address, options=(("grpc.enable_http_proxy", 0),))
        try:
            self.get_state = observations.bind_gripper_reader(self.channel, gripper_pb2.StateResponse)
            self.operations = {
                name: self.channel.unary_unary(
                    f"/gripper.GripperService/{name}", response_deserializer=gripper_pb2.Response.FromString
                )
                for name in (
                    "Enable",
                    "Initialization",
                    "SetTargetSpeed",
                    "SetTargetForce",
                    "SetTargetPosition",
                    "Stop",
                )
            }
        except (AttributeError, TypeError, ValueError, RuntimeError, self.rpc_error):
            self.channel.close()
            raise
        self.touched = False
        self.enabled = False
        self.force_level = args.force_level
        self.last_force: int | None = None
        self.last_position: int | None = None

    def read(self) -> observations.Sample:
        started = time.monotonic()
        try:
            state = self.get_state.future(b"", timeout=min(0.2, self.args.max_age)).result()
        except self.rpc_error as exc:
            raise RuntimeError(f"Gripper GetAllState failed: {exc}") from exc
        age = state.update_age_ms / 1000.0 + time.monotonic() - started
        if (
            not state.connected
            or not state.valid
            or state.stale
            or not math.isfinite(age)
            or not 0 <= age <= self.args.max_age
            or state.error != 0
        ):
            raise RuntimeError(
                f"Gripper is not connected/fresh/error-free: connected={state.connected}, valid={state.valid}, "
                f"stale={state.stale}, error={state.error}. Prepare its CAN connection with your approved driver."
            )
        return observations.Sample(state, started)

    def call(self, name: str, *, deadline: float, value: int | None = None) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Command deadline expired before gripper {name}")
        request = b"" if value is None else self.pb.ValueRequest(value=value).SerializeToString()
        self.touched = True
        try:
            response = self.operations[name](request, timeout=min(0.2, remaining))
        except self.rpc_error as exc:
            raise RuntimeError(f"Gripper {name} failed; command delivery is uncertain: {exc}") from exc
        if not response.success:
            raise RuntimeError(f"Gripper {name} rejected: {response.message}")

    def prepare(self) -> None:
        sample = self.read()
        if sample.value.init_state != 1:
            if not self.args.initialize_gripper:
                raise RuntimeError("Gripper is not initialized; --initialize-gripper additionally permits calibration.")
            if sample.value.init_state not in (0, 2):
                raise RuntimeError(f"Unknown gripper initialization state: {sample.value.init_state}")
            deadline = time.monotonic() + 15
            if sample.value.init_state == 0:
                self.call("Enable", deadline=deadline)
                self.call("Initialization", deadline=deadline)
            while sample.value.init_state != 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Gripper initialization exceeded 15 seconds")
                time.sleep(0.05)
                sample = self.read()
                if sample.value.init_state not in (0, 1, 2):
                    raise RuntimeError(f"Unknown gripper initialization state: {sample.value.init_state}")
        observations.gripper_position(
            sample.value, max_age=self.args.max_age, elapsed=time.monotonic() - sample.monotonic_time
        )

    def apply(self, action: np.ndarray, *, deadline: float) -> None:
        sample = self.read()
        observations.gripper_position(
            sample.value, max_age=self.args.max_age, elapsed=time.monotonic() - sample.monotonic_time
        )
        if not self.enabled:
            self.call("Enable", deadline=deadline)
            self.call("SetTargetSpeed", deadline=deadline, value=self.args.gripper_speed)
            self.enabled = True
        position, force = map(int, action[7:])
        if force != self.last_force:
            self.call("SetTargetForce", deadline=deadline, value=force)
            self.last_force = force
            self.force_level = force
        if position != self.last_position:
            self.call("SetTargetPosition", deadline=deadline, value=position)
            self.last_position = position

    def stop(self) -> None:
        if self.touched:
            self.call("Stop", deadline=time.monotonic() + 0.5)

    def close(self) -> None:
        self.channel.close()


def controller_positions(message: Any, joint_names: Sequence[str], field: str) -> np.ndarray:
    names = list(message.joint_names)
    values = list(getattr(message, field).positions)
    if len(names) != 7 or set(names) != set(joint_names) or len(values) != 7:
        raise observations.ObservationUnavailable("Controller joint names/positions do not match the seven AR3 joints")
    result = np.asarray([values[names.index(name)] for name in joint_names], dtype=np.float64)
    if not np.isfinite(result).all():
        raise observations.ObservationUnavailable("Nonfinite controller positions")
    return result


def controller_is_stopped(message: Any) -> bool:
    velocities = np.asarray(message.feedback.velocities)
    return velocities.shape == (7,) and np.isfinite(velocities).all() and np.max(np.abs(velocities)) <= 0.02


class ROSHardware:
    """Receive in a separate thread so feedback and stopping do not depend on stdin or model progress."""

    def __init__(self, args: argparse.Namespace, gripper: GripperCommands):
        self.args = args
        self.gripper = gripper
        self.samples: dict[str, observations.Sample] = {}
        self.description = ""
        self.controller: observations.Sample | None = None
        self.active_goals = False
        self.spin_error: str | None = None
        self.context = self.node = self.executor = self.publisher = self.thread = None
        self.closed = threading.Event()
        self.limits: JointLimits | None = None
        self.last_target: np.ndarray | None = None
        self.arm_touched = False

    def start(self) -> None:
        from action_msgs.msg import GoalStatusArray
        from control_msgs.msg import JointTrajectoryControllerState
        from controller_manager_msgs.srv import ListControllers
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy
        from rclpy.qos import QoSProfile
        from rclpy.qos import ReliabilityPolicy
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory

        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = Node(
            f"openpi_ar3_control_{uuid.uuid4().hex[:12]}",
            context=self.context,
            use_global_arguments=False,
            enable_rosout=False,
            start_parameter_services=False,
        )
        self.publisher = self.node.create_publisher(JointTrajectory, self.args.joint_command_topic, 1)
        observations.subscribe_observations(self.node, self.args, self.samples)
        feedback_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.node.create_subscription(String, "/robot_description", self.receive_description, latched_qos)
        self.node.create_subscription(
            JointTrajectoryControllerState,
            f"/{self.args.controller_name}/controller_state",
            self.receive_controller,
            feedback_qos,
        )
        self.node.create_subscription(
            GoalStatusArray,
            f"/{self.args.controller_name}/follow_joint_trajectory/_action/status",
            self.receive_goals,
            latched_qos,
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.spin, name="ar3-live-feedback", daemon=True)
        self.thread.start()
        client = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        try:
            if not client.wait_for_service(timeout_sec=self.args.timeout):
                raise TimeoutError("Controller manager list_controllers service is unavailable")
            future = client.call_async(ListControllers.Request())
            deadline = time.monotonic() + self.args.timeout
            while not future.done():
                if time.monotonic() >= deadline:
                    future.cancel()
                    raise TimeoutError("Controller manager readiness query timed out")
                time.sleep(0.01)
            response = future.result()
            controllers = [item for item in response.controller if item.name == self.args.controller_name]
            required = {f"{name}/position" for name in self.args.joint_names}
            if (
                len(controllers) != 1
                or controllers[0].state != "active"
                or controllers[0].type != "joint_trajectory_controller/JointTrajectoryController"
                or set(controllers[0].claimed_interfaces) != required
            ):
                raise RuntimeError(
                    "Selected controller must be active and exclusively claim all seven position interfaces"
                )
        finally:
            self.node.destroy_client(client)
        deadline = time.monotonic() + self.args.timeout
        while True:
            try:
                self.current_joints()
                if not self.description:
                    raise observations.ObservationUnavailable("Waiting for robot_description")
                self.limits = JointLimits.from_urdf(self.description, self.args.joint_names)
                break
            except observations.ObservationUnavailable:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def receive_description(self, message: Any) -> None:
        self.description = message.data

    def receive_controller(self, message: Any) -> None:
        self.controller = observations.Sample(message, time.monotonic())

    def receive_goals(self, message: Any) -> None:
        self.active_goals = any(item.status in (1, 2, 3) for item in message.status_list)

    def spin(self) -> None:
        try:
            while not self.closed.is_set() and self.context.ok():
                self.executor.spin_once(timeout_sec=0.02)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            self.spin_error = str(exc)
            logger.exception("Live ROS feedback thread failed")

    def check_ownership(self) -> None:
        own = (self.node.get_name(), self.node.get_namespace())
        publishers = self.node.get_publishers_info_by_topic(self.args.joint_command_topic)
        if any((item.node_name, item.node_namespace) != own for item in publishers):
            raise RuntimeError(
                "Another arm command publisher is present; stop teleoperation/replay before LIVE execution"
            )
        if len(publishers) != 1 or self.publisher.get_subscription_count() != 1:
            raise observations.ObservationUnavailable(
                "Require exactly one discovered command publisher and controller subscriber"
            )
        nodes = self.node.get_node_names_and_namespaces()
        if self.active_goals or any("gripper_teleop" in name or "cartesian_retargeting" in name for name, _ in nodes):
            raise RuntimeError("An action goal or teleoperation controller is active")
        if self.node.get_publishers_info_by_topic("/v3_gripper/action"):
            raise RuntimeError("Another gripper command owner is active")

    def current_joints(self) -> np.ndarray:
        if self.spin_error or not self.context.ok() or not self.thread.is_alive():
            raise RuntimeError(f"ROS feedback stopped: {self.spin_error}")
        self.check_ownership()
        now, wall = time.monotonic(), time.time()
        missing = [key for key in observations.ROS_INPUTS if key not in self.samples]
        if missing:
            raise observations.ObservationUnavailable(f"Missing ROS inputs: {', '.join(missing)}")
        for key in observations.ROS_INPUTS:
            observations.ros_source_age(self.samples[key], now=now, wall_now=wall, max_age=self.args.max_age)
        controller = self.controller
        if controller is None:
            raise observations.ObservationUnavailable("Missing controller feedback")
        observations.ros_source_age(controller, now=now, wall_now=wall, max_age=self.args.max_age)
        actual = controller_positions(controller.value, self.args.joint_names, "feedback")
        if self.last_target is None:
            reference = controller_positions(controller.value, self.args.joint_names, "reference")
            if (
                not controller_is_stopped(controller.value)
                or np.max(np.abs(actual - reference)) > self.args.max_joint_step
            ):
                raise RuntimeError(
                    "Controller must be idle near its measured position; do not take over an active move"
                )
        joints, _ = observations.ordered_joints(self.samples["joints"].value, self.args.joint_names)
        if np.max(np.abs(joints - actual)) > self.args.max_target_deviation:
            raise RuntimeError("Controller feedback and joint_states disagree")
        if self.last_target is not None and np.max(np.abs(actual - self.last_target)) > self.args.max_target_deviation:
            raise RuntimeError("Arm tracking error exceeds --max-target-deviation")
        return actual

    def observe(self) -> tuple[dict[str, np.ndarray], float]:
        self.current_joints()
        gripper = self.gripper.read()
        samples = self.samples.copy()
        now, wall = time.monotonic(), time.time()
        result = observations.make_observation(
            samples,
            gripper,
            now=now,
            wall_now=wall,
            max_age=self.args.max_age,
            force_level=self.gripper.force_level,
            joint_names=self.args.joint_names,
        )
        ages = [gripper.value.update_age_ms / 1000.0 + now - gripper.monotonic_time]
        for sample in samples.values():
            ages.extend(
                (
                    now - sample.monotonic_time,
                    observations.ros_source_age(sample, now=now, wall_now=wall, max_age=self.args.max_age),
                )
            )
        return result, now + self.args.max_age - max(ages)

    def publish(self, target: np.ndarray, *, hold: bool = False) -> float:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        if hold:
            point.velocities = [0.0] * 7
        nanoseconds = round(self.args.command_period * 1_000_000_000)
        point.time_from_start = Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)
        message = JointTrajectory(joint_names=list(self.args.joint_names), points=[point])
        self.arm_touched = True
        started = time.monotonic()
        self.publisher.publish(message)
        self.last_target = target.copy()
        return started

    def wait_target(self, target: np.ndarray, started: float, deadline: float, *, stopping: bool = False) -> None:
        while time.monotonic() < deadline:
            sample = self.controller
            if sample is not None and sample.monotonic_time > started:
                observations.ros_source_age(
                    sample, now=time.monotonic(), wall_now=time.time(), max_age=self.args.max_age
                )
                reference = controller_positions(sample.value, self.args.joint_names, "reference")
                if stopping:
                    actual = controller_positions(sample.value, self.args.joint_names, "feedback")
                    if (
                        not controller_is_stopped(sample.value)
                        or np.max(np.abs(actual - target)) > self.args.max_joint_step
                    ):
                        time.sleep(0.005)
                        continue
                if np.allclose(reference, target, rtol=0, atol=1e-4):
                    return
            time.sleep(0.005)
        raise TimeoutError("Controller did not acknowledge the hold/target before its deadline")

    def hold(self) -> None:
        if not self.arm_touched:
            return
        try:
            sample = self.controller
            if sample is None:
                raise observations.ObservationUnavailable("No controller feedback for hold")
            observations.ros_source_age(sample, now=time.monotonic(), wall_now=time.time(), max_age=self.args.max_age)
            target = controller_positions(sample.value, self.args.joint_names, "feedback")
            if self.limits is None or np.any(target < self.limits.lower) or np.any(target > self.limits.upper):
                raise observations.ObservationUnavailable("Controller hold feedback is outside joint limits")
        except observations.ObservationUnavailable as exc:
            logger.error("Fresh hold position unavailable (%s); retaining the last bounded target", exc)
            if self.last_target is None:
                raise RuntimeError("No known arm target to hold") from exc
            target = self.last_target
        started = self.publish(target, hold=True)
        self.wait_target(target, started, time.monotonic() + 1, stopping=True)

    def close(self) -> None:
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1)
        if self.node is not None:
            self.node.destroy_node()
        if self.context is not None:
            import rclpy

            rclpy.try_shutdown(context=self.context)


class ControlSession:
    def __init__(
        self, args: argparse.Namespace, hardware: ROSHardware, gripper: GripperCommands, *, clock=time.monotonic
    ):
        self.args, self.hardware, self.gripper, self.clock = args, hardware, gripper, clock
        self.lock = threading.RLock()
        self.pending: tuple[int, float] | None = None
        self.sequence = 0
        self.last_command: float | None = None
        self.previous_target: np.ndarray | None = None
        self.halted = False
        self.stop_error: str | None = None
        self.watchdog_exit = threading.Event()
        self.watchdog: threading.Thread | None = None

    def require_running(self) -> None:
        if self.halted:
            raise RuntimeError(f"Control is stopped; a new session is required. {self.stop_error or ''}")
        if self.last_command is not None and self.clock() - self.last_command >= self.args.command_timeout:
            raise TimeoutError("Command watchdog expired; paused inference cannot resume motion")

    def observe(self) -> dict[str, np.ndarray]:
        with self.lock:
            self.require_running()
            self.pending = None
        result, deadline = self.hardware.observe()
        with self.lock:
            self.require_running()
            self.sequence += 1
            self.pending = (self.sequence, deadline)
            return {**result, "__observation_id__": np.asarray(self.sequence, dtype=np.int64)}

    def execute(self, identifier: Any, action: Any) -> dict[str, np.ndarray]:
        with self.lock:
            self.require_running()
            pending, self.pending = self.pending, None
            if type(identifier) is not int or pending is None or identifier != pending[0]:
                raise ValueError("Execute requires the most recent, unused observation ID")
            now = self.clock()
            if now >= pending[1]:
                raise TimeoutError("Prediction observation expired before execution")
            measured = self.hardware.current_joints()
            if self.hardware.limits is None:
                raise RuntimeError("Robot joint limits are unavailable")
            applied = self.hardware.limits.bound_action(
                action,
                measured,
                period=self.args.command_period,
                max_step=self.args.max_joint_step,
                max_velocity=self.args.max_joint_velocity,
                max_deviation=self.args.max_target_deviation,
                previous_target=self.previous_target,
                elapsed=None if self.last_command is None else now - self.last_command,
            )
            self.last_command = now
            deadline = min(pending[1], now + self.args.command_timeout)
            self.gripper.apply(applied, deadline=deadline)
            if self.clock() >= deadline:
                raise TimeoutError("Command deadline expired before arm publication")
            self.hardware.current_joints()
            started = self.hardware.publish(applied[:7])
            self.hardware.wait_target(applied[:7], started, deadline)
            self.previous_target = applied[:7].copy()
            return {"__observation_id__": np.asarray(identifier), "__executed_action__": applied}

    def stop(self, reason: str, *, requested: bool = False) -> None:
        with self.lock:
            if not self.halted:
                self.halted = True
                self.pending = None
                logger.log(logging.INFO if requested else logging.WARNING, "Stopping LIVE execution: %s", reason)
                errors = []
                # Both must be attempted, even if one actuator's transport has failed.
                for name, stop in (("gripper", self.gripper.stop), ("arm", self.hardware.hold)):
                    try:
                        stop()
                    except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                        errors.append(f"{name}: {exc}")
                        logger.exception("%s stop was NOT confirmed; inspect hardware / emergency stop", name)
                self.stop_error = "; ".join(errors) or None
            if self.stop_error:
                raise RuntimeError(f"STOP NOT CONFIRMED: {self.stop_error}")

    def check_watchdog(self) -> None:
        with self.lock:
            if self.halted or self.last_command is None:
                return
            try:
                self.require_running()
                self.hardware.current_joints()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                self.stop(str(exc))

    def start_watchdog(self) -> None:
        def watch() -> None:
            try:
                while not self.watchdog_exit.wait(min(0.02, self.args.command_timeout / 4)):
                    self.check_watchdog()
            except (RuntimeError, ValueError, OSError):
                logger.exception("Watchdog stop failed; use the physical emergency stop if needed")

        self.watchdog = threading.Thread(target=watch, name="ar3-command-watchdog", daemon=True)
        self.watchdog.start()

    def close(self) -> None:
        try:
            self.stop("worker closing")
        finally:
            self.watchdog_exit.set()
            if self.watchdog is not None:
                self.watchdog.join(timeout=2)


def stopped_error(session: ControlSession | None, error: str) -> dict[str, np.ndarray]:
    stopped = True
    if session is not None:
        try:
            session.stop(error)
        except RuntimeError as exc:
            stopped = False
            error += f"; {exc}"
    return {"__error__": np.asarray(error), "__stopped__": np.asarray(stopped)}


def serve_requests(source: BinaryIO, output: BinaryIO, session: ControlSession) -> int:
    observations.write_frame(output, {"__ready__": np.asarray(a=True), "__control__": np.asarray(a=True)})
    try:
        while line := source.readline(4097):
            if len(line) > 4096 or not line.endswith(b"\n"):
                raise ValueError("Live requests must be newline-delimited JSON, at most 4096 bytes")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Expected a request object")
            operation = request.get("op")
            if operation == "close" and set(request) == {"op"}:
                session.stop("client closed", requested=True)
                observations.write_frame(output, {"__closed__": np.asarray(a=True), "__stopped__": np.asarray(a=True)})
                return 0
            if operation == "observe" and set(request) == {"op"}:
                result = session.observe()
            elif operation == "execute" and set(request) == {"op", "observation_id", "action"}:
                result = session.execute(request["observation_id"], request["action"])
            else:
                raise ValueError("Unsupported live request or fields")
            observations.write_frame(output, result)
        session.stop("client pipe closed")
        return 0
    except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
        logger.exception("Live request failed; stopping before responding")
        observations.write_frame(output, stopped_error(session, str(exc)))
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = observations.reserve_protocol_stdout()
    gripper = hardware = session = None
    lock = None
    handlers = {}
    status = 1
    try:
        path = Path(tempfile.gettempdir()) / f"openpi-ar3-control-{os.getuid()}.lock"
        lock = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def interrupt(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
        logger.warning("LIVE worker explicitly enabled. Keep the physical emergency stop accessible.")
        gripper = GripperCommands(args)
        hardware = ROSHardware(args, gripper)
        hardware.start()
        session = ControlSession(args, hardware, gripper)
        gripper.prepare()
        hardware.observe()
        session.start_watchdog()
        status = serve_requests(sys.stdin.buffer, output, session)
    except KeyboardInterrupt:
        logger.warning("LIVE worker interrupted; stopping hardware")
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
        logger.exception("LIVE startup/runtime failed")
        observations.write_frame(output, stopped_error(session, str(exc)))
    finally:
        for signum in handlers:
            signal.signal(signum, signal.SIG_IGN)
        if session is not None:
            try:
                session.close()
            except RuntimeError:
                logger.exception("Worker exit could not confirm actuator stop")
                status = 1
        for resource in (hardware, gripper):
            if resource is not None:
                try:
                    resource.close()
                except (ImportError, RuntimeError, TypeError, AttributeError, OSError):
                    logger.exception("Failed to close owned LIVE resources")
                    status = 1
        if lock is not None:
            os.close(lock)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        output.close()
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    raise SystemExit(main())
