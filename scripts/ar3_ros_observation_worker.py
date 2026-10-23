#!/usr/bin/env python3
"""Read-only AR3 ROS observations, isolated from the model's Python environment.

Run with the ROS Jazzy Python 3.12 environment containing G3_client. ROS_DOMAIN_ID
is inherited. Nothing here connects/enables/initializes the motor or sends actions:
the only gripper RPC is GetAllState with an empty protobuf request. An independent,
approved driver must already maintain valid, initialized motor feedback.

stdin: newline-delimited {"op":"observe"} / {"op":"close"} requests.
stdout: eight-byte big-endian length followed by an uncompressed, pickle-free NPZ.
The initial __ready__ frame confirms local subscription/channel setup, NOT sensor
or motor readiness. Errors are __error__ string scalars and terminate with code 1.
--check-gripper instead validates one real motor status and exits with text output,
without ROS subscriptions, protocol frames, or any motor commands.

Freshness policy: every ROS message must have both a recent host monotonic receipt
and a nonzero, normalized acquisition header stamp in the host wall-clock domain.
Both ages must be <= --max-age; stamps farther into the future than this budget
are rejected. Source timestamp spread (including gripper age) is also bounded by
--max-age. Zero/unstamped headers and simulated/device-uptime clocks are rejected,
not silently replaced. This is bounded software alignment, not hardware sync.
Gripper age is server update_age_ms PLUS the entire local elapsed time since the
read request started, conservatively including RPC latency. Invalid new messages
replace previous ones; stale/missing/invalid inputs never become zero-filled data.

Images are RGB uint8 HWC; depth is raw SDK float32, sensor_0 then sensor_2.
Wrenches remain raw local-frame Fx,Fy,Fz,Mx,My,Mz. State is joint1..joint7 in
radians, then unscaled SDK gripper position 0..1000 (closed..open). --force-level
is explicitly supplied command-setting metadata (20..100), NOT measured force
or a verified current setting: StateResponse exposes motor current but no force
level. This worker neither reads current as force nor writes the force setting.
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import io
import json
import math
import os
import struct
import sys
import time
from typing import Any, BinaryIO

import numpy as np

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ROS_INPUTS = ("joints", "front", "wrist", "depth_0", "depth_2", "wrench_0", "wrench_2")
GRIPPER_RPC = "/gripper.GripperService/GetAllState"
GRIPPER_PREPARATION = (
    "Prepare valid, initialized gripper feedback through your approved driver; "
    "this read-only worker will not connect or initialize the motor."
)


class ObservationUnavailable(RuntimeError):  # noqa: N818
    """Required real observations are missing, invalid, or no longer fresh."""


@dataclass(frozen=True)
class Sample:
    value: Any
    monotonic_time: float


def _finite_float32(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ObservationUnavailable(f"{name} must contain finite values with shape {shape}")
    return array


def _image_buffer(message: Any, bytes_per_pixel: int) -> memoryview:
    if message.height <= 0 or message.width <= 0 or message.step < message.width * bytes_per_pixel:
        raise ObservationUnavailable("Image has invalid dimensions or row step")
    data = memoryview(message.data).cast("B")
    if data.nbytes != message.height * message.step:
        raise ObservationUnavailable("Image payload length does not match height * step")
    return data


def rgb_image(message: Any) -> np.ndarray:
    """Decode rgb8/bgr8 without cv_bridge, retaining neither padding nor BGR order."""
    if message.encoding not in ("rgb8", "bgr8"):
        raise ObservationUnavailable(f"Unsupported RGB encoding: {message.encoding!r}")
    data = _image_buffer(message, 3)
    image = np.ndarray((message.height, message.width, 3), dtype=np.uint8, buffer=data, strides=(message.step, 3, 1))
    if message.encoding == "bgr8":
        image = image[..., ::-1]
    return image.copy(order="C")


def tactile_depth(message: Any) -> np.ndarray:
    """Decode raw 32FC1 SDK depth, including endian conversion and padded rows."""
    if message.encoding != "32FC1":
        raise ObservationUnavailable(f"Unsupported tactile encoding: {message.encoding!r}")
    if message.is_bigendian not in (0, 1):
        raise ObservationUnavailable("Image has invalid endianness")
    data = _image_buffer(message, 4)
    depth = np.ndarray(
        (message.height, message.width),
        dtype=">f4" if message.is_bigendian else "<f4",
        buffer=data,
        strides=(message.step, 4),
    ).astype(np.float32, copy=True)
    return _finite_float32(depth, (message.height, message.width), "Tactile depth")


def ordered_joints(message: Any, joint_names: Sequence[str] = JOINT_NAMES) -> tuple[np.ndarray, np.ndarray]:
    """Require actual positions AND efforts for all seven explicitly named joints."""
    if len(joint_names) != 7 or len(set(joint_names)) != 7 or any(not name for name in joint_names):
        raise ObservationUnavailable("Exactly seven unique, nonempty joint names are required")
    names = list(message.name)
    if len(set(names)) != len(names):
        raise ObservationUnavailable("JointState contains duplicate joint names")
    if len(message.position) != len(names) or len(message.effort) != len(names):
        raise ObservationUnavailable("JointState must provide positions and efforts for every named joint")
    missing = set(joint_names) - set(names)
    if missing:
        raise ObservationUnavailable(f"JointState is missing required joints: {sorted(missing)}")
    indices = [names.index(name) for name in joint_names]
    return (
        _finite_float32([message.position[i] for i in indices], (7,), "Joint positions"),
        _finite_float32([message.effort[i] for i in indices], (7,), "Joint efforts"),
    )


def raw_wrench(message: Any) -> np.ndarray:
    force, torque = message.wrench.force, message.wrench.torque
    return _finite_float32([force.x, force.y, force.z, torque.x, torque.y, torque.z], (6,), "Finger wrench")


def gripper_position(state: Any, *, max_age: float, elapsed: float = 0.0) -> np.float32:
    """Validate the actual G3 StateResponse; a responsive service is not a valid motor."""
    required = ("connected", "valid", "stale", "init_state", "grip_state", "error", "position", "update_age_ms")
    if any(not hasattr(state, name) for name in required):
        raise ObservationUnavailable(f"GetAllState is missing required status fields. {GRIPPER_PREPARATION}")
    if not state.connected or not state.valid or state.stale:
        raise ObservationUnavailable(
            f"Gripper feedback unavailable (connected={state.connected}, valid={state.valid}, stale={state.stale}, "
            f"init_state={state.init_state}). A responsive gRPC server is not motor readiness. "
            f"{GRIPPER_PREPARATION}"
        )
    if state.init_state != 1:
        raise ObservationUnavailable(
            f"Gripper is not initialized (init_state={state.init_state}). {GRIPPER_PREPARATION}"
        )
    if state.error != 0 or state.grip_state not in (0, 1, 2, 3):
        raise ObservationUnavailable(
            f"Gripper reports invalid motor state (error={state.error}, grip_state={state.grip_state}). "
            f"{GRIPPER_PREPARATION}"
        )
    age = float(state.update_age_ms) / 1000.0
    if not math.isfinite(age) or age < 0 or not math.isfinite(elapsed) or elapsed < 0 or age + elapsed > max_age:
        raise ObservationUnavailable(f"Gripper feedback has stale/invalid update_age_ms. {GRIPPER_PREPARATION}")
    position = float(state.position)
    if not math.isfinite(position) or not 0 <= position <= 1000 or not position.is_integer():
        raise ObservationUnavailable("Gripper position must be an actual integer SDK position in 0..1000")
    return np.float32(position)


def bind_gripper_reader(channel: Any, response_type: Any) -> Any:
    """Bind ONLY the read RPC, without constructing a motor-owning SDK wrapper."""
    fields = response_type.DESCRIPTOR.fields_by_name
    # Field number/type pairs from G3_client/DM_V3Gripper/gripper.proto (proto3).
    required = {
        "valid": (1, 8),
        "init_state": (2, 5),
        "grip_state": (3, 5),
        "position": (4, 5),
        "error": (7, 5),
        "update_age_ms": (9, 1),
        "stale": (10, 8),
        "connected": (14, 8),
    }
    for name, expected in required.items():
        field = fields.get(name)
        if field is None or (field.number, field.type) != expected:
            raise ObservationUnavailable(f"Installed G3 StateResponse schema is incompatible: {name}")
    return channel.unary_unary(GRIPPER_RPC, response_deserializer=response_type.FromString)


def ros_source_age(sample: Sample, *, now: float, wall_now: float, max_age: float) -> float:
    receipt_age = now - sample.monotonic_time
    if not math.isfinite(receipt_age) or not 0 <= receipt_age <= max_age:
        raise ObservationUnavailable("stale or invalid host receipt timestamp")
    stamp = sample.value.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000 or (stamp.sec == 0 and stamp.nanosec == 0):
        raise ObservationUnavailable("missing or invalid nonzero acquisition header timestamp")
    age = wall_now - (stamp.sec + stamp.nanosec / 1_000_000_000)
    if not math.isfinite(age) or not -max_age <= age <= max_age:
        raise ObservationUnavailable("stale/future acquisition header timestamp; use synchronized host wall clocks")
    return age


def make_observation(
    samples: Mapping[str, Sample],
    gripper: Sample | None,
    *,
    now: float,
    wall_now: float,
    max_age: float,
    force_level: float,
    joint_names: Sequence[str] = JOINT_NAMES,
) -> dict[str, np.ndarray]:
    missing = [name for name in ROS_INPUTS if name not in samples]
    if missing:
        raise ObservationUnavailable(f"Missing ROS inputs: {', '.join(missing)}")
    ages = []
    for name in ROS_INPUTS:
        try:
            ages.append(ros_source_age(samples[name], now=now, wall_now=wall_now, max_age=max_age))
        except (AttributeError, ValueError, ObservationUnavailable) as exc:
            raise ObservationUnavailable(f"{name}: {exc}") from exc
    if gripper is None:
        raise ObservationUnavailable(f"No valid GetAllState motor feedback. {GRIPPER_PREPARATION}")
    position = gripper_position(gripper.value, max_age=max_age, elapsed=now - gripper.monotonic_time)
    ages.append(gripper.value.update_age_ms / 1000.0 + now - gripper.monotonic_time)
    if max(ages) - min(ages) > max_age:
        raise ObservationUnavailable("Observation source timestamp spread exceeds --max-age")
    if not math.isfinite(force_level) or not 20 <= force_level <= 100:
        raise ObservationUnavailable("Declared --force-level must be in 20..100 (command setting, not measured force)")
    joints, efforts = ordered_joints(samples["joints"].value, joint_names)
    depths = [tactile_depth(samples[f"depth_{sensor}"].value) for sensor in (0, 2)]
    if depths[0].shape != depths[1].shape:
        raise ObservationUnavailable("sensor_0 and sensor_2 tactile depth shapes differ")
    return {
        "observation/front_image": rgb_image(samples["front"].value),
        "observation/wrist_image": rgb_image(samples["wrist"].value),
        "observation/state": np.concatenate((joints, np.asarray([position], dtype=np.float32))),
        "observation/arm_effort": efforts,
        "observation/finger_wrench_0": raw_wrench(samples["wrench_0"].value),
        "observation/finger_wrench_2": raw_wrench(samples["wrench_2"].value),
        "observation/tactile_depth": np.stack(depths),
        "observation/force_level": np.asarray(force_level, dtype=np.float32),
    }


def wait_for_observation(
    samples: Mapping[str, Sample],
    get_state: Any,
    *,
    spin_once: Callable[[float], None],
    is_running: Callable[[], bool],
    timeout: float,
    max_age: float,
    force_level: float,
    joint_names: Sequence[str] = JOINT_NAMES,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    rpc_errors: tuple[type[Exception], ...] = (),
) -> dict[str, np.ndarray]:
    """Spin callbacks during asynchronous, deadline-bounded read-only state polling."""
    deadline = clock() + timeout
    pending = None
    gripper = None
    next_read = 0.0
    requested_at = 0.0
    problem = "No complete snapshot received"
    gripper_problem = ""
    state_errors = (ObservationUnavailable, *rpc_errors)
    try:
        while is_running():
            now = clock()
            if now >= deadline:
                raise ObservationUnavailable(f"Observation timed out after {timeout:g}s: {problem}")
            if pending is None and now >= next_read:
                requested_at = now
                pending = get_state.future(b"", timeout=min(max_age, deadline - now))
            spin_once(min(0.01, max(0.0, deadline - clock())))
            now = clock()
            if pending is not None and pending.done():
                try:
                    state = pending.result()
                    gripper_position(state, max_age=max_age, elapsed=now - requested_at)
                    gripper = Sample(state, requested_at)
                    gripper_problem = ""
                except state_errors as exc:
                    gripper = None
                    gripper_problem = (
                        str(exc)
                        if isinstance(exc, ObservationUnavailable)
                        else f"GetAllState read failed ({type(exc).__name__}): {exc}. {GRIPPER_PREPARATION}"
                    )
                pending = None
                next_read = now + min(0.05, max_age / 2)
            if now >= deadline:
                continue
            try:
                return make_observation(
                    samples,
                    gripper,
                    now=now,
                    wall_now=wall_clock(),
                    max_age=max_age,
                    force_level=force_level,
                    joint_names=joint_names,
                )
            except (AttributeError, TypeError, ValueError, ObservationUnavailable) as exc:
                problem = str(exc)
                if gripper_problem:
                    problem += f"; {gripper_problem}"
        raise ObservationUnavailable("ROS context stopped while waiting for observations")
    finally:
        if pending is not None:
            pending.cancel()  # Cancels only the outstanding read RPC, never a motor command.


def write_frame(output: BinaryIO, arrays: Mapping[str, np.ndarray]) -> None:
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in converted.values()):
        raise ValueError("Object arrays/pickle are forbidden in observation responses")
    payload = io.BytesIO()
    np.savez(payload, **converted)
    data = payload.getvalue()
    output.write(struct.pack(">Q", len(data)))
    output.write(data)
    output.flush()


def serve_requests(source: BinaryIO, output: BinaryIO, observe: Callable[[], dict[str, np.ndarray]]) -> int:
    write_frame(output, {"__ready__": np.asarray(a=True)})
    while line := source.readline(65537):
        try:
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("Requests must be newline-delimited JSON, at most 65536 bytes")
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"op"}:
                raise ValueError("Expected a JSON object containing only 'op'")
            if request["op"] == "close":
                write_frame(output, {"__closed__": np.asarray(a=True)})
                return 0
            if request["op"] != "observe":
                raise ValueError("Supported operations are 'observe' and 'close' only")
            write_frame(output, observe())
        except (ObservationUnavailable, ValueError, TypeError) as exc:
            print(f"Read-only observation error: {exc}", file=sys.stderr)
            write_frame(output, {"__error__": np.asarray(str(exc))})
            return 1
    return 0


class _DiscardParameterEvents:
    def publish(self, message: Any) -> None:
        """Do not create or publish ROS parameter events."""


class SubscriptionOnlyNodeMixin:
    """Jazzy Node otherwise creates /parameter_events even with services disabled."""

    def create_publisher(self, msg_type: Any, topic: str, *args: Any, **kwargs: Any) -> _DiscardParameterEvents:
        if topic == "/parameter_events" and msg_type.__name__ == "ParameterEvent":
            return _DiscardParameterEvents()
        raise RuntimeError("Publishers are forbidden in the read-only observation worker")

    def create_client(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ROS clients are forbidden in the read-only observation worker")

    def create_service(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ROS services are forbidden in the read-only observation worker")


def add_observation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--front-topic", default="/front/color/image_raw")
    parser.add_argument("--wrist-topic", default="/wrist/color/image_raw")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--tactile-prefix", default="/v3_gripper/tactile")
    parser.add_argument("--gripper-address", default="192.168.127.10:50008")
    parser.add_argument(
        "--force-level", type=float, default=80, help="Declared 20..100 command setting, NOT measured force"
    )
    parser.add_argument("--max-age", type=float, default=0.5, help="Maximum input age and timestamp spread, seconds")
    parser.add_argument("--timeout", type=float, default=10, help="Deadline per observation request, seconds")
    parser.add_argument("--joint-names", nargs=7, default=JOINT_NAMES)


def validate_observation_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.force_level) or not 20 <= args.force_level <= 100:
        parser.error("--force-level must be finite and in 20..100")
    for name in ("max_age", "timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if len(set(args.joint_names)) != 7 or any(not name for name in args.joint_names):
        parser.error("--joint-names requires seven unique, nonempty names")
    if any(
        not getattr(args, name).strip()
        for name in ("front_topic", "wrist_topic", "joint_topic", "tactile_prefix", "gripper_address")
    ):
        parser.error("Topics, tactile prefix, and gripper address must not be empty")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_observation_arguments(parser)
    parser.add_argument(
        "--check-gripper", action="store_true", help="Check real motor feedback via GetAllState only, then exit"
    )
    args = parser.parse_args(argv)
    validate_observation_arguments(parser, args)
    return args


def subscribe_observations(node: Any, args: argparse.Namespace, samples: dict[str, Sample]) -> None:
    from geometry_msgs.msg import WrenchStamped
    from rclpy.qos import HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    topics = [
        ("joints", JointState, args.joint_topic),
        ("front", Image, args.front_topic),
        ("wrist", Image, args.wrist_topic),
    ]
    prefix = args.tactile_prefix.rstrip("/")
    for sensor in (0, 2):
        topics.extend(
            (
                (f"depth_{sensor}", Image, f"{prefix}/sensor_{sensor}/depth"),
                (f"wrench_{sensor}", WrenchStamped, f"{prefix}/sensor_{sensor}/wrench"),
            )
        )
    for key, msg_type, topic in topics:

        def receive(message: Any, key: str = key) -> None:
            samples[key] = Sample(message, time.monotonic())

        node.create_subscription(msg_type, topic, receive, qos)


def reserve_protocol_stdout() -> BinaryIO:
    sys.stdout.flush()
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    # Redirect native-library output as well as Python prints away from the protocol.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return output


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["RCUTILS_LOGGING_USE_STDOUT"] = "0"
    args = parse_args(argv)
    output = None if args.check_gripper else reserve_protocol_stdout()
    context = node = executor = channel = None
    try:
        from DM_V3Gripper import gripper_pb2
        import grpc

        if args.check_gripper:
            channel = grpc.insecure_channel(args.gripper_address, options=(("grpc.enable_http_proxy", 0),))
            get_state = bind_gripper_reader(channel, gripper_pb2.StateResponse)
            started = time.monotonic()
            try:
                state = get_state.future(b"", timeout=min(args.timeout, args.max_age)).result()
            except grpc.RpcError as exc:
                raise ObservationUnavailable(f"GetAllState read failed: {exc}. {GRIPPER_PREPARATION}") from exc
            position = gripper_position(state, max_age=args.max_age, elapsed=time.monotonic() - started)
            print(
                f"Read-only gripper feedback ready at {args.gripper_address}: SDK position={position:g}, "
                "connected, initialized and fresh. No motor commands were sent.",
                file=sys.stderr,
            )
            return 0

        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter

        class ObservationNode(SubscriptionOnlyNodeMixin, Node):
            pass

        context = Context()
        rclpy.init(args=[], context=context)
        node = ObservationNode(
            "ar3_read_only_observation",
            context=context,
            use_global_arguments=False,
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
            parameter_overrides=[Parameter("start_type_description_service", value=False)],
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        samples: dict[str, Sample] = {}
        subscribe_observations(node, args, samples)
        channel = grpc.insecure_channel(args.gripper_address, options=(("grpc.enable_http_proxy", 0),))
        get_state = bind_gripper_reader(channel, gripper_pb2.StateResponse)
        print(
            f"Read-only subscriptions set up (NOT sensor readiness): front={args.front_topic}, wrist={args.wrist_topic}, "
            f"joints={args.joint_topic}, tactile={args.tactile_prefix}, gripper={args.gripper_address}. "
            f"Declared force level={args.force_level:g} is NOT measured force. "
            "Motor feedback must already be prepared by an approved driver.",
            file=sys.stderr,
        )
        assert output is not None
        return serve_requests(
            sys.stdin.buffer,
            output,
            lambda: wait_for_observation(
                samples,
                get_state,
                spin_once=lambda timeout: executor.spin_once(timeout_sec=timeout),
                is_running=context.ok,
                timeout=args.timeout,
                max_age=args.max_age,
                force_level=args.force_level,
                joint_names=args.joint_names,
                rpc_errors=(grpc.RpcError,),
            ),
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        print(f"Read-only observation startup/runtime error: {exc}", file=sys.stderr)
        if output is not None:
            write_frame(output, {"__error__": np.asarray(str(exc))})
        return 1
    finally:
        for resource in (channel, executor, node, context):
            if resource is None:
                continue
            try:
                if resource is channel:
                    channel.close()
                elif resource is executor:
                    executor.shutdown(timeout_sec=1.0)
                elif resource is node:
                    node.destroy_node()
                else:
                    rclpy.try_shutdown(context=context)
            except (OSError, RuntimeError) as exc:
                print(f"Read-only worker local cleanup failed: {type(exc).__name__}", file=sys.stderr)
        if output is not None:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
