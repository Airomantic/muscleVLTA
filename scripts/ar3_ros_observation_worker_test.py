import io
import struct
import sys
import time
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ar3_ros_observation_worker as worker

WALL_TIME = 1_700_000_000.0


def _header(stamp: float = WALL_TIME) -> SimpleNamespace:
    seconds = int(stamp)
    return SimpleNamespace(stamp=SimpleNamespace(sec=seconds, nanosec=round((stamp - seconds) * 1_000_000_000)))


def _image(values: np.ndarray, *, encoding: str = "rgb8", padding: int = 0, endian: int = 0) -> SimpleNamespace:
    values = np.asarray(values)
    if encoding == "32FC1":
        values = values.astype(">f4" if endian else "<f4")
    rows = [row.tobytes() + b"\xff" * padding for row in values]
    return SimpleNamespace(
        header=_header(),
        encoding=encoding,
        height=values.shape[0],
        width=values.shape[1],
        step=len(rows[0]),
        data=b"".join(rows),
        is_bigendian=endian,
    )


def _joints() -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(),
        name=list(reversed(worker.JOINT_NAMES)),
        position=list(reversed(np.arange(7) / 10)),
        effort=list(reversed(np.arange(7) + 10)),
    )


def _wrench(offset: float = 0) -> SimpleNamespace:
    return SimpleNamespace(
        header=_header(),
        wrench=SimpleNamespace(
            force=SimpleNamespace(x=offset + 1, y=offset + 2, z=offset + 3),
            torque=SimpleNamespace(x=offset + 4, y=offset + 5, z=offset + 6),
        ),
    )


def _state(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "connected": True,
            "valid": True,
            "stale": False,
            "init_state": 1,
            "grip_state": 1,
            "error": 0,
            "position": 432,
            "update_age_ms": 10,
            "current": 999,
            **overrides,
        }
    )


def _samples() -> dict[str, worker.Sample]:
    messages = {
        "joints": _joints(),
        "front": _image(np.arange(18, dtype=np.uint8).reshape(2, 3, 3)),
        "wrist": _image(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), encoding="bgr8"),
        "depth_0": _image(np.full((2, 3), 1.25), encoding="32FC1"),
        "depth_2": _image(np.full((2, 3), -2.5), encoding="32FC1"),
        "wrench_0": _wrench(),
        "wrench_2": _wrench(-10),
    }
    return {key: worker.Sample(message, 10.0) for key, message in messages.items()}


def _observation(samples=None, gripper=None, **kwargs) -> dict[str, np.ndarray]:
    options = {"now": 10.1, "wall_now": WALL_TIME + 0.1, "max_age": 0.5, "force_level": 80}
    options.update(kwargs)
    return worker.make_observation(
        _samples() if samples is None else samples,
        worker.Sample(_state(), 10.0) if gripper is None else gripper,
        **options,
    )


def _frames(data: bytes) -> list[dict[str, np.ndarray]]:
    stream = io.BytesIO(data)
    frames = []
    while header := stream.read(8):
        assert len(header) == 8
        size = struct.unpack(">Q", header)[0]
        payload = stream.read(size)
        assert len(payload) == size
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            frames.append({name: archive[name] for name in archive.files})
    return frames


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8"])
@pytest.mark.parametrize("padding", [0, 5])
def test_rgb_conversion_preserves_shape_and_ignores_padding(encoding, padding):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    decoded = worker.rgb_image(_image(pixels, encoding=encoding, padding=padding))
    expected = pixels[..., ::-1] if encoding == "bgr8" else pixels
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == np.uint8
    assert decoded.flags.c_contiguous
    assert decoded.flags.owndata


@pytest.mark.parametrize("endian", [0, 1])
@pytest.mark.parametrize("padding", [0, 3, 8])
def test_tactile_depth_is_raw_float32_in_both_endian_orders(endian, padding):
    depth = np.array([[0.125, -2.5, 1000], [3.5, 4.75, 0]], dtype=np.float32)
    decoded = worker.tactile_depth(_image(depth, encoding="32FC1", padding=padding, endian=endian))
    np.testing.assert_array_equal(decoded, depth)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous


@pytest.mark.parametrize("encoding", ["mono8", "rgba8", "jpeg", "16UC1"])
def test_unknown_rgb_and_tactile_encodings_fail(encoding):
    message = _image(np.zeros((2, 3, 3), np.uint8), encoding=encoding)
    with pytest.raises(worker.ObservationUnavailable, match="Unsupported RGB encoding"):
        worker.rgb_image(message)
    with pytest.raises(worker.ObservationUnavailable, match="Unsupported tactile encoding"):
        worker.tactile_depth(message)


@pytest.mark.parametrize(("field", "value"), [("step", 2), ("height", 0), ("width", -1), ("data", b"")])
def test_invalid_image_layout_fails(field, value):
    message = _image(np.zeros((2, 3, 3), np.uint8))
    setattr(message, field, value)
    with pytest.raises(worker.ObservationUnavailable, match="Image"):
        worker.rgb_image(message)


def test_invalid_depth_endianness_fails():
    message = _image(np.zeros((2, 3)), encoding="32FC1", endian=2)
    with pytest.raises(worker.ObservationUnavailable, match="endianness"):
        worker.tactile_depth(message)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_depth_and_wrench_are_rejected(value):
    with pytest.raises(worker.ObservationUnavailable, match="finite"):
        worker.tactile_depth(_image(np.array([[value]]), encoding="32FC1"))
    wrench = _wrench()
    wrench.wrench.torque.y = value
    with pytest.raises(worker.ObservationUnavailable, match="finite"):
        worker.raw_wrench(wrench)


def test_joint_ordering_and_additional_joints():
    message = _joints()
    message.name.append("extra_joint")
    message.position.append(99)
    message.effort.append(100)
    positions, efforts = worker.ordered_joints(message)
    np.testing.assert_allclose(positions, np.arange(7) / 10)
    np.testing.assert_array_equal(efforts, np.arange(7) + 10)
    assert positions.dtype == efforts.dtype == np.float32
    positions, efforts = worker.ordered_joints(message, tuple(reversed(worker.JOINT_NAMES)))
    np.testing.assert_allclose(positions, np.arange(7)[::-1] / 10)
    np.testing.assert_array_equal(efforts, np.arange(7)[::-1] + 10)


@pytest.mark.parametrize("field", ["position", "effort"])
def test_missing_joint_positions_or_efforts_fail(field):
    message = _joints()
    setattr(message, field, [])
    with pytest.raises(worker.ObservationUnavailable, match="positions and efforts"):
        worker.ordered_joints(message)


@pytest.mark.parametrize("field", ["position", "effort"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e100])
def test_nonfinite_and_float32_overflowing_joint_values_fail(field, value):
    message = _joints()
    getattr(message, field)[0] = value
    with pytest.raises(worker.ObservationUnavailable, match="finite"):
        worker.ordered_joints(message)


def test_missing_duplicate_or_invalid_joint_names_fail():
    message = _joints()
    message.name[0] = "wrong_joint"
    with pytest.raises(worker.ObservationUnavailable, match="missing required joints"):
        worker.ordered_joints(message)
    message.name[0] = message.name[1]
    with pytest.raises(worker.ObservationUnavailable, match="duplicate"):
        worker.ordered_joints(message)
    with pytest.raises(worker.ObservationUnavailable, match="seven unique"):
        worker.ordered_joints(_joints(), ["joint1"] * 7)


@pytest.mark.parametrize("position", [0, 432, 1000])
def test_gripper_position_is_actual_unscaled_sdk_position(position):
    assert worker.gripper_position(_state(position=position), max_age=0.5) == np.float32(position)


@pytest.mark.parametrize(
    "overrides",
    [
        {"connected": False},
        {"valid": False},
        {"stale": True},
        {"init_state": 0},
        {"init_state": 2},
        {"init_state": 0xFFFF},
        {"grip_state": 0xFFFF},
        {"error": 5},
        {"update_age_ms": -1},
        {"update_age_ms": np.nan},
        {"update_age_ms": np.inf},
        {"update_age_ms": 501},
    ],
)
def test_responsive_but_invalid_gripper_state_requires_approved_driver(overrides):
    with pytest.raises(worker.ObservationUnavailable, match="approved driver"):
        worker.gripper_position(_state(**overrides), max_age=0.5)


@pytest.mark.parametrize("position", [-1, 1001, 0.5, np.nan, np.inf])
def test_invalid_gripper_positions_are_not_clipped_or_guessed(position):
    with pytest.raises(worker.ObservationUnavailable, match="SDK position"):
        worker.gripper_position(_state(position=position), max_age=0.5)


def test_gripper_age_includes_network_and_local_elapsed_time():
    with pytest.raises(worker.ObservationUnavailable, match="stale"):
        worker.gripper_position(_state(update_age_ms=300), max_age=0.5, elapsed=0.21)
    with pytest.raises(worker.ObservationUnavailable, match="missing required status"):
        worker.gripper_position(SimpleNamespace(position=0), max_age=0.5)


def _response_type():
    field_numbers_types = {
        "valid": (1, 8),
        "init_state": (2, 5),
        "grip_state": (3, 5),
        "position": (4, 5),
        "error": (7, 5),
        "update_age_ms": (9, 1),
        "stale": (10, 8),
        "connected": (14, 8),
    }

    class StateResponse:
        DESCRIPTOR = SimpleNamespace(
            fields_by_name={
                name: SimpleNamespace(number=number, type=field_type)
                for name, (number, field_type) in field_numbers_types.items()
            }
        )

        @staticmethod
        def FromString(data):  # noqa: N802
            assert data == b"fake response bytes"
            return _state()

    return StateResponse


class _Future:
    def __init__(self, result=None, *, done=None, error=None):
        self.state = _state() if result is None else result
        self.is_done = done or (lambda: True)
        self.error = error
        self.cancelled = False

    def done(self):
        return self.is_done()

    def result(self):
        if self.error is not None:
            raise self.error
        return self.state

    def cancel(self):
        self.cancelled = True


class _Reader:
    def __init__(self, future=None):
        self.response = future or _Future()
        self.requests = []

    def future(self, request, *, timeout):
        assert request == b""
        assert timeout > 0
        self.requests.append((request, timeout))
        return self.response


class _Channel:
    def __init__(self):
        self.bindings = []
        self.reader = _Reader()
        self.closed = False

    def unary_unary(self, path, *, response_deserializer):
        assert path == "/gripper.GripperService/GetAllState"
        self.bindings.append((path, response_deserializer))
        return self.reader

    def close(self):
        self.closed = True


def test_gripper_binding_exposes_only_get_all_state_with_actual_response_decoder():
    response = _response_type()
    channel = _Channel()
    reader = worker.bind_gripper_reader(channel, response)
    assert reader is channel.reader
    assert channel.bindings == [(worker.GRIPPER_RPC, response.FromString)]
    assert channel.bindings[0][1](b"fake response bytes").position == 432
    del response.DESCRIPTOR.fields_by_name["connected"]
    with pytest.raises(worker.ObservationUnavailable, match="schema is incompatible: connected"):
        worker.bind_gripper_reader(channel, response)
    assert len(channel.bindings) == 1


@pytest.mark.parametrize(
    ("overrides", "rpc_failure", "status", "message"),
    [
        ({}, False, 0, "gripper feedback ready"),
        ({"connected": False, "init_state": -1}, False, 1, "A responsive gRPC server is not motor readiness"),
        ({"update_age_ms": 1000}, False, 1, "stale/invalid update_age_ms"),
        ({}, True, 1, "GetAllState read failed"),
    ],
)
def test_gripper_readiness_checks_real_state_without_ros_or_protocol_frames(
    monkeypatch, capsys, overrides, rpc_failure, status, message
):
    channel = _Channel()
    response = _response_type()
    rpc_error = type("RpcError", (Exception,), {})
    channel.reader = _Reader(
        _Future(result=_state(**overrides), error=rpc_error("unavailable") if rpc_failure else None)
    )

    def insecure_channel(address, *, options):
        assert address == "192.168.127.10:50008"
        assert options == (("grpc.enable_http_proxy", 0),)
        return channel

    monkeypatch.setitem(
        sys.modules, "DM_V3Gripper", SimpleNamespace(gripper_pb2=SimpleNamespace(StateResponse=response))
    )
    monkeypatch.setitem(sys.modules, "grpc", SimpleNamespace(insecure_channel=insecure_channel, RpcError=rpc_error))
    monkeypatch.setitem(sys.modules, "rclpy", None)
    assert worker.main(["--check-gripper"]) == status
    assert channel.bindings == [(worker.GRIPPER_RPC, response.FromString)]
    assert len(channel.reader.requests) == 1
    assert channel.closed
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err
    assert ("gripper feedback ready" in captured.err) == (status == 0)


def test_complete_observation_has_exact_keys_shapes_units_and_sensor_order():
    observation = _observation()
    assert set(observation) == {
        "observation/front_image",
        "observation/wrist_image",
        "observation/state",
        "observation/arm_effort",
        "observation/finger_wrench_0",
        "observation/finger_wrench_2",
        "observation/tactile_depth",
        "observation/force_level",
    }
    np.testing.assert_allclose(observation["observation/state"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 432])
    np.testing.assert_array_equal(observation["observation/arm_effort"], np.arange(7) + 10)
    np.testing.assert_array_equal(observation["observation/finger_wrench_0"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(observation["observation/finger_wrench_2"], [-9, -8, -7, -6, -5, -4])
    np.testing.assert_array_equal(observation["observation/tactile_depth"][0], np.full((2, 3), 1.25))
    np.testing.assert_array_equal(observation["observation/tactile_depth"][1], np.full((2, 3), -2.5))
    assert observation["observation/force_level"].shape == ()
    assert observation["observation/force_level"] == 80
    for key, value in observation.items():
        assert value.dtype == (np.uint8 if key.endswith("_image") else np.float32)


def test_missing_inputs_missing_gripper_and_mismatched_depth_fail():
    samples = _samples()
    del samples["wrench_2"]
    with pytest.raises(worker.ObservationUnavailable, match="Missing ROS inputs: wrench_2"):
        _observation(samples)
    with pytest.raises(worker.ObservationUnavailable, match="No valid GetAllState"):
        worker.make_observation(_samples(), None, now=10.1, wall_now=WALL_TIME + 0.1, max_age=0.5, force_level=80)
    samples = _samples()
    samples["depth_2"] = worker.Sample(_image(np.zeros((4, 5)), encoding="32FC1"), 10)
    with pytest.raises(worker.ObservationUnavailable, match="shapes differ"):
        _observation(samples)


def test_invalid_new_message_replaces_previous_valid_data():
    samples = _samples()
    _observation(samples)
    invalid = _joints()
    invalid.effort = []
    samples["joints"] = worker.Sample(invalid, 10.05)
    with pytest.raises(worker.ObservationUnavailable, match="positions and efforts"):
        _observation(samples)


@pytest.mark.parametrize("receipt", [9, 11, float("nan")])
def test_host_receipt_freshness_is_checked_independently(receipt):
    samples = _samples()
    samples["front"] = worker.Sample(samples["front"].value, receipt)
    with pytest.raises(worker.ObservationUnavailable, match="front: stale or invalid host receipt"):
        _observation(samples)


@pytest.mark.parametrize("stamp", [WALL_TIME - 1, WALL_TIME + 1, 0])
def test_recent_receipt_does_not_hide_old_future_or_unstamped_acquisition(stamp):
    samples = _samples()
    samples["front"].value.header = _header(stamp)
    with pytest.raises(worker.ObservationUnavailable, match="front: .*header timestamp"):
        _observation(samples)


def test_invalid_nanoseconds_are_rejected():
    samples = _samples()
    samples["depth_0"].value.header.stamp.nanosec = 1_000_000_000
    with pytest.raises(worker.ObservationUnavailable, match="invalid.*header timestamp"):
        _observation(samples)


def test_source_timestamp_spread_cannot_exceed_budget():
    samples = _samples()
    samples["front"].value.header = _header(WALL_TIME - 0.39)
    samples["wrist"].value.header = _header(WALL_TIME + 0.4)
    with pytest.raises(worker.ObservationUnavailable, match="timestamp spread"):
        _observation(samples)


@pytest.mark.parametrize("force_level", [0, 19, 101, np.nan, np.inf])
def test_force_level_metadata_must_be_an_explicit_valid_setting(force_level):
    with pytest.raises(worker.ObservationUnavailable, match="force-level"):
        _observation(force_level=force_level)


class _Clock:
    def __init__(self):
        self.now = 10.0
        self.spins = 0

    def monotonic(self):
        return self.now

    def wall(self):
        return WALL_TIME + self.now - 10

    def spin(self, timeout):
        self.spins += 1
        self.now += timeout


def _wait(samples, reader, clock, **overrides):
    options = {
        "spin_once": clock.spin,
        "is_running": lambda: True,
        "timeout": 0.2,
        "max_age": 0.5,
        "force_level": 80,
        "clock": clock.monotonic,
        "wall_clock": clock.wall,
    }
    options.update(overrides)
    return worker.wait_for_observation(samples, reader, **options)


def test_wait_spins_callbacks_while_gripper_read_is_pending():
    clock = _Clock()
    samples = {}
    future = _Future(done=lambda: clock.spins >= 3)
    reader = _Reader(future)

    def spin(timeout):
        clock.spin(timeout)
        if clock.spins == 2:
            samples.update(_samples())

    result = _wait(samples, reader, clock, spin_once=spin)
    assert clock.spins == 3
    assert result["observation/state"][-1] == 432
    assert len(reader.requests) == 1
    assert reader.requests[0][1] <= 0.2


def test_wait_times_out_with_missing_inputs_and_motor_preparation_details():
    clock = _Clock()
    reader = _Reader(_Future(result=_state(connected=False)))
    with pytest.raises(worker.ObservationUnavailable, match="timed out.*Missing ROS inputs.*approved driver"):
        _wait({}, reader, clock)
    assert clock.spins > 1
    assert len(reader.requests) > 1


def test_rpc_failure_never_becomes_zero_gripper_position():
    clock = _Clock()
    reader = _Reader(_Future(error=RuntimeError("fake RPC unavailable")))
    with pytest.raises(worker.ObservationUnavailable, match="GetAllState read failed.*approved driver"):
        _wait(_samples(), reader, clock, rpc_errors=(RuntimeError,))


def test_unexpected_state_reader_errors_are_not_retried_until_timeout():
    clock = _Clock()
    reader = _Reader(_Future(error=TypeError("broken RPC adapter")))
    with pytest.raises(TypeError, match="broken RPC adapter"):
        _wait(_samples(), reader, clock)


def test_timeout_cancels_pending_read_and_ros_shutdown_is_not_stop_iteration():
    clock = _Clock()
    future = _Future(done=lambda: False)
    with pytest.raises(worker.ObservationUnavailable, match="timed out"):
        _wait(_samples(), _Reader(future), clock)
    assert future.cancelled
    with pytest.raises(worker.ObservationUnavailable, match="ROS context stopped"):
        _wait(_samples(), _Reader(), clock, is_running=lambda: False)


def test_protocol_handshake_observation_and_close_are_framed_npz_only():
    output = io.BytesIO()
    status = worker.serve_requests(io.BytesIO(b'{"op":"observe"}\n{"op":"close"}\n'), output, _observation)
    assert status == 0
    ready, observation, closed = _frames(output.getvalue())
    assert ready["__ready__"].item() is True
    assert closed["__closed__"].item() is True
    assert ready["__ready__"].shape == closed["__closed__"].shape == ()
    for key, value in _observation().items():
        np.testing.assert_array_equal(observation[key], value)


@pytest.mark.parametrize(
    "request_line",
    [b"no json\n", b'{"op":"move"}\n', b'{"op":"observe","action":[1]}\n', b"[]\n", b'{"op":"observe"}', b"x" * 65537],
)
def test_protocol_invalid_requests_have_one_error_and_nonzero_exit(request_line, capsys):
    output = io.BytesIO()
    assert worker.serve_requests(io.BytesIO(request_line), output, _observation) == 1
    ready, error = _frames(output.getvalue())
    assert ready["__ready__"]
    assert error["__error__"].dtype.kind == "U"
    assert error["__error__"].shape == ()
    assert capsys.readouterr().out == ""


def test_protocol_observation_failure_is_fatal_and_eof_is_clean(capsys):
    def unavailable():
        raise worker.ObservationUnavailable("Missing effort")

    output = io.BytesIO()
    assert worker.serve_requests(io.BytesIO(b'{"op":"observe"}\n{"op":"close"}\n'), output, unavailable) == 1
    assert _frames(output.getvalue())[1]["__error__"].item() == "Missing effort"
    assert capsys.readouterr().out == ""
    output = io.BytesIO()
    assert worker.serve_requests(io.BytesIO(), output, unavailable) == 0
    assert len(_frames(output.getvalue())) == 1


def test_protocol_never_serializes_object_arrays():
    output = io.BytesIO()
    with pytest.raises(ValueError, match="pickle"):
        worker.write_frame(output, {"bad": np.array([object()], dtype=object)})
    assert output.getvalue() == b""


def test_subscription_only_mixin_disables_even_default_parameter_publisher():
    node = worker.SubscriptionOnlyNodeMixin()
    event_type = type("ParameterEvent", (), {})
    sink = node.create_publisher(event_type, "/parameter_events", object())
    assert sink.publish(event_type()) is None
    with pytest.raises(RuntimeError, match="Publishers are forbidden"):
        node.create_publisher(object, "/command")
    with pytest.raises(RuntimeError, match="Publishers are forbidden"):
        node.create_publisher(object, "/parameter_events")
    with pytest.raises(RuntimeError, match="clients are forbidden"):
        node.create_client(object, "/control")
    with pytest.raises(RuntimeError, match="services are forbidden"):
        node.create_service(object, "/control")


def test_cli_contract_defaults_and_custom_joint_order():
    args = worker.parse_args([])
    assert args.front_topic == "/front/color/image_raw"
    assert args.wrist_topic == "/wrist/color/image_raw"
    assert args.joint_topic == "/joint_states"
    assert args.tactile_prefix == "/v3_gripper/tactile"
    assert args.gripper_address == "192.168.127.10:50008"
    assert (args.force_level, args.max_age, args.timeout) == (80, 0.5, 10)
    assert tuple(args.joint_names) == worker.JOINT_NAMES
    args = worker.parse_args(["--joint-names", *reversed(worker.JOINT_NAMES), "--force-level", "65"])
    assert tuple(args.joint_names) == tuple(reversed(worker.JOINT_NAMES))
    assert args.force_level == 65


@pytest.mark.parametrize(
    "args",
    [
        ["--force-level", "nan"],
        ["--force-level", "101"],
        ["--max-age", "0"],
        ["--timeout", "inf"],
        ["--joint-names", *(["joint1"] * 7)],
        ["--front-topic", ""],
    ],
)
def test_invalid_cli_configuration_fails_before_dependencies_are_loaded(args):
    with pytest.raises(SystemExit) as exc:
        worker.parse_args(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("invalid_schema", [False, True])
def test_main_with_fake_ros_and_grpc_has_only_subscriptions_and_read_rpc(monkeypatch, invalid_schema):
    events = []
    channel = _Channel()
    response = _response_type()
    if invalid_schema:
        del response.DESCRIPTOR.fields_by_name["connected"]

    class Context:
        def ok(self):
            return True

    class Node:
        def __init__(self, name, **kwargs):
            assert name == "ar3_read_only_observation"
            assert kwargs["enable_rosout"] is False
            assert kwargs["start_parameter_services"] is False
            assert kwargs["enable_logger_service"] is False
            assert kwargs["use_global_arguments"] is False
            assert kwargs["parameter_overrides"][0].name == "start_type_description_service"
            assert kwargs["parameter_overrides"][0].value is False
            self.subscriptions = []
            self.create_publisher(type("ParameterEvent", (), {}), "/parameter_events", object()).publish(None)

        def create_subscription(self, msg_type, topic, callback, qos):
            assert qos.depth == 1
            assert qos.reliability == "best_effort"
            self.subscriptions.append((topic, callback))
            events.append(("subscription", topic))

        def destroy_node(self):
            events.append("destroy_node")

    class Executor:
        def __init__(self, **kwargs):
            pass

        def add_node(self, node):
            self.node = node

        def spin_once(self, *, timeout_sec):
            messages = _samples()
            keys = ("joints", "front", "wrist", "depth_0", "wrench_0", "depth_2", "wrench_2")
            for (_, callback), key in zip(self.node.subscriptions, keys, strict=True):
                message = messages[key].value
                message.header = _header(time.time())
                callback(message)

        def shutdown(self, **kwargs):
            events.append("executor_shutdown")

    class Output(io.BytesIO):
        def close(self):
            events.append("output_closed")

    output = Output()

    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)

    def insecure_channel(address, *, options):
        assert address == "192.168.127.10:50008"
        assert options == (("grpc.enable_http_proxy", 0),)
        events.append("read_channel_created")
        return channel

    module("grpc", insecure_channel=insecure_channel, RpcError=type("RpcError", (Exception,), {}))
    module(
        "rclpy",
        init=lambda **kwargs: events.append("ros_init"),
        try_shutdown=lambda **kwargs: events.append("ros_shutdown"),
    )
    module("rclpy.context", Context=Context)
    module("rclpy.executors", SingleThreadedExecutor=Executor)
    module("rclpy.node", Node=Node)
    module("rclpy.parameter", Parameter=lambda name, **kwargs: SimpleNamespace(name=name, **kwargs))
    module(
        "rclpy.qos",
        QoSProfile=SimpleNamespace,
        HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"),
        ReliabilityPolicy=SimpleNamespace(BEST_EFFORT="best_effort"),
    )
    module("sensor_msgs", msg=None)
    module("sensor_msgs.msg", Image=type("Image", (), {}), JointState=type("JointState", (), {}))
    module("geometry_msgs", msg=None)
    module("geometry_msgs.msg", WrenchStamped=type("WrenchStamped", (), {}))
    module("DM_V3Gripper", gripper_pb2=SimpleNamespace(StateResponse=response))
    monkeypatch.setattr(worker, "reserve_protocol_stdout", lambda: output)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"op":"observe"}\n{"op":"close"}\n')))
    monkeypatch.setenv("ROS_DOMAIN_ID", "111")
    monkeypatch.setenv("RCUTILS_LOGGING_USE_STDOUT", "1")

    assert worker.main([]) == int(invalid_schema)
    assert worker.os.environ["ROS_DOMAIN_ID"] == "111"
    assert worker.os.environ["RCUTILS_LOGGING_USE_STDOUT"] == "0"
    assert channel.closed
    assert events[-4:] == ["executor_shutdown", "destroy_node", "ros_shutdown", "output_closed"]
    assert [event[1] for event in events if isinstance(event, tuple)] == [
        "/joint_states",
        "/front/color/image_raw",
        "/wrist/color/image_raw",
        "/v3_gripper/tactile/sensor_0/depth",
        "/v3_gripper/tactile/sensor_0/wrench",
        "/v3_gripper/tactile/sensor_2/depth",
        "/v3_gripper/tactile/sensor_2/wrench",
    ]
    frames = _frames(output.getvalue())
    if invalid_schema:
        assert len(frames) == 1
        assert "schema is incompatible" in frames[0]["__error__"].item()
        assert not channel.bindings
    else:
        assert frames[0]["__ready__"]
        assert frames[1]["observation/state"][-1] == 432
        assert frames[2]["__closed__"]
        assert len(channel.bindings) == len(channel.reader.requests) == 1
