import io
import json
import logging
import struct
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import ar3_ros_control_worker as worker


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class FakeGripper:
    def __init__(self, events, clock):
        self.events, self.clock = events, clock
        self.force_level = 80
        self.failure = None
        self.stop_failure = None
        self.delay = 0
        self.actions = []

    def read(self):
        return worker.observations.Sample(state(), self.clock())

    def apply(self, action, *, deadline):
        self.events.append("gripper")
        if self.failure:
            raise RuntimeError(self.failure)
        self.actions.append(action.copy())
        self.force_level = int(action[8])
        self.clock.now += self.delay

    def stop(self):
        self.events.append("gripper-stop")
        if self.stop_failure:
            raise RuntimeError(self.stop_failure)

    def close(self):
        self.events.append("gripper-close")


class FakeHardware:
    def __init__(self, args, events, clock):
        self.args, self.events, self.clock = args, events, clock
        self.limits = worker.JointLimits(-np.ones(7), np.ones(7), np.ones(7))
        self.targets = []
        self.failure = None
        self.stop_failure = None
        self.stopped = threading.Event()

    def start(self):
        self.events.append("start")

    def current_joints(self):
        if self.failure == "feedback":
            raise RuntimeError("stale feedback")
        return np.zeros(7)

    def observe(self):
        self.events.append("observe")
        return {"observation/state": np.zeros(8)}, self.clock() + self.args.max_age

    def publish(self, target):
        self.events.append("arm")
        self.targets.append(target.copy())
        return self.clock()

    def wait_target(self, target, started, deadline):
        self.events.append("ack")
        if self.failure == "ack":
            raise TimeoutError("arm acknowledgement failed")

    def hold(self):
        self.events.append("arm-hold")
        self.stopped.set()
        if self.stop_failure:
            raise RuntimeError(self.stop_failure)

    def close(self):
        self.events.append("arm-close")


@pytest.fixture
def rig():
    args = worker.parse_args(["--execute"])
    events, clock = [], Clock()
    gripper = FakeGripper(events, clock)
    hardware = FakeHardware(args, events, clock)
    session = worker.ControlSession(args, hardware, gripper, clock=clock)
    return SimpleNamespace(args=args, events=events, clock=clock, gripper=gripper, hardware=hardware, session=session)


def state(**values):
    return SimpleNamespace(
        **{
            "connected": True,
            "valid": True,
            "stale": False,
            "init_state": 1,
            "grip_state": 1,
            "error": 0,
            "position": 432,
            "update_age_ms": 0,
            **values,
        }
    )


def action():
    return [0.2] * 7 + [432.2, 80.7]


def frames(output):
    stream, result = io.BytesIO(output.getvalue()), []
    while header := stream.read(8):
        size = struct.unpack(">Q", header)[0]
        with np.load(io.BytesIO(stream.read(size)), allow_pickle=False) as archive:
            result.append({key: archive[key] for key in archive.files})
    return result


def requests(*items):
    return io.BytesIO(b"".join((json.dumps(item) + "\n").encode() for item in items))


def execute_request():
    return {"op": "execute", "observation_id": 1, "action": action()}


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--initialize-gripper"],
        ["--execute", "--command-period", "nan"],
        ["--execute", "--command-period", "0.3"],
        ["--execute", "--command-timeout", "0.02"],
        ["--execute", "--command-timeout", "3"],
        ["--execute", "--gripper-speed", "101"],
        ["--execute", "--joint-command-topic", "/other/controller"],
    ],
)
def test_invalid_opt_in_or_limits_never_open_hardware(monkeypatch, argv):
    monkeypatch.setattr(worker, "GripperCommands", lambda args: pytest.fail("Must not open hardware"))
    with pytest.raises(SystemExit) as exc:
        worker.main(argv)
    assert exc.value.code == 2


def test_observation_authorizes_one_bounded_target_with_exact_acknowledgement(rig):
    observation = rig.session.observe()
    result = rig.session.execute(int(observation["__observation_id__"]), action())
    np.testing.assert_allclose(result["__executed_action__"][:7], 0.2 / 15)
    np.testing.assert_array_equal(result["__executed_action__"][7:], [432, 81])
    np.testing.assert_array_equal(rig.gripper.actions[0], result["__executed_action__"])
    np.testing.assert_array_equal(rig.hardware.targets[0], result["__executed_action__"][:7])
    assert result["__observation_id__"].item() == 1
    assert rig.events == ["observe", "gripper", "arm", "ack"]
    with pytest.raises(ValueError, match="unused observation"):
        rig.session.execute(1, action())
    assert len(rig.hardware.targets) == 1


@pytest.mark.parametrize("identifier", [True, 1.0, None, 0, 2, "1"])
def test_wrong_or_coerced_observation_identifiers_cannot_move(rig, identifier):
    rig.session.observe()
    with pytest.raises(ValueError, match="observation ID"):
        rig.session.execute(identifier, action())
    assert rig.events == ["observe"]


def test_old_and_expired_observations_cannot_move(rig):
    rig.session.observe()
    rig.session.observe()
    with pytest.raises(ValueError, match="unused"):
        rig.session.execute(1, action())
    rig.session.observe()
    rig.clock.now += rig.args.max_age
    with pytest.raises(TimeoutError, match="observation expired"):
        rig.session.execute(3, action())
    assert not rig.gripper.actions
    assert not rig.hardware.targets


@pytest.mark.parametrize("close_request", [True, False])
def test_protocol_close_or_eof_stops_both_actuators(rig, close_request, caplog):
    items = [{"op": "observe"}, execute_request()]
    if close_request:
        items.append({"op": "close"})
    output = io.BytesIO()
    with caplog.at_level(logging.INFO, logger=worker.__name__):
        assert worker.serve_requests(requests(*items), output, rig.session) == 0
    replies = frames(output)
    assert replies[0]["__control__"].item() is True
    assert replies[2]["__observation_id__"].item() == 1
    assert rig.events[-2:] == ["gripper-stop", "arm-hold"]
    if close_request:
        assert replies[-1]["__stopped__"].item() is True
    stop_log = next(record for record in caplog.records if "Stopping LIVE execution:" in record.message)
    assert stop_log.levelno == (logging.INFO if close_request else logging.WARNING)


@pytest.mark.parametrize("target", [[0] * 8, [0.8] * 7 + [432, 80], [0] * 7 + [1001, 80]])
def test_invalid_model_targets_abort_before_any_motion(rig, target):
    output = io.BytesIO()
    request = execute_request() | {"action": target}
    assert worker.serve_requests(requests({"op": "observe"}, request), output, rig.session) == 1
    assert frames(output)[-1]["__stopped__"].item() is True
    assert not rig.hardware.targets
    assert not rig.gripper.actions


@pytest.mark.parametrize(
    "payload", [b"not json\n", b"[]\n", b'{"op":"observe","extra":1}\n', b"x" * 4097, b'{"op":"close"}']
)
def test_bad_protocol_aborts_and_acknowledges_stop(rig, payload):
    output = io.BytesIO()
    assert worker.serve_requests(io.BytesIO(payload), output, rig.session) == 1
    assert "__error__" in frames(output)[-1]
    assert frames(output)[-1]["__stopped__"].item() is True
    assert rig.events == ["gripper-stop", "arm-hold"]


@pytest.mark.parametrize("failure", ["gripper", "deadline", "ack"])
def test_partial_execution_failure_stops_before_error_response(rig, failure):
    if failure == "gripper":
        rig.gripper.failure = "gripper command failed"
    elif failure == "deadline":
        rig.gripper.delay = 0.6
    else:
        rig.hardware.failure = "ack"
    output = io.BytesIO()
    assert worker.serve_requests(requests({"op": "observe"}, execute_request()), output, rig.session) == 1
    reply = frames(output)[-1]
    assert "__executed_action__" not in reply
    assert reply["__stopped__"].item() is True
    assert rig.events[-2:] == ["gripper-stop", "arm-hold"]
    assert len(rig.hardware.targets) == (1 if failure == "ack" else 0)
    with pytest.raises(RuntimeError, match="new session"):
        rig.session.observe()


@pytest.mark.parametrize("failure", ["timeout", "feedback"])
def test_watchdog_stops_without_another_client_request_and_never_resumes(rig, failure):
    rig.session.observe()
    rig.session.execute(1, action())
    rig.session.start_watchdog()
    try:
        if failure == "timeout":
            rig.clock.now += rig.args.command_timeout
        else:
            rig.hardware.failure = "feedback"
        assert rig.hardware.stopped.wait(1), "Independent watchdog did not attempt a hold"
        with pytest.raises(RuntimeError, match="new session"):
            rig.session.execute(1, action())
        assert len(rig.hardware.targets) == 1
    finally:
        rig.session.close()
    assert not rig.session.watchdog.is_alive()


@pytest.mark.parametrize("failed_stops", [("gripper",), ("arm",), ("gripper", "arm")])
def test_stop_failure_never_prevents_other_stop_or_reports_success(rig, failed_stops):
    if "gripper" in failed_stops:
        rig.gripper.stop_failure = "Stop RPC failed"
    if "arm" in failed_stops:
        rig.hardware.stop_failure = "Hold feedback failed"
    output = io.BytesIO()
    assert worker.serve_requests(requests({"op": "close"}), output, rig.session) == 1
    reply = frames(output)[-1]
    assert reply["__stopped__"].item() is False
    assert "STOP NOT CONFIRMED" in reply["__error__"].item()
    assert rig.events == ["gripper-stop", "arm-hold"]


@pytest.fixture
def rpc(monkeypatch):
    class RPCError(Exception):
        pass

    calls, bound, states = [], [], [state()]
    failures = {}
    closed = []

    def bind(name):
        def call(request, *, timeout):
            calls.append((name, request, timeout))
            failure = failures.get(name)
            if isinstance(failure, Exception):
                raise failure
            if name == "GetAllState":
                return states.pop(0) if len(states) > 1 else states[0]
            return SimpleNamespace(success=not failure, message=failure or "")

        call.future = lambda request, **kwargs: SimpleNamespace(result=lambda: call(request, **kwargs))
        return call

    def unary_unary(path, **kwargs):
        name = path.rsplit("/", 1)[-1]
        bound.append(name)
        return bind(name)

    channel = SimpleNamespace(unary_unary=unary_unary, close=lambda: closed.append(True))
    pb = SimpleNamespace(
        StateResponse=object(),
        Response=SimpleNamespace(FromString=lambda payload: payload),
        ValueRequest=lambda value: SimpleNamespace(SerializeToString=lambda: str(value).encode()),
    )
    monkeypatch.setitem(sys.modules, "DM_V3Gripper", SimpleNamespace(gripper_pb2=pb))
    monkeypatch.setitem(
        sys.modules, "grpc", SimpleNamespace(RpcError=RPCError, insecure_channel=lambda *args, **kwargs: channel)
    )
    monkeypatch.setattr(worker.observations, "bind_gripper_reader", lambda *args: bind("GetAllState"))
    return SimpleNamespace(calls=calls, bound=bound, states=states, failures=failures, closed=closed, error=RPCError)


def test_gripper_preparation_is_read_only_until_first_explicit_action(rpc):
    gripper = worker.GripperCommands(worker.parse_args(["--execute"]))
    try:
        gripper.prepare()
        assert [item[0] for item in rpc.calls] == ["GetAllState"]
        command = np.asarray([0] * 7 + [432, 81])
        gripper.apply(command, deadline=time.monotonic() + 1)
        gripper.apply(command, deadline=time.monotonic() + 1)
        writes = [(name, payload) for name, payload, _ in rpc.calls if name != "GetAllState"]
        assert writes == [
            ("Enable", b""),
            ("SetTargetSpeed", b"20"),
            ("SetTargetForce", b"81"),
            ("SetTargetPosition", b"432"),
        ]
        assert gripper.force_level == 81
        gripper.stop()
        assert rpc.calls[-1][0] == "Stop"
        assert all(0 < timeout <= 0.2 for _, _, timeout in rpc.calls)
        assert "Connect" not in rpc.bound
        assert "Disconnect" not in rpc.bound
    finally:
        gripper.close()
    assert rpc.closed == [True]


@pytest.mark.parametrize("allow_init", [False, True])
def test_gripper_initialization_needs_additional_explicit_permission(rpc, allow_init):
    rpc.states[:] = [state(init_state=0), state()]
    args = worker.parse_args(["--execute", *(["--initialize-gripper"] if allow_init else [])])
    gripper = worker.GripperCommands(args)
    try:
        if allow_init:
            gripper.prepare()
            assert [name for name, _, _ in rpc.calls] == ["GetAllState", "Enable", "Initialization", "GetAllState"]
        else:
            with pytest.raises(RuntimeError, match="additionally permits"):
                gripper.prepare()
            gripper.stop()
            assert [name for name, _, _ in rpc.calls] == ["GetAllState"]
    finally:
        gripper.close()


@pytest.mark.parametrize("failure_type", ["rejected", "rpc"])
def test_gripper_failed_force_command_does_not_update_metadata_or_send_position(rpc, failure_type):
    rpc.failures["SetTargetForce"] = "force rejected" if failure_type == "rejected" else rpc.error("lost response")
    gripper = worker.GripperCommands(worker.parse_args(["--execute"]))
    try:
        with pytest.raises(RuntimeError, match="SetTargetForce"):
            gripper.apply(np.asarray([0] * 7 + [432, 81]), deadline=time.monotonic() + 1)
        assert gripper.force_level == 80
        assert "SetTargetPosition" not in [name for name, _, _ in rpc.calls]
        gripper.stop()
        assert rpc.calls[-1][0] == "Stop"
    finally:
        gripper.close()


def test_gripper_partial_constructor_failure_closes_local_channel(rpc, monkeypatch):
    def incompatible(*args):
        raise worker.observations.ObservationUnavailable("incompatible schema")

    monkeypatch.setattr(worker.observations, "bind_gripper_reader", incompatible)
    with pytest.raises(RuntimeError, match="incompatible schema"):
        worker.GripperCommands(worker.parse_args(["--execute"]))
    assert rpc.closed == [True]
    assert not rpc.calls


def header(age=0):
    seconds = time.time() - age
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(seconds), nanosec=int((seconds % 1) * 1_000_000_000)))


@pytest.fixture
def hardware():
    args = worker.parse_args(["--execute"])
    gripper = FakeGripper([], time.monotonic)
    result = worker.ROSHardware(args, gripper)
    own = SimpleNamespace(node_name="worker", node_namespace="/")
    publishers = {args.joint_command_topic: [own], "/v3_gripper/action": []}
    result.node = SimpleNamespace(
        get_name=lambda: "worker",
        get_namespace=lambda: "/",
        get_publishers_info_by_topic=lambda topic: publishers[topic],
        get_node_names_and_namespaces=list,
    )
    result.publisher = SimpleNamespace(get_subscription_count=lambda: 1)
    result.context = SimpleNamespace(ok=lambda: True)
    result.thread = SimpleNamespace(is_alive=lambda: True)
    result.samples = {
        key: worker.observations.Sample(SimpleNamespace(header=header()), time.monotonic())
        for key in worker.observations.ROS_INPUTS
    }
    joints = result.samples["joints"].value
    joints.name, joints.position, joints.effort = list(args.joint_names), [0.0] * 7, [0.0] * 7
    result.receive_controller(
        SimpleNamespace(
            header=header(),
            joint_names=list(reversed(args.joint_names)),
            reference=SimpleNamespace(positions=[0.0] * 7),
            feedback=SimpleNamespace(positions=[0.0] * 7, velocities=[0.0] * 7),
        )
    )
    result.limits = worker.JointLimits(-np.ones(7), np.ones(7), np.ones(7))
    return result


@pytest.mark.parametrize("conflict", ["publisher", "gripper", "goal", "moving", "tracking", "stale"])
def test_controller_conflicts_and_bad_feedback_prevent_commands(hardware, conflict):
    np.testing.assert_array_equal(hardware.current_joints(), np.zeros(7))
    if conflict == "publisher":
        hardware.node.get_publishers_info_by_topic(hardware.args.joint_command_topic).append(
            SimpleNamespace(node_name="another_owner", node_namespace="/")
        )
    elif conflict == "gripper":
        hardware.node.get_publishers_info_by_topic("/v3_gripper/action").append(object())
    elif conflict == "goal":
        hardware.receive_goals(SimpleNamespace(status_list=[SimpleNamespace(status=2)]))
    elif conflict == "moving":
        hardware.controller.value.feedback.velocities[0] = 0.1
    elif conflict == "tracking":
        hardware.last_target = np.full(7, 0.3)
    else:
        hardware.controller.value.header = header(age=1)
    with pytest.raises(RuntimeError):
        hardware.current_joints()
    assert not hardware.arm_touched


def test_observation_deadline_preserves_original_sensor_age(hardware, monkeypatch):
    hardware.samples["front"].value.header = header(age=0.3)
    monkeypatch.setattr(worker.observations, "make_observation", lambda *args, **kwargs: {})
    before = time.monotonic()
    _, deadline = hardware.observe()
    assert before + 0.15 < deadline <= before + 0.21


def test_missing_tactile_reports_all_missing_inputs_without_commands(hardware):
    for key in ("depth_0", "depth_2", "wrench_0", "wrench_2"):
        del hardware.samples[key]
    with pytest.raises(RuntimeError, match="Missing ROS inputs: depth_0, depth_2, wrench_0, wrench_2"):
        hardware.current_joints()
    assert not hardware.arm_touched


def test_ros_trajectory_contains_one_short_ordered_radian_target(hardware, monkeypatch):
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", SimpleNamespace(Duration=SimpleNamespace))
    monkeypatch.setitem(
        sys.modules,
        "trajectory_msgs.msg",
        SimpleNamespace(JointTrajectory=SimpleNamespace, JointTrajectoryPoint=lambda: SimpleNamespace(velocities=[])),
    )
    messages = []
    hardware.publisher.publish = messages.append
    target = np.arange(7, dtype=np.float64) / 10
    hardware.publish(target)
    message = messages[-1]
    assert message.joint_names == list(hardware.args.joint_names)
    assert len(message.points) == 1
    assert message.points[0].positions == target.tolist()
    assert message.points[0].velocities == []
    assert message.points[0].time_from_start.nanosec == 66_666_667
    hardware.publish(target, hold=True)
    assert messages[-1].points[0].velocities == [0.0] * 7


@pytest.mark.parametrize("bad_hold", ["velocity", "position", "reference", "invalid", "old"])
def test_hold_acknowledgement_requires_fresh_matching_feedback(hardware, bad_hold):
    target = np.zeros(7)
    started = hardware.controller.monotonic_time - 0.01
    hardware.wait_target(target, started, time.monotonic() + 0.03, stopping=True)
    if bad_hold == "velocity":
        hardware.controller.value.feedback.velocities[0] = 0.2
    elif bad_hold == "position":
        hardware.controller.value.feedback.positions[0] = 0.3
    elif bad_hold == "reference":
        hardware.controller.value.reference.positions[0] = 0.3
    elif bad_hold == "invalid":
        hardware.controller.value.feedback.positions[0] = np.nan
    else:
        started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError)):
        hardware.wait_target(target, started, time.monotonic() + 0.02, stopping=True)


def test_startup_failure_stops_before_error_frame_and_closes_resources(rig, monkeypatch, tmp_path):
    class Output(io.BytesIO):
        def close(self):
            self.saved = self.getvalue()
            super().close()

    output = Output()

    def prepare():
        raise RuntimeError("initialization failed")

    original_write = worker.observations.write_frame

    def write(stream, values):
        if "__error__" in values:
            rig.events.append("error-frame")
        original_write(stream, values)

    rig.gripper.prepare = prepare
    monkeypatch.setattr(worker, "GripperCommands", lambda args: rig.gripper)
    monkeypatch.setattr(worker, "ROSHardware", lambda *args: rig.hardware)
    monkeypatch.setattr(worker.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(worker.observations, "reserve_protocol_stdout", lambda: output)
    monkeypatch.setattr(worker.observations, "write_frame", write)
    assert worker.main(["--execute"]) == 1
    assert rig.events == ["start", "gripper-stop", "arm-hold", "error-frame", "arm-close", "gripper-close"]
    assert frames(io.BytesIO(output.saved))[-1]["__stopped__"].item() is True
