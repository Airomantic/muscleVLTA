from collections import deque
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_ar3_gripper as helper


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RpcError(Exception):
    pass


def state(**values):
    return SimpleNamespace(
        **{
            "connected": True,
            "valid": True,
            "stale": False,
            "init_state": 1,
            "grip_state": 1,
            "position": 432,
            "error": 0,
            "update_age_ms": 0.0,
            "message": "",
            **values,
        }
    )


class ConnectConfig:
    def __init__(self, **values):
        self.values = values

    def SerializeToString(self):  # noqa: N802
        return json.dumps(self.values).encode()


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    calls, bindings = [], []
    states = deque([state()])
    responses = {}
    fields = {
        name: SimpleNamespace(number=number, type=kind)
        for name, number, kind in (
            ("valid", 1, 8),
            ("init_state", 2, 5),
            ("grip_state", 3, 5),
            ("position", 4, 5),
            ("error", 7, 5),
            ("update_age_ms", 9, 1),
            ("stale", 10, 8),
            ("connected", 14, 8),
        )
    }
    protobuf = SimpleNamespace(
        StateResponse=SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(fields_by_name=fields), FromString=lambda value: value
        ),
        Response=SimpleNamespace(FromString=lambda value: value),
        ConnectConfig=ConnectConfig,
    )

    class Channel:
        closed = False
        read_delay = 0.0

        def unary_unary(self, path, **kwargs):
            name = path.rsplit("/", 1)[-1]
            bindings.append(name)

            def rpc(request, *, timeout):
                assert timeout > 0
                calls.append((name, request, timeout))
                if name == "GetAllState":
                    clock.now += self.read_delay
                    result = states.popleft() if len(states) > 1 else states[0]
                else:
                    result = responses.get(name, SimpleNamespace(success=True, message="ok"))
                if isinstance(result, BaseException):
                    raise result
                return result

            return rpc

        def close(self):
            self.closed = True

    channel = Channel()
    opened = []

    def open_channel(address, *, options):
        opened.append((address, options))
        return channel

    monkeypatch.setitem(sys.modules, "grpc", SimpleNamespace(insecure_channel=open_channel, RpcError=RpcError))
    monkeypatch.setitem(sys.modules, "DM_V3Gripper", SimpleNamespace(gripper_pb2=protobuf))
    return SimpleNamespace(
        clock=clock,
        channel=channel,
        calls=calls,
        bindings=bindings,
        states=states,
        responses=responses,
        protobuf=protobuf,
        opened=opened,
    )


def run(rig, *argv):
    return helper.prepare(helper.parse_args(["--connect", *argv]), clock=rig.clock, sleep=rig.clock.sleep)


def writes(rig):
    return [name for name, _, _ in rig.calls if name != "GetAllState"]


@pytest.mark.parametrize(
    "argv", [[], ["--initialize"], ["--connect", "--timeout", "nan"], ["--connect", "--motor-id", "0"]]
)
def test_cli_requires_explicit_connection_and_valid_parameters_before_opening_hardware(rig, argv):
    with pytest.raises(SystemExit) as exc:
        helper.main(argv)
    assert exc.value.code == 2
    assert not rig.opened


def test_disconnected_connects_once_then_requires_new_fresh_feedback(rig):
    rig.states.extendleft(
        reversed([state(connected=False, valid=False, stale=True, init_state=-1), state(valid=False, stale=True)])
    )
    run(rig)
    assert writes(rig) == ["Connect"]
    _, payload, timeout = next(call for call in rig.calls if call[0] == "Connect")
    assert json.loads(payload) == {
        "channel": "can0",
        "motor_id": 1,
        "read_interval_s": 0.05,
        "read_timeout_s": 0.1,
        "retry_count": 2,
        "stale_timeout_s": 0.5,
        "command_min_interval_s": 0.02,
    }
    assert timeout == 5.0
    assert [name for name, _, _ in rig.calls].count("GetAllState") == 3
    assert rig.channel.closed
    assert "Disconnect" not in rig.bindings


def test_already_ready_needs_no_write_rpc_and_preserves_server_connection(rig):
    run(rig, "--initialize")
    assert rig.bindings == ["GetAllState"]
    assert not writes(rig)
    assert rig.channel.closed


def test_uninitialized_default_connects_but_never_enables_or_calibrates(rig):
    rig.states.clear()
    rig.states.extend([state(connected=False), state(init_state=0)])
    with pytest.raises(RuntimeError, match="infer_oneclick.sh --ready-only --initialize-gripper"):
        run(rig)
    assert writes(rig) == ["Connect"]
    assert rig.channel.closed


def test_explicit_initialization_waits_for_actual_completion_without_target_commands(rig):
    rig.states.clear()
    rig.states.extend([state(init_state=0), state(init_state=2, position=-12), state()])
    run(rig, "--initialize")
    assert writes(rig) == ["Enable", "Initialization"]
    assert rig.channel.closed


def test_existing_initialization_only_waits_even_without_motion_authorization(rig):
    rig.states.appendleft(state(init_state=2, position=-12))
    run(rig)
    assert not writes(rig)


def test_existing_initialization_timeout_does_not_stop_other_owners_motor(rig):
    rig.states[0] = state(init_state=2)
    with pytest.raises(TimeoutError, match="initialization timed out"):
        run(rig, "--initialize", "--timeout", "0.2")
    assert not writes(rig)
    assert rig.channel.closed


def test_transient_read_error_never_causes_a_blind_connect(rig):
    rig.states.appendleft(RpcError("temporarily unavailable"))
    run(rig)
    assert not writes(rig)
    assert rig.channel.closed


@pytest.mark.parametrize("name", ["Enable", "Initialization"])
@pytest.mark.parametrize("failure", [SimpleNamespace(success=False, message="rejected"), RpcError("timeout")])
def test_initialization_write_failures_attempt_stop_and_close_channel(rig, name, failure, capsys):
    rig.states[0] = state(init_state=0)
    rig.responses[name] = failure
    with pytest.raises(RuntimeError, match=name):
        run(rig, "--initialize")
    assert writes(rig)[-1] == "Stop"
    assert "physical stop is unverified" in capsys.readouterr().err
    assert rig.channel.closed
    assert "Disconnect" not in rig.bindings


def test_interruption_during_calibration_attempts_stop(rig):
    rig.states.clear()
    rig.states.extend([state(init_state=0), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        run(rig, "--initialize")
    assert writes(rig) == ["Enable", "Initialization", "Stop"]
    assert rig.channel.closed


def test_stop_failure_is_reported_without_claiming_stop_success(rig, capsys):
    rig.states[0] = state(init_state=0)
    rig.responses["Initialization"] = RpcError("initialization timeout")
    rig.responses["Stop"] = SimpleNamespace(success=False, message="CAN unreachable")
    with pytest.raises(RuntimeError, match="Initialization"):
        run(rig, "--initialize")
    error = capsys.readouterr().err
    assert "cleanup Stop failed" in error
    assert "CAN unreachable" in error
    assert "Stop RPC acknowledged" not in error


def test_connect_failure_does_not_stop_or_disconnect_motor(rig):
    rig.states[0] = state(connected=False)
    rig.responses["Connect"] = RpcError("no CAN response")
    with pytest.raises(RuntimeError, match="Connect"):
        run(rig)
    assert writes(rig) == ["Connect"]
    assert rig.channel.closed


@pytest.mark.parametrize("timeout", [0.2, 30.0])
def test_initialization_has_fifteen_second_limit_and_respects_total_timeout(rig, timeout):
    rig.states[0] = state(init_state=0)
    with pytest.raises(TimeoutError, match="initialization timed out.*init_state=0"):
        run(rig, "--initialize", "--timeout", str(timeout))
    assert rig.clock.now == pytest.approx(10.0 + min(timeout, 15.0))
    assert writes(rig) == ["Enable", "Initialization", "Stop"]
    assert all(limit <= min(timeout, 5.0) for name, _, limit in rig.calls if name in ("Enable", "Initialization"))


@pytest.mark.parametrize(
    "overrides",
    [{"valid": False}, {"stale": True}, {"update_age_ms": -1}, {"update_age_ms": 501}, {"update_age_ms": float("nan")}],
)
def test_missing_or_stale_feedback_times_out_with_actual_state_and_no_writes(rig, overrides):
    rig.states[0] = state(**overrides)
    with pytest.raises(TimeoutError, match="CAN feedback timed out.*connected=True"):
        run(rig, "--timeout", "0.2")
    assert not writes(rig)
    assert rig.channel.closed


def test_feedback_age_includes_rpc_latency(rig):
    rig.channel.read_delay = 0.6
    with pytest.raises(TimeoutError, match="CAN feedback timed out"):
        run(rig, "--timeout", "1")
    assert not writes(rig)


def test_feedback_received_after_total_deadline_cannot_report_ready(rig):
    rig.channel.read_delay = 0.2
    with pytest.raises(TimeoutError, match="CAN feedback timed out"):
        run(rig, "--timeout", "0.1")
    assert not writes(rig)


@pytest.mark.parametrize("overrides", [{"error": 5}, {"init_state": -1}, {"position": -1}, {"grip_state": -1}])
def test_invalid_fresh_feedback_is_rejected_without_writes(rig, overrides):
    rig.states[0] = state(**overrides)
    with pytest.raises(RuntimeError):
        run(rig, "--initialize")
    assert not writes(rig)
    assert rig.channel.closed


def test_incompatible_feedback_protocol_is_rejected_before_write_binding(rig):
    rig.protobuf.StateResponse.DESCRIPTOR.fields_by_name.pop("connected")
    with pytest.raises(helper.observations.ObservationUnavailable, match="schema is incompatible"):
        run(rig)
    assert not rig.bindings
    assert rig.channel.closed


def test_cli_reports_timeout_as_failure_without_traceback(monkeypatch, capsys):
    def timeout(args):
        raise TimeoutError("CAN feedback timed out")

    monkeypatch.setattr(helper, "prepare", timeout)
    assert helper.main(["--connect"]) == 1
    assert "Gripper preparation failed: CAN feedback timed out" in capsys.readouterr().err
