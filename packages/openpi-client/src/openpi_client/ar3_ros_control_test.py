import json
import sys

import numpy as np
from openpi_client.ar3_ros_control import LiveAR3Robot
import pytest


@pytest.fixture
def worker(tmp_path):
    script = tmp_path / "control fixture.py"
    script.write_text("""
import io, json, struct, sys
from pathlib import Path
import numpy as np
mode = sys.argv[2]
def send(values):
    stream = io.BytesIO()
    np.savez(stream, **values)
    payload = stream.getvalue()
    sys.stdout.buffer.write(struct.pack('!Q', len(payload)) + payload)
    sys.stdout.buffer.flush()
send({'__ready__': np.array(True), '__control__': np.array(mode != 'readonly')})
observations = 0
for line in sys.stdin:
    request = json.loads(line)
    with Path(sys.argv[1]).open('a') as log:
        log.write(json.dumps(request) + '\\n')
    if request['op'] == 'close':
        stopped = 'true' if mode == 'badstop' else mode != 'nostop'
        send({'__closed__': np.array(True), '__stopped__': np.array(stopped)})
        break
    if request['op'] == 'observe':
        observations += 1
        if mode == 'observe-error' and observations > 1:
            send({'__error__': np.array('observation failed')})
            continue
        send({'__observation_id__': np.array(1), 'observation/state': np.arange(8, dtype=np.float32)})
    if request['op'] == 'execute':
        if mode == 'stopped-error':
            send({'__error__': np.array('expired prediction'), '__stopped__': np.array(True)})
            break
        identifier = 1.0 if mode == 'floatack' else 2 if mode == 'badack' else 1
        send({'__observation_id__': np.array(identifier),
              '__executed_action__': np.asarray(request['action'], dtype=np.float32) * 0.5})
""")
    setup = tmp_path / "setup.bash"
    setup.write_text("")
    log = tmp_path / "calls"
    args = dict(
        worker_script=script,
        ros_python=sys.executable,
        ros_setup=setup,
        worker_args=[str(log), "normal"],
        timeout=5,
    )
    return args, log


def test_live_action_is_sent_once_and_acknowledged_target_is_returned(worker):
    args, log = worker
    robot = LiveAR3Robot(**args)
    try:
        assert robot.read_only is False
        observation = robot.observe()
        assert "__observation_id__" not in observation
        command = np.arange(9, dtype=np.float32)
        np.testing.assert_array_equal(robot.execute(command), command * 0.5)
        with pytest.raises(RuntimeError, match="Observe before each"):
            robot.execute(command)
    finally:
        robot.close()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["op"] for call in calls] == ["observe", "execute", "close"]
    assert calls[1]["observation_id"] == 1
    assert calls[1]["action"] == command.tolist()
    assert robot._process.poll() == 0


def test_live_transport_refuses_readonly_worker(worker):
    args, _ = worker
    args["worker_args"][-1] = "readonly"
    with pytest.raises(RuntimeError, match="explicitly armed"):
        LiveAR3Robot(**args)


@pytest.mark.parametrize("mode", ["badack", "floatack"])
def test_live_transport_requires_matching_acknowledgement(worker, mode):
    args, _ = worker
    args["worker_args"][-1] = mode
    robot = LiveAR3Robot(**args)
    try:
        robot.observe()
        with pytest.raises(RuntimeError, match="different observation"):
            robot.execute(np.zeros(9))
    finally:
        robot.close()


@pytest.mark.parametrize("mode", ["nostop", "badstop"])
def test_missing_stop_acknowledgement_is_not_silenced(worker, mode):
    args, _ = worker
    args["worker_args"][-1] = mode
    robot = LiveAR3Robot(**args)
    with pytest.raises(RuntimeError, match="stop was not acknowledged"):
        robot.close()
    assert robot._closed
    assert robot._process.poll() == 0


def test_nonfinite_live_actions_never_reach_worker(worker):
    args, log = worker
    robot = LiveAR3Robot(**args)
    try:
        robot.observe()
        with pytest.raises(ValueError, match="finite 9D"):
            robot.execute(np.full(9, np.nan))
    finally:
        robot.close()
    assert "execute" not in log.read_text()


def test_worker_error_with_confirmed_stop_preserves_original_error_on_close(worker):
    args, log = worker
    args["worker_args"][-1] = "stopped-error"
    robot = LiveAR3Robot(**args)
    try:
        robot.observe()
        with pytest.raises(RuntimeError, match="expired prediction"):
            robot.execute(np.zeros(9))
    finally:
        robot.close()
    assert robot._stop_confirmed
    assert [json.loads(line)["op"] for line in log.read_text().splitlines()] == ["observe", "execute"]


def test_failed_observation_invalidates_previous_authorization(worker):
    args, log = worker
    args["worker_args"][-1] = "observe-error"
    robot = LiveAR3Robot(**args)
    try:
        robot.observe()
        with pytest.raises(RuntimeError, match="observation failed"):
            robot.observe()
        with pytest.raises(RuntimeError, match="Observe before each"):
            robot.execute(np.zeros(9))
    finally:
        robot.close()
    assert "execute" not in log.read_text()


def test_worker_exit_without_stop_acknowledgement_is_reported(worker):
    args, _ = worker
    robot = LiveAR3Robot(**args)
    robot._process.terminate()
    robot._process.wait(timeout=3)
    with pytest.raises(RuntimeError, match="without a stop acknowledgement"):
        robot.close()
    assert robot._closed
