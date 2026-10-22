import json
from pathlib import Path
import sys

import numpy as np
from openpi_client.ar3_ros_readonly import ReadOnlyAR3Robot
import pytest


@pytest.fixture
def fake_worker(tmp_path):
    script = tmp_path / "fake worker.py"
    script.write_text("""
import io, json, struct, sys
from pathlib import Path
import numpy as np
def send(values):
    data = io.BytesIO()
    np.savez(data, **values)
    payload = data.getvalue()
    sys.stdout.buffer.write(struct.pack('!Q', len(payload)) + payload)
    sys.stdout.buffer.flush()
send({'__ready__': np.array(True)})
for line in sys.stdin:
    request = json.loads(line)
    with Path(sys.argv[1]).open('a') as output:
        output.write(json.dumps(request) + '\\n')
    if request['op'] == 'close':
        send({'__closed__': np.array(True)})
        break
    send({'observation/state': np.arange(8, dtype=np.float32)})
""")
    setup = tmp_path / "ROS setup.bash"
    setup.write_text("")
    calls = tmp_path / "calls.jsonl"
    return {
        "worker_script": script,
        "ros_python": sys.executable,
        "ros_setup": setup,
        "worker_args": [str(calls)],
        "timeout": 5,
    }, calls


def test_observes_real_worker_but_never_sends_an_execute_operation(fake_worker):
    arguments, calls = fake_worker
    robot = ReadOnlyAR3Robot(**arguments)
    try:
        observation = robot.observe()
        np.testing.assert_array_equal(observation["observation/state"], np.arange(8))
        preview = np.arange(9, dtype=np.float32)
        robot.execute(preview)
        np.testing.assert_array_equal(robot.last_prediction, preview)
        assert robot.read_only
    finally:
        robot.close()
    robot.close()
    assert [json.loads(line)["op"] for line in calls.read_text().splitlines()] == ["observe", "close"]
    assert robot._process.poll() == 0


@pytest.mark.parametrize("action", [np.zeros(8), np.full(9, np.nan)])
def test_invalid_preview_is_not_sent_to_worker(fake_worker, action):
    arguments, calls = fake_worker
    robot = ReadOnlyAR3Robot(**arguments)
    try:
        with pytest.raises(ValueError, match="finite 9D"):
            robot.execute(action)
    finally:
        robot.close()
    assert [json.loads(line)["op"] for line in calls.read_text().splitlines()] == ["close"]


def test_worker_error_is_surfaced_and_owned_process_is_closed(fake_worker):
    arguments, _ = fake_worker
    path = Path(arguments["worker_script"])
    path.write_text(
        path.read_text().replace(
            "send({'__ready__': np.array(True)})",
            "send({'__error__': np.array('gripper state unavailable')})",
        )
    )
    with pytest.raises(RuntimeError, match="gripper state unavailable"):
        ReadOnlyAR3Robot(**arguments)


def test_rejects_invalid_timeout_before_launching(fake_worker):
    arguments, calls = fake_worker
    arguments["timeout"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        ReadOnlyAR3Robot(**arguments)
    assert not calls.exists()
