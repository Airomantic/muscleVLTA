import socket
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy
import pytest

from openpi.serving import websocket_policy_server as _server

_CACHE = {key: f"cache_{key}" for key in _server._REACTIVE_CACHE_KEYS}


class FakeReactivePolicy:
    """Mimics the reactive API of openpi.policies.policy.Policy; action i of a chunk has the value i."""

    def __init__(self, horizon: int = 6) -> None:
        self.horizon = horizon
        self.replans = []
        self.steps = []

    def infer(self, obs: dict) -> dict:
        return {"actions": np.full((self.horizon, 2), obs["value"], dtype=np.float32)}

    def reactive_replan(self, obs: dict) -> dict:
        self.replans.append(obs)
        return {"action_horizon": self.horizon, "action_dim": 2, "sequence_id": 0, **_CACHE}

    def reactive_get_k_actions(self, *, obs: dict, k: int, current_step: int, sequence_id: int, **cache) -> dict:
        assert cache == _CACHE
        self.steps.append((current_step, obs))
        next_step = min(current_step + k, self.horizon)
        actions = np.repeat(np.arange(self.horizon, dtype=np.float32)[:, None], 2, axis=1)[current_step:next_step]
        # Like Policy.reactive_get_k_actions: padded to the action horizon with the last action.
        padded = np.concatenate([actions, np.repeat(actions[-1:], self.horizon - len(actions), axis=0)])
        return {"actions": padded, "sequence_id": sequence_id, "next_step": next_step}


def test_reactive_session_replans_once_and_streams_readings():
    policy = FakeReactivePolicy(horizon=6)
    session = _server.ReactiveSession(policy)

    out = session.handle({"op": "replan", "k": 4}, {"image": 1, "force": np.zeros(3)})
    np.testing.assert_array_equal(out["actions"][:, 0], [0, 1, 2, 3])
    assert (out["start_step"], out["next_step"], out["action_horizon"]) == (0, 4, 6)

    out = session.handle({"op": "step", "k": 4}, {"force": np.ones((5, 3))})
    np.testing.assert_array_equal(out["actions"][:, 0], [4, 5])
    assert (out["start_step"], out["next_step"]) == (4, 6)
    # The readings replace the chunk-start entries, the rest of the replan observation is kept.
    current_step, obs = policy.steps[-1]
    assert current_step == 4
    assert obs["image"] == 1
    assert obs["force"].shape == (5, 3)
    assert len(policy.replans) == 1

    with pytest.raises(ValueError, match="replan a new chunk"):
        session.handle({"op": "step", "k": 1}, {})
    out = session.handle({"op": "step", "k": 1, "current_step": 2}, {})
    np.testing.assert_array_equal(out["actions"][:, 0], [2])


def test_reactive_session_rejects_invalid_requests():
    session = _server.ReactiveSession(FakeReactivePolicy())
    with pytest.raises(ValueError, match="preceding 'replan'"):
        session.handle({"op": "step", "k": 1}, {})
    with pytest.raises(ValueError, match="Unknown reactive op"):
        session.handle({"op": "reset"}, {})

    class InferOnlyPolicy:
        def infer(self, obs):
            return obs

    with pytest.raises(ValueError, match="does not support reactive inference"):
        _server.ReactiveSession(InferOnlyPolicy())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def test_websocket_server_routes_infer_and_reactive_requests():
    policy = FakeReactivePolicy(horizon=6)
    port = _free_port()
    server = _server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={"action_horizon": 6})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _wait_for_port(port)

    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
    assert client.get_server_metadata() == {"action_horizon": 6}
    np.testing.assert_array_equal(client.infer({"value": np.float32(3.0)})["actions"], np.full((6, 2), 3.0))

    out = client.reactive_replan({"value": np.float32(1.0), "force": np.zeros(3, np.float32)}, k=2)
    assert out["actions"].shape == (2, 2)
    assert (out["start_step"], out["next_step"]) == (0, 2)
    out = client.reactive_step({"force": np.ones((3, 3), np.float32)}, k=2)
    np.testing.assert_array_equal(out["actions"][:, 0], [2, 3])
    assert "infer_ms" in out["server_timing"]
    # The reserved request key never reaches the policy.
    assert all(websocket_client_policy.REACTIVE_REQUEST_KEY not in obs for _, obs in policy.steps)
    assert policy.steps[-1][1]["force"].shape == (3, 3)
