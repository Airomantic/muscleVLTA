import json
import pathlib

import cv2
import numpy as np
from openpi_client import msgpack_numpy
import pytest

from openpi.data import lerobot_v3
from openpi.policies import ar3_force_visuotactile_policy as ar3
from openpi.serving import websocket_policy_server
from scripts import ar3_force_visuotactile_client_example as client_example

DATASET = pathlib.Path(__file__).resolve().parents[1] / "dataset_example"
TACTILE_SHAPE = (72, 96)


@pytest.fixture(scope="module")
def tactile_cache_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("tactile_cache"))


def _wire(data: dict) -> dict:
    """Round trip through the WebSocket serialization."""
    return msgpack_numpy.unpackb(msgpack_numpy.packb(data))


class FakeReactivePolicy:
    """Reactive policy stub for ReactiveSession: action j of a chunk is the chunk-start joints + 0.001 * j."""

    def __init__(self, action_horizon: int) -> None:
        self.action_horizon = action_horizon
        self.replans: list[dict] = []
        self.calls: list[tuple[int, int, dict]] = []

    def reactive_replan(self, obs: dict) -> dict:
        self.replans.append(obs)
        cache = dict.fromkeys(websocket_policy_server._REACTIVE_CACHE_KEYS)
        return {**cache, "action_horizon": self.action_horizon, "sequence_id": len(self.replans)}

    def reactive_get_k_actions(self, *, obs: dict, k: int, current_step: int, sequence_id: int, **cache) -> dict:
        self.calls.append((current_step, k, obs))
        state = np.asarray(obs[ar3.STATE_KEY], np.float32)
        actions = np.zeros((self.action_horizon, ar3.ACTION_DIM), np.float32)
        actions[:, : ar3.NUM_ARM_JOINTS] = state[: ar3.NUM_ARM_JOINTS] + 0.001 * np.arange(self.action_horizon)[:, None]
        actions[:, 7] = state[7]
        actions[:, 8] = 80.0
        # Like Policy.reactive_get_k_actions: the actions of steps [current_step, next_step), padded to the horizon.
        next_step = min(current_step + k, self.action_horizon)
        actions = actions[current_step:next_step]
        actions = np.concatenate([actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)])
        return {"actions": actions, "next_step": next_step, "sequence_id": sequence_id}


class InProcessClient:
    """WebsocketClientPolicy stand-in that runs the server-side ReactiveSession in process."""

    def __init__(self, policy: FakeReactivePolicy, metadata: dict | None = None) -> None:
        self._session = websocket_policy_server.ReactiveSession(policy)
        self._metadata = metadata or {}

    def get_server_metadata(self) -> dict:
        return self._metadata

    def reactive_replan(self, obs: dict, *, k: int) -> dict:
        return _wire(self._session.handle({"op": "replan", "k": k}, _wire(obs)))

    def reactive_step(self, readings: dict, *, k: int, current_step: int | None = None) -> dict:
        request = {"op": "step", "k": k}
        if current_step is not None:
            request["current_step"] = current_step
        return _wire(self._session.handle(request, _wire(readings)))


def test_client_layout_matches_policy_transforms():
    assert client_example.FRONT_IMAGE_KEY == ar3.FRONT_IMAGE_KEY
    assert client_example.WRIST_IMAGE_KEY == ar3.WRIST_IMAGE_KEY
    assert client_example.STATE_KEY == ar3.STATE_KEY
    assert client_example.ARM_EFFORT_KEY == ar3.ARM_EFFORT_KEY
    assert client_example.FINGER_WRENCH_KEYS == ar3.FINGER_WRENCH_KEYS
    assert client_example.TACTILE_KEY == ar3.TACTILE_KEY
    assert client_example.FORCE_LEVEL_KEY == ar3.FORCE_LEVEL_KEY
    assert client_example.JOINT_NAMES == ar3.ARM_JOINT_NAMES
    assert client_example.TACTILE_SENSORS == ar3.TACTILE_SENSORS
    assert client_example.FORCE_LEVEL_RANGE == ar3.FORCE_LEVEL_RANGE
    assert (client_example.STATE_DIM, client_example.ACTION_DIM) == (ar3.STATE_DIM, ar3.ACTION_DIM)
    assert client_example.FORCE_LEVEL_INDEX == ar3.FORCE_LEVEL_ACTION_INDEX
    assert ar3.ACTION_NAMES[client_example.GRIPPER_INDEX] == "gripper.position_sdk"


def test_prepare_observation_downsamples_tactile_like_training():
    depth = np.random.default_rng(0).random((2, 288, 384), dtype=np.float32) * 1.4
    observation = client_example.prepare_observation(
        {ar3.TACTILE_KEY: depth, ar3.STATE_KEY: [0.0] * 8, ar3.FORCE_LEVEL_KEY: 80}, TACTILE_SHAPE
    )
    tactile = observation[ar3.TACTILE_KEY]
    assert tactile.dtype == np.float16
    assert tactile.shape == (2, *TACTILE_SHAPE)
    # Same as the training cache: cv2.INTER_AREA on float32, stored as float16.
    expected = np.stack([cv2.resize(image, (96, 72), interpolation=cv2.INTER_AREA) for image in depth])
    np.testing.assert_allclose(tactile.astype(np.float32), expected.astype(np.float16).astype(np.float32), atol=1e-3)
    assert observation[ar3.STATE_KEY].dtype == np.float32
    assert observation[ar3.FORCE_LEVEL_KEY].dtype == np.float32

    # Models without tactile input do not receive it; non-integer factors are resized by the server.
    assert ar3.TACTILE_KEY not in client_example.prepare_observation({ar3.TACTILE_KEY: depth}, None)
    odd = client_example.prepare_observation({ar3.TACTILE_KEY: depth[:, :287]}, TACTILE_SHAPE)[ar3.TACTILE_KEY]
    assert odd.dtype == np.float32
    assert odd.shape == (2, 287, 384)


def test_limit_action_clips_joint_steps_and_gripper_ranges():
    state = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 500.0], np.float32)
    action = np.array([0.05, 0.9, -0.9, -0.2, 0.0, 0.0, 0.0, 1200.0, 10.0], np.float32)

    limited = client_example.limit_action(action, state, max_joint_step=0.1)
    np.testing.assert_allclose(limited, [0.05, 0.6, -0.6, -0.1, 0.0, 0.0, 0.0, 1000.0, 20.0], atol=1e-6)
    unlimited = client_example.limit_action(action, state)
    np.testing.assert_allclose(unlimited[:7], action[:7])
    np.testing.assert_allclose(unlimited[7:], [1000.0, 20.0])
    assert action[7] == 1200.0  # not modified in place


def test_chunk_mode_executes_the_first_actions_of_each_chunk():
    class ChunkClient:
        def __init__(self) -> None:
            self.observations: list[dict] = []

        def infer(self, obs: dict) -> dict:
            self.observations.append(_wire(obs))
            actions = np.zeros((4, ar3.ACTION_DIM), np.float32)
            actions[:, 0] = 10 * len(self.observations) + np.arange(4)
            actions[:, 7] = 2000.0
            actions[:, 8] = 50.0
            return {"actions": actions, "server_timing": {"infer_ms": 1.0}}

    client = ChunkClient()
    recorder = client_example.EpisodeRecorder()
    steps = client_example.run_policy(
        client,
        client_example.FakeAR3Robot(),
        mode="chunk",
        action_horizon=4,
        prompt="test prompt",
        tactile_shape=TACTILE_SHAPE,
        execute_steps=2,
        num_chunks=3,
        callbacks=[recorder],
    )

    assert steps == 6
    assert len(client.observations) == 3
    observation = client.observations[0]
    assert observation["prompt"] == "test prompt"
    assert observation[ar3.TACTILE_KEY].dtype == np.float16
    assert observation[ar3.TACTILE_KEY].shape == (2, *TACTILE_SHAPE)
    data = recorder.arrays()
    np.testing.assert_array_equal(data["policy_action"][:, 0], [10, 11, 20, 21, 30, 31])
    np.testing.assert_array_equal(data["chunk_step"], [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(data["action"][:, 7], 1000.0)
    summary = recorder.summary()
    assert summary["chunks"] == 3
    assert summary["latency_ms"]["infer"]["count"] == 3
    assert summary["latency_ms"]["infer"]["server_mean"] == 1.0
    assert "evaluation" not in summary


def test_dataset_replay_streams_the_training_force_and_tactile_sequences(tactile_cache_dir):
    policy = FakeReactivePolicy(action_horizon=4)
    robot = client_example.DatasetReplayRobot(
        str(DATASET),
        tactile_shape=TACTILE_SHAPE,
        start_frame=5,
        missing_video="black",
        tactile_cache_dir=tactile_cache_dir,
    )
    recorder = client_example.EpisodeRecorder()
    steps = client_example.run_policy(
        InProcessClient(policy),
        robot,
        mode="reactive",
        action_horizon=4,
        prompt=robot.task,
        tactile_shape=TACTILE_SHAPE,
        reactive_k=2,
        max_steps=10,
        callbacks=[recorder],
    )
    assert steps == 10
    assert robot.frame == 15

    # The training inputs of every chunk start: force / tactile sequences over [t, t + horizon).
    training = lerobot_v3.LeRobotV3Dataset(
        DATASET,
        sequence_keys=("observation.arm_effort", "observation.gripper_wrench.sensor_0"),
        sequence_length=4,
        options=lerobot_v3.LeRobotV3Options(
            decode_videos=False,
            require_valid=False,
            tactile=lerobot_v3.TactileOptions(resolution=TACTILE_SHAPE, cache_dir=tactile_cache_dir),
        ),
    )
    chunk_starts = [5, 9, 13]
    assert len(policy.replans) == len(chunk_starts)
    for frame, obs in zip(chunk_starts, policy.replans, strict=True):
        item = training[frame]
        assert obs["prompt"] == item["task"]
        np.testing.assert_allclose(obs[ar3.STATE_KEY], item["observation.state"][:8])
        np.testing.assert_allclose(obs[ar3.ARM_EFFORT_KEY], item["observation.arm_effort"][0])
        np.testing.assert_array_equal(obs[ar3.TACTILE_KEY], item["observation.tactile_depth"][0])
        assert obs[ar3.FORCE_LEVEL_KEY] == item["action"][8]
        assert obs[ar3.FRONT_IMAGE_KEY].shape == (480, 640, 3)

    # Reactive updates at chunk step 2 carry the readings of chunk steps 0..2, exactly the training sequence prefix.
    updates = [(step, obs) for step, _, obs in policy.calls if step > 0]
    assert [step for step, _ in updates] == [2, 2]
    for frame, (_, obs) in zip(chunk_starts, updates, strict=False):
        item = training[frame]
        np.testing.assert_allclose(obs[ar3.ARM_EFFORT_KEY], item["observation.arm_effort"][:3])
        np.testing.assert_allclose(obs[ar3.FINGER_WRENCH_KEYS[0]], item["observation.gripper_wrench.sensor_0"][:3])
        np.testing.assert_array_equal(obs[ar3.TACTILE_KEY], item["observation.tactile_depth"][:3])
        assert obs[ar3.TACTILE_KEY].dtype == np.float16
        np.testing.assert_allclose(obs[ar3.STATE_KEY], item["observation.state"][:8])  # chunk-start state is kept

    # Executed actions follow the chunk structure (chunk-start joints + 0.001 * chunk step).
    data = recorder.arrays()
    np.testing.assert_array_equal(data["frame_index"], np.arange(5, 15))
    np.testing.assert_array_equal(data["chunk_step"], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    chunk_start_state = data["state"][np.arange(10) - data["chunk_step"]]
    np.testing.assert_allclose(
        data["policy_action"][:, :7], chunk_start_state[:, :7] + 0.001 * data["chunk_step"][:, None], atol=1e-6
    )
    evaluation = recorder.summary()["evaluation"]
    assert evaluation["frames"] == int(data["valid"].sum()) > 0
    assert len(evaluation["joint_mae_deg_by_chunk_step"]) == 4
    assert evaluation["force_level_mae"] == pytest.approx(0.0)


def test_rerun_logger_writes_a_recording(tmp_path):
    rr = pytest.importorskip("rerun")
    path = tmp_path / "replay.rrd"
    logger = client_example.RerunLogger(str(path), application_id="test")
    observation = client_example.prepare_observation(client_example.FakeAR3Robot().observe(), TACTILE_SHAPE)
    for index in range(3):
        logger(
            client_example.Step(
                index=index,
                chunk=0,
                chunk_step=index,
                time_s=0.1 * index,
                observation=observation,
                policy_action=np.full(9, 50.0, np.float32),
                action=np.full(9, 50.0, np.float32),
                recorded={"frame_index": index, "timestamp": index / 15, "action": np.full(9, 40.0), "valid": True},
                request="replan" if index == 0 else None,
                round_trip_ms=5.0 if index == 0 else None,
                server_ms=4.0 if index == 0 else None,
                num_actions=4 if index == 0 else None,
            )
        )
    logger.close()

    assert path.stat().st_size > 0
    if hasattr(rr, "dataframe") and hasattr(rr.dataframe, "load_recording"):
        schema = rr.dataframe.load_recording(str(path)).schema()
        entities = {column.entity_path for column in schema.component_columns()}
        for entity in (
            "/observation.images.front",
            "/observation.tactile_depth.sensor_0",
            "/observation.gripper_wrench.sensor_2",
            "/action",
            "/policy/action",
            "/gripper/target",
            "/compare/joint1.pos/policy",
            "/inference/round_trip_ms",
        ):
            assert entity in entities


def test_main_replays_a_dataset_episode_with_outputs(monkeypatch, tmp_path, tactile_cache_dir):
    policy = FakeReactivePolicy(action_horizon=4)
    metadata = {
        "config_name": "fake",
        "action_horizon": 4,
        "reactive": True,
        "cross_attention_latency": 1,
        "tactile_shape": list(TACTILE_SHAPE),
    }
    monkeypatch.setattr(
        client_example.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda host, port: InProcessClient(policy, metadata),
    )
    output_dir = tmp_path / "replay"
    argv = [
        "--source",
        "dataset",
        "--dataset",
        str(DATASET),
        "--allow-missing-videos",
        "--tactile-cache-dir",
        tactile_cache_dir,
        "--max-steps",
        "9",
        "--output-dir",
        str(output_dir),
    ]
    summary = client_example.main(argv)

    assert summary["mode"] == "reactive"
    assert summary["reactive_k"] == 2
    assert summary["episode"] == 0
    assert summary["prompt"] == "Unplug the white charging plug from the white socket and place it on the table"
    assert summary["steps"] == 9
    assert summary["latency_ms"]["replan"]["count"] == 3
    assert summary["latency_ms"]["step"]["count"] == 2  # chunks start at frames 0, 4, 8; updates at 2 and 6
    assert summary["evaluation"]["frames"] > 0
    rollout = np.load(output_dir / "rollout.npz")
    np.testing.assert_array_equal(rollout["frame_index"], np.arange(9))
    assert rollout["recorded_action"].shape == (9, 9)
    assert json.loads((output_dir / "summary.json").read_text())["steps"] == 9
