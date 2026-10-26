import json
import logging
import pathlib
from types import SimpleNamespace

import cv2
import numpy as np
from openpi_client import msgpack_numpy
import pyarrow.parquet as pq
import pytest

from openpi.data import lerobot_v3
from openpi.policies import ar3_force_visuotactile_policy as ar3
from openpi.serving import websocket_policy_server
from scripts import ar3_force_visuotactile_client_example as client_example
from scripts.ar3_action_limits import JointLimits

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
    assert (client_example.STATE_NAMES, client_example.ACTION_NAMES) == (ar3.STATE_NAMES, ar3.ACTION_NAMES)
    assert client_example.ACTION_NAMES[client_example.GRIPPER_INDEX] == "gripper.position_sdk"


def test_readonly_source_uses_real_worker_without_motion_api(monkeypatch):
    from openpi_client import ar3_ros_readonly

    requested = {}
    expected = object()

    def create_worker(**kwargs):
        requested.update(kwargs)
        return expected

    monkeypatch.setattr(ar3_ros_readonly, "ReadOnlyAR3Robot", create_worker)
    args = client_example.parse_args(["--source", "ros-readonly", "--force-level", "75", "--ros-domain-id", "112"])
    assert client_example.make_robot(args, {}) is expected
    assert requested["worker_script"].name == "ar3_ros_observation_worker.py"
    assert requested["domain_id"] == 112
    assert requested["worker_args"][requested["worker_args"].index("--force-level") + 1] == "75.0"
    assert "--execute" not in requested["worker_args"]


def test_live_source_threads_explicit_limits_and_initialization_to_worker(monkeypatch):
    from openpi_client import ar3_ros_control

    requested = {}
    monkeypatch.setattr(ar3_ros_control, "LiveAR3Robot", lambda **kwargs: requested.update(kwargs))
    args = client_example.parse_args(
        ["--source", "ros", "--execute", "--fps", "15", "--initialize-gripper", "--max-joint-velocity", "0.1"]
    )
    client_example.make_robot(args, {})
    assert requested["worker_script"].name == "ar3_ros_control_worker.py"
    assert "--execute" in requested["worker_args"]
    assert "--initialize-gripper" in requested["worker_args"]
    assert requested["worker_args"][requested["worker_args"].index("--max-joint-velocity") + 1] == "0.1"


@pytest.mark.parametrize(
    "options",
    [
        ["--source", "ros"],
        ["--source", "ros-readonly", "--execute"],
        ["--source", "ros", "--execute", "--fps", "100"],
        ["--source", "ros", "--execute", "--fps", "15", "--command-timeout", "0.05"],
        ["--source", "ros", "--execute", "--fps", "15", "--command-timeout", "3"],
    ],
)
def test_live_invalid_opt_in_or_deadlines_fail_before_policy_or_worker(options):
    with pytest.raises(ValueError, match="Hardware execution|Live execution|--command-timeout"):
        client_example.run(client_example.parse_args(options))


def test_live_warms_whole_chunk_before_worker_and_uses_explicit_prompt(monkeypatch, caplog):
    policy = FakeReactivePolicy(action_horizon=4)
    client = InProcessClient(policy, {"action_horizon": 4, "reactive": True, "cross_attention_latency": 1})
    events, actions = [], []

    class Robot(client_example.FakeAR3Robot):
        handles_action_limits = True

        def execute(self, target):
            actions.append(target.copy())
            return super().execute(target)

        def close(self):
            events.append("close")

    def create_robot(args, metadata):
        assert len(policy.replans) == 1
        assert [step for step, _, _ in policy.calls] == [0, 2]
        events.append("create")
        return Robot()

    monkeypatch.setattr(client_example, "make_robot", create_robot)
    monkeypatch.setattr(client_example.time, "sleep", lambda seconds: None)
    args = client_example.parse_args(
        ["--source", "ros", "--execute", "--fps", "15", "--max-steps", "3", "--prompt", "a different task"]
    )
    with caplog.at_level(logging.INFO):
        summary = client_example.run(args, policy_client=client)
    assert summary["prompt"] == "a different task"
    assert all(observation["prompt"] == "a different task" for observation in policy.replans)
    assert len(actions) == 3
    assert events == ["create", "close"]
    assert "Control step budget reached (--max-steps=3); requesting normal shutdown." in caplog.text


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


@pytest.mark.parametrize("mode", ["chunk", "reactive"])
def test_live_gripper_predictions_are_bounded_without_hiding_raw_joints(mode, caplog):
    predictions = np.asarray(
        [
            [0.08] * 7 + [-0.25, 19.75],
            [0.08] * 7 + [1000.25, 100.25],
            [0.08] * 7 + [400.25, 79.75],
            [0.08] * 7 + [0, 100],
        ],
        dtype=np.float32,
    )
    original = predictions.copy()
    requested = []
    limits = JointLimits(-np.ones(7), np.ones(7), np.ones(7))

    class Client:
        def infer(self, observation):
            return {"actions": predictions}

        def reactive_replan(self, observation, *, k):
            self.next_step = k
            return {"actions": predictions[:k]}

        def reactive_step(self, readings, *, k):
            start = self.next_step
            self.next_step += k
            return {"start_step": start, "actions": predictions[start : self.next_step]}

    class Robot(client_example.FakeAR3Robot):
        handles_action_limits = True

        def execute(self, target):
            requested.append(target.copy())
            applied = limits.bound_action(
                target, self.joints, period=1 / 15, max_step=0.03, max_velocity=0.1, max_deviation=0.25
            )
            super().execute(applied)
            return applied

    recorder = client_example.EpisodeRecorder()
    with caplog.at_level(logging.WARNING):
        steps = client_example.run_policy(
            Client(),
            Robot(),
            mode=mode,
            action_horizon=4,
            prompt="test prompt",
            reactive_k=2,
            num_chunks=2,
            max_joint_step=0.001,
            callbacks=[recorder],
        )

    assert steps == 8
    data = recorder.arrays()
    expected_raw = np.tile(original, (2, 1))
    np.testing.assert_array_equal(predictions, original)
    np.testing.assert_array_equal(data["policy_action"], expected_raw)
    np.testing.assert_array_equal(np.asarray(requested)[:, :7], expected_raw[:, :7])
    np.testing.assert_array_equal(data["action"][:, 7:], np.tile([[0, 20], [1000, 100], [400, 80], [0, 100]], (2, 1)))
    np.testing.assert_allclose(data["action"][0, :7], 0.1 / 15)
    assert "position -0.25 -> 0" in caplog.text
    assert "position 1000.25 -> 1000" in caplog.text
    assert "force level 19.75 -> 20" in caplog.text
    assert "force level 100.25 -> 100" in caplog.text
    assert len([record for record in caplog.records if "clipped gripper targets" in record.message]) == 4


@pytest.mark.parametrize(("index", "value"), [(7, np.nan), (7, -np.inf), (8, np.inf)])
def test_nonfinite_gripper_predictions_fail_before_saturation_or_execution(monkeypatch, index, value):
    predictions = np.asarray([[0.0] * 7 + [500, 80]], dtype=np.float32)
    predictions[0, index] = value
    robot = client_example.FakeAR3Robot()
    robot.handles_action_limits = True
    monkeypatch.setattr(robot, "execute", lambda target: pytest.fail("Invalid predictions must never be executed"))
    client = SimpleNamespace(infer=lambda observation: {"actions": predictions})
    with pytest.raises(ValueError, match="non-finite actions"):
        client_example.run_policy(client, robot, mode="chunk", action_horizon=1, prompt="test", max_steps=1)


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


@pytest.mark.parametrize("recorded", [True, False])
def test_rerun_shows_recorded_actions_only_for_sources_with_recordings(recorded):
    rrb = pytest.importorskip("rerun.blueprint")
    names, stack = set(), [client_example.RerunLogger.blueprint(rrb, recorded=recorded).root_container]
    while stack:
        item = stack.pop()
        names.add(item.name)
        if isinstance(item, rrb.Container):
            stack.extend(item.contents)
    assert "action (policy)" in names
    # Live sources have no recorded (teleoperation) actions: no empty "action (recorded)" panel.
    assert ("action (recorded)" in names) == recorded
    assert ("policy vs recorded" in names) == recorded
    assert ("policy vs measured state" in names) != recorded


def test_record_parquet_appends_inference_episodes_without_videos_or_tactile(monkeypatch, tmp_path):
    policy = FakeReactivePolicy(action_horizon=4)
    metadata = {"config_name": "fake", "action_horizon": 4, "reactive": True, "cross_attention_latency": 1}
    metadata["tactile_shape"] = list(TACTILE_SHAPE)
    monkeypatch.setattr(
        client_example.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda host, port: InProcessClient(policy, metadata),
    )
    root = tmp_path / "dataRecordParquet"
    # A tiny joint step limit, so that executed and predicted joint targets differ.
    argv = ["--source", "fake", "--fps", "0", "--max-joint-step", "0.0005", "--record-parquet", str(root)]
    summary = client_example.main([*argv, "--max-steps", "5"])
    assert summary["record_parquet"] == {"root": str(root), "episode_index": 0, "frames": 5, "outcome": "completed"}

    class InterruptedRobot(client_example.FakeAR3Robot):
        observations = 0

        def observe(self):
            self.observations += 1
            if self.observations > 3:
                raise KeyboardInterrupt  # Ctrl+C, or VS Code Stop with "onTerminate": "KeyboardInterrupt"
            return super().observe()

    monkeypatch.setattr(client_example, "make_robot", lambda args, metadata: InterruptedRobot())
    summary = client_example.main([*argv, "--prompt", "other task"])
    assert summary["record_parquet"] == {"root": str(root), "episode_index": 1, "frames": 3, "outcome": "interrupted"}

    meta = lerobot_v3.LeRobotV3Metadata(root)
    assert (meta.fps, meta.info["total_episodes"], meta.info["total_frames"]) == (15.0, 2, 8)
    assert meta.tasks == {0: client_example.PROMPT, 1: "other task"}
    assert not [key for key in meta.features if any(word in key for word in ("image", "wrench", "tactile"))]
    assert [meta.episodes[index]["inference/outcome"] for index in (0, 1)] == ["completed", "interrupted"]
    assert json.loads(meta.episodes[1]["inference/args"])["max_joint_step"] == 0.0005
    dataset = lerobot_v3.LeRobotV3Dataset(root)
    assert (len(dataset), dataset.episode_bounds(1)) == (8, (5, 8))
    item = dataset[6]
    assert (item["task"], int(item["frame_index"]), int(item["index"])) == ("other task", 1, 6)
    assert json.loads((root / "meta/stats.json").read_text())["action"]["count"] == [8]

    table = pq.read_table(meta.data_file(0))
    state, action, predicted = (
        lerobot_v3.column_to_numpy(table.column(key)) for key in ("observation.state", "action", "policy.action")
    )
    np.testing.assert_array_equal(lerobot_v3.column_to_numpy(table.column("policy.chunk_step")), [0, 1, 2, 3, 0])
    # The state is measured before the executed action of the step (the fake robot reaches its targets).
    np.testing.assert_array_equal(state[1:], action[:-1, :8])
    assert np.all(np.abs(action[:, :7] - state[:, :7]) <= 0.0005 + 1e-7)
    assert np.any(predicted[:, :7] != action[:, :7])

    # Training datasets, other fps and unrelated non-empty directories are never appended to.
    for directory, fps in ((DATASET, 15), (root, 10), (tmp_path, 15)):
        with pytest.raises(ValueError, match="choose another directory"):
            client_example.ParquetDatasetRecorder(directory, fps=fps)


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
