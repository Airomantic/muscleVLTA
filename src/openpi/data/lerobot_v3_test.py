import dataclasses
import json
import pathlib
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.data import lerobot_v3

FPS = 10
IMAGE_SIZE = 32
TACTILE_SHAPE = (8, 12)
TACTILE_PERIOD = 0.07
SENSORS = ("sensor_a", "sensor_b")
VIDEO_KEY = "observation.images.cam"
EXAMPLE_ROOT = pathlib.Path(__file__).resolve().parents[3] / "dataset_example"


def _fixed_list(values: np.ndarray) -> pa.Array:
    values = np.asarray(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def _frame_host_time(episode: int, frame: int) -> float:
    return 100.0 + 10.0 * episode + frame / FPS


def _tactile_host_times(episode: int, num_frames: int) -> np.ndarray:
    # Asynchronous samples (~14 Hz) starting slightly before the first frame.
    num_samples = int(num_frames / FPS / TACTILE_PERIOD) + 2
    return 100.0 + 10.0 * episode - 0.01 + TACTILE_PERIOD * np.arange(num_samples)


def _color(global_frame: int) -> int:
    return 10 + 25 * global_frame


def _write_video(path: pathlib.Path, num_frames: int) -> None:
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = IMAGE_SIZE
        stream.height = IMAGE_SIZE
        stream.pix_fmt = "yuv420p"
        for index in range(num_frames):
            image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), _color(index), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_dataset(
    root: pathlib.Path,
    lengths: tuple[int, ...] = (5, 4),
    *,
    invalid: tuple[tuple[int, int], ...] = (),
    with_video: bool = True,
    with_tactile: bool = True,
) -> pathlib.Path:
    features = {
        "observation.state": {"dtype": "float32", "shape": [3]},
        "observation.state_valid": {"dtype": "float32", "shape": [3]},
        "action": {"dtype": "float32", "shape": [2]},
        VIDEO_KEY: {"dtype": "video", "shape": [IMAGE_SIZE, IMAGE_SIZE, 3]},
        "sync.target_host_s": {"dtype": "float64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table({"task_index": [0, 1], "__index_level_0__": ["task zero", "task one"]}),
        root / "meta" / "tasks.parquet",
    )

    rows = []
    episodes = []
    global_index = 0
    for episode, length in enumerate(lengths):
        episodes.append(
            {
                "episode_index": episode,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": 0,
                f"videos/{VIDEO_KEY}/from_timestamp": global_index / FPS,
                "stats/action/mean": [0.0, 0.0],
            }
        )
        for frame in range(length):
            rows.append(
                {
                    "state": [episode, frame, global_index],
                    "valid": [0.0 if (episode, frame) in invalid else 1.0] * 3,
                    "action": [100 * episode + frame, -frame],
                    "host": _frame_host_time(episode, frame),
                    "timestamp": frame / FPS,
                    "frame": frame,
                    "episode": episode,
                    "index": global_index,
                    "task": episode % 2,
                }
            )
            global_index += 1
    # Store the frames out of order: the reader must sort them by (episode, frame).
    rows.reverse()

    def column(name, dtype):
        return np.asarray([row[name] for row in rows], dtype=dtype)

    table = pa.table(
        {
            "observation.state": _fixed_list(column("state", np.float32)),
            "observation.state_valid": _fixed_list(column("valid", np.float32)),
            "action": _fixed_list(column("action", np.float32)),
            "sync.target_host_s": column("host", np.float64),
            "timestamp": column("timestamp", np.float32),
            "frame_index": column("frame", np.int64),
            "episode_index": column("episode", np.int64),
            "index": column("index", np.int64),
            "task_index": column("task", np.int64),
        }
    )
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if with_video:
        _write_video(root / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4", global_index)

    if with_tactile:
        for episode, length in enumerate(lengths):
            directory = root / "tactile" / f"episode-{episode:06d}"
            directory.mkdir(parents=True)
            host_times = _tactile_host_times(episode, length)
            streams = {}
            for sensor_offset, sensor in enumerate(SENSORS):
                values = np.stack(
                    [
                        np.full(TACTILE_SHAPE, sample + 0.5 * sensor_offset, np.float32)
                        for sample in range(len(host_times))
                    ]
                )
                sidecar = pa.table(
                    {
                        "episode_index": np.full(len(host_times), episode, np.int64),
                        "sample_index": np.arange(len(host_times)),
                        "host_timestamp_s": host_times,
                        "value": _fixed_list(values.reshape(len(host_times), -1)),
                    }
                )
                # Small row groups, as written by the recorder.
                pq.write_table(sidecar, directory / f"tactile_depth.{sensor}.parquet", row_group_size=3)
                streams[f"tactile_depth.{sensor}"] = {"dtype": "float32", "shape": list(TACTILE_SHAPE)}
            (directory / "info.json").write_text(
                json.dumps({"episode_index": episode, "start_host_s": _frame_host_time(episode, 0), "streams": streams})
            )
    return root


def _tactile_options(tmp_path: pathlib.Path, **kwargs) -> lerobot_v3.TactileOptions:
    return lerobot_v3.TactileOptions(
        sensors=SENSORS, resolution=(4, 6), cache_dir=str(tmp_path / "tactile_cache"), **kwargs
    )


def _expected_tactile_sample(episode: int, frame: int, length: int) -> int:
    host_times = _tactile_host_times(episode, length)
    return max(int(np.searchsorted(host_times, _frame_host_time(episode, frame), side="right")) - 1, 0)


def test_is_lerobot_v3_dataset(tmp_path):
    root = _write_dataset(tmp_path / "v3", with_video=False, with_tactile=False)
    assert lerobot_v3.is_lerobot_v3_dataset(root)
    assert not lerobot_v3.is_lerobot_v3_dataset(tmp_path / "missing")

    (tmp_path / "v21" / "meta").mkdir(parents=True)
    (tmp_path / "v21" / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
    assert not lerobot_v3.is_lerobot_v3_dataset(tmp_path / "v21")
    with pytest.raises(ValueError, match="v3.0"):
        lerobot_v3.LeRobotV3Metadata(tmp_path / "v21")


def test_metadata(tmp_path):
    meta = lerobot_v3.LeRobotV3Metadata(_write_dataset(tmp_path / "ds", with_video=False, with_tactile=False))
    assert meta.fps == FPS
    assert meta.tasks == {0: "task zero", 1: "task one"}
    assert sorted(meta.episodes) == [0, 1]
    assert not any(key.startswith("stats/") for key in meta.episodes[0])
    assert meta.video_keys == [VIDEO_KEY]
    assert VIDEO_KEY not in meta.numeric_keys
    assert "action" in meta.numeric_keys
    assert meta.video_start(VIDEO_KEY, 1) == pytest.approx(0.5)
    assert meta.data_file(1) == tmp_path / "ds" / "data" / "chunk-000" / "file-000.parquet"


def test_items_and_sequences(tmp_path):
    root = _write_dataset(tmp_path / "ds", with_video=False, with_tactile=False)
    dataset = lerobot_v3.LeRobotV3Dataset(
        root,
        sequence_keys=("action",),
        sequence_length=3,
        options=lerobot_v3.LeRobotV3Options(video_keys=()),
    )
    assert len(dataset) == 9

    item = dataset[0]
    np.testing.assert_array_equal(item["observation.state"], [0, 0, 0])
    np.testing.assert_array_equal(item["action"], [[0, 0], [1, -1], [2, -2]])
    np.testing.assert_array_equal(item["action_is_pad"], [False, False, False])
    assert item["task"] == "task zero"
    assert int(item["frame_index"]) == 0
    assert "sync.target_host_s" not in item

    # Last frame of episode 0: the sequence is clamped to the episode end and flagged as padding.
    item = dataset[4]
    np.testing.assert_array_equal(item["action"], [[4, -4], [4, -4], [4, -4]])
    np.testing.assert_array_equal(item["action_is_pad"], [False, True, True])

    # First frame of episode 1 (frames were stored out of order).
    item = dataset[5]
    np.testing.assert_array_equal(item["observation.state"], [1, 0, 5])
    np.testing.assert_array_equal(item["action"], [[100, 0], [101, -1], [102, -2]])
    assert item["task"] == "task one"
    assert dataset[-1]["observation.state"][1] == 3
    with pytest.raises(IndexError):
        dataset[9]

    # Items are copies: in-place transforms must not modify the dataset.
    dataset[5]["observation.state"][:] = -1
    dataset[5]["action"][:] = -1
    np.testing.assert_array_equal(dataset[5]["observation.state"], [1, 0, 5])
    np.testing.assert_array_equal(dataset[5]["action"][0], [100, 0])


def test_valid_flags_and_episode_subset(tmp_path):
    root = _write_dataset(tmp_path / "ds", invalid=((0, 1), (1, 3)), with_video=False, with_tactile=False)
    options = lerobot_v3.LeRobotV3Options(video_keys=())
    dataset = lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("action",), sequence_length=2, options=options)
    assert len(dataset) == 7
    frames = [(int(dataset[i]["episode_index"]), int(dataset[i]["frame_index"])) for i in range(len(dataset))]
    assert (0, 1) not in frames
    assert (1, 3) not in frames
    # Invalid frames are still part of the action sequences of the previous frames.
    np.testing.assert_array_equal(dataset[0]["action"], [[0, 0], [1, -1]])

    dataset = lerobot_v3.LeRobotV3Dataset(root, options=dataclasses.replace(options, require_valid=False))
    assert len(dataset) == 9

    dataset = lerobot_v3.LeRobotV3Dataset(root, options=dataclasses.replace(options, episodes=(1,)))
    assert len(dataset) == 3
    assert {int(dataset[i]["episode_index"]) for i in range(len(dataset))} == {1}

    with pytest.raises(ValueError, match="not part of the dataset"):
        lerobot_v3.LeRobotV3Dataset(root, options=dataclasses.replace(options, episodes=(7,)))
    with pytest.raises(ValueError, match="Unknown sequence keys"):
        lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("missing",), options=options)


def test_video_decoding(tmp_path):
    root = _write_dataset(tmp_path / "ds", with_tactile=False)
    dataset = lerobot_v3.LeRobotV3Dataset(root)
    for index in (0, 3, 4, 5, 8):
        item = dataset[index]
        image = item[VIDEO_KEY]
        assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert image.dtype == np.uint8
        # Episode 1 starts at its from_timestamp in the concatenated video.
        expected = _color(int(item["index"]))
        assert abs(float(image.mean()) - expected) < 6, (index, image.mean(), expected)
    # Random access backwards still decodes the right frame.
    assert abs(float(dataset[1][VIDEO_KEY].mean()) - _color(1)) < 6

    # Open video containers are dropped when pickling (worker processes).
    restored = pickle.loads(pickle.dumps(dataset))
    assert abs(float(restored[2][VIDEO_KEY].mean()) - _color(2)) < 6


def test_missing_videos(tmp_path):
    root = _write_dataset(tmp_path / "ds", with_video=False, with_tactile=False)
    with pytest.raises(FileNotFoundError):
        lerobot_v3.LeRobotV3Dataset(root)[0]

    dataset = lerobot_v3.LeRobotV3Dataset(root, options=lerobot_v3.LeRobotV3Options(missing_video="black"))
    image = dataset[0][VIDEO_KEY]
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert not image.any()

    dataset = lerobot_v3.LeRobotV3Dataset(root, options=lerobot_v3.LeRobotV3Options(decode_videos=False))
    assert dataset[0][VIDEO_KEY].dtype == np.uint8


def test_tactile_alignment_and_cache(tmp_path):
    lengths = (5, 4)
    root = _write_dataset(tmp_path / "ds", lengths, with_video=False)
    options = lerobot_v3.LeRobotV3Options(video_keys=(), tactile=_tactile_options(tmp_path))
    dataset = lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("action",), sequence_length=3, options=options)

    for index in range(len(dataset)):
        item = dataset[index]
        episode, frame = int(item["episode_index"]), int(item["frame_index"])
        tactile = item["observation.tactile_depth"]
        assert tactile.shape == (3, len(SENSORS), 4, 6)
        for step in range(3):
            expected = _expected_tactile_sample(episode, min(frame + step, lengths[episode] - 1), lengths[episode])
            np.testing.assert_allclose(tactile[step, 0], expected)
            np.testing.assert_allclose(tactile[step, 1], expected + 0.5)

    cache_dir = dataset.tactile_cache_dir()
    cache_files = sorted(cache_dir.glob("episode_*.npy"))
    assert [path.name for path in cache_files] == ["episode_000000.npy", "episode_000001.npy"]
    mtimes = [path.stat().st_mtime_ns for path in cache_files]

    # The cache is reused ...
    restored = pickle.loads(pickle.dumps(dataset))
    lerobot_v3.LeRobotV3Dataset(root, sequence_keys=("action",), sequence_length=3, options=options)
    assert [path.stat().st_mtime_ns for path in cache_files] == mtimes
    np.testing.assert_array_equal(restored[2]["observation.tactile_depth"], dataset[2]["observation.tactile_depth"])

    # ... and rebuilt when a sidecar changes.
    sidecar = root / "tactile" / "episode-000001" / f"tactile_depth.{SENSORS[0]}.parquet"
    table = pq.read_table(sidecar)
    values = np.full((table.num_rows, np.prod(TACTILE_SHAPE)), 3.0, np.float32)
    pq.write_table(table.set_column(table.schema.get_field_index("value"), "value", _fixed_list(values)), sidecar)
    dataset = lerobot_v3.LeRobotV3Dataset(root, options=options)
    np.testing.assert_allclose(dataset[5]["observation.tactile_depth"][0, 0], 3.0)
    np.testing.assert_allclose(dataset[0]["observation.tactile_depth"][0, 0], _expected_tactile_sample(0, 0, 5))


def test_tactile_without_frame_host_times(tmp_path):
    # Falls back to start_host_s + timestamp when the frame table has no host time column.
    root = _write_dataset(tmp_path / "ds", with_video=False)
    options = lerobot_v3.LeRobotV3Options(
        video_keys=(), tactile=_tactile_options(tmp_path, frame_time_key="sync.unknown")
    )
    dataset = lerobot_v3.LeRobotV3Dataset(root, options=options)
    np.testing.assert_allclose(dataset[3]["observation.tactile_depth"][0, 0], _expected_tactile_sample(0, 3, 5))


def test_missing_tactile_sidecars(tmp_path):
    root = _write_dataset(tmp_path / "ds", with_video=False, with_tactile=False)
    options = lerobot_v3.LeRobotV3Options(video_keys=(), tactile=_tactile_options(tmp_path))
    with pytest.raises(FileNotFoundError):
        lerobot_v3.LeRobotV3Dataset(root, options=options)


def test_metadata_paths_must_stay_inside_root(tmp_path):
    root = _write_dataset(tmp_path / "ds", with_video=False, with_tactile=False)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["data_path"] = "../outside/file-{file_index:03d}.parquet"
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="outside of the dataset root"):
        lerobot_v3.LeRobotV3Dataset(root, options=lerobot_v3.LeRobotV3Options(video_keys=()))


@pytest.mark.skipif(not (EXAMPLE_ROOT / "meta" / "info.json").is_file(), reason="dataset_example is not available")
def test_dataset_example(tmp_path):
    meta = lerobot_v3.LeRobotV3Metadata(EXAMPLE_ROOT)
    assert meta.fps == 15
    assert meta.video_keys == ["observation.images.front", "observation.images.wrist"]

    options = lerobot_v3.LeRobotV3Options(
        decode_videos=False,
        tactile=lerobot_v3.TactileOptions(cache_dir=str(tmp_path / "tactile_cache")),
    )
    sequence_keys = ("action", "observation.arm_effort", "observation.gripper_wrench.sensor_0")
    dataset = lerobot_v3.LeRobotV3Dataset(
        EXAMPLE_ROOT, sequence_keys=sequence_keys, sequence_length=15, options=options
    )
    assert len(dataset) == meta.episodes[0]["length"]

    item = dataset[100]
    assert item["observation.state"].shape == (27,)
    assert item["action"].shape == (15, 9)
    assert item["observation.arm_effort"].shape == (15, 7)
    assert item["observation.gripper_wrench.sensor_0"].shape == (15, 6)
    assert item["observation.gripper_wrench.sensor_2"].shape == (6,)
    assert item["observation.tactile_depth"].shape == (15, 2, 72, 96)
    assert item["task"] == meta.tasks[0]
    # The 27D state repeats the joint torques and finger wrenches of the dedicated force features.
    np.testing.assert_allclose(item["observation.state"][8:15], item["observation.arm_effort"][0], atol=1e-5)
    np.testing.assert_allclose(
        item["observation.state"][15:21], item["observation.gripper_wrench.sensor_0"][0], atol=1e-5
    )
    np.testing.assert_allclose(item["observation.state"][21:27], item["observation.gripper_wrench.sensor_2"], atol=1e-5)

    tactile = np.load(sorted(dataset.tactile_cache_dir().glob("episode_*.npy"))[0])
    assert tactile.shape == (len(dataset), 2, 72, 96)
    assert np.isfinite(tactile).all()
    # Contact during the grasp produces clearly visible depth on both fingers.
    assert tactile[:, 0].max() > 0.1
    assert tactile[:, 1].max() > 0.1
