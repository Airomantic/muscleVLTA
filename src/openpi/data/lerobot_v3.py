"""Native reader for LeRobot v3.0 datasets.

openpi pins the LeRobot v2.1 library (``lerobot.common.datasets``), which cannot read the v3.0 layout written by
recent LeRobot recorders (several episodes per parquet / mp4 file, episode metadata stored as parquet). This module
reads v3.0 datasets directly with pyarrow and PyAV, so they can be used for training without a lossy conversion:

* ``meta/info.json``, ``meta/tasks.parquet`` and ``meta/episodes/**/*.parquet``
* ``data/chunk-XXX/file-XXX.parquet``: the frame table (may contain several episodes)
* ``videos/<video_key>/chunk-XXX/file-XXX.mp4``: concatenated episodes, an episode starts at the
  ``videos/<video_key>/from_timestamp`` of its metadata record
* optional native tactile sidecars ``tactile/episode-XXXXXX/<stream>.<sensor>.parquet`` (muscleVLTA AR3 recorder).
  They are recorded asynchronously on the host clock (~26 Hz), so they are aligned causally to the frame grid (the
  latest sample with ``host_timestamp_s <= sync.target_host_s`` of the frame), area-downsampled and cached per episode.

Items follow the LeRobot conventions used by openpi: the current frame of every numeric feature, ``sequence_length``
consecutive frames for the sequence keys (clamped at the episode end, with a ``<key>_is_pad`` flag), decoded uint8
HWC images for the video keys and the task string under ``"task"``.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openpi.shared import download
from openpi.shared.lerobot_reader import resolve_lerobot_dataset_root

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
TACTILE_CACHE_ENV = "OPENPI_TACTILE_CACHE_DIR"

_NON_NUMERIC_DTYPES = frozenset({"video", "image", "string"})
_BOOKKEEPING_KEYS = ("episode_index", "frame_index", "timestamp", "task_index", "index")
_TACTILE_CACHE_VERSION = 1
_PLACEHOLDER_IMAGE_SHAPE = (16, 16, 3)


@dataclasses.dataclass(frozen=True)
class TactileOptions:
    """Native tactile image sidecars (e.g. the Daimon visuo-tactile depth maps of the two gripper fingers)."""

    # Sensor names used in the sidecar file names, e.g. ``tactile_depth.sensor_0.parquet``.
    sensors: tuple[str, ...] = ("sensor_0", "sensor_2")
    # Sidecar stream name.
    stream: str = "tactile_depth"
    # Output key of the dataset items. Values have shape [sequence_length, num_sensors, height, width] (float16).
    key: str = "observation.tactile_depth"
    # Spatial resolution (height, width) after area downsampling.
    resolution: tuple[int, int] = (72, 96)
    # Per-frame host time column of the frame table, used to align the asynchronous sidecar samples. If the column
    # is missing, ``start_host_s`` of the sidecar ``info.json`` + ``timestamp`` is used instead.
    frame_time_key: str = "sync.target_host_s"
    # Cache directory for the aligned and downsampled arrays. Defaults to ``$OPENPI_TACTILE_CACHE_DIR`` or
    # ``$OPENPI_DATA_HOME/lerobot_v3_tactile`` (``~/.cache/openpi/lerobot_v3_tactile``).
    cache_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class LeRobotV3Options:
    """Options of the native LeRobot v3.0 reader."""

    # Video features to decode. ``None`` decodes every video feature listed in ``meta/info.json``.
    video_keys: tuple[str, ...] | None = None
    # If false, video features are returned as small black placeholders without decoding (e.g. to compute
    # normalization statistics, which do not depend on the images).
    decode_videos: bool = True
    # Behaviour for missing video files: "error" raises, "black" returns black frames (only for debugging).
    missing_video: Literal["error", "black"] = "error"
    # Skip frames whose ``*_valid`` features (e.g. ``observation.state_valid``, ``action_valid``) are not all 1.
    require_valid: bool = True
    # Optional native tactile sidecars.
    tactile: TactileOptions | None = None
    # Optional subset of episode indices. ``None`` uses all episodes.
    episodes: tuple[int, ...] | None = None
    # Rows before the current one prepended to the sequences of ``history_keys`` (sequence keys and / or the tactile
    # key), clamped to the first row of the episode: those sequences hold history_length + sequence_length rows.
    history_length: int = 0
    history_keys: tuple[str, ...] = ()
    # Number of video files kept open per (worker) process.
    max_open_videos: int = 16
    # Number of per-episode tactile caches kept memory-mapped per (worker) process.
    max_open_tactile: int = 64


def is_lerobot_v3_dataset(repo_id_or_root: str | os.PathLike[str]) -> bool:
    """Returns True if the dataset can be resolved locally and its ``meta/info.json`` declares codebase v3.0."""
    try:
        root = resolve_lerobot_dataset_root(repo_id_or_root)
    except (FileNotFoundError, ValueError):
        return False
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    return json.loads(info_path.read_text()).get("codebase_version") == CODEBASE_VERSION


def _safe_path(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset metadata points outside of the dataset root {root}: {relative}")
    return path


def column_to_numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    """Converts a (chunked) arrow column of scalars or fixed-width lists to a numpy array of shape [N] or [N, D]."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_fixed_size_list(array.type):
        values = array.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(array), array.type.list_size)
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
        if len(lengths) and np.any(lengths != lengths[0]):
            raise ValueError(f"Variable-length list column of type {array.type} is not supported")
        values = array.flatten().to_numpy(zero_copy_only=False)
        return values.reshape(len(array), int(lengths[0]) if len(lengths) else 0)
    return array.to_numpy(zero_copy_only=False)


class LeRobotV3Metadata:
    """Metadata (``meta/``) of a local LeRobot v3.0 dataset."""

    def __init__(self, repo_id_or_root: str | os.PathLike[str]):
        self.root = resolve_lerobot_dataset_root(repo_id_or_root)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
        self.info: dict[str, Any] = json.loads(info_path.read_text())
        version = self.info.get("codebase_version")
        if version != CODEBASE_VERSION:
            raise ValueError(f"Expected a LeRobot {CODEBASE_VERSION} dataset at {self.root}, got {version}")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Invalid fps in {info_path}: {self.info['fps']}")
        self.features: dict[str, dict[str, Any]] = self.info["features"]
        self.tasks = self._load_tasks(self.root / "meta" / "tasks.parquet")
        self.episodes = self._load_episodes(self.root / "meta" / "episodes")

    @property
    def video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    @property
    def numeric_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") not in _NON_NUMERIC_DTYPES]

    def data_file(self, episode_index: int) -> pathlib.Path:
        episode = self.episodes[episode_index]
        relative = self.info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
        )
        return _safe_path(self.root, relative)

    def video_file(self, video_key: str, episode_index: int) -> pathlib.Path:
        episode = self.episodes[episode_index]
        relative = self.info["video_path"].format(
            video_key=video_key,
            chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode[f"videos/{video_key}/file_index"]),
        )
        return _safe_path(self.root, relative)

    def video_start(self, video_key: str, episode_index: int) -> float:
        return float(self.episodes[episode_index][f"videos/{video_key}/from_timestamp"])

    @staticmethod
    def _load_tasks(path: pathlib.Path) -> dict[int, str]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing LeRobot tasks file: {path}")
        table = pq.read_table(path)
        names = table.column_names
        # The task string is stored as the (named) pandas index of the dataframe written by LeRobot.
        task_column = "task" if "task" in names else "__index_level_0__"
        if "task_index" not in names or task_column not in names:
            raise ValueError(f"Unexpected columns in {path}: {names}")
        indices = table.column("task_index").to_pylist()
        texts = table.column(task_column).to_pylist()
        return {int(index): str(text) for index, text in zip(indices, texts, strict=True)}

    @staticmethod
    def _load_episodes(directory: pathlib.Path) -> dict[int, dict[str, Any]]:
        files = sorted(directory.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata found under {directory}")
        episodes: dict[int, dict[str, Any]] = {}
        for path in files:
            schema = pq.read_schema(path)
            # Skip the (large) per-episode statistics.
            columns = [name for name in schema.names if not name.startswith("stats/")]
            for record in pq.read_table(path, columns=columns).to_pylist():
                index = int(record["episode_index"])
                if index in episodes:
                    raise ValueError(f"Duplicate metadata for episode {index} in {directory}")
                episodes[index] = record
        return dict(sorted(episodes.items()))


class _VideoDecoder:
    """Decodes single frames with PyAV, keeping a small LRU cache of open containers."""

    def __init__(self, max_open: int):
        self._max_open = max(1, max_open)
        self._containers: collections.OrderedDict[pathlib.Path, Any] = collections.OrderedDict()

    def _open(self, path: pathlib.Path):
        container = self._containers.pop(path, None)
        if container is None:
            import av

            container = av.open(str(path))
            while len(self._containers) >= self._max_open:
                _, oldest = self._containers.popitem(last=False)
                oldest.close()
        self._containers[path] = container
        return container

    def decode(self, path: pathlib.Path, timestamp: float, fps: float) -> np.ndarray:
        container = self._open(path)
        stream = container.streams.video[0]
        # Seek to the last keyframe before the requested time and decode forward to the closest frame.
        start = max(0.0, timestamp - 1.0 / fps)
        container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        previous = None
        current = None
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"Video frame without presentation timestamp in {path}")
            previous, current = current, frame
            if frame.time >= timestamp:
                break
        candidates = [frame for frame in (previous, current) if frame is not None]
        if not candidates:
            raise ValueError(f"No video frames decoded from {path} around {timestamp:.4f}s")
        best = min(candidates, key=lambda frame: abs(frame.time - timestamp))
        if abs(best.time - timestamp) > 0.5 / fps + 1e-3:
            raise ValueError(
                f"No video frame near {timestamp:.4f}s in {path} (closest is {best.time:.4f}s). "
                "The video does not match the frame table."
            )
        return best.to_ndarray(format="rgb24")

    def close(self) -> None:
        while self._containers:
            _, container = self._containers.popitem(last=False)
            container.close()


class LeRobotV3Dataset:
    """Random-access frame dataset over a local LeRobot v3.0 dataset (map-style, picklable for worker processes)."""

    def __init__(
        self,
        repo_id_or_root: str | os.PathLike[str],
        *,
        sequence_keys: Sequence[str] = (),
        sequence_length: int = 1,
        options: LeRobotV3Options | None = None,
    ):
        self.meta = LeRobotV3Metadata(repo_id_or_root)
        self.options = options or LeRobotV3Options()
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        self.sequence_length = int(sequence_length)

        episodes = list(self.meta.episodes) if self.options.episodes is None else list(self.options.episodes)
        if missing_episodes := sorted(set(episodes) - set(self.meta.episodes)):
            raise ValueError(f"Episodes {missing_episodes} are not part of the dataset {self.meta.root}")
        if not episodes:
            raise ValueError(f"No episodes selected from {self.meta.root}")
        self.episode_indices = sorted(episodes)

        numeric_keys = self.meta.numeric_keys
        if unknown := sorted(set(sequence_keys) - set(numeric_keys)):
            raise ValueError(f"Unknown sequence keys {unknown}. Numeric features: {numeric_keys}")
        self.sequence_keys = tuple(sequence_keys)
        if self.options.history_length < 0:
            raise ValueError(f"history_length must be non-negative, got {self.options.history_length}")
        history_targets = set(self.sequence_keys) | ({self.options.tactile.key} if self.options.tactile else set())
        if unknown := sorted(set(self.options.history_keys) - history_targets):
            raise ValueError(f"History keys {unknown} are neither sequence keys nor the tactile key")
        self.video_keys = tuple(self.meta.video_keys if self.options.video_keys is None else self.options.video_keys)
        if unknown := sorted(set(self.video_keys) - set(self.meta.video_keys)):
            raise ValueError(f"Unknown video keys {unknown}. Video features: {self.meta.video_keys}")

        # Host-clock synchronisation columns are only needed to align the tactile sidecars.
        self.frame_keys = tuple(
            key for key in numeric_keys if not key.startswith("sync.") and key not in self.sequence_keys
        )
        tactile = self.options.tactile
        extra_keys = [tactile.frame_time_key] if tactile is not None else []
        self._arrays = self._load_frames([*self.frame_keys, *self.sequence_keys, *_BOOKKEEPING_KEYS], extra_keys)

        episode_column = self._arrays["episode_index"]
        self._episode_bounds: dict[int, tuple[int, int]] = {}
        for episode_index in self.episode_indices:
            start = int(np.searchsorted(episode_column, episode_index, side="left"))
            end = int(np.searchsorted(episode_column, episode_index, side="right"))
            length = int(self.meta.episodes[episode_index]["length"])
            if end - start != length:
                raise ValueError(f"Episode {episode_index}: expected {length} frames, found {end - start}")
            if not np.array_equal(self._arrays["frame_index"][start:end], np.arange(length)):
                raise ValueError(f"Episode {episode_index}: frame indices are not consecutive from 0")
            self._episode_bounds[episode_index] = (start, end)
        # First / last row (inclusive) of the episode of every row, used to clamp sequences to the episode.
        self._episode_first_row = np.empty(len(episode_column), dtype=np.int64)
        self._episode_last_row = np.empty(len(episode_column), dtype=np.int64)
        for start, end in self._episode_bounds.values():
            self._episode_first_row[start:end] = start
            self._episode_last_row[start:end] = end - 1

        valid = np.ones(len(episode_column), dtype=bool)
        if self.options.require_valid:
            for key in numeric_keys:
                if key.endswith("_valid") and key in self._arrays:
                    flags = self._arrays[key]
                    valid &= np.all(flags.reshape(len(flags), -1) >= 0.5, axis=-1)
        self._samples = np.flatnonzero(valid)
        if not len(self._samples):
            raise ValueError(f"No valid frames in {self.meta.root}")
        if skipped := len(valid) - len(self._samples):
            logger.info("Skipping %d of %d frames whose *_valid flags are not set", skipped, len(valid))

        self._tactile_files: dict[int, pathlib.Path] = {}
        if tactile is not None:
            self._tactile_files = self._prepare_tactile_cache(tactile)
        self._warned_missing_videos: set[str] = set()
        self._video_decoder: _VideoDecoder | None = None
        self._tactile_maps: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()

        logger.info(
            "Loaded LeRobot v3.0 dataset %s: %d episodes, %d frames (%d samples), videos=%s, tactile=%s",
            self.meta.root,
            len(self.episode_indices),
            len(episode_column),
            len(self._samples),
            list(self.video_keys),
            list(tactile.sensors) if tactile is not None else None,
        )

    def __getstate__(self) -> dict[str, Any]:
        # Open video containers and memory maps are per-process resources.
        state = self.__dict__.copy()
        state["_video_decoder"] = None
        state["_tactile_maps"] = collections.OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def num_frames(self) -> int:
        return len(self._arrays["episode_index"])

    def episode_bounds(self, episode_index: int) -> tuple[int, int]:
        """Returns the [start, end) row range of an episode in the frame table of this dataset."""
        return self._episode_bounds[episode_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not -len(self) <= index < len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        row = int(self._samples[index])
        episode_index = int(self._arrays["episode_index"][row])
        # Copies, so that in-place transforms cannot modify the in-memory frame table.
        item: dict[str, Any] = {key: np.array(self._arrays[key][row]) for key in self.frame_keys}
        for key in _BOOKKEEPING_KEYS:
            item.setdefault(key, self._arrays[key][row])

        last_row = int(self._episode_last_row[row])
        rows = np.arange(row, row + self.sequence_length)
        is_pad = rows > last_row
        rows = np.minimum(rows, last_row)
        history = self.options.history_length
        history_rows = np.arange(row - history, row + self.sequence_length)
        history_is_pad = (history_rows < self._episode_first_row[row]) | (history_rows > last_row)
        history_rows = np.clip(history_rows, self._episode_first_row[row], last_row)
        for key in self.sequence_keys:
            key_rows, key_is_pad = (
                (history_rows, history_is_pad) if key in self.options.history_keys else (rows, is_pad)
            )
            item[key] = self._arrays[key][key_rows]
            item[f"{key}_is_pad"] = key_is_pad

        timestamp = float(self._arrays["timestamp"][row])
        for key in self.video_keys:
            item[key] = self._decode_video(key, episode_index, timestamp)

        if (tactile := self.options.tactile) is not None:
            start, _ = self._episode_bounds[episode_index]
            tactile_rows = history_rows if tactile.key in self.options.history_keys else rows
            item[tactile.key] = np.asarray(self._tactile_map(episode_index)[tactile_rows - start])

        item["task"] = self.meta.tasks[int(self._arrays["task_index"][row])]
        return item

    def sequence_change_flags(self, key: str, dim: int, *, threshold: float = 0.0) -> np.ndarray:
        """Per sample: whether ``key[..., dim]`` varies by more than ``threshold`` within the sample's window.

        The window holds the sequence_length rows from the sample on, clamped to the episode end (e.g. the action
        chunk), so the flags mark e.g. chunks containing a gripper open / close action.
        """
        if key not in self._arrays:
            raise ValueError(f"Unknown key {key!r}")
        column = self._arrays[key]
        values = column.reshape(len(column), -1)[:, dim].astype(np.float64)
        rows = self._samples[:, None] + np.arange(self.sequence_length)[None]
        rows = np.minimum(rows, self._episode_last_row[self._samples][:, None])
        return np.ptp(values[rows], axis=1) > threshold

    def _load_frames(self, keys: Sequence[str], optional_keys: Sequence[str]) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(keys))
        files = list(dict.fromkeys(self.meta.data_file(index) for index in self.episode_indices))
        selected = pa.array(self.episode_indices, type=pa.int64())
        columns: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        for path in files:
            names = set(pq.read_schema(path).names)
            if missing := sorted(set(keys) - names):
                raise ValueError(f"Missing columns {missing} in {path}")
            read_keys = keys + [key for key in optional_keys if key in names and key not in keys]
            table = pq.read_table(path, columns=read_keys)
            table = table.filter(pc.is_in(table.column("episode_index").cast(pa.int64()), value_set=selected))
            for key in read_keys:
                columns[key].append(column_to_numpy(table.column(key)))
        arrays = {key: np.concatenate(values) for key, values in columns.items() if len(values) == len(files)}
        order = np.lexsort((arrays["frame_index"], arrays["episode_index"]))
        return {key: value[order] for key, value in arrays.items()}

    def _decode_video(self, key: str, episode_index: int, timestamp: float) -> np.ndarray:
        if not self.options.decode_videos:
            return np.zeros(_PLACEHOLDER_IMAGE_SHAPE, dtype=np.uint8)
        path = self.meta.video_file(key, episode_index)
        if not path.is_file():
            if self.options.missing_video != "black":
                raise FileNotFoundError(f"Missing video file {path}")
            if key not in self._warned_missing_videos:
                logger.warning("Video %s is missing, using black frames for '%s'", path, key)
                self._warned_missing_videos.add(key)
            return np.zeros(tuple(self.meta.features[key]["shape"]), dtype=np.uint8)
        if self._video_decoder is None:
            self._video_decoder = _VideoDecoder(self.options.max_open_videos)
        return self._video_decoder.decode(path, self.meta.video_start(key, episode_index) + timestamp, self.meta.fps)

    def _tactile_map(self, episode_index: int) -> np.ndarray:
        array = self._tactile_maps.pop(episode_index, None)
        if array is None:
            array = np.load(self._tactile_files[episode_index], mmap_mode="r")
            while len(self._tactile_maps) >= max(1, self.options.max_open_tactile):
                self._tactile_maps.popitem(last=False)
        self._tactile_maps[episode_index] = array
        return array

    def tactile_cache_dir(self) -> pathlib.Path:
        tactile = self.options.tactile
        if tactile is None:
            raise ValueError("Tactile sidecars are not enabled")
        base = tactile.cache_dir or os.environ.get(TACTILE_CACHE_ENV)
        base_dir = pathlib.Path(base).expanduser() if base else download.get_cache_dir() / "lerobot_v3_tactile"
        digest = hashlib.sha1(str(self.meta.root).encode()).hexdigest()[:10]
        height, width = tactile.resolution
        name = f"{tactile.stream}.{'+'.join(tactile.sensors)}.{height}x{width}.v{_TACTILE_CACHE_VERSION}"
        return base_dir / f"{self.meta.root.name}-{digest}" / name

    def _frame_host_times(self, tactile: TactileOptions, episode_index: int) -> np.ndarray:
        start, end = self._episode_bounds[episode_index]
        if tactile.frame_time_key in self._arrays:
            return self._arrays[tactile.frame_time_key][start:end].astype(np.float64).reshape(end - start)
        info = self._tactile_info(episode_index)
        return float(info["start_host_s"]) + self._arrays["timestamp"][start:end].astype(np.float64)

    def _tactile_info(self, episode_index: int) -> dict[str, Any]:
        path = self.meta.root / "tactile" / f"episode-{episode_index:06d}" / "info.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing tactile sidecar metadata {path}")
        return json.loads(path.read_text())

    def _prepare_tactile_cache(self, tactile: TactileOptions) -> dict[int, pathlib.Path]:
        cache_dir = self.tactile_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for episode_index in self.episode_indices:
            frame_times = self._frame_host_times(tactile, episode_index)
            sources = [
                self.meta.root / "tactile" / f"episode-{episode_index:06d}" / f"{tactile.stream}.{sensor}.parquet"
                for sensor in tactile.sensors
            ]
            if missing := [str(path) for path in sources if not path.is_file()]:
                raise FileNotFoundError(f"Missing tactile sidecars {missing}")
            signature = {
                "version": _TACTILE_CACHE_VERSION,
                "resolution": list(tactile.resolution),
                "sources": [[path.name, path.stat().st_size, path.stat().st_mtime_ns] for path in sources],
                "frame_times": hashlib.sha1(np.ascontiguousarray(frame_times).tobytes()).hexdigest(),
            }
            path = cache_dir / f"episode_{episode_index:06d}.npy"
            signature_path = path.with_suffix(".json")
            if not (
                path.is_file() and signature_path.is_file() and json.loads(signature_path.read_text()) == signature
            ):
                logger.info("Building tactile cache for episode %d in %s", episode_index, cache_dir)
                array = np.stack(
                    [
                        self._align_tactile(source, frame_times, tactile, self._tactile_info(episode_index))
                        for source in sources
                    ],
                    axis=1,
                )
                _atomic_save(path, array)
                tmp_signature = signature_path.with_name(f".{signature_path.name}.{os.getpid()}.tmp")
                tmp_signature.write_text(json.dumps(signature))
                os.replace(tmp_signature, signature_path)
            files[episode_index] = path
        return files

    @staticmethod
    def _align_tactile(
        path: pathlib.Path, frame_times: np.ndarray, tactile: TactileOptions, info: dict[str, Any]
    ) -> np.ndarray:
        """Returns the causally aligned and downsampled tactile images of one sensor: [num_frames, height, width]."""
        import cv2

        height, width = tactile.resolution
        stream_name = path.name.removesuffix(".parquet")
        source_shape = tuple(info.get("streams", {}).get(stream_name, {}).get("shape", ()))
        parquet = pq.ParquetFile(path)
        host_times = column_to_numpy(parquet.read(columns=["host_timestamp_s"]).column(0)).astype(np.float64)
        if not len(host_times):
            logger.warning("Tactile sidecar %s is empty, using zeros", path)
            return np.zeros((len(frame_times), height, width), dtype=np.float16)

        # Latest sample received at or before the frame time. Frames before the first sample use the first one.
        order = np.argsort(host_times, kind="stable")
        positions = np.searchsorted(host_times[order], frame_times, side="right") - 1
        rows = order[np.clip(positions, 0, len(order) - 1)]
        ages = frame_times - host_times[rows]
        if np.max(np.abs(ages)) > 0.5:
            logger.warning("Tactile samples in %s are up to %.3fs away from the frame times", path, np.abs(ages).max())

        needed = np.unique(rows)
        group_ends = np.cumsum([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
        resized: dict[int, np.ndarray] = {}
        for group, group_end in enumerate(group_ends):
            group_start = group_end - parquet.metadata.row_group(group).num_rows
            selected = needed[(needed >= group_start) & (needed < group_end)]
            if not len(selected):
                continue
            values = column_to_numpy(parquet.read_row_group(group, columns=["value"]).column(0))
            for row in selected:
                image = np.asarray(values[row - group_start], dtype=np.float32)
                image = image.reshape(source_shape if len(source_shape) == 2 else image.shape)
                if image.ndim != 2:
                    raise ValueError(f"Cannot infer the image shape of {path}; add 'streams' to its info.json")
                resized[int(row)] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.stack([resized[int(row)] for row in rows]).astype(np.float16)


def _atomic_save(path: pathlib.Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as file:
        np.save(file, array)
    os.replace(tmp_path, path)
