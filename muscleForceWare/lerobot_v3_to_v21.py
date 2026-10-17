"""Convert local LeRobot v3 datasets using the repository's LeRobot v2.1 writer.

Run from the repository root:
    uv run python muscleForceWare/lerobot_v3_to_v21.py --help

Feature names, numeric values, task text, and episode boundaries are preserved.
Videos are decoded at episode timestamps and re-encoded by LeRobot; this is not
a byte-preserving video conversion. Native tactile sidecars remain unaligned.
"""

import argparse
from collections.abc import Iterator
import copy
import json
import logging
from pathlib import Path
import shutil

import av
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


def source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source metadata points outside the dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def video_frames(path: Path, timestamps: np.ndarray, fps: int) -> Iterator[np.ndarray]:
    """Decode bounded-memory RGB frames, including episodes in shared videos."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = max(0.0, float(timestamps[0]) - 1.0 / fps)
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        decoded = iter(container.decode(stream))
        previous = None
        current = next(decoded, None)
        for timestamp in timestamps:
            while current is not None:
                if current.time is None:
                    raise ValueError(f"Video frame has no presentation timestamp: {path}")
                if current.time >= timestamp:
                    break
                previous, current = current, next(decoded, None)
            candidates = [frame for frame in (previous, current) if frame is not None]
            if not candidates:
                raise ValueError(f"No video frames in {path}")
            frame = min(candidates, key=lambda item: abs(item.time - timestamp))
            if abs(frame.time - timestamp) > 0.5 / fps + 1e-3:
                raise ValueError(f"No video frame near {timestamp:.6f}s in {path}; nearest is {frame.time}s")
            yield frame.to_ndarray(format="rgb24")


def numeric_value(value, feature: dict, key: str) -> np.ndarray | str:
    if feature["dtype"] == "string":
        if not isinstance(value, str):
            raise ValueError(f"{key}: expected string, got {type(value).__name__}")
        return value
    array = np.asarray(value, dtype=feature["dtype"])
    shape = tuple(feature["shape"])
    if array.ndim == 0 and shape == (1,):
        array = array.reshape(1)
    if array.shape != shape:
        raise ValueError(f"{key}: expected shape {shape}, got {array.shape}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"{key}: non-finite values")
    return array


def copy_extensions(source: Path, output: Path, episode_count: int, *, skip_tactile: bool) -> None:
    # Copy after save_episode: v2.1's writer counts all Parquet files below root.
    for path in sorted(source.iterdir()):
        if path.name in {"meta", "data", "videos", "images"}:
            continue
        if path.name == "tactile":
            if skip_tactile:
                logger.warning("Skipping native tactile sidecars by explicit request")
                continue
            for episode in range(episode_count):
                sidecar = path / f"episode-{episode:06d}"
                if not sidecar.is_dir():
                    raise FileNotFoundError(f"Missing native tactile episode: {sidecar}")
                shutil.copytree(sidecar, output / "tactile" / sidecar.name)
        elif path.is_dir():
            shutil.copytree(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    managed = {
        "info.json",
        "stats.json",
        "tasks.parquet",
        "tasks.jsonl",
        "episodes",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "conversion.json",
    }
    for path in sorted((source / "meta").iterdir()):
        if path.name in managed:
            continue
        if path.is_dir():
            shutil.copytree(path, output / "meta" / path.name)
        else:
            shutil.copy2(path, output / "meta" / path.name)


def convert(
    source: Path,
    output: Path,
    repo_id: str,
    *,
    max_episodes: int | None = None,
    skip_tactile: bool = False,
) -> None:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Use this repository's v2.1 LeRobot environment, not {CODEBASE_VERSION}")
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output directories must not contain one another")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    info = read_json(source / "meta/info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected v3.0 input, got {info.get('codebase_version')}")
    fps = int(info["fps"])
    if fps <= 0 or fps != info["fps"]:
        raise ValueError(f"Expected positive integer FPS, got {info['fps']}")
    episode_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No meta/episodes/**/*.parquet files")
    episodes = sorted(
        (row for path in episode_files for row in pq.read_table(path).to_pylist()),
        key=lambda row: row["episode_index"],
    )
    if [row["episode_index"] for row in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Expected one metadata record per episode, numbered consecutively from zero")
    if sum(row["length"] for row in episodes) != info["total_frames"]:
        raise ValueError("Episode lengths do not match info.json total_frames")
    tasks = {row["task_index"]: row["task"] for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("No episodes to convert")

    features = {}
    for key, value in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        if value["dtype"] == "image":
            raise ValueError(f"Embedded image feature {key} is not supported; this converter supports video datasets")
        features[key] = {
            "dtype": value["dtype"],
            "shape": tuple(value["shape"]),
            "names": copy.deepcopy(value.get("names")),
        }
    video_keys = [key for key, value in features.items() if value["dtype"] == "video"]
    numeric_keys = [key for key in features if key not in video_keys]
    data = pads.dataset(source / "data", format="parquet")
    required = {*numeric_keys, "timestamp", "frame_index", "episode_index", "task_index"}
    if missing := required - set(data.schema.names):
        raise ValueError(f"Missing data columns: {sorted(missing)}")

    writer = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type=info.get("robot_type"),
        features=features,
        use_videos=bool(video_keys),
        video_backend="pyav",
    )
    marker = output / ".conversion_incomplete"
    marker.write_text("Conversion has not completed. Do not train on this directory.\n")
    try:
        for episode in tqdm(episodes, desc="Episodes"):
            index = episode["episode_index"]
            table = data.to_table(columns=sorted(required), filter=pads.field("episode_index") == index)
            table = table.sort_by([("frame_index", "ascending")])
            rows = table.to_pylist()
            if len(rows) != episode["length"] or not rows:
                raise ValueError(f"Episode {index}: metadata length does not match data")
            if [row["frame_index"] for row in rows] != list(range(len(rows))):
                raise ValueError(f"Episode {index}: non-consecutive frame indices")
            timestamps = np.array([row["timestamp"] for row in rows], dtype=np.float64)
            if not np.allclose(timestamps, np.arange(len(rows)) / fps, atol=1e-4, rtol=0):
                raise ValueError(f"Episode {index}: timestamps are not a regular zero-based {fps} Hz grid")
            streams = {}
            try:
                for key in video_keys:
                    prefix = f"videos/{key}"
                    path = source_path(
                        source,
                        info["video_path"].format(
                            video_key=key,
                            chunk_index=episode[f"{prefix}/chunk_index"],
                            file_index=episode[f"{prefix}/file_index"],
                        ),
                    )
                    start, end = episode[f"{prefix}/from_timestamp"], episode[f"{prefix}/to_timestamp"]
                    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
                        raise ValueError(f"Episode {index}: invalid video interval for {key}")
                    if start + timestamps[-1] >= end + 1e-4:
                        raise ValueError(f"Episode {index}: frames exceed video interval for {key}")
                    streams[key] = video_frames(path, timestamps + start, fps)
                for row in tqdm(rows, desc=f"Episode {index}", leave=False):
                    frame = {key: numeric_value(row[key], features[key], key) for key in numeric_keys}
                    frame.update({key: next(stream) for key, stream in streams.items()})
                    frame["task"] = tasks[row["task_index"]]
                    writer.add_frame(frame)
                writer.save_episode()
            finally:
                for stream in streams.values():
                    stream.close()
    finally:
        writer.stop_image_writer()

    logger.info("Copying extension files, including native tactile data unless excluded")
    copy_extensions(source, output, len(episodes), skip_tactile=skip_tactile)
    report = {
        "source": str(source),
        "source_version": info["codebase_version"],
        "output_version": CODEBASE_VERSION,
        "episodes": len(episodes),
        "frames": sum(episode["length"] for episode in episodes),
        "feature_names_preserved": True,
        "standard_indices_and_timestamps_regenerated": True,
        "videos_reencoded": bool(video_keys),
        "tactile_copied": (source / "tactile").is_dir() and not skip_tactile,
        "tactile_aligned": False,
        "max_episodes": max_episodes,
    }
    (output / "meta/conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    marker.unlink()
    logger.info("Converted %d episodes to %s", len(episodes), output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Existing local LeRobot v3.0 dataset")
    parser.add_argument("--output", required=True, type=Path, help="New local output directory; never overwritten")
    parser.add_argument("--repo-id", required=True, help="Output dataset identifier, e.g. ar3/charging_unplug_v21")
    parser.add_argument("--max-episodes", type=int, help="Convert only the first N episodes for a smoke run")
    parser.add_argument(
        "--skip-tactile", action="store_true", help="Explicitly omit potentially large tactile sidecars"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.source, args.output, args.repo_id, max_episodes=args.max_episodes, skip_tactile=args.skip_tactile)


if __name__ == "__main__":
    main()
