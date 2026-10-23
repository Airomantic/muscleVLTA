"""Compute normalization statistics after applying a config's data transforms.

Run with --config-name CONFIG after setting its dataset repo_id. Statistics are
written to <config.assets_dirs>/<asset_id>/norm_stats.json (by default,
assets/<config>/<asset_id>/norm_stats.json, where asset_id defaults to the
repo_id), not to a pretrained checkpoint. This is the location from which
training loads them. Reuse pretrained statistics only if their observation and
action conventions match your data.

--max-frames optionally limits the dataset scan. State, action, and wrench
statistics are computed only for fields present in the transformed batches.
Videos of LeRobot v3.0 datasets are not decoded, since images do not enter the
statistics (tactile sidecars are aligned and cached on the first run).
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.data.lerobot_v3 as _lerobot_v3
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.lerobot_v3 is not None or (
        data_config.repo_id is not None and _lerobot_v3.is_lerobot_v3_dataset(data_config.repo_id)
    ):
        options = data_config.lerobot_v3 or _lerobot_v3.LeRobotV3Options()
        data_config = dataclasses.replace(data_config, lerobot_v3=dataclasses.replace(options, decode_videos=False))

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions", "wrench"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key not in batch:
                continue
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stat.get_statistics() for key, stat in stats.items() if stat.count > 1}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
