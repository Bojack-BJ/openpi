"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""
import os
import pathlib
import time
import contextlib
from itertools import chain

# Reduce TensorFlow/XLA startup log noise (especially with multiprocessing workers).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _first_present(flat: dict, keys: tuple[str, ...], field_name: str):
    for key in keys:
        if key in flat:
            return flat[key]
    preview = ", ".join(sorted(str(key) for key in flat)[:20])
    raise KeyError(f"Could not find {field_name}; tried {keys}. Available keys start with: {preview}")


class ImageFreeFastUMINormRepack(transforms.DataTransformFn):
    """Repack FastUMI samples for norm stats without touching image/mask/video fields."""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        return {
            "state": _first_present(flat, ("observation.state", "observation/state", "state"), "state"),
            "actions": _first_present(flat, ("action", "actions"), "actions"),
        }


def _transform_names(transform_group: transforms.Group) -> list[str]:
    return [type(transform).__name__ for transform in transform_group.inputs]


def _uses_fastumi_image_transforms(data_config: _config.DataConfig) -> bool:
    names = set(_transform_names(data_config.repack_transforms) + _transform_names(data_config.data_transforms))
    return "FastUMIInputs" in names


def _norm_stats_transforms(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    if not _uses_fastumi_image_transforms(data_config):
        return [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ]

    skipped_names = {
        "ComposePromptWithSubtask",
        "FastUMIInputs",
        "OverlayMasksOnImages",
        "ResizeImages",
    }
    data_transforms = [
        transform for transform in data_config.data_transforms.inputs if type(transform).__name__ not in skipped_names
    ]
    kept = [type(transform).__name__ for transform in data_transforms]
    print(
        "[stats] using image-free FastUMI norm transforms; "
        f"skipped={sorted(skipped_names)} kept={kept}",
        flush=True,
    )
    return [
        ImageFreeFastUMINormRepack(),
        *data_transforms,
        RemoveStrings(),
    ]


def _preview_columns(columns: list[str], limit: int = 8) -> str:
    preview = ", ".join(columns[:limit])
    if len(columns) > limit:
        preview += f", ... ({len(columns)} total)"
    return preview


def _is_visual_feature(column: str, feature: dict | None) -> bool:
    dtype = str((feature or {}).get("dtype", "")).lower()
    shape = tuple((feature or {}).get("shape", ()) or ())
    normalized = column.replace(".", "/").lower()
    return (
        dtype in {"image", "video"}
        or "image" in normalized
        or "video" in normalized
        or "/images/" in normalized
        or "/videos/" in normalized
        or ("mask" in normalized and len(shape) >= 2)
    )


def _lerobot_non_visual_columns(data_config: _config.DataConfig) -> list[str] | None:
    if data_config.repo_id is None or not _uses_fastumi_image_transforms(data_config):
        return None
    try:
        from lerobot.common.datasets import lerobot_dataset

        meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
        features = dict(getattr(meta, "features", {}) or {})
    except Exception as exc:
        print(
            f"[stats] warning: failed to inspect LeRobot metadata for column projection: {exc!r}. "
            "Falling back to full parquet columns.",
            flush=True,
        )
        return None

    columns = [column for column, feature in features.items() if not _is_visual_feature(column, feature)]
    skipped = [column for column, feature in features.items() if _is_visual_feature(column, feature)]
    if not columns or not skipped:
        return None

    print(
        "[stats] projecting LeRobot parquet columns for norm stats; "
        f"kept={_preview_columns(columns)} skipped_visual={_preview_columns(skipped)}",
        flush=True,
    )
    return columns


@contextlib.contextmanager
def _project_lerobot_parquet_columns(columns: list[str] | None):
    if not columns:
        yield
        return

    import openpi.training.lerobot_dataset as openpi_lerobot_dataset

    original_load_dataset = openpi_lerobot_dataset.load_dataset
    logged = False

    def load_dataset_with_projection(path, *args, **kwargs):
        nonlocal logged
        if path == "parquet" and "columns" not in kwargs:
            kwargs["columns"] = columns
            if not logged:
                print(f"[stats] load_dataset('parquet') columns={_preview_columns(columns)}", flush=True)
                logged = True
        return original_load_dataset(path, *args, **kwargs)

    openpi_lerobot_dataset.load_dataset = load_dataset_with_projection
    try:
        yield
    finally:
        openpi_lerobot_dataset.load_dataset = original_load_dataset


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
    with _project_lerobot_parquet_columns(_lerobot_non_visual_columns(data_config)):
        dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        _norm_stats_transforms(data_config),
    )
    target_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    local_batch_size = min(batch_size, target_frames)
    if local_batch_size <= 0:
        raise ValueError(f"Dataset for repo_id={data_config.repo_id!r} is empty; cannot compute normalization stats.")
    if target_frames < len(dataset):
        num_batches = max(1, target_frames // local_batch_size)
        shuffle = True
    else:
        num_batches = max(1, target_frames // local_batch_size)
        shuffle = False
    print(
        f"[stats] dataset_len={len(dataset)} target_frames={target_frames} "
        f"local_batch_size={local_batch_size} num_batches={num_batches} shuffle={shuffle}",
        flush=True,
    )
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
        convert_to_jax=False,
    )
    return data_loader, num_batches


def _get_output_path(config: _config.TrainConfig, data_config: _config.DataConfig):
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must define either asset_id or repo_id to save normalization statistics.")

    data_factory_assets = getattr(getattr(config.data, "assets", None), "assets_dir", None)
    assets_dir = pathlib.Path(data_factory_assets) if data_factory_assets is not None else config.assets_dirs
    return assets_dir / asset_id


def main(config_name: str, max_frames: int | None = None, num_workers: int | None = None):
    os.environ.setdefault("OPENPI_TORCH_DATALOADER_MP_CONTEXT", "fork")
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    if data_config.rlds_data_dir is not None:
        print(f"Using RLDS data path: {data_config.rlds_data_dir}")
    else:
        print(f"Using dataset repo_id: {data_config.repo_id}")
        
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        effective_num_workers = min(config.num_workers, 8) if num_workers is None else num_workers
        print(f"Using norm-stats dataloader workers: {effective_num_workers}")
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            effective_num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    print("Fetching first batch for norm stats...")
    first_batch_t0 = time.perf_counter()
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(f"Fetched first batch in {time.perf_counter() - first_batch_t0:.1f} seconds.")

    for batch in tqdm.tqdm(chain([first_batch], data_iter), total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = _get_output_path(config, data_config)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
