"""Profile OpenPI data loading stages for a train config."""

import argparse
import os
import statistics
import time
from typing import Any

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def _parse_indices(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_index(index: int, length: int) -> int:
    return length + index if index < 0 else index


def _summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    values_sorted = sorted(values)
    p50 = values_sorted[len(values_sorted) // 2]
    p90 = values_sorted[min(len(values_sorted) - 1, int(0.9 * (len(values_sorted) - 1)))]
    return (
        f"n={len(values)} mean={statistics.fmean(values):.4f}s "
        f"p50={p50:.4f}s p90={p90:.4f}s min={min(values):.4f}s max={max(values):.4f}s"
    )


def _time_call(label: str, fn) -> Any:
    start = time.perf_counter()
    value = fn()
    print(f"[{label}] {time.perf_counter() - start:.3f}s", flush=True)
    return value


def _profile_transform_chain(raw_dataset, data_config: _config.DataConfig, indices: list[int], *, skip_norm_stats: bool):
    norm_stats = {} if skip_norm_stats else data_config.norm_stats
    if norm_stats is None:
        raise ValueError("Normalization stats not found. Use --skip-norm-stats to profile without Normalize.")

    transform_chain = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    timings: dict[str, list[float]] = {type(transform).__name__: [] for transform in transform_chain}
    for index in indices:
        sample = raw_dataset[index]
        for transform in transform_chain:
            name = type(transform).__name__
            start = time.perf_counter()
            sample = transform(sample)
            timings[name].append(time.perf_counter() - start)

    print("[transform_profile]")
    for name, values in timings.items():
        print(f"  {name}: {_summarize(values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="Training config name, e.g. sponge_visual_guided_pi0")
    parser.add_argument("--num-workers", type=int, default=None, help="Override config.num_workers")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of DataLoader batches to time")
    parser.add_argument("--sample-indices", type=str, default="0,100,-1", help="Comma-separated dataset indices")
    parser.add_argument("--skip-norm-stats", action="store_true", help="Skip normalization transforms")
    parser.add_argument("--framework", choices=["jax", "pytorch"], default="jax", help="Profile JAX sharding or CPU tensors")
    parser.add_argument("--profile-transforms", action="store_true", help="Time each transform on sample-indices")
    args = parser.parse_args()

    config = _time_call("config", lambda: _config.get_config(args.config_name))
    data_config = _time_call("data_config", lambda: config.data.create(config.assets_dirs, config.model))
    num_workers = config.num_workers if args.num_workers is None else args.num_workers

    print(f"[config] batch_size={config.batch_size} action_horizon={config.model.action_horizon}")
    print(f"[config] num_workers={num_workers} framework={args.framework}")
    print(f"[config] mp_context={os.environ.get('OPENPI_TORCH_DATALOADER_MP_CONTEXT', 'spawn')}")
    print(f"[config] repo_id={data_config.repo_id} dataset_columns={data_config.dataset_columns}")

    raw_dataset = _time_call(
        "create_raw_dataset",
        lambda: _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model),
    )
    print(f"[dataset] len={len(raw_dataset)}")

    transformed_dataset = _time_call(
        "wrap_transforms",
        lambda: _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=args.skip_norm_stats),
    )

    indices = [_resolve_index(index, len(raw_dataset)) for index in _parse_indices(args.sample_indices)]
    indices = [index for index in indices if 0 <= index < len(raw_dataset)]
    if indices:
        raw_times = []
        transformed_times = []
        for index in indices:
            start = time.perf_counter()
            _ = raw_dataset[index]
            raw_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            _ = transformed_dataset[index]
            transformed_times.append(time.perf_counter() - start)
        print(f"[raw_getitem] {_summarize(raw_times)}")
        print(f"[transformed_getitem] {_summarize(transformed_times)}")
        if args.profile_transforms:
            _profile_transform_chain(raw_dataset, data_config, indices, skip_norm_stats=args.skip_norm_stats)

    loader = _time_call(
        "create_torch_dataloader",
        lambda: _data_loader.TorchDataLoader(
            transformed_dataset,
            local_batch_size=config.batch_size,
            shuffle=True,
            num_batches=args.num_batches,
            num_workers=num_workers,
            seed=config.seed,
            framework=args.framework,
        ),
    )

    batch_times = []
    start = time.perf_counter()
    iterator = iter(loader)
    print(f"[iter(loader)] {time.perf_counter() - start:.3f}s")
    for batch_index in range(args.num_batches):
        start = time.perf_counter()
        _ = next(iterator)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)
        print(f"[batch {batch_index:04d}] next={elapsed:.4f}s", flush=True)

    print(f"[batch_next] {_summarize(batch_times)}")
    if batch_times:
        steady = batch_times[1:] if len(batch_times) > 1 else batch_times
        print(f"[batch_next_steady] {_summarize(steady)}")


if __name__ == "__main__":
    main()
