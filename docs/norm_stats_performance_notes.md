# Norm Stats Performance Notes

This note records the main performance issues we hit in `scripts/compute_norm_stats.py`, what we changed, and what still remains if norm-stats generation is slower than expected.

## Scope

This doc is about:

- startup latency before the progress bar appears
- per-iteration cost while computing normalization stats
- data-loader and worker-side overhead

This doc is not about:

- how normalization stats are used during training and inference
- action space definitions
- model runtime performance

See [norm_stats.md](/Users/lxt/Github/Lumos/openpi/docs/norm_stats.md) for the functional background.

## Main Findings

## 1. A Slow Start Usually Meant DataLoader Startup, Not Statistics Math

The first big delay often happened before the tqdm progress bar appeared.

That time was mostly:

- DataLoader worker startup
- dataset shard opening
- first-sample transform cost
- worker import overhead

It was not the `RunningStats` math itself.

To make this visible, [compute_norm_stats.py](/Users/lxt/Github/Lumos/openpi/scripts/compute_norm_stats.py) now prints:

- `Fetching first batch for norm stats...`
- `Fetched first batch in X.Y seconds.`

This separates:

- startup latency
- steady-state per-batch cost

## 2. Norm Stats Did Not Need JAX Arrays

Originally, the norm-stats path still reused a JAX-oriented data path:

- batches could be converted to JAX arrays
- RLDS path could still materialize JAX arrays

But `compute_norm_stats.py` only needs:

- numpy arrays
- CPU-side batch iteration
- mean/std/quantile accumulation

So we changed the script to use a pure CPU/numpy style path:

- [compute_norm_stats.py](/Users/lxt/Github/Lumos/openpi/scripts/compute_norm_stats.py) now creates torch-style loaders for stats collection
- RLDS stats path uses `convert_to_jax=False`

This does not change the meaning of the computed statistics. It only removes unnecessary runtime overhead.

## 3. Worker Startup Was Paying Unnecessary JAX Import Cost

Another startup issue was that spawned data workers could end up importing JAX-heavy modules even though they were only reading data.

We reduced that by changing:

- [data_loader.py](/Users/lxt/Github/Lumos/openpi/src/openpi/training/data_loader.py)
- [transforms.py](/Users/lxt/Github/Lumos/openpi/src/openpi/transforms.py)

Key change:

- remove top-level JAX imports from the worker-facing path
- delay JAX usage until the actual main-process JAX sharding path needs it

This matters most when:

- `num_workers > 0`
- worker start method is `spawn`
- the config previously used very large worker counts like `32` or `64`

## 4. Norm Stats Should Not Blindly Reuse the Training Worker Count

Training configs often use a large `num_workers`, but that is not always appropriate for norm-stats generation.

For stats generation, very large worker counts can make startup worse instead of better because:

- more worker processes must spawn
- more modules must import
- more dataset handles and shards open at once

So [compute_norm_stats.py](/Users/lxt/Github/Lumos/openpi/scripts/compute_norm_stats.py) now:

- accepts an explicit `num_workers`
- if not provided, uses `min(config.num_workers, 8)` for torch-style datasets

This is a pragmatic cap for the stats script only. It does not change the training config itself.

## Quantile vs Z-Score

This repository supports two normalization styles:

- `z-score norm`
  - uses `mean` and `std`
- `quantile norm`
  - uses `q01` and `q99`

Even though some configs do not currently rely on quantile normalization, we intentionally keep full quantile computation enabled by default in the norm-stats script because future `pi05` workflows may need it.

That means:

- `scripts/compute_norm_stats.py` still computes full `q01/q99`
- optimization work should avoid changing semantics unless explicitly requested

## Current Improvements

After this round, the main norm-stats optimizations are:

- first-batch latency logging in [compute_norm_stats.py](/Users/lxt/Github/Lumos/openpi/scripts/compute_norm_stats.py)
- pure CPU/numpy batch path for stats generation
- `RLDSDataLoader(convert_to_jax=False)` support in [data_loader.py](/Users/lxt/Github/Lumos/openpi/src/openpi/training/data_loader.py)
- worker-facing import path no longer pays unnecessary top-level JAX initialization in:
  - [data_loader.py](/Users/lxt/Github/Lumos/openpi/src/openpi/training/data_loader.py)
  - [transforms.py](/Users/lxt/Github/Lumos/openpi/src/openpi/transforms.py)
- configurable worker count with a more conservative default cap for stats generation

## What Still Likely Dominates If It Is Still Slow

If norm-stats generation is still slow after the changes above, the most likely cause is no longer JAX startup.

The next most likely bottleneck is dataset I/O, especially if the loader is still touching image or video fields while only `state` and `actions` are actually needed for the statistics.

That would show up as:

- first batch improved
- but steady-state `s/it` still high

In that case, the next optimization should be:

- create a stats-only dataset path that skips image/video decoding entirely when computing state/action normalization

## Practical Guidance

If norm-stats generation feels slow, check in this order:

1. Look at the first-batch timing.
   - If this is huge, the problem is startup/worker/data-open overhead.
2. Lower `--num-workers` explicitly.
   - Start with `4` or `8`.
3. Compare steady-state `s/it` after the progress bar starts.
   - If this is still high, suspect dataset I/O, not JAX.
4. Only after that consider deeper stats-specific optimizations.

## Useful Command Pattern

Example:

```bash
python scripts/compute_norm_stats.py your_config_name --num-workers 4
```

This is often a better starting point than blindly reusing the full training worker count.
