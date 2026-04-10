# JAX Qwen Performance Notes

This note records the main performance failures we hit while bringing `qwen2_5_vl` and `qwen3_5_vl` into the JAX `Pi0` path, plus the fixes and screening checklist we should reuse for future backbones.

## Scope

This doc is about:

- JAX compile time
- JAX step time
- backend-specific low-performance paths
- how to separate runtime issues from model-quality issues

This doc is not about:

- HF checkpoint key alignment
- DDP startup problems on the PyTorch trainer
- dataset quality

## Short Version

The core finding from this round was:

- `qwen2.5` is slow to compile for heterogeneous prefix/expert pairs, but runtime is still close to normal transformer behavior.
- `qwen3.5` is a different class of problem: the current JAX path was running a low-performance reference implementation for the hybrid `linear_attention` blocks.
- The main bottleneck was not "thinking mode" and not token count. It was the actual JAX implementation of the recurrent linear-attention path.

## Symptoms We Observed

### Qwen2.5

- `qwen2_5 3B + 3B`:
  - first-step compile roughly `60s`
  - runtime around `1s/it`
- `qwen2_5 3B + 700M` and `3B + 400M`:
  - first-step compile roughly `300s` to `400s`
  - runtime still near the normal training regime

Interpretation:

- compile cost is very sensitive to heterogeneous branch shapes
- runtime is still acceptable because the text stack is mostly standard transformer compute

### Qwen3.5

- `qwen3_5 4B + 400M`:
  - first-step compile can be very long
  - runtime was observed around `240s/it`
- this is far worse than the `qwen2.5` cases and cannot be explained by token length alone

Interpretation:

- this is not just "harder compile"
- it indicates the actual training step is falling onto an inefficient implementation path

## Main Root Causes

## 1. Qwen3.5 JAX Was Running Reference Linear-Attention Code

The main issue was in [text.py](/Users/lxt/Github/Lumos/openpi/src/openpi/models/qwen3_5/text.py).

Important facts:

- `qwen3.5` uses a hybrid pattern:
  - `3 x linear_attention`
  - `1 x full_attention`
- `qwen3_5_4b` has 32 layers, so 24 of them are `linear_attention`
- the JAX path did not have any fused or custom kernel for the linear-attention blocks

What this means in practice:

- no Triton kernel
- no Pallas kernel
- no custom call
- no fast fused recurrent kernel
- no flash-attention-like path for the hybrid blocks

So the bottleneck was not "missing one optimization flag". The whole linear-attention section was already the slow path.

## 2. Depthwise Short Convolution Was Implemented as Token-Wise Scan

Original training/pre-fill behavior:

- `DepthwiseShortConv1D` used `jax.lax.scan` over tokens
- each step built a sliding window with `concatenate`
- this is a reference implementation, not a high-throughput conv path

Why this hurts:

- training uses the `cache is None` path
- that means this slow path was active on the main forward/backward path
- it was repeated in every linear-attention layer

## 3. Gated Delta Recurrence Was Also a Reference Scan

The main recurrent update in `_gated_delta_recurrence(...)` used:

- `jax.lax.scan`
- multiple `einsum`
- explicit recurrent state updates
- `float32` compute state even when activations are lower precision

This is correct and readable, but slow.

It also originally materialized repeated K/V-style shapes in a less efficient way than necessary.

## 4. Remat Helps Memory but Can Hurt Runtime

`qwen3.5` already uses `scan + remat` at the group level.

This helps:

- compile graph size
- activation memory

But it also means:

- backward will recompute more work
- runtime can get worse if the underlying block is already expensive

So `remat` was not the root cause, but it can amplify runtime pain when the block itself is slow.

## 5. Token Count Was Not the Main Explanation

We explicitly checked the likely sequence lengths.

For 224x224 images:

- `qwen3.5` vision uses `patch_size=16`, `merge=2`
  - about `49` tokens per image after merge
- `qwen2.5` vision uses `patch_size=14`, `merge=2`
  - about `64` tokens per image after merge

With 3 cameras plus prompt plus suffix, `qwen3.5` can actually have fewer total tokens than `qwen2.5`.

Conclusion:

- the huge runtime gap was not explained by token workload
- it came from the compute path itself

## Changes We Already Made

## 1. Reduced Compile-Only Pain

These changes help compile and workflow ergonomics, but do not by themselves solve slow runtime:

- `qwen3.5` text stack moved from Python per-layer expansion to group-level `scan + remat`
- first-step compile logging was added in [train.py](/Users/lxt/Github/Lumos/openpi/scripts/train.py)
- JAX persistent compilation cache was added in [train.py](/Users/lxt/Github/Lumos/openpi/scripts/train.py) and [config.py](/Users/lxt/Github/Lumos/openpi/src/openpi/training/config.py)
- train-state/sharding logs were made quieter by default

## 2. Fixed an Actual Runtime Bottleneck in Qwen3.5 Short Conv

Current change in [text.py](/Users/lxt/Github/Lumos/openpi/src/openpi/models/qwen3_5/text.py):

- when `cache is None`, `DepthwiseShortConv1D` now uses `jax.lax.conv_general_dilated`
- only incremental decode keeps the token-wise scan path

Why this matters:

- training and prefix prefill run through `cache is None`
- so this replaces a slow scan with a real conv on the main training path

## 3. Tightened the Gated Delta Recurrence

Current change in [text.py](/Users/lxt/Github/Lumos/openpi/src/openpi/models/qwen3_5/text.py):

- removed unnecessary `repeat_kv`-style materialization in the linear-attention path
- restructured recurrence state into grouped shape
- added `unroll=8` to the recurrent scan
- moved `exp(g)` outside the recurrent scan
- replaced inner-step `einsum` contractions with simpler broadcast + reduction updates
- switched normalization to a lighter `rsqrt(sum(square(x)))` form
- added named scopes around projection, short-conv, and recurrence hotspots for profiling

Why this matters:

- it reduces unnecessary tensor expansion
- it keeps the reference recurrence but makes it less wasteful
- it gives the training path a more runtime-friendly scan body without changing weight layout

## 4. Tuned the Training Scan for Runtime

Current change in [text.py](/Users/lxt/Github/Lumos/openpi/src/openpi/models/qwen3_5/text.py):

- decode/incremental cache path still uses the smaller scan unroll
- training/prefill recurrence now uses a larger scan unroll than decode

Why this matters:

- training sequence lengths are much larger than decode
- a slightly larger unroll can improve GPU utilization on the hot recurrent path
- we keep the decode path conservative to avoid unnecessary compile bloat there

## Screening Checklist for Future Backbones

Before claiming a new JAX backbone is "working", always check these points.

## 1. Separate Compile Problems from Runtime Problems

Measure both:

- first-step compile+execute time
- steady-state step time

If compile is bad but runtime is good:

- look at graph size
- look at `scan`
- look at heterogeneous branch shapes
- use persistent compilation cache

If compile is fine but runtime is terrible:

- assume low-performance kernels or reference implementations first

## 2. Look for Reference Implementations in the Main Path

These are red flags:

- token-wise `lax.scan` inside training forward
- Python `for` loops over layers
- repeated `concatenate` inside inner loops
- large `einsum` chains in recurrent updates
- explicit materialization of repeated K/V heads
- always-on `float32` recurrent state in very deep stacks

## 3. Ask Whether the Fast Path Exists at All

Do not only ask:

- "Did we fail to hit the fused kernel?"

Also ask:

- "Did we ever implement a fused kernel or efficient path in the first place?"

If the answer is "no", the problem is structural, not a missed flag.

## 4. Verify Token Workload with Real Numbers

Always print or derive:

- image token counts after patching and merge
- prompt token length
- suffix token length
- full prefix length

Do not rely on intuition. Many "it is slower because the sequence is longer" guesses are false.

## 5. Watch for Environment-Level GPU Fallbacks

Example from this migration:

- CUDA below `12.3` cannot support XLA command-buffer tracing on the tested setup
- this caused:
  - `StreamBeginCaptureToGraph is not implemented for CUDA below version 12.3`

This is different from model-path slowness, but it is easy to confuse the two.

## 6. Treat Remat as a Tradeoff, Not a Free Win

`remat` can be correct and necessary, but it is not free.

It can:

- reduce activation memory
- reduce compile graph size
- increase step runtime through recomputation

So if runtime is the pain point and memory headroom exists, `remat` should be reconsidered.

## Recommended Debug Flow

When a new backend is slow, use this order.

1. Confirm it is not a startup-only issue.
   - Compare first-step compile vs later steps.
2. Confirm it is not an environment issue.
   - CUDA version
   - command buffer / tracing support
   - sharding actually enabled
3. Confirm token workload.
   - prefix image tokens
   - prompt tokens
   - suffix tokens
4. Inspect the block implementation.
   - reference scan?
   - Python loops?
   - repeated materialization?
   - forced float32 recurrence?
5. Compare against the nearest fast backend.
   - here, `qwen2.5` was the best control
6. Optimize the hottest reference path first.
   - replace token-wise conv with real conv
   - reduce recurrence materialization
   - only then look at secondary issues like remat tuning

## What to Avoid in Future JAX Backbone Ports

- Do not stop at "shape-compatible and loss runs".
- Do not assume official architecture parity implies efficient kernels.
- Do not land deep recurrent blocks without asking what the high-performance training path is.
- Do not add heterogeneous prefix/expert structures without checking compile-time effects.
- Do not use wandb image previews as evidence that model inputs are black or corrupted.

## Next Likely Improvements

The current `qwen3.5` path is better than before, but it is still not a truly high-performance implementation.

Most promising next steps:

- make `remat` configurable for `qwen3.5`
- add named scopes or profiler labels around:
  - short conv
  - gated delta recurrence
  - full attention
- benchmark forward-only vs backward-inclusive step time
- consider a more specialized recurrent implementation if `qwen3.5` will remain a primary training target
