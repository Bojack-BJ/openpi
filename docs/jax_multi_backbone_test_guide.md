# JAX Multi-Backbone Test Guide

This note summarizes how to test the current JAX `Pi0` runtime after the multi-backbone refactor.

For performance-specific findings and the JAX Qwen compile/runtime debugging checklist, see
[jax_qwen_performance_notes.md](/Users/lxt/Github/Lumos/openpi/docs/jax_qwen_performance_notes.md).

## Current Status

### PaliGemma

- JAX `paligemma` remains the most stable backend.
- It is still the reference path for checkpoint-compatible JAX training and inference.

### Qwen2.5-VL

- JAX `qwen2_5_vl` now has:
  - a native JAX backend adapter
  - a vision tower under `src/openpi/models/qwen2_5/vision.py`
  - a projector into the text hidden size
  - backend-owned multimodal positions
  - a 3-stream rotary path in the text stack
- This means the JAX `Pi0` prefix/suffix path is now structurally connected for Qwen.
- It is **not** yet an HF-weight-compatible Qwen2.5-VL port.
- Today, JAX Qwen testing should be treated as:
  - graph/runtime integration validation
  - shape/cache/position-path validation
  - not pretrained-quality validation

### Qwen3.5

- Hugging Face exposes official `Qwen3.5` model repos such as:
  - `Qwen/Qwen3.5-2B`
  - `Qwen/Qwen3.5-4B`
- At the **architecture layer**, `Qwen3.5` is **not** the same as `Qwen2.5`:
  - the official model card and config describe a hybrid layout with `Gated DeltaNet` plus `Gated Attention`
  - `2B` uses `6 × (3 linear-attention + 1 full-attention)` blocks
  - `4B` uses `8 × (3 linear-attention + 1 full-attention)` blocks
- The current repo's `qwen3_5_vl` JAX path now implements:
  - an official-style hybrid `Qwen3.5` text backbone
  - an official-style JAX Qwen3.5 vision tower
  - `GatedDeltaNet` linear-attention blocks
  - `GatedAttention` full-attention blocks
  - a causal sequence mask in `Pi0` for the `qwen3_5_vl` backend
- It now also includes:
  - a `Qwen3_5WeightLoader` aligned to the current official HF text-tree assumptions

In other words:

- `qwen2_5_vl` is the more mature JAX multimodal runtime path today
- `qwen3_5_vl` now exercises a much more faithful official 3.5 text + vision architecture
- `qwen3_5_vl` now targets the official HF Qwen3.5 checkpoint tree more directly, but still needs runtime validation in your environment

### Qwen Pi05

JAX Qwen now supports the `Pi0Config(pi05=True, vlm_backend="qwen2_5_vl" | "qwen3_5_vl")` path.

The important Pi05-specific behavior is:

- robot state must be encoded as discrete language text in the prompt/prefix
- `ModelTransformFactory` should use `discrete_state_input=True`
- Qwen tokenizers format the state through the Pi05 prompt text path before model input
- `Pi0.embed_suffix(...)` skips the continuous state token for Pi05, so the suffix contains action tokens only
- the Qwen action expert uses AdaRMS timestep conditioning through `use_adarms=[False, True]`

The config now guards this invariant:

- Qwen Pi05 with `discrete_state_input=False` raises a clear `ValueError`
- PaliGemma Pi05 behavior is unchanged

HF loading expectation:

- `Qwen2_5WeightLoader(..., load_action_expert=False)` and `Qwen3_5WeightLoader(..., load_action_expert=False)` remain the intended Pi05 defaults
- pretrained Qwen VLM params are loaded from HF/local snapshots
- action-expert AdaRMS params are newly initialized
- old Pi0 Qwen full train states should not be expected to resume directly as Pi05 full train states

## Prerequisites

Your Python environment should have at least:

- `jax`
- `flax`
- `pytest`

Optional but useful:

- CUDA-enabled JAX for GPU validation
- dataset assets for any non-fake-data training config

## 1. Unit Tests

Run the JAX model tests first.

```bash
python -m pytest src/openpi/models/pi0_test.py -q
python -m pytest src/openpi/models/qwen2_5_test.py -q
python -m pytest src/openpi/models/qwen3_5_test.py -q
python -m pytest src/openpi/models/model_test.py -k paligemma -q
```

What these cover:

- `Pi0` still builds under JAX
- legacy `PaliGemma/...` checkpoint remapping still works
- Qwen JAX `embed_prefix()` now runs instead of failing on image embedding
- Qwen JAX backend-owned multimodal positions are produced with shape `[3, B, T]`
- Qwen3.5 JAX vision embedding and merged image grid metadata are produced with the expected shapes
- Qwen Pi05 validates discrete state input and builds AdaRMS action-expert params

### Qwen Pi05 targeted unit tests

Run these when touching Pi05 state handling, Qwen action-expert norms, or Qwen config defaults.

```bash
python -m pytest src/openpi/models/pi0_test.py::test_qwen_pi05_requires_discrete_state_input -q
python -m pytest \
  src/openpi/models/qwen2_5_test.py::test_qwen2_5_pi05_builds_with_adarms_and_suffix_excludes_state \
  src/openpi/models/qwen2_5_test.py::test_qwen2_5_pi05_adarms_params_exist \
  -q
python -m pytest \
  src/openpi/models/qwen3_5_test.py::test_qwen3_5_pi05_builds_with_adarms_and_suffix_excludes_state \
  src/openpi/models/qwen3_5_test.py::test_qwen3_5_pi05_adarms_params_exist \
  -q
```

Expected:

- Qwen Pi05 rejects `discrete_state_input=False`
- suffix token count is exactly `action_horizon`
- suffix no longer includes a continuous state token
- `adarms_cond` is produced for the action expert branch
- adaptive norm params exist under action-expert norms and final norm

If the environment does not have `pytest`, use this as a syntax-only fallback before remote testing:

```bash
python -m py_compile \
  src/openpi/models/pi0_config.py \
  src/openpi/models/qwen2_5/text.py \
  src/openpi/models/qwen3_5/text.py \
  src/openpi/models/qwen2_5/adapter.py \
  src/openpi/models/qwen3_5/adapter.py \
  src/openpi/models/pi0_test.py \
  src/openpi/models/qwen2_5_test.py \
  src/openpi/models/qwen3_5_test.py \
  src/openpi/training/config.py
```

## 2. Minimal Forward Smoke

These tests do not require a training loop. They are the fastest way to check whether the backend can:

- build the model
- embed images
- compute loss
- sample actions

### PaliGemma forward smoke

```bash
python - <<'PY'
import jax
from openpi.models import pi0_config

cfg = pi0_config.Pi0Config(vlm_backend="paligemma")
model = cfg.create(jax.random.key(0))
obs, act = cfg.fake_obs(2), cfg.fake_act(2)

loss = model.compute_loss(jax.random.key(1), obs, act)
actions = model.sample_actions(jax.random.key(2), obs, num_steps=2)

print("loss shape:", loss.shape)
print("sampled actions shape:", actions.shape)
PY
```

Expected:

- `loss shape` is `(2, action_horizon)`
- `sampled actions shape` is `(2, action_horizon, action_dim)`

### Qwen forward smoke

This validates the JAX Qwen backend with real Qwen geometry.

```bash
python - <<'PY'
import jax
from openpi.models import pi0_config

cfg = pi0_config.Pi0Config(
    vlm_backend="qwen2_5_vl",
    vlm_backbone_variant="qwen2_5_3b",
    action_expert_variant="qwen2_5_3b",
)
model = cfg.create(jax.random.key(0))
obs, act = cfg.fake_obs(2), cfg.fake_act(2)

loss = model.compute_loss(jax.random.key(1), obs, act)
actions = model.sample_actions(jax.random.key(2), obs, num_steps=2)

print("loss shape:", loss.shape)
print("sampled actions shape:", actions.shape)
PY
```

Expected:

- model creation succeeds
- image embedding succeeds
- multimodal positions do not crash
- `compute_loss()` and `sample_actions()` both return tensors with the expected shapes

If this fails, treat it as a JAX Qwen integration bug, not a dataset issue.

### Qwen3.5 forward smoke

Use this to smoke-test the official-style JAX `Qwen3.5` text + vision path.

```bash
python - <<'PY'
import jax
from openpi.models import pi0_config

cfg = pi0_config.Pi0Config(
    vlm_backend="qwen3_5_vl",
    vlm_backbone_variant="qwen3_5_2b",
    action_expert_variant="qwen3_5_2b",
)
model = cfg.create(jax.random.key(0))
obs, act = cfg.fake_obs(2), cfg.fake_act(2)

loss = model.compute_loss(jax.random.key(1), obs, act)
actions = model.sample_actions(jax.random.key(2), obs, num_steps=2)

print("loss shape:", loss.shape)
print("sampled actions shape:", actions.shape)
PY
```

Interpretation:

- if this passes, it means the JAX `Qwen3.5` hybrid text + vision backbone is wired into `Pi0`
- it still does **not** by itself guarantee full end-to-end checkpointed training parity

### Qwen Pi05 forward smoke

This checks the Pi05 prefix/suffix split without entering the trainer.

```bash
python - <<'PY'
import jax
import jax.numpy as jnp
from openpi.models import pi0_config

cfg = pi0_config.Pi0Config(
    pi05=True,
    vlm_backend="qwen2_5_vl",
    vlm_backbone_variant="qwen2_5_3b",
    action_expert_variant="qwen2_5_3b_action_400m",
)
model = cfg.create(jax.random.key(0))
obs, act = cfg.fake_obs(2), cfg.fake_act(2)

prefix_tokens, prefix_mask, _, _ = model.embed_prefix(obs)
suffix_tokens, suffix_mask, _, adarms_cond = model.embed_suffix(
    obs,
    act,
    jnp.ones((2,), dtype=jnp.float32),
)

print("prefix shape:", prefix_tokens.shape)
print("suffix shape:", suffix_tokens.shape)
print("suffix mask shape:", suffix_mask.shape)
print("adarms cond shape:", adarms_cond.shape)
PY
```

Expected:

- prefix embedding accepts tokenized prompts that already include discretized state
- suffix shape is `(batch, action_horizon, expert_width)`
- suffix mask shape is `(batch, action_horizon)`
- `adarms_cond` is present for the action expert

Repeat the same smoke for Qwen3.5 by changing:

```python
vlm_backend="qwen3_5_vl"
vlm_backbone_variant="qwen3_5_2b"
action_expert_variant="qwen3_5_2b_action_400m"
```

## 3. Tiny Training Smoke

Use fake data first. This checks whether the JAX trainer, sharding, optimizer, and model loop are still coherent.

### PaliGemma tiny training smoke

```bash
python scripts/train.py debug --exp_name jax_paligemma_debug --overwrite --no-wandb-enabled
```

This uses:

- fake data
- tiny dummy model geometry
- JAX trainer

Expected:

- training starts
- checkpoints are written under `checkpoints/debug/jax_paligemma_debug`

### Qwen tiny training smoke

There is no dedicated `debug_qwen_jax` config yet, so the simplest tiny backend smoke is to reuse `debug` and override the model backend.

```bash
python scripts/train.py \
  debug \
  --exp_name jax_qwen_debug \
  --overwrite \
  --no-wandb-enabled \
  --model.vlm-backend qwen2_5_vl \
  --model.vlm-backbone-variant dummy \
  --model.action-expert-variant dummy
```

What this checks:

- backend dispatch to JAX Qwen
- JAX Qwen image path on fake images
- backend-owned multimodal positions inside the trainer
- end-to-end trainer loop on a tiny configuration

What it does **not** check:

- real Qwen geometry
- real dataset loading
- real Qwen checkpoint loading

### Qwen2.5 pretrained tiny training smoke

The repo now also includes a named config for loading HF Qwen2.5 text weights in JAX:

```bash
python scripts/train.py debug_qwen2_5_pretrained --overwrite --no-wandb-enabled
```

That config uses:

- `model.vlm_backend=qwen2_5_vl`
- `model.vlm_hf_model_id=Qwen/Qwen2.5-VL-3B-Instruct`
- `weight_loader=Qwen2_5WeightLoader("Qwen/Qwen2.5-VL-3B-Instruct")`

You can also load from a direct local directory:

- `weight_loader=Qwen2_5WeightLoader("Qwen/Qwen2.5-VL-3B-Instruct", local_snapshot_path="/path/to/local/qwen2.5")`

Current expectation:

- the JAX `qwen2_5_vl` vision path is still pragmatic, so `Qwen2_5WeightLoader` loads text-stack weights only
- `Qwen2_5WeightLoader` initializes the VLM branch by default; the action expert remains random-init unless `load_action_expert=True`

### Qwen3.5 tiny training smoke

Use this to validate JAX trainer integration for the new hybrid `Qwen3.5` text + vision path.

```bash
python scripts/train.py \
  debug \
  --exp_name jax_qwen3_5_debug \
  --overwrite \
  --no-wandb-enabled \
  --model.vlm-backend qwen3_5_vl \
  --model.vlm-backbone-variant qwen3_5_2b \
  --model.action-expert-variant qwen3_5_2b
```

Expected success criteria:

- backend dispatch works
- model initializes
- one or more steps run on fake data

This should be interpreted as:

- a trainer/runtime test for the official-style `Qwen3.5` text + vision backbone
- not by itself a pretrained-quality validation path

### Qwen3.5 pretrained tiny training smoke

The repo includes a named config for the official loader path:

```bash
python scripts/train.py debug_qwen3_5_pretrained --overwrite --no-wandb-enabled
```

That config uses:

- `model.vlm_backend=qwen3_5_vl`
- `model.vlm_backbone_variant=qwen3_5_2b`
- `model.action_expert_variant=qwen3_5_2b`
- `model.vlm_hf_model_id=Qwen/Qwen3.5-2B`
- `weight_loader=Qwen3_5WeightLoader("Qwen/Qwen3.5-2B")`

You can also bypass Hugging Face cache layout and load from a direct local directory:

- `weight_loader=Qwen3_5WeightLoader("Qwen/Qwen3.5-2B", local_snapshot_path="/path/to/local/qwen3.5-2b")`

The directory must contain one or more `.safetensors` files (recursively).

Current expectation:

- this path should exercise the HF checkpoint loader against the current JAX `qwen3_5_vl` text + vision tree
- treat it as a real compatibility smoke, not just a structural fake-data run
- successful execution still needs to be validated in your own JAX runtime
- `Qwen3_5WeightLoader` initializes the VLM branch by default; the action expert remains random-init unless `load_action_expert=True`

### Qwen Pi05 tiny training smoke

Use these named configs to validate the JAX Qwen Pi05 trainer path on fake data.

```bash
python scripts/train.py debug_qwen2_5_pi05 \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 1
```

```bash
python scripts/train.py debug_qwen3_5_pi05 \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 1
```

What this checks:

- `Pi0Config(pi05=True, vlm_backend=...)` builds for Qwen
- discrete state remains on the prefix/tokenizer side
- suffix contains action tokens only
- action-expert AdaRMS params are initialized
- one fake-data train step can compile and execute

### Qwen Pi05 pretrained-loader smoke

Use these only when the local snapshot paths exist or after updating the config paths.

```bash
python scripts/train.py debug_qwen2_5_pi05_pretrained \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 1
```

```bash
python scripts/train.py debug_qwen3_5_pi05_pretrained \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 1
```

Expected:

- Qwen VLM params load from HF/local safetensors
- action-expert AdaRMS params remain random-init
- the run should not attempt to load HF weights into action-expert AdaRMS params

## 4. Real-Geometry Qwen Training Smoke

If the tiny Qwen smoke passes, move on to the real JAX Qwen config already present in the repo:

```bash
python scripts/train.py \
  fruit_classification_Aa_qwen \
  --exp_name jax_qwen_real_smoke \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 2
```

Important notes:

- this config uses `vlm_backend="qwen2_5_vl"`
- this is a JAX structural validation path, not a pretrained Qwen restore path
- `weight_loader` is currently `NoOpWeightLoader`, so this tests random-init runtime behavior
- you must have the dataset/assets required by `fruit_classification_Aa_qwen`

Expected success criteria:

- the model builds
- data loader initializes
- train state initializes
- at least one or two train steps complete

### Qwen3.5 real-geometry smoke

The repo now also includes a named JAX config for real-geometry structural testing:

```bash
python scripts/train.py \
  fruit_classification_Aa_qwen3_5 \
  --exp_name jax_qwen3_5_real_smoke \
  --overwrite \
  --no-wandb-enabled \
  --num_train_steps 2
```

Important notes:

- this config uses `vlm_backend="qwen3_5_vl"`
- it initializes from `Qwen/Qwen3.5-2B` through `Qwen3_5WeightLoader`
- you must have the dataset/assets required by `fruit_classification_Aa_qwen3_5`
- this is the most meaningful current end-to-end Qwen3.5 JAX validation path in the repo

### Small Qwen-native action experts

The repo now includes smaller Qwen2.5/Qwen3.5 action-expert variants that preserve each family's attention/KV
interface while shrinking only the expert width/MLP capacity:

- `qwen2_5_7b_action_1b`
- `qwen2_5_3b_action_700m`
- `qwen2_5_3b_action_400m`
- `qwen3_5_4b_action_1b`
- `qwen3_5_4b_action_700m`
- `qwen3_5_4b_action_400m`
- `qwen3_5_2b_action_700m`
- `qwen3_5_2b_action_400m`

This follows the original OpenPI PaliGemma pattern more closely:

- the VLM branch is initialized from pretrained HF weights
- the action expert can stay randomly initialized and specialize for flow-matching control
- layer-wise KV/cache reuse remains valid because the expert keeps the same attention geometry as its paired VLM

Examples:

```bash
python scripts/train.py debug_qwen2_5_pretrained \
  --overwrite \
  --no-wandb-enabled \
  --model.action-expert-variant qwen2_5_3b_action_400m
```

```bash
python scripts/train.py debug_qwen3_5_pretrained \
  --overwrite \
  --no-wandb-enabled \
  --model.vlm-backbone-variant qwen3_5_4b \
  --model.action-expert-variant qwen3_5_4b_action_700m
```

```bash
python scripts/train.py debug_qwen3_5_pretrained \
  --overwrite \
  --no-wandb-enabled \
  --model.action-expert-variant qwen3_5_2b_action_400m
```

## 5. Recommended Test Order

Run tests in this order:

1. `pytest` unit tests
2. minimal forward smoke for `paligemma`
3. minimal forward smoke for `qwen2_5_vl`
4. optional structural forward smoke for `qwen3_5_vl`
5. `debug` tiny training smoke for `paligemma`
6. `debug` tiny training smoke for `qwen2_5_vl`
7. optional `debug_qwen2_5_pretrained` tiny training smoke
8. optional `debug` tiny training smoke for `qwen3_5_vl`
9. Qwen Pi05 targeted unit tests
10. `debug_qwen2_5_pi05` tiny training smoke
11. `debug_qwen3_5_pi05` tiny training smoke
12. optional `debug_qwen2_5_pi05_pretrained` loader smoke
13. optional `debug_qwen3_5_pi05_pretrained` loader smoke
14. `fruit_classification_Aa_qwen` real-geometry smoke
15. optional `fruit_classification_Aa_qwen3_5` real-geometry smoke with official loader

This order keeps failures easy to localize:

- unit test failure: model code bug
- forward smoke failure: backend integration bug
- tiny training failure: trainer/sharding/optimizer integration bug
- real-geometry failure: memory/config/data/runtime issue specific to the large model path

## 6. How To Interpret Results

### If PaliGemma passes and Qwen fails

That means the JAX trainer and common `Pi0` path are still healthy, and the bug is in the JAX Qwen backend.

### If both PaliGemma and Qwen fail

That points to a broader JAX runtime, trainer, or environment issue.

### If tiny Qwen passes but real-geometry Qwen fails

That usually means one of:

- memory pressure
- sharding behavior under larger geometry
- dataset-dependent issue
- large-model initialization cost

## 7. Known Limitations

Current JAX Qwen limitations:

- not HF `Qwen2.5-VL` weight-compatible yet
- vision path is pragmatic rather than checkpoint-faithful
- `Qwen2_5WeightLoader` currently loads text weights only
- this should be treated as a structural/runtime migration milestone, not final parity
- Qwen Pi05 should use VLM-only HF loading by default; action-expert AdaRMS params are new trainable params
- old Pi0 Qwen full train states are not expected to resume as Pi05 full train states because the action-expert norm structure changed

Current `qwen3_5_vl` limitations:

- official checkpoint loading is newly realigned to the current HF text-tree assumptions and still needs end-to-end runtime validation
- the current implementation is designed for image-only Pi0 usage, not full video/multiframe Qwen3.5-VL parity
- should be treated as a strong JAX Qwen3.5 migration milestone rather than a finished HF parity claim until full checkpointed training/inference runs pass
