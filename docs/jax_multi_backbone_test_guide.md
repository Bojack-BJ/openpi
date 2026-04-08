# JAX Multi-Backbone Test Guide

This note summarizes how to test the current JAX `Pi0` runtime after the multi-backbone refactor.

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
python -m pytest src/openpi/models/model_test.py -k paligemma -q
```

What these cover:

- `Pi0` still builds under JAX
- legacy `PaliGemma/...` checkpoint remapping still works
- Qwen JAX `embed_prefix()` now runs instead of failing on image embedding
- Qwen JAX backend-owned multimodal positions are produced with shape `[3, B, T]`

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

## 3. Tiny Training Smoke

Use fake data first. This checks whether the JAX trainer, sharding, optimizer, and model loop are still coherent.

### PaliGemma tiny training smoke

```bash
python scripts/train.py debug --exp_name jax_paligemma_debug --overwrite True --wandb_enabled False
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
  --overwrite True \
  --wandb_enabled False \
  --model.vlm_backend qwen2_5_vl \
  --model.vlm_backbone_variant dummy \
  --model.action_expert_variant dummy
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

## 4. Real-Geometry Qwen Training Smoke

If the tiny Qwen smoke passes, move on to the real JAX Qwen config already present in the repo:

```bash
python scripts/train.py \
  fruit_classification_Aa_qwen \
  --exp_name jax_qwen_real_smoke \
  --overwrite True \
  --wandb_enabled False \
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

## 5. Recommended Test Order

Run tests in this order:

1. `pytest` unit tests
2. minimal forward smoke for `paligemma`
3. minimal forward smoke for `qwen2_5_vl`
4. `debug` tiny training smoke for `paligemma`
5. `debug` tiny training smoke for `qwen2_5_vl`
6. `fruit_classification_Aa_qwen` real-geometry smoke

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
- no dedicated JAX Qwen pretrained weight loader yet
- this should be treated as a structural/runtime migration milestone, not final parity
