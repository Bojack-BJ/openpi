# PI0 VLM Backend Migration Plan

This project is currently hard-wired to a `PaliGemma + Gemma expert` stack in several layers:

- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/models_pytorch/gemma_pytorch.py`
- `src/openpi/models/tokenizer.py`
- `src/openpi/training/weight_loaders.py`
- `src/openpi/training/config.py`

The first migration step implemented in this branch is structural decoupling:

- `Pi0Config` now has `vlm_backend` and `vlm_hf_model_id`
- `PI0Pytorch` now routes VLM creation through `models_pytorch/vlm_backbone.py`
- the runtime path now uses a neutral `vlm_with_expert` handle instead of hard-coding all logic to `paligemma_with_expert`
- the existing PaliGemma implementation remains the default backend

## Recommended rollout order

1. Keep `paligemma` as the only production backend until the adapter API is stable.
2. Implement `qwen2_vl` first.
3. Implement `internvl3` second.

`Qwen2-VL` is the lower-risk first target because the Hugging Face ecosystem and processor path are typically simpler than InternVL-style custom wrappers. `InternVL3` should be added after the generic adapter seams are proven.

## What must change for each new backend

### 1. Backend adapter

Add a dedicated module under `src/openpi/models_pytorch/` that exposes the same runtime surface currently used by `PI0Pytorch`:

- `embed_image(image)`
- `embed_language_tokens(tokens)`
- `forward(...)`
- `set_gradient_checkpointing_enabled(enabled)`
- `prefix_q_proj_dtype()`
- `set_prefix_attention_implementation(implementation)`
- `set_suffix_attention_implementation(implementation)`
- `to_bfloat16_for_selected_params(precision)`

If the new VLM cannot support the current shared prefix/suffix attention scheme, then the adapter layer is not enough and `PI0Pytorch.forward()` plus `denoise_step()` must be redesigned.

### 2. Hidden-size compatibility

The current architecture assumes:

- prefix language tower hidden size == suffix expert hidden size
- shared attention can concatenate Q/K/V from prefix and suffix streams
- rotary embedding conventions are compatible

Before wiring in a backend, verify:

- hidden size
- head count
- kv head count
- head dim
- rope implementation

If these do not match the action expert, you need one of:

- projection adapters between prefix and suffix streams
- a rewritten joint-attention block
- an expert model family that matches the VLM text tower geometry

### 3. Tokenizer / processor

Current prompt tokenization is tied to PaliGemma SentencePiece and, for some paths, to reserving the tail of the PaliGemma vocabulary for action tokens.

For `qwen2_vl` or `internvl3`, add a backend-specific tokenizer or processor layer:

- text tokenization
- BOS/EOS behavior
- image placeholder token strategy
- maximum sequence length policy
- action-token mapping strategy for FAST/baseline paths if those paths must still work

Do not try to reuse `PaligemmaTokenizer` unchanged for non-PaliGemma backends.

### 4. Weight loading

Current loader support is PaliGemma-specific:

- `PaliGemmaWeightLoader` loads official PaliGemma `.npz`

For new backends, add dedicated loaders, for example:

- `Qwen2VLWeightLoader`
- `InternVL3WeightLoader`

Those loaders should define how pretrained VLM weights initialize:

- vision tower
- multimodal projector
- text tower
- any new projection layers

The action expert weights can still be initialized separately or left random depending on the experiment.

### 5. Training config and transforms

Update:

- `src/openpi/training/config.py`
- `src/openpi/transforms.py`
- any dataset/model transform factory that instantiates `PaligemmaTokenizer`

The config should choose tokenizer/processor based on `model.vlm_backend`, not on hard-coded PaliGemma assumptions.

## Backend-specific notes

### Qwen2-VL

Recommended implementation path:

1. Build `Qwen2VLWithExpertModel`.
2. Reuse the current action expert at first only if tensor geometry matches.
3. If geometry does not match, add explicit prefix-to-expert projection layers before attempting shared attention.
4. Add a `Qwen2VLTokenizer` or processor wrapper in `models/tokenizer.py`.
5. Add a dedicated pretrained weight loader.
6. Validate training forward first, then inference cache path.

Primary risk:

- the current implementation assumes a PaliGemma/Gemma-like text stack and direct access to layer internals for joint attention.

### InternVL3

Recommended implementation path:

1. Build `InternVL3WithExpertModel`.
2. Treat the processor path as custom from day one.
3. Expect additional normalization or projector differences in the image-text bridge.
4. Validate image embedding output shape before integrating the action expert.

Primary risk:

- custom model wrappers and processor conventions are more likely to diverge from the assumptions used by `PI0Pytorch`.

## Minimal execution plan from here

### Phase A: completed in this branch

- decouple runtime naming from `paligemma_with_expert`
- add `vlm_backend` config
- add VLM backend factory

### Phase B: next implementation target

1. Add `Qwen2VLWithExpertModel`.
2. Add `Qwen2VLTokenizer` or processor wrapper.
3. Add Qwen weight loader.
4. Add one tiny forward smoke test for prefix-only, suffix-only, and joint mode.

Status update:

- `Qwen2.5-VL-7B-Instruct` backbone adapter now exists in `src/openpi/models_pytorch/qwen2_vl_pytorch.py`
- prompt tokenizer routing is now backend-aware via `src/openpi/models/tokenizer.py:create_prompt_tokenizer`
- `Qwen2VLTokenizer` now handles prompt tokenization for `qwen2_vl` / `qwen2_5_vl`
- current supported configuration is:
  - `vlm_backend="qwen2_5_vl"` or `vlm_backend="qwen2_vl"`
  - `vlm_hf_model_id="Qwen/Qwen2.5-VL-7B-Instruct"`
  - `paligemma_variant="qwen2_5_7b"`
  - `action_expert_variant="qwen2_5_7b"`
- current limitations:
  - only the prompt tokenizer path has been migrated; FAST/action-tokenizer paths are still PaliGemma-specific
  - `pi05` / AdaRMS is not supported for the Qwen backend

### Phase C: after Qwen is stable

1. Add `InternVL3WithExpertModel`.
2. Add InternVL3 tokenizer/processor wrapper.
3. Add InternVL3 weight loader.
4. Validate training and autoregressive cache path.
