import dataclasses
import logging
import pathlib
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.models.vlm_backbone as _vlm_backbone
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.

    JAX multi-backend migration note:
    legacy official weights are still loaded, but the params will be remapped onto the neutral
    `vlm_with_expert` runtime root when the reference tree expects it. Non-PaliGemma backends
    should use dedicated backend-specific loaders.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        root_key = (
            _vlm_backbone.RUNTIME_VLM_ROOT if _vlm_backbone.RUNTIME_VLM_ROOT in params else _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT
        )
        loaded_params = {root_key: flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class Qwen3_5WeightLoader(WeightLoader):
    """Loads weights from a Hugging Face Qwen3.5 checkpoint.

    This loader targets the JAX `qwen3_5_vl` runtime path and maps official language-model and
    vision-model tensors into the OpenPI parameter tree. By default it initializes only the VLM
    branch; the action expert remains randomly initialized unless `load_action_expert=True`.
    """

    # Hugging Face repo id (for example "Qwen/Qwen3.5-2B") or an absolute/local
    # snapshot directory containing .safetensors files.
    model_id: str
    # Optional explicit local directory override. If provided, loading will ignore
    # `model_id` and read safetensors directly from this directory.
    local_snapshot_path: str | None = None
    local_files_only: bool = False
    # By default only initialize the VLM branch from HF weights. The action expert remains
    # randomly initialized, matching the original OpenPI PaliGemma pattern.
    load_action_expert: bool = False

    def load(self, params: at.Params) -> at.Params:
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        hf_tensors = _load_hf_safetensors_snapshot(
            self.model_id,
            local_snapshot_path=self.local_snapshot_path,
            local_files_only=self.local_files_only,
        )

        flat_loaded: dict[str, np.ndarray] = {}
        _load_qwen3_5_text_weights(
            flat_loaded,
            flat_ref,
            hf_tensors,
            branch_suffixes=("", "_1") if self.load_action_expert else ("",),
        )
        _load_qwen3_5_vision_weights(flat_loaded, flat_ref, hf_tensors)

        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class Qwen2_5WeightLoader(WeightLoader):
    """Loads text weights from a Hugging Face Qwen2.5(-VL) checkpoint.

    This loader maps Qwen2.5 language-model tensors into the JAX `qwen2_5_vl` text stack.
    The current JAX Qwen2.5 vision stack is a pragmatic SigLIP-based path, so official HF
    visual tensors are intentionally not loaded here. By default it initializes only the VLM
    branch; the action expert remains randomly initialized unless `load_action_expert=True`.
    """

    model_id: str
    local_snapshot_path: str | None = None
    local_files_only: bool = False
    load_action_expert: bool = False

    def load(self, params: at.Params) -> at.Params:
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        hf_tensors = _load_hf_safetensors_snapshot(
            self.model_id,
            local_snapshot_path=self.local_snapshot_path,
            local_files_only=self.local_files_only,
        )

        flat_loaded: dict[str, np.ndarray] = {}
        _load_qwen2_5_text_weights(
            flat_loaded,
            flat_ref,
            hf_tensors,
            branch_suffixes=("", "_1") if self.load_action_expert else ("",),
        )

        loaded_params = flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    loaded_params = _vlm_backbone.remap_legacy_vlm_checkpoint_root_for_reference(loaded_params, params)

    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _load_hf_safetensors_snapshot(
    model_id: str, *, local_snapshot_path: str | None = None, local_files_only: bool
) -> dict[str, np.ndarray]:
    try:
        from huggingface_hub import snapshot_download
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise ImportError(
            "Loading official Qwen3.5 weights requires `huggingface_hub` and `safetensors` "
            "to be installed in the active Python environment."
        ) from exc

    if local_snapshot_path is not None:
        snapshot_dir = pathlib.Path(local_snapshot_path).expanduser().resolve()
        logger.info("Loading HF safetensors from explicit local snapshot path: %s", snapshot_dir)
    else:
        model_path = pathlib.Path(model_id).expanduser()
        if model_path.exists():
            snapshot_dir = model_path.resolve()
            logger.info("Loading HF safetensors from local model_id directory: %s", snapshot_dir)
        else:
            snapshot_dir = pathlib.Path(
                snapshot_download(
                    repo_id=model_id,
                    allow_patterns=["*.safetensors", "*.json"],
                    cache_dir=str(download.get_cache_dir() / "huggingface"),
                    local_files_only=local_files_only,
                )
            )
            logger.info(
                "Loading HF safetensors via snapshot_download repo_id=%s local_files_only=%s snapshot_dir=%s",
                model_id,
                local_files_only,
                snapshot_dir,
            )

    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Qwen3.5 weights snapshot directory does not exist: {snapshot_dir}")
    if not snapshot_dir.is_dir():
        raise ValueError(f"Qwen3.5 weights snapshot path must be a directory: {snapshot_dir}")

    safetensor_files = sorted(snapshot_dir.rglob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No .safetensors files found under {snapshot_dir}. "
            "Make sure this directory contains a Hugging Face checkpoint snapshot."
        )

    tensors: dict[str, np.ndarray] = {}
    for path in safetensor_files:
        tensors.update(load_file(str(path)))
    return tensors


def _resolve_ref_key(flat_ref: dict[str, at.Array], suffix: str) -> str:
    matches = [key for key in flat_ref if key.endswith(suffix)]
    if not matches:
        raise KeyError(f"Could not find parameter ending with '{suffix}' in current JAX model.")
    if len(matches) > 1:
        raise KeyError(f"Parameter suffix '{suffix}' is ambiguous in current JAX model: {matches}")
    return matches[0]


def _target_shape(flat_ref: dict[str, at.Array], suffix: str) -> tuple[str, tuple[int, ...]]:
    key = _resolve_ref_key(flat_ref, suffix)
    return key, tuple(flat_ref[key].shape)


def _store_loaded(flat_loaded: dict[str, np.ndarray], flat_ref: dict[str, at.Array], suffix: str, value: np.ndarray):
    key, expected_shape = _target_shape(flat_ref, suffix)
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"Shape mismatch for {suffix}: expected {expected_shape}, got {value.shape}")
    flat_loaded[key] = np.asarray(value)


def _dense_to_kernel(weight: np.ndarray) -> np.ndarray:
    return np.asarray(weight).T


def _conv1d_depthwise_to_kernel(weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight)
    if weight.ndim == 3:
        if weight.shape[1] != 1:
            raise ValueError(f"Expected depthwise conv1d weight with shape [channels, 1, kernel], got {weight.shape}")
        weight = weight[:, 0, :]
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D depthwise conv1d kernel after squeeze, got {weight.shape}")
    return weight.T[::-1]


def _conv3d_to_kernel(weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight)
    if weight.ndim != 5:
        raise ValueError(f"Expected Conv3D weight with shape [out, in, kt, kh, kw], got {weight.shape}")
    return np.transpose(weight, (2, 3, 4, 1, 0))


def _hf_norm_to_rms_scale(weight: np.ndarray) -> np.ndarray:
    return np.asarray(weight) - 1.0


def _full_attention_qg_kernel(q_weight: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    num_heads, input_width, doubled_head_dim = expected_shape
    head_dim = doubled_head_dim // 2
    return _dense_to_kernel(q_weight).reshape(input_width, num_heads, 2 * head_dim).transpose(1, 0, 2)


def _full_attention_head_kernel(weight: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    num_heads, input_width, head_dim = expected_shape
    return _dense_to_kernel(weight).reshape(input_width, num_heads, head_dim).transpose(1, 0, 2)


def _full_attention_output_kernel(weight: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    num_heads, head_dim, output_width = expected_shape
    kernel = _dense_to_kernel(weight)
    return kernel.reshape(num_heads, head_dim, output_width)


def _linear_attention_output_kernel(weight: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    num_value_heads, value_dim, output_width = expected_shape
    kernel = _dense_to_kernel(weight)
    return kernel.reshape(num_value_heads, value_dim, output_width)


def _load_qwen3_5_text_weights(
    flat_loaded: dict[str, np.ndarray],
    flat_ref: dict[str, at.Array],
    hf_tensors: dict[str, np.ndarray],
    *,
    branch_suffixes: tuple[str, ...] = ("", "_1"),
):
    layer_prefixes: dict[int, str] = {}
    for key in flat_ref:
        match = re.search(r"llm/layers_(\d+)/", key)
        if match is not None:
            layer_idx = int(match.group(1))
            layer_prefixes.setdefault(layer_idx, f"llm/layers_{layer_idx}")
            continue

        match = re.search(r"llm/layers/(\d+)/layers_(\d+)/", key)
        if match is not None:
            group_idx = int(match.group(1))
            local_idx = int(match.group(2))
            layer_idx = group_idx * 4 + local_idx
            layer_prefixes.setdefault(layer_idx, f"llm/layers/{group_idx}/layers_{local_idx}")

    _store_loaded(
        flat_loaded,
        flat_ref,
        "llm/embedder/input_embedding",
        np.asarray(hf_tensors["model.language_model.embed_tokens.weight"]),
    )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "llm/final_norm/scale",
        _hf_norm_to_rms_scale(hf_tensors["model.language_model.norm.weight"]),
    )
    if "_1" in branch_suffixes:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "llm/final_norm_1/scale",
            _hf_norm_to_rms_scale(hf_tensors["model.language_model.norm.weight"]),
        )

    layer_indices = sorted(
        layer_prefixes
    )

    for layer_idx in layer_indices:
        target_prefix = layer_prefixes[layer_idx]
        hf_prefix = f"model.language_model.layers.{layer_idx}."
        input_norm = _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}input_layernorm.weight"])
        post_attn_norm = _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}post_attention_layernorm.weight"])
        for branch_suffix in branch_suffixes:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/pre_attention_norm{branch_suffix}/scale",
                input_norm,
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/pre_ffw_norm{branch_suffix}/scale",
                post_attn_norm,
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/mlp{branch_suffix}/gate_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.gate_proj.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/mlp{branch_suffix}/up_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.up_proj.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/mlp{branch_suffix}/down_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.down_proj.weight"]),
            )

        has_linear = f"{target_prefix}/self_attn/in_proj_qkv/w" in flat_ref
        has_full = f"{target_prefix}/self_attn/qg_einsum/w" in flat_ref

        if has_full:
            for branch_suffix in branch_suffixes:
                qg_suffix = f"{target_prefix}/self_attn/qg_einsum{branch_suffix}/w"
                _, qg_shape = _target_shape(flat_ref, qg_suffix)
                _store_loaded(
                    flat_loaded,
                    flat_ref,
                    qg_suffix,
                    _full_attention_qg_kernel(hf_tensors[f"{hf_prefix}self_attn.q_proj.weight"], qg_shape),
                )
                head_suffixes = (
                    ("k_einsum", "self_attn.k_proj.weight"),
                    ("v_einsum", "self_attn.v_proj.weight"),
                )
                for target_name, hf_name in head_suffixes:
                    target_suffix = f"{target_prefix}/self_attn/{target_name}{branch_suffix}/w"
                    _, target_shape = _target_shape(flat_ref, target_suffix)
                    _store_loaded(
                        flat_loaded,
                        flat_ref,
                        target_suffix,
                        _full_attention_head_kernel(hf_tensors[f"{hf_prefix}{hf_name}"], target_shape),
                    )
                o_suffix = f"{target_prefix}/self_attn/o_einsum{branch_suffix}/w"
                _, o_shape = _target_shape(flat_ref, o_suffix)
                _store_loaded(
                    flat_loaded,
                    flat_ref,
                    o_suffix,
                    _full_attention_output_kernel(hf_tensors[f"{hf_prefix}self_attn.o_proj.weight"], o_shape),
                )
                _store_loaded(
                    flat_loaded,
                    flat_ref,
                    f"{target_prefix}/self_attn/q_norm{branch_suffix}/scale",
                    _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}self_attn.q_norm.weight"]),
                )
                _store_loaded(
                    flat_loaded,
                    flat_ref,
                    f"{target_prefix}/self_attn/k_norm{branch_suffix}/scale",
                    _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}self_attn.k_norm.weight"]),
                )
        elif has_linear:
            branch_linear_weights = (
                ("in_proj_qkv", "linear_attn.in_proj_qkv.weight"),
                ("in_proj_z", "linear_attn.in_proj_z.weight"),
                ("in_proj_b", "linear_attn.in_proj_b.weight"),
                ("in_proj_a", "linear_attn.in_proj_a.weight"),
                ("out_proj", "linear_attn.out_proj.weight"),
            )
            for branch_suffix in branch_suffixes:
                for target_name, hf_name in branch_linear_weights:
                    target_suffix = f"{target_prefix}/self_attn/{target_name}{branch_suffix}/w"
                    target_key, target_shape = _target_shape(flat_ref, target_suffix)
                    value = hf_tensors[f"{hf_prefix}{hf_name}"]
                    if target_name == "out_proj":
                        mapped = _linear_attention_output_kernel(value, target_shape)
                    else:
                        mapped = _dense_to_kernel(value)
                    if tuple(mapped.shape) != target_shape:
                        raise ValueError(
                            f"Shape mismatch for {target_suffix}: expected {target_shape}, got {mapped.shape}"
                        )
                    flat_loaded[target_key] = np.asarray(mapped)

            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/self_attn/short_conv/kernel",
                _conv1d_depthwise_to_kernel(hf_tensors[f"{hf_prefix}linear_attn.conv1d.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/self_attn/norm/scale",
                _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}linear_attn.norm.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/self_attn/A_log",
                np.asarray(hf_tensors[f"{hf_prefix}linear_attn.A_log"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"{target_prefix}/self_attn/dt_bias",
                np.asarray(hf_tensors[f"{hf_prefix}linear_attn.dt_bias"]),
            )
        else:
            raise ValueError(f"Could not determine Qwen3.5 layer type for layer {layer_idx} from the JAX reference tree.")


def _load_qwen3_5_vision_weights(
    flat_loaded: dict[str, np.ndarray],
    flat_ref: dict[str, at.Array],
    hf_tensors: dict[str, np.ndarray],
):
    _store_loaded(
        flat_loaded,
        flat_ref,
        "vision/encoder/patch_embed/proj/kernel",
        _conv3d_to_kernel(hf_tensors["model.visual.patch_embed.proj.weight"]),
    )
    if "model.visual.patch_embed.proj.bias" in hf_tensors:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "vision/encoder/patch_embed/proj/bias",
            np.asarray(hf_tensors["model.visual.patch_embed.proj.bias"]),
        )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "vision/encoder/pos_embed/embedding",
        np.asarray(hf_tensors["model.visual.pos_embed.weight"]),
    )

    block_indices = sorted(
        {
            int(match.group(1))
            for key in flat_ref
            for match in [re.search(r"vision/encoder/blocks_(\d+)/", key)]
            if match is not None
        }
    )
    for block_idx in block_indices:
        hf_prefix = f"model.visual.blocks.{block_idx}."
        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/norm1/scale",
            np.asarray(hf_tensors[f"{hf_prefix}norm1.weight"]),
        )
        if f"{hf_prefix}norm1.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/norm1/bias",
                np.asarray(hf_tensors[f"{hf_prefix}norm1.bias"]),
            )
        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/norm2/scale",
            np.asarray(hf_tensors[f"{hf_prefix}norm2.weight"]),
        )
        if f"{hf_prefix}norm2.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/norm2/bias",
                np.asarray(hf_tensors[f"{hf_prefix}norm2.bias"]),
            )

        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/attn/qkv/kernel",
            _dense_to_kernel(hf_tensors[f"{hf_prefix}attn.qkv.weight"]),
        )
        if f"{hf_prefix}attn.qkv.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/attn/qkv/bias",
                np.asarray(hf_tensors[f"{hf_prefix}attn.qkv.bias"]),
            )
        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/attn/proj/kernel",
            _dense_to_kernel(hf_tensors[f"{hf_prefix}attn.proj.weight"]),
        )
        if f"{hf_prefix}attn.proj.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/attn/proj/bias",
                np.asarray(hf_tensors[f"{hf_prefix}attn.proj.bias"]),
            )

        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/mlp/linear_fc1/kernel",
            _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.linear_fc1.weight"]),
        )
        if f"{hf_prefix}mlp.linear_fc1.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/mlp/linear_fc1/bias",
                np.asarray(hf_tensors[f"{hf_prefix}mlp.linear_fc1.bias"]),
            )
        _store_loaded(
            flat_loaded,
            flat_ref,
            f"vision/encoder/blocks_{block_idx}/mlp/linear_fc2/kernel",
            _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.linear_fc2.weight"]),
        )
        if f"{hf_prefix}mlp.linear_fc2.bias" in hf_tensors:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"vision/encoder/blocks_{block_idx}/mlp/linear_fc2/bias",
                np.asarray(hf_tensors[f"{hf_prefix}mlp.linear_fc2.bias"]),
            )

    _store_loaded(
        flat_loaded,
        flat_ref,
        "vision/encoder/merger/norm/scale",
        np.asarray(hf_tensors["model.visual.merger.norm.weight"]),
    )
    if "model.visual.merger.norm.bias" in hf_tensors:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "vision/encoder/merger/norm/bias",
            np.asarray(hf_tensors["model.visual.merger.norm.bias"]),
        )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "vision/encoder/merger/linear_fc1/kernel",
        _dense_to_kernel(hf_tensors["model.visual.merger.linear_fc1.weight"]),
    )
    if "model.visual.merger.linear_fc1.bias" in hf_tensors:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "vision/encoder/merger/linear_fc1/bias",
            np.asarray(hf_tensors["model.visual.merger.linear_fc1.bias"]),
        )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "vision/encoder/merger/linear_fc2/kernel",
        _dense_to_kernel(hf_tensors["model.visual.merger.linear_fc2.weight"]),
    )
    if "model.visual.merger.linear_fc2.bias" in hf_tensors:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "vision/encoder/merger/linear_fc2/bias",
            np.asarray(hf_tensors["model.visual.merger.linear_fc2.bias"]),
        )


def _first_present(mapping: dict[str, np.ndarray], candidates: list[str]) -> str:
    for key in candidates:
        if key in mapping:
            return key
    raise KeyError(f"Could not find any of keys: {candidates}")


def _first_present_or_none(mapping: dict[str, np.ndarray], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in mapping:
            return key
    return None


def _load_qwen2_5_text_weights(
    flat_loaded: dict[str, np.ndarray],
    flat_ref: dict[str, at.Array],
    hf_tensors: dict[str, np.ndarray],
    *,
    branch_suffixes: tuple[str, ...] = ("", "_1"),
):
    embed_key = _first_present(
        hf_tensors,
        [
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ],
    )
    final_norm_key = _first_present(
        hf_tensors,
        [
            "model.norm.weight",
            "model.language_model.norm.weight",
        ],
    )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "llm/embedder/input_embedding",
        np.asarray(hf_tensors[embed_key]),
    )
    _store_loaded(
        flat_loaded,
        flat_ref,
        "llm/final_norm/scale",
        _hf_norm_to_rms_scale(hf_tensors[final_norm_key]),
    )
    if "_1" in branch_suffixes:
        _store_loaded(
            flat_loaded,
            flat_ref,
            "llm/final_norm_1/scale",
            _hf_norm_to_rms_scale(hf_tensors[final_norm_key]),
        )

    if any(key.startswith("model.layers.") for key in hf_tensors):
        layer_prefix_root = "model.layers"
    elif any(key.startswith("model.language_model.layers.") for key in hf_tensors):
        layer_prefix_root = "model.language_model.layers"
    else:
        raise KeyError("Could not detect Qwen2.5 layer prefix in HF checkpoint tensors.")

    layer_indices = sorted(
        {
            int(match.group(1))
            for key in flat_ref
            for match in [re.search(r"llm/layers/(\d+)/", key)]
            if match is not None
        }
    )

    for layer_idx in layer_indices:
        hf_prefix = f"{layer_prefix_root}.{layer_idx}."
        input_norm = _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}input_layernorm.weight"])
        post_attn_norm = _hf_norm_to_rms_scale(hf_tensors[f"{hf_prefix}post_attention_layernorm.weight"])

        for branch_suffix in branch_suffixes:
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"llm/layers/{layer_idx}/pre_attention_norm{branch_suffix}/scale",
                input_norm,
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"llm/layers/{layer_idx}/pre_ffw_norm{branch_suffix}/scale",
                post_attn_norm,
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"llm/layers/{layer_idx}/mlp{branch_suffix}/gate_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.gate_proj.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"llm/layers/{layer_idx}/mlp{branch_suffix}/up_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.up_proj.weight"]),
            )
            _store_loaded(
                flat_loaded,
                flat_ref,
                f"llm/layers/{layer_idx}/mlp{branch_suffix}/down_proj/w",
                _dense_to_kernel(hf_tensors[f"{hf_prefix}mlp.down_proj.weight"]),
            )

            q_suffix = f"llm/layers/{layer_idx}/attn/q_einsum{branch_suffix}/w"
            _, q_shape = _target_shape(flat_ref, q_suffix)
            _store_loaded(
                flat_loaded,
                flat_ref,
                q_suffix,
                _full_attention_head_kernel(hf_tensors[f"{hf_prefix}self_attn.q_proj.weight"], q_shape),
            )
            k_suffix = f"llm/layers/{layer_idx}/attn/k_einsum{branch_suffix}/w"
            _, k_shape = _target_shape(flat_ref, k_suffix)
            _store_loaded(
                flat_loaded,
                flat_ref,
                k_suffix,
                _full_attention_head_kernel(hf_tensors[f"{hf_prefix}self_attn.k_proj.weight"], k_shape),
            )
            v_suffix = f"llm/layers/{layer_idx}/attn/v_einsum{branch_suffix}/w"
            _, v_shape = _target_shape(flat_ref, v_suffix)
            _store_loaded(
                flat_loaded,
                flat_ref,
                v_suffix,
                _full_attention_head_kernel(hf_tensors[f"{hf_prefix}self_attn.v_proj.weight"], v_shape),
            )
            o_suffix = f"llm/layers/{layer_idx}/attn/o_einsum{branch_suffix}/w"
            _, o_shape = _target_shape(flat_ref, o_suffix)
            _store_loaded(
                flat_loaded,
                flat_ref,
                o_suffix,
                _full_attention_output_kernel(hf_tensors[f"{hf_prefix}self_attn.o_proj.weight"], o_shape),
            )
