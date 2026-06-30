from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
import math
import pathlib
from typing import Any

import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample


PROPRIO_FRAME_TOKEN = "<|hl_proprio_frame|>"
PROPRIO_SUMMARY_TOKEN = "<|hl_proprio_summary|>"
PROPRIO_STATE_FILENAME = "hl_proprio_projector.pt"
PROPRIO_CONFIG_FILENAME = "hl_proprio_config.json"
_DEFAULT_TRANSLATION_DIMS = (0, 1, 2, 7, 8, 9)
_DEFAULT_ROTATION_DIMS = (3, 4, 5, 10, 11, 12)
_DEFAULT_GRIPPER_DIMS = (6, 13)


class ProprioTokenProjector(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        noise_std: float,
        projector_mode: str = "joint",
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.dropout = float(dropout)
        self.noise_std = float(noise_std)
        self.projector_mode = projector_mode
        input_dim = self.state_dim * 2
        self.frame_mlp = (
            torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, self.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden_dim, self.output_dim),
            )
            if projector_mode == "joint"
            else None
        )
        if projector_mode == "split":
            translation_dim = len(_group_dims_for_state_dim(self.state_dim, group="translation")) * 2
            rotation_dim = len(_group_dims_for_state_dim(self.state_dim, group="rotation")) * 2
            gripper_dim = len(_group_dims_for_state_dim(self.state_dim, group="gripper")) * 2
            self.translation_mlp = _make_split_branch(translation_dim, self.hidden_dim, self.output_dim)
            self.rotation_mlp = _make_split_branch(rotation_dim, self.hidden_dim, self.output_dim)
            self.gripper_mlp = _make_split_branch(gripper_dim, self.hidden_dim, self.output_dim)
            self.split_out = torch.nn.Sequential(
                torch.nn.LayerNorm(self.output_dim * 3),
                torch.nn.Linear(self.output_dim * 3, self.output_dim),
            )
        else:
            self.translation_mlp = None
            self.rotation_mlp = None
            self.gripper_mlp = None
            self.split_out = None
        self.summary_query = torch.nn.Parameter(torch.zeros(self.output_dim))
        self.summary_key = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.summary_value = torch.nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.summary_out = torch.nn.Linear(self.output_dim, self.output_dim)

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or masks.ndim != 3:
            raise ValueError("Proprio states and masks must have shape [batch, time, dim].")
        if states.shape != masks.shape:
            raise ValueError(f"Proprio states/masks shape mismatch: {tuple(states.shape)} vs {tuple(masks.shape)}.")
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected proprio state_dim={self.state_dim}, got {states.shape[-1]}.")

        masks = masks.to(dtype=states.dtype)
        if self.training and self.noise_std > 0.0:
            states = states + torch.randn_like(states) * self.noise_std * masks
        if self.training and self.dropout > 0.0:
            keep = (torch.rand((states.shape[0], 1, 1), device=states.device) >= self.dropout).to(dtype=states.dtype)
            states = states * keep
            masks = masks * keep

        valid_time = masks.to(dtype=torch.bool).any(dim=-1)
        if self.projector_mode == "joint":
            if self.frame_mlp is None:
                raise RuntimeError("Joint proprio projector is not initialized.")
            inputs = torch.cat([states * masks, masks], dim=-1)
            frame_tokens = self.frame_mlp(inputs)
        else:
            frame_tokens = self._split_frame_tokens(states, masks)
        frame_tokens = frame_tokens * valid_time.unsqueeze(-1).to(dtype=frame_tokens.dtype)

        keys = self.summary_key(frame_tokens)
        values = self.summary_value(frame_tokens)
        query = self.summary_query.to(dtype=frame_tokens.dtype, device=frame_tokens.device)
        scores = torch.einsum("bth,h->bt", keys, query) / math.sqrt(float(self.output_dim))
        scores = scores.masked_fill(~valid_time, -1.0e4)
        all_invalid = ~valid_time.any(dim=-1)
        if bool(all_invalid.any()):
            scores = scores.clone()
            scores[all_invalid] = 0.0
        attn = torch.softmax(scores, dim=-1) * valid_time.to(dtype=frame_tokens.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        summary = torch.einsum("bt,bth->bh", attn, values)
        summary = self.summary_out(summary)
        return frame_tokens, summary

    def _split_frame_tokens(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if self.translation_mlp is None or self.rotation_mlp is None or self.gripper_mlp is None or self.split_out is None:
            raise RuntimeError("Split proprio projector is not initialized.")
        translation = self.translation_mlp(_select_group_inputs(states, masks, group="translation"))
        rotation = self.rotation_mlp(_select_group_inputs(states, masks, group="rotation"))
        gripper = self.gripper_mlp(_select_group_inputs(states, masks, group="gripper"))
        return self.split_out(torch.cat([translation, rotation, gripper], dim=-1))


class ProprioEmbeddingWrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        base_embedding: torch.nn.Module,
        projector: ProprioTokenProjector,
        frame_token_id: int,
        summary_token_id: int,
        token_mode: str,
    ):
        super().__init__()
        self.base_embedding = base_embedding
        self.projector = projector
        self.frame_token_id = int(frame_token_id)
        self.summary_token_id = int(summary_token_id)
        self.token_mode = token_mode
        self._states: torch.Tensor | None = None
        self._masks: torch.Tensor | None = None
        self._last_grad_summary: dict[str, float] = {}

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_embedding.weight

    def set_proprio_batch(
        self,
        states: torch.Tensor | None,
        masks: torch.Tensor | None,
        *,
        track_gradients: bool = False,
    ) -> None:
        if states is not None and track_gradients:
            states = states.detach().requires_grad_(True)
            states.register_hook(self._capture_state_gradients)
        else:
            self._last_grad_summary = {}
        self._states = states
        self._masks = masks

    def grad_summary(self) -> dict[str, float]:
        return dict(self._last_grad_summary)

    def _capture_state_gradients(self, grad: torch.Tensor) -> None:
        grad = grad.detach()
        total = float(grad.abs().mean().item())
        translation = float(_group_grad_abs_mean(grad, group="translation"))
        rotation = float(_group_grad_abs_mean(grad, group="rotation"))
        gripper = float(_group_grad_abs_mean(grad, group="gripper"))
        denom = max(translation + rotation + gripper, 1.0e-12)
        self._last_grad_summary = {
            "proprio_grad_abs_mean": total,
            "proprio_grad_abs_mean_translation": translation,
            "proprio_grad_abs_mean_rotation": rotation,
            "proprio_grad_abs_mean_gripper": gripper,
            "proprio_grad_share_translation": translation / denom,
            "proprio_grad_share_rotation": rotation / denom,
            "proprio_grad_share_gripper": gripper / denom,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_embedding(input_ids)
        if self._states is None or self._masks is None:
            return embeddings
        projector_dtype = next(self.projector.parameters()).dtype
        states = self._states.to(device=input_ids.device, dtype=projector_dtype)
        masks = self._masks.to(device=input_ids.device, dtype=projector_dtype)
        frame_tokens, summary_tokens = self.projector(states, masks)
        frame_tokens = frame_tokens.to(dtype=embeddings.dtype)
        summary_tokens = summary_tokens.to(dtype=embeddings.dtype)
        output = embeddings.clone()
        for batch_index in range(input_ids.shape[0]):
            row_ids = input_ids[batch_index]
            has_summary_token = bool((row_ids == self.summary_token_id).any())
            has_frame_token = bool((row_ids == self.frame_token_id).any())
            if not has_summary_token and not has_frame_token:
                # During autoregressive generation, only the prefill step contains the prompt
                # proprio placeholders. Later decode steps should use normal token embeddings.
                continue
            if self.token_mode in {"summary", "per_frame_plus_summary"}:
                summary_positions = torch.nonzero(
                    row_ids == self.summary_token_id,
                    as_tuple=False,
                ).flatten()
                if summary_positions.numel() != 1:
                    raise ValueError(
                        f"Expected exactly one proprio summary token in row {batch_index}, got {summary_positions.numel()}."
                    )
                output[batch_index, summary_positions[0]] = summary_tokens[batch_index]
            if self.token_mode in {"per_frame", "per_frame_plus_summary"}:
                frame_positions = torch.nonzero(
                    row_ids == self.frame_token_id,
                    as_tuple=False,
                ).flatten()
                if frame_positions.numel() > frame_tokens.shape[1]:
                    raise ValueError(
                        f"Expected at most {frame_tokens.shape[1]} proprio frame tokens in row {batch_index}, "
                        f"got {frame_positions.numel()}."
                    )
                for token_index, position in enumerate(frame_positions.tolist()):
                    output[batch_index, position] = frame_tokens[batch_index, token_index]
        return output


def configure_model_for_proprio(model: torch.nn.Module, processor: Any, config: HLMemoryConfig) -> None:
    if not config.proprio_enabled:
        return
    tokenizer = getattr(processor, "tokenizer", processor)
    _add_proprio_tokens(tokenizer)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    base_embedding = model.get_input_embeddings()
    if isinstance(base_embedding, ProprioEmbeddingWrapper):
        return
    output_dim = _embedding_output_dim(base_embedding)
    frame_token_id = int(tokenizer.convert_tokens_to_ids(PROPRIO_FRAME_TOKEN))
    summary_token_id = int(tokenizer.convert_tokens_to_ids(PROPRIO_SUMMARY_TOKEN))
    projector = ProprioTokenProjector(
        state_dim=config.proprio_state_dim,
        hidden_dim=config.proprio_hidden_dim,
        output_dim=output_dim,
        dropout=config.proprio_dropout,
        noise_std=config.proprio_noise_std,
        projector_mode=config.proprio_projector_mode,
    )
    wrapper = ProprioEmbeddingWrapper(
        base_embedding=base_embedding,
        projector=projector,
        frame_token_id=frame_token_id,
        summary_token_id=summary_token_id,
        token_mode=config.proprio_token_mode,
    )
    model.set_input_embeddings(wrapper)


def load_proprio_state_if_available(model: torch.nn.Module, checkpoint_dir: pathlib.Path | str | None) -> None:
    wrapper = find_proprio_embedding_wrapper(model)
    if wrapper is None or checkpoint_dir is None:
        return
    path = pathlib.Path(checkpoint_dir) / PROPRIO_STATE_FILENAME
    if not path.is_file():
        return
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("projector", payload)
    wrapper.projector.load_state_dict(state_dict)


def save_proprio_state_if_available(model: torch.nn.Module, output_dir: pathlib.Path | str, config: HLMemoryConfig) -> None:
    wrapper = find_proprio_embedding_wrapper(model)
    if wrapper is None:
        return
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"projector": wrapper.projector.state_dict()}, output_dir / PROPRIO_STATE_FILENAME)
    payload = {
        "proprio_enabled": config.proprio_enabled,
        "proprio_token_mode": config.proprio_token_mode,
        "proprio_state_dim": config.proprio_state_dim,
        "proprio_hidden_dim": config.proprio_hidden_dim,
        "proprio_dropout": config.proprio_dropout,
        "proprio_noise_std": config.proprio_noise_std,
        "proprio_projector_mode": config.proprio_projector_mode,
        "frame_token": PROPRIO_FRAME_TOKEN,
        "summary_token": PROPRIO_SUMMARY_TOKEN,
    }
    (output_dir / PROPRIO_CONFIG_FILENAME).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


@contextmanager
def temporarily_unwrap_proprio_embeddings(model: torch.nn.Module) -> Iterator[None]:
    wrapper = find_proprio_embedding_wrapper(model)
    if wrapper is None:
        yield
        return
    model.set_input_embeddings(wrapper.base_embedding)
    try:
        yield
    finally:
        model.set_input_embeddings(wrapper)


def find_proprio_embedding_wrapper(model: torch.nn.Module) -> ProprioEmbeddingWrapper | None:
    accessor = getattr(model, "get_input_embeddings", None)
    if callable(accessor):
        module = accessor()
        if isinstance(module, ProprioEmbeddingWrapper):
            return module
    for module in model.modules():
        if isinstance(module, ProprioEmbeddingWrapper):
            return module
    return None


def proprio_base_embedding_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    wrapper = find_proprio_embedding_wrapper(model)
    return () if wrapper is None else (wrapper.base_embedding,)


def set_model_proprio_batch(
    model: torch.nn.Module,
    samples: Sequence[ExportedHLMemorySample],
    config: HLMemoryConfig,
    *,
    device: torch.device,
) -> None:
    if not config.proprio_enabled:
        return
    wrapper = find_proprio_embedding_wrapper(model)
    if wrapper is None:
        raise ValueError("Proprio is enabled but the model input embedding has not been configured.")
    states, masks = build_proprio_batch(samples, config, device=device)
    wrapper.set_proprio_batch(states, masks, track_gradients=config.proprio_debug_gradients)


def clear_model_proprio_batch(model: torch.nn.Module) -> None:
    wrapper = find_proprio_embedding_wrapper(model)
    if wrapper is not None:
        wrapper.set_proprio_batch(None, None)


def summarize_proprio_gradients(model: torch.nn.Module) -> dict[str, float]:
    wrapper = find_proprio_embedding_wrapper(model)
    return {} if wrapper is None else wrapper.grad_summary()


def build_proprio_batch(
    samples: Sequence[ExportedHLMemorySample],
    config: HLMemoryConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[list[float]]] = []
    mask_rows: list[list[list[float]]] = []
    max_time = 0
    for sample in samples:
        if not sample.recent_robot_states:
            if sample.recent_valid_length == 0:
                rows.append([])
                mask_rows.append([])
                continue
            raise ValueError(
                f"Sample {sample.sample_id} has no recent_robot_states. Re-export the dataset with --proprio-enabled."
            )
        if len(sample.recent_robot_states) != sample.recent_valid_length:
            raise ValueError(
                f"Sample {sample.sample_id} has {len(sample.recent_robot_states)} proprio states but "
                f"recent_valid_length={sample.recent_valid_length}."
            )
        states = [list(row) for row in sample.recent_robot_states]
        masks = (
            [list(row) for row in sample.recent_robot_state_masks]
            if sample.recent_robot_state_masks
            else [[1.0] * len(row) for row in sample.recent_robot_states]
        )
        for row in states:
            if len(row) != config.proprio_state_dim:
                raise ValueError(
                    f"Sample {sample.sample_id} proprio state dim mismatch: expected {config.proprio_state_dim}, got {len(row)}."
                )
        for row in masks:
            if len(row) != config.proprio_state_dim:
                raise ValueError(
                    f"Sample {sample.sample_id} proprio mask dim mismatch: expected {config.proprio_state_dim}, got {len(row)}."
                )
        rows.append(states)
        mask_rows.append(masks)
        max_time = max(max_time, len(states))
    batch_size = len(samples)
    state_tensor = torch.zeros((batch_size, max_time, config.proprio_state_dim), dtype=torch.float32, device=device)
    mask_tensor = torch.zeros_like(state_tensor)
    for batch_index, (states, masks) in enumerate(zip(rows, mask_rows, strict=True)):
        if states:
            state_tensor[batch_index, : len(states)] = torch.tensor(states, dtype=torch.float32, device=device)
            mask_tensor[batch_index, : len(masks)] = torch.tensor(masks, dtype=torch.float32, device=device)
    return state_tensor, mask_tensor


def apply_proprio_ablation(
    sample: ExportedHLMemorySample,
    *,
    mode: str,
) -> ExportedHLMemorySample:
    if mode == "none":
        return sample
    if not sample.recent_robot_states:
        return sample

    states = torch.tensor(sample.recent_robot_states, dtype=torch.float32)
    if sample.recent_robot_state_masks:
        masks = torch.tensor(sample.recent_robot_state_masks, dtype=torch.float32)
    else:
        masks = torch.ones_like(states)

    if mode == "zero":
        states.zero_()
        masks.zero_()
    elif mode == "gripper_only":
        states, masks = _keep_only_groups(states, masks, {"gripper"})
    elif mode == "pose_only":
        states, masks = _keep_only_groups(states, masks, {"translation", "rotation"})
    elif mode == "translation_only":
        states, masks = _keep_only_groups(states, masks, {"translation"})
    elif mode == "rotation_only":
        states, masks = _keep_only_groups(states, masks, {"rotation"})
    elif mode == "time_shuffle":
        states, masks = _permute_time(states, masks, reverse=False)
    elif mode == "reverse_time":
        states, masks = _permute_time(states, masks, reverse=True)
    else:
        raise ValueError(f"Unsupported proprio ablation mode: {mode}")

    return dataclasses.replace(
        sample,
        recent_robot_states=states.tolist(),
        recent_robot_state_masks=masks.tolist(),
    )


def render_proprio_token_text(sample: ExportedHLMemorySample, config: HLMemoryConfig) -> str:
    if not config.proprio_enabled:
        return ""
    if not sample.recent_robot_states:
        if sample.recent_valid_length == 0:
            return "No proprio token is available because Pass A proposed no candidate frame.\n"
        raise ValueError(
            f"Sample {sample.sample_id} has no recent_robot_states. Re-export the dataset with --proprio-enabled."
        )
    tokens: list[str] = []
    if config.proprio_token_mode in {"summary", "per_frame_plus_summary"}:
        tokens.append(PROPRIO_SUMMARY_TOKEN)
    if config.proprio_token_mode in {"per_frame", "per_frame_plus_summary"}:
        tokens.extend(PROPRIO_FRAME_TOKEN for _ in range(len(sample.recent_robot_states)))
    return (
        "The following learned proprio tokens are aligned with the recent observation window "
        "(oldest to newest; the last state is current): "
        + " ".join(tokens)
        + "\n"
    )


def _add_proprio_tokens(tokenizer: Any) -> None:
    tokens = [PROPRIO_FRAME_TOKEN, PROPRIO_SUMMARY_TOKEN]
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged = existing + [token for token in tokens if token not in existing]
    try:
        tokenizer.add_special_tokens({"additional_special_tokens": merged}, replace_additional_special_tokens=False)
    except TypeError:
        tokenizer.add_special_tokens({"additional_special_tokens": merged})


def _embedding_output_dim(embedding: torch.nn.Module) -> int:
    weight = getattr(embedding, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) >= 2:
        return int(weight.shape[-1])
    embedding_dim = getattr(embedding, "embedding_dim", None)
    if embedding_dim is not None:
        return int(embedding_dim)
    raise ValueError(f"Cannot infer embedding dimension from {type(embedding).__name__}.")


def _make_split_branch(input_dim: int, hidden_dim: int, output_dim: int) -> torch.nn.Module:
    effective_input_dim = max(int(input_dim), 1)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(effective_input_dim),
        torch.nn.Linear(effective_input_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def _group_dims_for_state_dim(state_dim: int, *, group: str) -> tuple[int, ...]:
    if state_dim == 14:
        if group == "translation":
            return _DEFAULT_TRANSLATION_DIMS
        if group == "rotation":
            return _DEFAULT_ROTATION_DIMS
        if group == "gripper":
            return _DEFAULT_GRIPPER_DIMS
    if group == "translation":
        return tuple(range(state_dim))
    return ()


def _select_group_inputs(states: torch.Tensor, masks: torch.Tensor, *, group: str) -> torch.Tensor:
    dims = _group_dims_for_state_dim(states.shape[-1], group=group)
    if not dims:
        zeros = torch.zeros((*states.shape[:-1], 1), dtype=states.dtype, device=states.device)
        return torch.cat([zeros, zeros], dim=-1)
    state_subset = states[..., list(dims)]
    mask_subset = masks[..., list(dims)]
    return torch.cat([state_subset * mask_subset, mask_subset], dim=-1)


def _group_grad_abs_mean(grad: torch.Tensor, *, group: str) -> float:
    dims = _group_dims_for_state_dim(grad.shape[-1], group=group)
    if not dims:
        return 0.0
    return float(grad[..., list(dims)].abs().mean().item())


def _keep_only_groups(
    states: torch.Tensor,
    masks: torch.Tensor,
    groups: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_dims: set[int] = set()
    for group in groups:
        keep_dims.update(_group_dims_for_state_dim(states.shape[-1], group=group))
    if len(keep_dims) == states.shape[-1]:
        return states, masks
    keep_mask = torch.zeros(states.shape[-1], dtype=torch.bool, device=states.device)
    if keep_dims:
        keep_mask[list(sorted(keep_dims))] = True
    states = states * keep_mask.view(1, 1, -1).to(dtype=states.dtype)
    masks = masks * keep_mask.view(1, 1, -1).to(dtype=masks.dtype)
    return states, masks


def _permute_time(
    states: torch.Tensor,
    masks: torch.Tensor,
    *,
    reverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.shape[0] <= 1:
        return states, masks
    if reverse:
        order = torch.arange(states.shape[0] - 1, -1, -1, device=states.device)
    else:
        order = torch.randperm(states.shape[0], device=states.device)
    return states[order], masks[order]
