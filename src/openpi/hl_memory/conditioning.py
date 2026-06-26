from __future__ import annotations

import json
import pathlib
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.keyframe_auxiliary import _find_final_language_norm
from openpi.hl_memory.keyframe_auxiliary import _resolve_hidden_size
from openpi.hl_memory.proprio import build_proprio_batch


CONDITIONING_STATE_FILENAME = "hl_conditioning.pt"
CONDITIONING_CONFIG_FILENAME = "hl_conditioning_config.json"
HL_PROGRESS_INPUT_IDS_KEY = "hl_progress_condition_input_ids"
HL_PROGRESS_ATTENTION_MASK_KEY = "hl_progress_condition_attention_mask"
HL_STATE_VALUES_KEY = "hl_state_condition_values"
HL_STATE_MASKS_KEY = "hl_state_condition_masks"
HL_CONDITION_STAGE_IDS_KEY = "hl_condition_stage_ids"
STAGE_PREDICT = 0
STAGE_CONFIRM = 1
STAGE_HORIZON = 2


class TextProgressEncoder(torch.nn.Module):
    def __init__(self, *, hidden_size: int, condition_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, condition_dim),
        )

    def forward(
        self,
        embedding: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = embedding(input_ids).detach()
        mask = attention_mask.to(device=embedded.device, dtype=embedded.dtype).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled.to(dtype=next(self.parameters()).dtype))


class StateConditionEncoder(torch.nn.Module):
    def __init__(self, *, state_dim: int, condition_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_dim = int(state_dim)
        self.frame_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(state_dim * 2),
            torch.nn.Linear(state_dim * 2, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, condition_dim),
        )

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or masks.ndim != 3 or states.shape != masks.shape:
            raise ValueError(
                f"State condition expects states/masks [B,T,D] with matching shapes, got "
                f"{tuple(states.shape)} and {tuple(masks.shape)}."
            )
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state condition dim={self.state_dim}, got {states.shape[-1]}.")
        parameter = next(self.parameters())
        states = states.to(dtype=parameter.dtype)
        masks = masks.to(dtype=parameter.dtype)
        valid = masks.to(dtype=torch.bool).any(dim=-1)
        encoded = self.frame_mlp(torch.cat([states * masks, masks], dim=-1))
        encoded = encoded * valid.unsqueeze(-1).to(dtype=encoded.dtype)
        denom = valid.sum(dim=1, keepdim=True).to(dtype=encoded.dtype).clamp_min(1.0)
        return encoded.sum(dim=1) / denom


class FiLMProjector(torch.nn.Module):
    def __init__(self, *, condition_dim: int, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(condition_dim, hidden_size * 2)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, condition: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.proj(condition).chunk(2, dim=-1)
        return gamma.to(dtype=dtype), beta.to(dtype=dtype)


class HLConditioningModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, config: HLMemoryConfig):
        super().__init__()
        self.base_model = base_model
        self.hl_config = config
        hidden_size = _resolve_hidden_size(base_model)
        self.progress_encoder = (
            TextProgressEncoder(
                hidden_size=hidden_size,
                condition_dim=config.progress_condition_dim,
                hidden_dim=config.progress_condition_hidden_dim,
                dropout=config.progress_condition_dropout,
            )
            if config.progress_condition_enabled
            else None
        )
        self.progress_film = (
            FiLMProjector(condition_dim=config.progress_condition_dim, hidden_size=hidden_size)
            if config.progress_condition_enabled
            else None
        )
        self.state_encoder = (
            StateConditionEncoder(
                state_dim=config.proprio_state_dim,
                condition_dim=config.state_condition_dim,
                hidden_dim=config.state_condition_hidden_dim,
                dropout=config.state_condition_dropout,
            )
            if config.state_condition_enabled and config.state_condition_mode in {"film", "both"}
            else None
        )
        self.state_film = (
            FiLMProjector(condition_dim=config.state_condition_dim, hidden_size=hidden_size)
            if self.state_encoder is not None
            else None
        )
        final_norm = _find_final_language_norm(base_model, hidden_size=hidden_size)
        self._final_norm_ref = weakref.ref(final_norm)

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def generate(self, *args, **kwargs):
        condition_kwargs = _pop_condition_kwargs(kwargs)
        with self._conditioning_hook(**condition_kwargs):
            return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        condition_kwargs = _pop_condition_kwargs(kwargs)
        with self._conditioning_hook(**condition_kwargs):
            return self.base_model(*args, **kwargs)

    def _conditioning_hook(
        self,
        *,
        hl_progress_condition_input_ids: torch.Tensor | None = None,
        hl_progress_condition_attention_mask: torch.Tensor | None = None,
        hl_state_condition_values: torch.Tensor | None = None,
        hl_state_condition_masks: torch.Tensor | None = None,
        hl_condition_stage_ids: torch.Tensor | None = None,
    ):
        model = self

        class _HookContext:
            def __enter__(self_inner):
                final_norm = model._final_norm_ref()
                if final_norm is None:
                    raise RuntimeError("The conditioning final language norm module is no longer available.")
                hook = final_norm.register_forward_hook(
                    lambda _module, _inputs, output: model._apply_conditioning(
                        output,
                        progress_input_ids=hl_progress_condition_input_ids,
                        progress_attention_mask=hl_progress_condition_attention_mask,
                        state_values=hl_state_condition_values,
                        state_masks=hl_state_condition_masks,
                        stage_ids=hl_condition_stage_ids,
                    )
                )
                self_inner._hook = hook
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                self_inner._hook.remove()

        return _HookContext()

    def _apply_conditioning(
        self,
        output,
        *,
        progress_input_ids: torch.Tensor | None,
        progress_attention_mask: torch.Tensor | None,
        state_values: torch.Tensor | None,
        state_masks: torch.Tensor | None,
        stage_ids: torch.Tensor | None,
    ):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            return output
        modulated = hidden
        if self.progress_encoder is not None and progress_input_ids is not None:
            if progress_attention_mask is None:
                progress_attention_mask = torch.ones_like(progress_input_ids)
            condition = self.progress_encoder(
                self.base_model.get_input_embeddings(),
                progress_input_ids.to(device=hidden.device),
                progress_attention_mask.to(device=hidden.device),
            )
            strength = _stage_strengths(
                stage_ids,
                batch_size=hidden.shape[0],
                device=hidden.device,
                dtype=condition.dtype,
                predict_strength=self.hl_config.progress_condition_predict_strength,
                horizon_strength=self.hl_config.progress_condition_horizon_strength,
                confirm_strength=self.hl_config.progress_condition_confirm_strength,
            )
            condition = condition * strength.unsqueeze(-1)
            modulated = _apply_film(modulated, self.progress_film, condition)
        if self.state_encoder is not None and state_values is not None:
            if state_masks is None:
                state_masks = torch.ones_like(state_values)
            condition = self.state_encoder(
                state_values.to(device=hidden.device),
                state_masks.to(device=hidden.device),
            )
            modulated = _apply_film(modulated, self.state_film, condition)
        if isinstance(output, tuple):
            return (modulated, *output[1:])
        return modulated


def configure_conditioning_model(
    model: torch.nn.Module,
    config: HLMemoryConfig,
    *,
    checkpoint_dir: pathlib.Path | str | None = None,
) -> torch.nn.Module:
    if not (config.progress_condition_enabled or config.state_condition_enabled):
        return model
    wrapped = find_conditioning_model(model)
    if wrapped is not None:
        if checkpoint_dir is not None:
            load_conditioning_state_if_available(model, checkpoint_dir)
        return model
    wrapped = HLConditioningModel(model, config)
    if checkpoint_dir is not None:
        load_conditioning_state_if_available(wrapped, checkpoint_dir)
    return wrapped


def load_conditioning_state_if_available(model: torch.nn.Module, checkpoint_dir: pathlib.Path | str | None) -> None:
    wrapped = find_conditioning_model(model)
    if wrapped is None or checkpoint_dir is None:
        return
    path = pathlib.Path(checkpoint_dir) / CONDITIONING_STATE_FILENAME
    if not path.is_file():
        return
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("conditioning", payload)
    wrapped.load_state_dict(state_dict, strict=False)


def save_conditioning_state_if_available(
    model: torch.nn.Module,
    output_dir: pathlib.Path | str,
    config: HLMemoryConfig,
    *,
    full_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    wrapped = find_conditioning_model(model)
    if wrapped is None:
        return
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if full_state_dict is None:
        state_dict = wrapped.state_dict()
    else:
        state_dict = _extract_conditioning_state(full_state_dict)
    torch.save({"conditioning": state_dict}, output_dir / CONDITIONING_STATE_FILENAME)
    payload = {
        "progress_condition_enabled": config.progress_condition_enabled,
        "progress_condition_input_mode": config.progress_condition_input_mode,
        "progress_condition_dim": config.progress_condition_dim,
        "progress_condition_hidden_dim": config.progress_condition_hidden_dim,
        "progress_condition_dropout": config.progress_condition_dropout,
        "progress_condition_predict_strength": config.progress_condition_predict_strength,
        "progress_condition_horizon_strength": config.progress_condition_horizon_strength,
        "progress_condition_confirm_strength": config.progress_condition_confirm_strength,
        "state_condition_enabled": config.state_condition_enabled,
        "state_condition_mode": config.state_condition_mode,
        "state_condition_dim": config.state_condition_dim,
        "state_condition_hidden_dim": config.state_condition_hidden_dim,
        "state_condition_dropout": config.state_condition_dropout,
    }
    (output_dir / CONDITIONING_CONFIG_FILENAME).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def find_conditioning_model(model: torch.nn.Module) -> HLConditioningModel | None:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HLConditioningModel):
            return current
        for attribute in ("module", "base_model"):
            next_model = getattr(current, attribute, None)
            if isinstance(next_model, torch.nn.Module):
                current = next_model
                break
        else:
            break
    return None


def unwrap_conditioning_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = find_conditioning_model(model)
    return wrapped.base_model if wrapped is not None else model


def build_conditioning_batch(
    samples: Sequence[ExportedHLMemorySample],
    config: HLMemoryConfig,
    tokenizer: Any,
    *,
    device: torch.device,
    stage_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if config.progress_condition_enabled:
        texts = [render_progress_condition_text(sample, config) for sample in samples]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tensors[HL_PROGRESS_INPUT_IDS_KEY] = encoded["input_ids"].to(device)
        tensors[HL_PROGRESS_ATTENTION_MASK_KEY] = encoded.get(
            "attention_mask",
            torch.ones_like(encoded["input_ids"]),
        ).to(device)
    if config.state_condition_enabled and config.state_condition_mode in {"film", "both"}:
        states, masks = build_proprio_batch(samples, config, device=device)
        tensors[HL_STATE_VALUES_KEY] = states
        tensors[HL_STATE_MASKS_KEY] = masks
    if tensors and stage_ids is not None:
        tensors[HL_CONDITION_STAGE_IDS_KEY] = stage_ids.to(device=device)
    return tensors


def render_progress_condition_text(sample: ExportedHLMemorySample, config: HLMemoryConfig) -> str:
    mode = config.progress_condition_input_mode
    if mode == "full":
        memory = sample.language_memory.strip() or "No accepted completed event yet."
    else:
        memory = _completed_only_memory(sample.language_memory)
    parts = [f"Completed events: {memory or 'none'}"]
    if mode == "structured":
        last_completed = sample.last_objective.strip()
        if last_completed:
            parts.append(f"Last completed objective: {last_completed}")
    return "\n".join(parts)


def _completed_only_memory(memory: str) -> str:
    lines: list[str] = []
    forbidden = ("current objective", "current subtask", "phase", "subtask progress", "should advance")
    for raw_line in memory.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key = line.split(":", 1)[0].strip().lower()
        if any(token in key for token in forbidden):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _apply_film(
    hidden: torch.Tensor,
    projector: FiLMProjector | None,
    condition: torch.Tensor,
) -> torch.Tensor:
    if projector is None:
        return hidden
    gamma, beta = projector(condition, dtype=hidden.dtype)
    return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


def _stage_strengths(
    stage_ids: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    predict_strength: float,
    horizon_strength: float,
    confirm_strength: float,
) -> torch.Tensor:
    if stage_ids is None:
        return torch.full((batch_size,), predict_strength, device=device, dtype=dtype)
    stage_ids = stage_ids.to(device=device)
    if stage_ids.shape != (batch_size,):
        raise ValueError(f"Expected condition stage ids shape {(batch_size,)}, got {tuple(stage_ids.shape)}.")
    predict = torch.as_tensor(predict_strength, device=device, dtype=dtype)
    horizon = torch.as_tensor(horizon_strength, device=device, dtype=dtype)
    confirm = torch.as_tensor(confirm_strength, device=device, dtype=dtype)
    return torch.where(
        stage_ids == STAGE_CONFIRM,
        confirm,
        torch.where(stage_ids == STAGE_HORIZON, horizon, predict),
    )


def _pop_condition_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "hl_progress_condition_input_ids": kwargs.pop(HL_PROGRESS_INPUT_IDS_KEY, None),
        "hl_progress_condition_attention_mask": kwargs.pop(HL_PROGRESS_ATTENTION_MASK_KEY, None),
        "hl_state_condition_values": kwargs.pop(HL_STATE_VALUES_KEY, None),
        "hl_state_condition_masks": kwargs.pop(HL_STATE_MASKS_KEY, None),
        "hl_condition_stage_ids": kwargs.pop(HL_CONDITION_STAGE_IDS_KEY, None),
    }


def _extract_conditioning_state(full_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    markers = (
        "progress_encoder.",
        "progress_film.",
        "state_encoder.",
        "state_film.",
    )
    for key, value in full_state_dict.items():
        for marker in markers:
            if marker in key:
                result[key[key.index(marker) :]] = value
                break
    if not result:
        raise ValueError("Full checkpoint state did not contain HL conditioning parameters.")
    return result
