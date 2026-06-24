from __future__ import annotations

import json
import pathlib
import weakref
from collections.abc import Mapping
from typing import Any

import torch


KEYFRAME_AUX_STATE_FILENAME = "hl_keyframe_auxiliary.pt"
KEYFRAME_AUX_CONFIG_FILENAME = "hl_keyframe_auxiliary_config.json"
KEYFRAME_AUX_ANCHOR_POSITIONS_KEY = "_hl_keyframe_aux_anchor_positions"
KEYFRAME_AUX_MASK_KEY = "_hl_keyframe_aux_mask"
KEYFRAME_AUX_POSITION_TARGETS_KEY = "_hl_keyframe_aux_position_targets"
KEYFRAME_AUX_EVENT_TARGETS_KEY = "_hl_keyframe_aux_event_targets"
KEYFRAME_AUX_UPDATE_TARGETS_KEY = "_hl_keyframe_aux_update_targets"
KEYFRAME_AUX_CANONICAL_POSITIONS_KEY = "_hl_keyframe_aux_canonical_positions"
KEYFRAME_AUX_VALID_POSITIONS_KEY = "_hl_keyframe_aux_valid_positions"


class KeyframeAuxiliaryHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_positions: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.position_head = torch.nn.Linear(hidden_dim, num_positions)
        self.event_head = torch.nn.Linear(hidden_dim, 1)
        self.update_head = torch.nn.Linear(hidden_dim, 3)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        parameter = next(self.parameters())
        features = self.trunk(hidden.to(dtype=parameter.dtype))
        return {
            "position_logits": self.position_head(features),
            "event_logits": self.event_head(features).squeeze(-1),
            "update_logits": self.update_head(features),
        }


class KeyframeAuxiliaryModel(torch.nn.Module):
    """Runs the auxiliary head inside the wrapped model forward for FSDP safety."""

    def __init__(self, base_model: torch.nn.Module, *, hidden_dim: int, num_positions: int):
        super().__init__()
        self.base_model = base_model
        self.auxiliary_head = KeyframeAuxiliaryHead(
            input_dim=_resolve_hidden_size(base_model),
            hidden_dim=hidden_dim,
            num_positions=num_positions,
        )
        final_norm = _find_final_language_norm(base_model, hidden_size=_resolve_hidden_size(base_model))
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

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def forward(
        self,
        *args,
        hl_keyframe_aux_anchor_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        captured: list[torch.Tensor] = []
        hook = None
        if hl_keyframe_aux_anchor_positions is not None:
            final_norm = self._final_norm_ref()
            if final_norm is None:
                raise RuntimeError("The keyframe auxiliary final language norm module is no longer available.")

            def _capture_hidden(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    raise ValueError("Final language norm hook did not receive a tensor output.")
                captured.append(hidden)

            hook = final_norm.register_forward_hook(_capture_hidden)
        try:
            outputs = self.base_model(*args, **kwargs)
        finally:
            if hook is not None:
                hook.remove()
        if hl_keyframe_aux_anchor_positions is None:
            return outputs
        if not captured:
            raise ValueError("Keyframe auxiliary hook did not capture the final language hidden state.")
        last_hidden = captured[-1]
        if last_hidden.ndim != 3:
            raise ValueError(f"Expected final hidden state [B, T, D], got {tuple(last_hidden.shape)}.")
        anchors = hl_keyframe_aux_anchor_positions.to(device=last_hidden.device, dtype=torch.long)
        if anchors.shape != (last_hidden.shape[0],):
            raise ValueError(
                f"Expected one keyframe auxiliary anchor per row, got {tuple(anchors.shape)} "
                f"for hidden states {tuple(last_hidden.shape)}."
            )
        row_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        auxiliary = self.auxiliary_head(last_hidden[row_indices, anchors])
        for name, value in auxiliary.items():
            setattr(outputs, f"hl_keyframe_aux_{name}", value)
        return outputs


def configure_keyframe_auxiliary_model(
    model: torch.nn.Module,
    *,
    hidden_dim: int,
    num_positions: int,
    checkpoint_dir: pathlib.Path | str | None = None,
) -> KeyframeAuxiliaryModel:
    if isinstance(model, KeyframeAuxiliaryModel):
        wrapped = model
    else:
        wrapped = KeyframeAuxiliaryModel(
            model,
            hidden_dim=hidden_dim,
            num_positions=num_positions,
        )
    if checkpoint_dir is not None:
        path = pathlib.Path(checkpoint_dir) / KEYFRAME_AUX_STATE_FILENAME
        if path.is_file():
            payload = torch.load(path, map_location="cpu")
            wrapped.auxiliary_head.load_state_dict(payload.get("auxiliary_head", payload))
    return wrapped


def save_keyframe_auxiliary_state(
    model: torch.nn.Module,
    output_dir: pathlib.Path | str,
    *,
    hidden_dim: int,
    num_positions: int,
    full_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> None:
    wrapped = find_keyframe_auxiliary_model(model)
    if wrapped is None:
        return
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if full_state_dict is None:
        auxiliary_state = wrapped.auxiliary_head.state_dict()
    else:
        marker = "auxiliary_head."
        auxiliary_state = {
            key.split(marker, 1)[1]: value
            for key, value in full_state_dict.items()
            if marker in key
        }
        if not auxiliary_state:
            raise ValueError("Full checkpoint state did not contain keyframe auxiliary head parameters.")
    torch.save({"auxiliary_head": auxiliary_state}, output_dir / KEYFRAME_AUX_STATE_FILENAME)
    payload = {
        "hidden_dim": int(hidden_dim),
        "num_positions": int(num_positions),
        "update_classes": ["reject", "add", "duplicate"],
    }
    (output_dir / KEYFRAME_AUX_CONFIG_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    )


def find_keyframe_auxiliary_model(model: torch.nn.Module) -> KeyframeAuxiliaryModel | None:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, KeyframeAuxiliaryModel):
            return current
        for attribute in ("module", "base_model"):
            next_model = getattr(current, attribute, None)
            if isinstance(next_model, torch.nn.Module):
                current = next_model
                break
        else:
            break
    return None


def unwrap_keyframe_auxiliary_model(model: torch.nn.Module) -> torch.nn.Module:
    wrapped = find_keyframe_auxiliary_model(model)
    return wrapped.base_model if wrapped is not None else model


def _resolve_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates: list[Any] = [
        getattr(config, "hidden_size", None),
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(getattr(config, "language_config", None), "hidden_size", None),
    ]
    for candidate in candidates:
        if candidate is not None and int(candidate) > 0:
            return int(candidate)
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    raise ValueError("Could not resolve VLM hidden size for keyframe auxiliary head.")


def _find_final_language_norm(model: torch.nn.Module, *, hidden_size: int) -> torch.nn.Module:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if not name.endswith("norm") or any(token in lowered for token in ("visual", "vision")):
            continue
        normalized_shape = getattr(module, "normalized_shape", None)
        if normalized_shape is not None:
            shape = tuple(int(value) for value in normalized_shape)
            if shape and shape[-1] != hidden_size:
                continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.numel() != hidden_size:
            continue
        candidates.append((name, module))
    if not candidates:
        raise ValueError("Could not locate the final language-model norm for keyframe auxiliary features.")
    preferred = [
        item
        for item in candidates
        if any(token in item[0].lower() for token in ("language_model", "text_model", "model.norm"))
    ]
    return (preferred or candidates)[-1][1]


def auxiliary_outputs(outputs: Any) -> Mapping[str, torch.Tensor] | None:
    position_logits = getattr(outputs, "hl_keyframe_aux_position_logits", None)
    event_logits = getattr(outputs, "hl_keyframe_aux_event_logits", None)
    update_logits = getattr(outputs, "hl_keyframe_aux_update_logits", None)
    if not all(isinstance(value, torch.Tensor) for value in (position_logits, event_logits, update_logits)):
        return None
    return {
        "position_logits": position_logits,
        "event_logits": event_logits,
        "update_logits": update_logits,
    }
