from __future__ import annotations

import torch

from openpi.hl_memory.keyframe_auxiliary import auxiliary_outputs
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_ANCHOR_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_CANONICAL_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_EVENT_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_MASK_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_POSITION_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_UPDATE_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_VALID_POSITIONS_KEY


HL_STAGE_IDS_KEY = "_hl_stage_ids"
HL_FIELD_IDS_KEY = "_hl_field_ids"
HL_FIELD_VALUE_MASK_KEY = "_hl_field_value_mask"
HL_LOSS_FIELD_IDS_BY_NAME = {
    "current_objective": 1,
    "horizon_current_objective": 2,
    "keyframe_candidate_positions": 3,
    "completed_objective": 4,
}
HL_LOSS_FIELD_NAMES_BY_ID = {value: key for key, value in HL_LOSS_FIELD_IDS_BY_NAME.items()}


def compute_hl_target_loss(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    two_pass_predict_weight: float = 1.0,
    two_pass_confirm_weight: float = 1.0,
    keyframe_aux_position_weight: float = 0.0,
    keyframe_aux_event_weight: float = 0.0,
    keyframe_aux_timing_weight: float = 0.0,
    keyframe_aux_update_weight: float = 0.0,
    keyframe_aux_event_pos_weight: float = 1.0,
    field_current_objective_weight: float = 1.0,
    field_horizon_objective_weight: float = 1.0,
    field_keyframe_candidate_positions_weight: float = 0.1,
    field_completed_objective_weight: float = 1.0,
    field_template_weight: float = 0.1,
) -> torch.Tensor:
    if two_pass_predict_weight <= 0.0 or two_pass_confirm_weight <= 0.0:
        raise ValueError("Two-pass loss weights must be positive.")
    loss, _ = compute_hl_target_loss_with_metrics(
        model,
        inputs,
        two_pass_predict_weight=two_pass_predict_weight,
        two_pass_confirm_weight=two_pass_confirm_weight,
        keyframe_aux_position_weight=keyframe_aux_position_weight,
        keyframe_aux_event_weight=keyframe_aux_event_weight,
        keyframe_aux_timing_weight=keyframe_aux_timing_weight,
        keyframe_aux_update_weight=keyframe_aux_update_weight,
        keyframe_aux_event_pos_weight=keyframe_aux_event_pos_weight,
        field_current_objective_weight=field_current_objective_weight,
        field_horizon_objective_weight=field_horizon_objective_weight,
        field_keyframe_candidate_positions_weight=field_keyframe_candidate_positions_weight,
        field_completed_objective_weight=field_completed_objective_weight,
        field_template_weight=field_template_weight,
    )
    return loss


def compute_hl_target_loss_with_metrics(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    two_pass_predict_weight: float = 1.0,
    two_pass_confirm_weight: float = 1.0,
    keyframe_aux_position_weight: float = 0.0,
    keyframe_aux_event_weight: float = 0.0,
    keyframe_aux_timing_weight: float = 0.0,
    keyframe_aux_update_weight: float = 0.0,
    keyframe_aux_event_pos_weight: float = 1.0,
    field_current_objective_weight: float = 1.0,
    field_horizon_objective_weight: float = 1.0,
    field_keyframe_candidate_positions_weight: float = 0.1,
    field_completed_objective_weight: float = 1.0,
    field_template_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if two_pass_predict_weight <= 0.0 or two_pass_confirm_weight <= 0.0:
        raise ValueError("Two-pass loss weights must be positive.")

    labels = inputs["labels"]
    stage_ids = inputs.get(HL_STAGE_IDS_KEY)
    field_ids = inputs.get(HL_FIELD_IDS_KEY)
    field_value_mask = inputs.get(HL_FIELD_VALUE_MASK_KEY)
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key != "labels" and not key.startswith("_hl_")
    }
    auxiliary_enabled = any(
        weight > 0.0
        for weight in (
            keyframe_aux_position_weight,
            keyframe_aux_event_weight,
            keyframe_aux_timing_weight,
            keyframe_aux_update_weight,
        )
    )
    if auxiliary_enabled:
        model_inputs["hl_keyframe_aux_anchor_positions"] = inputs[KEYFRAME_AUX_ANCHOR_POSITIONS_KEY]
    logits_to_keep = target_logit_positions(labels)
    effective_logits_to_keep = logits_to_keep
    if logits_to_keep is None:
        outputs = model(**model_inputs)
        logits = outputs.logits[:, :-1]
        shifted_labels = labels[:, 1:]
    else:
        try:
            outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
            logits = outputs.logits
            shifted_labels = torch.nn.functional.pad(labels, (0, 1), value=-100).index_select(
                1,
                logits_to_keep + 1,
            )
        except TypeError:
            effective_logits_to_keep = None
            outputs = model(**model_inputs)
            logits = outputs.logits[:, :-1]
            shifted_labels = labels[:, 1:]

    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    supervised = shifted_labels != -100
    metrics: dict[str, torch.Tensor] = {
        "loss": token_losses.sum() / supervised.sum().clamp_min(1),
        "tokens": supervised.sum().to(dtype=torch.float32),
    }
    if isinstance(field_ids, torch.Tensor):
        if effective_logits_to_keep is None:
            shifted_field_ids = field_ids[:, 1:]
            shifted_field_value_mask = (
                field_value_mask[:, 1:] if isinstance(field_value_mask, torch.Tensor) else None
            )
        else:
            shifted_field_ids = torch.nn.functional.pad(field_ids, (0, 1), value=-1).index_select(
                1,
                effective_logits_to_keep + 1,
            )
            shifted_field_value_mask = (
                torch.nn.functional.pad(field_value_mask, (0, 1), value=False).index_select(
                    1,
                    effective_logits_to_keep + 1,
                )
                if isinstance(field_value_mask, torch.Tensor)
                else None
            )
        metrics.update(_field_loss_metrics(token_losses, supervised, shifted_field_ids))
        language_token_losses, language_supervised_weights = _apply_field_loss_weights(
            token_losses,
            supervised,
            shifted_field_ids,
            shifted_field_value_mask,
            current_weight=field_current_objective_weight,
            horizon_weight=field_horizon_objective_weight,
            keyframe_weight=field_keyframe_candidate_positions_weight,
            completed_weight=field_completed_objective_weight,
            template_weight=field_template_weight,
        )
        if auxiliary_enabled and "loss_field_keyframe_candidate_positions" in metrics:
            metrics["loss_field_keyframe_candidate_positions_token_ce"] = metrics[
                "loss_field_keyframe_candidate_positions"
            ]
    else:
        language_token_losses = token_losses
        language_supervised_weights = supervised.to(dtype=token_losses.dtype)
    language_loss = language_token_losses.sum() / language_supervised_weights.sum().clamp_min(1.0)
    metrics["loss_language_unweighted"] = metrics["loss"]
    metrics["loss"] = language_loss
    if stage_ids is None:
        stage_weighted_loss = language_loss
    else:
        if stage_ids.ndim != 1 or stage_ids.shape[0] != shifted_labels.shape[0]:
            raise ValueError(
                f"Expected one two-pass stage id per row, got {tuple(stage_ids.shape)} "
                f"for labels {tuple(shifted_labels.shape)}."
            )
        per_row_loss = language_token_losses.sum(dim=1) / language_supervised_weights.sum(dim=1).clamp_min(1.0)
        metrics.update(_stage_loss_metrics(per_row_loss, supervised, stage_ids.to(device=per_row_loss.device)))
        row_weights = torch.where(
            stage_ids.to(device=per_row_loss.device) == 1,
            torch.as_tensor(two_pass_confirm_weight, device=per_row_loss.device, dtype=per_row_loss.dtype),
            torch.as_tensor(two_pass_predict_weight, device=per_row_loss.device, dtype=per_row_loss.dtype),
        )
        stage_weighted_loss = (per_row_loss * row_weights).sum() / row_weights.sum()

    auxiliary_loss, auxiliary_metrics = _keyframe_auxiliary_loss(
        model,
        outputs,
        inputs,
        position_weight=keyframe_aux_position_weight,
        event_weight=keyframe_aux_event_weight,
        timing_weight=keyframe_aux_timing_weight,
        update_weight=keyframe_aux_update_weight,
        event_pos_weight=keyframe_aux_event_pos_weight,
    )
    metrics.update(auxiliary_metrics)
    if auxiliary_enabled and "loss_aux_total" in auxiliary_metrics:
        metrics["loss_field_keyframe_candidate_positions"] = auxiliary_metrics["loss_aux_total"]
    total_loss = stage_weighted_loss + auxiliary_loss
    metrics["loss_language"] = stage_weighted_loss
    metrics["loss"] = total_loss
    return total_loss, metrics


def _keyframe_auxiliary_loss(
    model: torch.nn.Module,
    outputs,
    inputs: dict[str, torch.Tensor],
    *,
    position_weight: float,
    event_weight: float,
    timing_weight: float,
    update_weight: float,
    event_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = (position_weight, event_weight, timing_weight, update_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Keyframe auxiliary loss weights must be non-negative.")
    if event_pos_weight <= 0.0:
        raise ValueError("keyframe_aux_event_pos_weight must be positive.")
    reference = outputs.logits
    zero = reference.sum() * 0.0
    if not any(weight > 0.0 for weight in weights):
        return zero, {}
    auxiliary = auxiliary_outputs(outputs, model=model)
    if auxiliary is None:
        raise ValueError(
            "Keyframe auxiliary loss is enabled, but the model did not return auxiliary logits. "
            "Configure the keyframe auxiliary model before FSDP/DDP wrapping."
        )
    row_mask = inputs[KEYFRAME_AUX_MASK_KEY].to(device=reference.device, dtype=torch.bool)
    event_targets = inputs[KEYFRAME_AUX_EVENT_TARGETS_KEY].to(device=reference.device)
    position_targets = inputs[KEYFRAME_AUX_POSITION_TARGETS_KEY].to(device=reference.device)
    update_targets = inputs[KEYFRAME_AUX_UPDATE_TARGETS_KEY].to(device=reference.device)
    canonical_positions = inputs[KEYFRAME_AUX_CANONICAL_POSITIONS_KEY].to(device=reference.device)
    valid_positions = inputs[KEYFRAME_AUX_VALID_POSITIONS_KEY].to(device=reference.device, dtype=torch.bool)

    event_logits = auxiliary["event_logits"].float()
    position_logits = auxiliary["position_logits"].float()
    update_logits = auxiliary["update_logits"].float()
    metrics: dict[str, torch.Tensor] = {}
    total = zero

    if bool(row_mask.any()):
        pos_weight = torch.as_tensor(event_pos_weight, device=reference.device, dtype=event_logits.dtype)
        event_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            event_logits[row_mask],
            event_targets[row_mask].float(),
            pos_weight=pos_weight,
        )
        update_loss = torch.nn.functional.cross_entropy(
            update_logits[row_mask],
            update_targets[row_mask].long(),
        )
        metrics["accuracy_aux_event"] = (
            (event_logits[row_mask] >= 0.0) == (event_targets[row_mask] >= 0.5)
        ).float().mean()
        event_predictions = event_logits[row_mask] >= 0.0
        event_positive = event_targets[row_mask] >= 0.5
        metrics["recall_aux_event"] = (
            (event_predictions & event_positive).float().sum() / event_positive.float().sum().clamp_min(1.0)
        )
        metrics["empty_rate_aux_event"] = (~event_predictions).float().mean()
        metrics["accuracy_aux_update"] = (
            update_logits[row_mask].argmax(dim=-1) == update_targets[row_mask]
        ).float().mean()
    else:
        event_loss = zero
        update_loss = zero
        metrics["accuracy_aux_event"] = zero
        metrics["recall_aux_event"] = zero
        metrics["empty_rate_aux_event"] = zero
        metrics["accuracy_aux_update"] = zero
    metrics["loss_aux_event"] = event_loss
    metrics["loss_aux_update"] = update_loss
    total = total + event_weight * event_loss + update_weight * update_loss

    positive_mask = row_mask & (event_targets > 0.5) & (position_targets.sum(dim=1) > 0.0)
    if bool(positive_mask.any()):
        masked_logits = position_logits.masked_fill(~valid_positions, torch.finfo(position_logits.dtype).min)
        log_probs = torch.nn.functional.log_softmax(masked_logits[positive_mask], dim=-1)
        position_loss = -(position_targets[positive_mask] * log_probs).sum(dim=-1).mean()

        probabilities = torch.nn.functional.softmax(masked_logits[positive_mask], dim=-1)
        indices = torch.arange(position_logits.shape[-1], device=position_logits.device, dtype=torch.float32)
        expected_positions = (probabilities * indices).sum(dim=-1)
        canonical = canonical_positions[positive_mask]
        canonical_mask = canonical >= 0.0
        timing_loss = (
            torch.nn.functional.smooth_l1_loss(
                expected_positions[canonical_mask],
                canonical[canonical_mask],
            )
            if bool(canonical_mask.any())
            else zero
        )
        metrics["error_aux_position_abs"] = (
            (expected_positions[canonical_mask] - canonical[canonical_mask]).abs().mean()
            if bool(canonical_mask.any())
            else zero
        )
        top1 = masked_logits[positive_mask].argmax(dim=-1).to(dtype=torch.float32)
        target_positive = position_targets[positive_mask] > 0.0
        metrics["recall_aux_position_top1"] = (
            target_positive.gather(1, top1.to(dtype=torch.long).unsqueeze(-1)).float().mean()
            if target_positive.numel() > 0
            else zero
        )
    else:
        position_loss = zero
        timing_loss = zero
        metrics["error_aux_position_abs"] = zero
        metrics["recall_aux_position_top1"] = zero
    metrics["loss_aux_position"] = position_loss
    metrics["loss_aux_timing"] = timing_loss
    total = total + position_weight * position_loss + timing_weight * timing_loss
    metrics["loss_aux_total"] = total
    return total, metrics


def _apply_field_loss_weights(
    token_losses: torch.Tensor,
    supervised: torch.Tensor,
    field_ids: torch.Tensor,
    field_value_mask: torch.Tensor | None,
    *,
    current_weight: float,
    horizon_weight: float,
    keyframe_weight: float,
    completed_weight: float,
    template_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    for name, value in (
        ("field_current_objective_weight", current_weight),
        ("field_horizon_objective_weight", horizon_weight),
        ("field_keyframe_candidate_positions_weight", keyframe_weight),
        ("field_completed_objective_weight", completed_weight),
        ("field_template_weight", template_weight),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    weights = supervised.to(dtype=token_losses.dtype)
    if field_value_mask is None:
        field_value_mask = field_ids >= 0
    else:
        field_value_mask = field_value_mask.to(device=field_ids.device, dtype=torch.bool)
    field_template_mask = (field_ids >= 0) & ~field_value_mask
    weights = torch.where(
        field_template_mask,
        torch.as_tensor(template_weight, device=token_losses.device, dtype=token_losses.dtype),
        weights,
    )
    weights = torch.where(
        (field_ids == HL_LOSS_FIELD_IDS_BY_NAME["current_objective"]) & field_value_mask,
        torch.as_tensor(current_weight, device=token_losses.device, dtype=token_losses.dtype),
        weights,
    )
    weights = torch.where(
        (field_ids == HL_LOSS_FIELD_IDS_BY_NAME["horizon_current_objective"]) & field_value_mask,
        torch.as_tensor(horizon_weight, device=token_losses.device, dtype=token_losses.dtype),
        weights,
    )
    weights = torch.where(
        (field_ids == HL_LOSS_FIELD_IDS_BY_NAME["keyframe_candidate_positions"]) & field_value_mask,
        torch.as_tensor(keyframe_weight, device=token_losses.device, dtype=token_losses.dtype),
        weights,
    )
    weights = torch.where(
        (field_ids == HL_LOSS_FIELD_IDS_BY_NAME["completed_objective"]) & field_value_mask,
        torch.as_tensor(completed_weight, device=token_losses.device, dtype=token_losses.dtype),
        weights,
    )
    weights = weights * supervised.to(dtype=token_losses.dtype)
    return token_losses * weights, weights


def _stage_loss_metrics(
    per_row_loss: torch.Tensor,
    supervised: torch.Tensor,
    stage_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for stage_id, stage_name in ((0, "predict"), (2, "horizon"), (1, "confirm")):
        row_mask = stage_ids == stage_id
        if not bool(row_mask.any()):
            continue
        metrics[f"loss_stage_{stage_name}"] = per_row_loss[row_mask].mean()
        metrics[f"rows_stage_{stage_name}"] = row_mask.sum().to(dtype=torch.float32)
        metrics[f"tokens_stage_{stage_name}"] = supervised[row_mask].sum().to(dtype=torch.float32)
    return metrics


def _field_loss_metrics(
    token_losses: torch.Tensor,
    supervised: torch.Tensor,
    shifted_field_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for field_id, field_name in HL_LOSS_FIELD_NAMES_BY_ID.items():
        mask = supervised & (shifted_field_ids.to(device=supervised.device) == field_id)
        if not bool(mask.any()):
            continue
        metrics[f"loss_field_{field_name}"] = token_losses[mask].mean()
        metrics[f"tokens_field_{field_name}"] = mask.sum().to(dtype=torch.float32)
    return metrics


def target_logit_positions(labels: torch.Tensor) -> torch.Tensor | None:
    # A logit at position i predicts the label at position i + 1. Prompt and
    # padding labels are -100, so vocab projection is only needed immediately
    # before supervised target tokens.
    if labels.shape[1] < 2:
        return None
    supervised_next_token = labels[:, 1:] != -100
    positions = torch.nonzero(supervised_next_token.any(dim=0), as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return positions.to(device=labels.device, dtype=torch.long)
