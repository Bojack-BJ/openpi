from __future__ import annotations

import torch


HL_STAGE_IDS_KEY = "_hl_stage_ids"
HL_FIELD_IDS_KEY = "_hl_field_ids"
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
) -> torch.Tensor:
    if two_pass_predict_weight <= 0.0 or two_pass_confirm_weight <= 0.0:
        raise ValueError("Two-pass loss weights must be positive.")
    loss, _ = compute_hl_target_loss_with_metrics(
        model,
        inputs,
        two_pass_predict_weight=two_pass_predict_weight,
        two_pass_confirm_weight=two_pass_confirm_weight,
    )
    return loss


def compute_hl_target_loss_with_metrics(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    two_pass_predict_weight: float = 1.0,
    two_pass_confirm_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if two_pass_predict_weight <= 0.0 or two_pass_confirm_weight <= 0.0:
        raise ValueError("Two-pass loss weights must be positive.")

    labels = inputs["labels"]
    stage_ids = inputs.get(HL_STAGE_IDS_KEY)
    field_ids = inputs.get(HL_FIELD_IDS_KEY)
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key != "labels" and not key.startswith("_hl_")
    }
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
        else:
            shifted_field_ids = torch.nn.functional.pad(field_ids, (0, 1), value=-1).index_select(
                1,
                effective_logits_to_keep + 1,
            )
        metrics.update(_field_loss_metrics(token_losses, supervised, shifted_field_ids))
    if stage_ids is None:
        return metrics["loss"], metrics

    if stage_ids.ndim != 1 or stage_ids.shape[0] != shifted_labels.shape[0]:
        raise ValueError(
            f"Expected one two-pass stage id per row, got {tuple(stage_ids.shape)} "
            f"for labels {tuple(shifted_labels.shape)}."
        )
    per_row_loss = token_losses.sum(dim=1) / supervised.sum(dim=1).clamp_min(1)
    metrics.update(_stage_loss_metrics(per_row_loss, supervised, stage_ids.to(device=per_row_loss.device)))
    row_weights = torch.where(
        stage_ids.to(device=per_row_loss.device) == 0,
        torch.as_tensor(two_pass_predict_weight, device=per_row_loss.device, dtype=per_row_loss.dtype),
        torch.as_tensor(two_pass_confirm_weight, device=per_row_loss.device, dtype=per_row_loss.dtype),
    )
    weighted_loss = (per_row_loss * row_weights).sum() / row_weights.sum()
    metrics["loss"] = weighted_loss
    return weighted_loss, metrics


def _stage_loss_metrics(
    per_row_loss: torch.Tensor,
    supervised: torch.Tensor,
    stage_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for stage_id, stage_name in ((0, "predict"), (1, "confirm")):
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
