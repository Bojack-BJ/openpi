from __future__ import annotations

import dataclasses
from typing import Any

import torch


COMPLETED_OBJECTIVE_FIELD = "completed_objective"
CURRENT_OBJECTIVE_FIELD = "current_objective"
HORIZON_OBJECTIVE_FIELD = "horizon_current_objective"
KEYFRAME_POSITIONS_FIELD = "keyframe_candidate_positions"


@dataclasses.dataclass(frozen=True)
class TypedAttentionSpans:
    recent_source_start: int
    recent_source_end: int
    target_start: int
    current_objective_start: int
    horizon_objective_start: int
    horizon_objective_end: int
    completed_objective_start: int | None


def build_qwen25_typed_attention_mask(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    dtype: torch.dtype,
    labels: torch.Tensor | None = None,
    target_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[TypedAttentionSpans | None, ...]]:
    """Build a 4D additive causal mask for keyframe-gated Qwen2.5 decoding.

    Horizon target rows cannot attend the teacher-forced current-objective
    target field. Completed-objective rows cannot directly attend the recent
    visual source span. Source prompt context remains visible to both fields.
    """
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must both have shape [batch, sequence].")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids/attention_mask shape mismatch: {tuple(input_ids.shape)} vs {tuple(attention_mask.shape)}."
        )
    if (labels is None) == (target_starts is None):
        raise ValueError("Provide exactly one of labels or target_starts.")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError(f"labels shape mismatch: {tuple(labels.shape)} vs {tuple(input_ids.shape)}.")
    if target_starts is not None and target_starts.shape != (input_ids.shape[0],):
        raise ValueError(
            f"target_starts must have shape [{input_ids.shape[0]}], got {tuple(target_starts.shape)}."
        )
    if not dtype.is_floating_point:
        raise ValueError(f"Typed additive masks require a floating dtype, got {dtype}.")

    valid = attention_mask.to(dtype=torch.bool)
    sequence_length = input_ids.shape[1]
    positions = torch.arange(sequence_length, device=input_ids.device)
    causal = positions[None, :, None] >= positions[None, None, :]
    allow = causal & valid[:, :, None] & valid[:, None, :]

    video_token_id = _resolve_special_token_id(tokenizer, "video_token_id", "<|video_pad|>")
    vision_start_token_id = _resolve_special_token_id(tokenizer, "vision_start_token_id", "<|vision_start|>")
    vision_end_token_id = _resolve_special_token_id(tokenizer, "vision_end_token_id", "<|vision_end|>")
    current_marker_candidates = _field_marker_candidates(tokenizer, CURRENT_OBJECTIVE_FIELD)
    horizon_marker_candidates = _field_marker_candidates(tokenizer, HORIZON_OBJECTIVE_FIELD)
    keyframe_marker_candidates = _field_marker_candidates(tokenizer, KEYFRAME_POSITIONS_FIELD)
    completed_marker_candidates = _field_marker_candidates(tokenizer, COMPLETED_OBJECTIVE_FIELD)

    spans: list[TypedAttentionSpans | None] = []
    for batch_index in range(input_ids.shape[0]):
        row_ids = input_ids[batch_index]
        row_valid = valid[batch_index]
        target_start = (
            _target_start_from_labels(labels[batch_index])
            if labels is not None
            else int(target_starts[batch_index].item())
        )
        span = _locate_typed_spans(
            row_ids=row_ids,
            row_valid=row_valid,
            target_start=target_start,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            current_marker_candidates=current_marker_candidates,
            horizon_marker_candidates=horizon_marker_candidates,
            keyframe_marker_candidates=keyframe_marker_candidates,
            completed_marker_candidates=completed_marker_candidates,
        )
        spans.append(span)
        if span is None:
            continue
        allow[
            batch_index,
            span.horizon_objective_start : span.horizon_objective_end,
            span.current_objective_start : span.horizon_objective_start,
        ] = False
        if span.completed_objective_start is not None:
            allow[
                batch_index,
                span.completed_objective_start :,
                span.recent_source_start : span.recent_source_end,
            ] = False

    additive_mask = torch.zeros(
        (input_ids.shape[0], 1, sequence_length, sequence_length),
        dtype=dtype,
        device=input_ids.device,
    )
    additive_mask.masked_fill_(~allow[:, None, :, :], torch.finfo(dtype).min)
    return additive_mask, tuple(spans)


def _locate_typed_spans(
    *,
    row_ids: torch.Tensor,
    row_valid: torch.Tensor,
    target_start: int,
    video_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    current_marker_candidates: tuple[tuple[int, ...], ...],
    horizon_marker_candidates: tuple[tuple[int, ...], ...],
    keyframe_marker_candidates: tuple[tuple[int, ...], ...],
    completed_marker_candidates: tuple[tuple[int, ...], ...],
) -> TypedAttentionSpans | None:
    valid_positions = torch.nonzero(row_valid, as_tuple=False).flatten().tolist()
    if not valid_positions:
        raise ValueError("Typed attention received a fully padded sequence.")
    first_valid = int(valid_positions[0])
    last_valid_exclusive = int(valid_positions[-1]) + 1
    if not first_valid <= target_start < last_valid_exclusive:
        raise ValueError(
            f"Target start {target_start} is outside valid sequence range [{first_valid}, {last_valid_exclusive})."
        )

    video_runs = _contiguous_token_runs(row_ids, row_valid, video_token_id, stop=target_start)
    if len(video_runs) < 2:
        raise ValueError(f"Expected memory and recent video token runs before target, found {len(video_runs)}.")
    recent_video_start, recent_video_end = video_runs[1]
    recent_source_start = _nearest_token_before(
        row_ids,
        token_id=vision_start_token_id,
        start=first_valid,
        stop=recent_video_start,
    )
    if recent_source_start is None:
        recent_source_start = recent_video_start
    recent_source_end_token = _nearest_token_after(
        row_ids,
        token_id=vision_end_token_id,
        start=recent_video_end,
        stop=target_start,
    )
    recent_source_end = recent_video_end if recent_source_end_token is None else recent_source_end_token + 1

    current_start = _find_first_subsequence(
        row_ids,
        current_marker_candidates,
        start=target_start,
        stop=last_valid_exclusive,
    )
    horizon_start = _find_first_subsequence(
        row_ids,
        horizon_marker_candidates,
        start=target_start,
        stop=last_valid_exclusive,
    )
    if current_start is None or horizon_start is None:
        return None
    keyframe_start = _find_first_subsequence(
        row_ids,
        keyframe_marker_candidates,
        start=horizon_start + 1,
        stop=last_valid_exclusive,
    )
    horizon_end = keyframe_start if keyframe_start is not None else last_valid_exclusive
    completed_start = _find_first_subsequence(
        row_ids,
        completed_marker_candidates,
        start=horizon_start + 1,
        stop=last_valid_exclusive,
    )
    return TypedAttentionSpans(
        recent_source_start=recent_source_start,
        recent_source_end=recent_source_end,
        target_start=target_start,
        current_objective_start=current_start,
        horizon_objective_start=horizon_start,
        horizon_objective_end=horizon_end,
        completed_objective_start=completed_start,
    )


def _target_start_from_labels(labels: torch.Tensor) -> int:
    positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("Typed attention requires at least one supervised target token.")
    return int(positions[0].item())


def _resolve_special_token_id(tokenizer: Any, attribute: str, token: str) -> int:
    value = getattr(tokenizer, attribute, None)
    if value is None:
        converter = getattr(tokenizer, "convert_tokens_to_ids", None)
        if converter is None:
            raise ValueError(f"Tokenizer does not expose {attribute} or convert_tokens_to_ids().")
        value = converter(token)
    value = int(value)
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    if value < 0 or (unknown_token_id is not None and value == int(unknown_token_id)):
        raise ValueError(f"Could not resolve tokenizer special token {token!r}.")
    return value


def _field_marker_candidates(tokenizer: Any, field_name: str) -> tuple[tuple[int, ...], ...]:
    encoder = getattr(tokenizer, "encode", None)
    if encoder is None:
        raise ValueError("Tokenizer must expose encode() for typed target span detection.")
    candidates: list[tuple[int, ...]] = []
    for text in (f'"{field_name}"', field_name):
        token_ids = tuple(int(value) for value in encoder(text, add_special_tokens=False))
        if token_ids and token_ids not in candidates:
            candidates.append(token_ids)
    if not candidates:
        raise ValueError(f"Tokenizer produced no marker tokens for field {field_name!r}.")
    return tuple(candidates)


def _contiguous_token_runs(
    row_ids: torch.Tensor,
    row_valid: torch.Tensor,
    token_id: int,
    *,
    stop: int,
) -> list[tuple[int, int]]:
    positions = torch.nonzero(
        row_valid[:stop] & (row_ids[:stop] == token_id),
        as_tuple=False,
    ).flatten().tolist()
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    run_start = previous = int(positions[0])
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position != previous + 1:
            runs.append((run_start, previous + 1))
            run_start = position
        previous = position
    runs.append((run_start, previous + 1))
    return runs


def _nearest_token_before(
    row_ids: torch.Tensor,
    *,
    token_id: int,
    start: int,
    stop: int,
) -> int | None:
    positions = torch.nonzero(row_ids[start:stop] == token_id, as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return start + int(positions[-1].item())


def _nearest_token_after(
    row_ids: torch.Tensor,
    *,
    token_id: int,
    start: int,
    stop: int,
) -> int | None:
    positions = torch.nonzero(row_ids[start:stop] == token_id, as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    return start + int(positions[0].item())


def _find_first_subsequence(
    row_ids: torch.Tensor,
    candidates: tuple[tuple[int, ...], ...],
    *,
    start: int,
    stop: int,
) -> int | None:
    values = row_ids[start:stop].tolist()
    matches: list[int] = []
    for candidate in candidates:
        max_start = len(values) - len(candidate)
        for offset in range(max_start + 1):
            if tuple(values[offset : offset + len(candidate)]) == candidate:
                matches.append(start + offset)
                break
    return min(matches) if matches else None
