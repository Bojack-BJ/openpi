import torch

from openpi.hl_memory.typed_attention import build_qwen25_typed_attention_mask


class _FakeTokenizer:
    video_token_id = 11
    vision_start_token_id = 10
    vision_end_token_id = 12
    unk_token_id = 999

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<|video_pad|>": self.video_token_id,
            "<|vision_start|>": self.vision_start_token_id,
            "<|vision_end|>": self.vision_end_token_id,
        }.get(token, self.unk_token_id)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text in {'"current_objective"', "current_objective"}:
            return [20]
        if text in {'"horizon_current_objective"', "horizon_current_objective"}:
            return [21]
        if text in {'"keyframe_candidate_positions"', "keyframe_candidate_positions"}:
            return [25]
        if text == '"completed_objective"':
            return [30, 31]
        if text == "completed_objective":
            return [31]
        return []


def test_completed_objective_cannot_attend_recent_visual_but_can_attend_prompt_bridge():
    input_ids = torch.tensor(
        [[1, 10, 11, 11, 12, 2, 10, 11, 11, 12, 3, 4, 20, 50, 21, 51, 25, 52, 30, 31, 40]]
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, 12:] = input_ids[:, 12:]

    additive_mask, spans = build_qwen25_typed_attention_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        tokenizer=_FakeTokenizer(),
        dtype=torch.float32,
    )

    blocked = torch.finfo(torch.float32).min
    assert spans[0] is not None
    assert spans[0].recent_source_start == 6
    assert spans[0].recent_source_end == 10
    assert spans[0].target_start == 12
    assert spans[0].current_objective_start == 12
    assert spans[0].horizon_objective_start == 14
    assert spans[0].horizon_objective_end == 16
    assert spans[0].completed_objective_start == 18
    assert torch.all(additive_mask[0, 0, 14:16, 12:14] == blocked)
    assert torch.all(additive_mask[0, 0, 18:, 6:10] == blocked)
    assert torch.all(additive_mask[0, 0, 18:, 10:12] == 0)
    assert additive_mask[0, 0, 12, 7] == 0
    assert additive_mask[0, 0, 14, 12] == blocked
    assert additive_mask[0, 0, 14, 7] == 0
    assert additive_mask[0, 0, 12, 13] == blocked


def test_generation_mask_activates_horizon_before_completed_field_is_generated():
    tokenizer = _FakeTokenizer()
    prompt_ids = [1, 10, 11, 12, 2, 10, 11, 12, 3]
    before_marker = torch.tensor([prompt_ids + [20, 50]])
    before_attention = torch.ones_like(before_marker)

    before_mask, before_spans = build_qwen25_typed_attention_mask(
        input_ids=before_marker,
        attention_mask=before_attention,
        target_starts=torch.tensor([len(prompt_ids)]),
        tokenizer=tokenizer,
        dtype=torch.float32,
    )

    assert before_spans == (None,)
    assert before_mask[0, 0, -1, 6] == 0

    after_marker = torch.tensor([prompt_ids + [20, 50, 21, 51]])
    after_attention = torch.ones_like(after_marker)
    after_mask, after_spans = build_qwen25_typed_attention_mask(
        input_ids=after_marker,
        attention_mask=after_attention,
        target_starts=torch.tensor([len(prompt_ids)]),
        tokenizer=tokenizer,
        dtype=torch.float32,
    )

    assert after_spans[0] is not None
    blocked = torch.finfo(torch.float32).min
    assert after_spans[0].completed_objective_start is None
    assert after_mask[0, 0, -1, len(prompt_ids)] == blocked
    assert after_mask[0, 0, -1, 6] == 0

    after_completed = torch.tensor([prompt_ids + [20, 50, 21, 51, 25, 52, 30, 31, 40]])
    completed_attention = torch.ones_like(after_completed)
    completed_mask, completed_spans = build_qwen25_typed_attention_mask(
        input_ids=after_completed,
        attention_mask=completed_attention,
        target_starts=torch.tensor([len(prompt_ids)]),
        tokenizer=tokenizer,
        dtype=torch.float32,
    )

    assert completed_spans[0] is not None
    assert completed_mask[0, 0, -1, 5] == blocked
    assert completed_mask[0, 0, -1, len(prompt_ids)] == 0


def test_padding_queries_and_keys_remain_blocked():
    input_ids = torch.tensor([[0, 0, 1, 10, 11, 12, 2, 10, 11, 12, 20, 50, 21, 51, 25, 30, 31, 40]])
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, :2] = 0
    labels = torch.full_like(input_ids, -100)
    labels[:, 10:] = input_ids[:, 10:]

    additive_mask, _ = build_qwen25_typed_attention_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        tokenizer=_FakeTokenizer(),
        dtype=torch.bfloat16,
    )

    blocked = torch.finfo(torch.bfloat16).min
    assert torch.all(additive_mask[0, 0, :, :2] == blocked)
    assert torch.all(additive_mask[0, 0, :2, :] == blocked)
