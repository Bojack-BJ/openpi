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
        if text == '"completed_objective"':
            return [30, 31]
        if text == "completed_objective":
            return [31]
        return []


def test_completed_objective_cannot_attend_recent_source_or_prompt_bridge():
    input_ids = torch.tensor([[1, 10, 11, 11, 12, 2, 10, 11, 11, 12, 3, 4, 20, 21, 30, 31, 40]])
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
    assert spans[0].completed_objective_start == 14
    assert torch.all(additive_mask[0, 0, 14:, 6:12] == blocked)
    assert additive_mask[0, 0, 12, 7] == 0
    assert additive_mask[0, 0, 14, 12] == 0
    assert additive_mask[0, 0, 12, 13] == blocked


def test_generation_mask_activates_only_after_completed_field_is_generated():
    tokenizer = _FakeTokenizer()
    prompt_ids = [1, 10, 11, 12, 2, 10, 11, 12, 3]
    before_marker = torch.tensor([prompt_ids + [20, 21]])
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

    after_marker = torch.tensor([prompt_ids + [20, 21, 30, 31, 40]])
    after_attention = torch.ones_like(after_marker)
    after_mask, after_spans = build_qwen25_typed_attention_mask(
        input_ids=after_marker,
        attention_mask=after_attention,
        target_starts=torch.tensor([len(prompt_ids)]),
        tokenizer=tokenizer,
        dtype=torch.float32,
    )

    blocked = torch.finfo(torch.float32).min
    assert after_spans[0] is not None
    assert after_mask[0, 0, -1, 5] == blocked
    assert after_mask[0, 0, -1, len(prompt_ids)] == 0


def test_padding_queries_and_keys_remain_blocked():
    input_ids = torch.tensor([[0, 0, 1, 10, 11, 12, 2, 10, 11, 12, 20, 30, 31, 40]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
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
