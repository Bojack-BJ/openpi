from types import SimpleNamespace

import torch

from openpi.hl_memory.keyframe_auxiliary import configure_keyframe_auxiliary_model


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 16)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, logits_to_keep=None):
        hidden = self.model.norm(self.embed_tokens(input_ids))
        logits = self.lm_head(hidden)
        if logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=logits)


def test_auxiliary_wrapper_reads_prompt_anchor_without_hidden_state_stack():
    model = configure_keyframe_auxiliary_model(
        _TinyCausalLM(),
        hidden_dim=4,
        num_positions=3,
    )

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        hl_keyframe_aux_anchor_positions=torch.tensor([2]),
    )

    assert not hasattr(outputs, "hidden_states")
    assert outputs.hl_keyframe_aux_position_logits.shape == (1, 3)
    assert outputs.hl_keyframe_aux_event_logits.shape == (1,)
    assert outputs.hl_keyframe_aux_update_logits.shape == (1, 3)
