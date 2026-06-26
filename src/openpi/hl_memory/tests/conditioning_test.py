from types import SimpleNamespace

import torch

from openpi.hl_memory.conditioning import configure_conditioning_model
from openpi.hl_memory.conditioning import HL_CONDITION_STAGE_IDS_KEY
from openpi.hl_memory.conditioning import HL_PROGRESS_ATTENTION_MASK_KEY
from openpi.hl_memory.conditioning import HL_PROGRESS_INPUT_IDS_KEY
from openpi.hl_memory.conditioning import HL_STATE_MASKS_KEY
from openpi.hl_memory.conditioning import HL_STATE_VALUES_KEY
from openpi.hl_memory.conditioning import STAGE_CONFIRM
from openpi.hl_memory.conditioning import STAGE_HORIZON
from openpi.hl_memory.conditioning import STAGE_PREDICT
from openpi.hl_memory.conditioning import _stage_strengths
from openpi.hl_memory.config import HLMemoryConfig


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.model = torch.nn.Module()
        self.model.norm = torch.nn.LayerNorm(8)
        self.lm_head = torch.nn.Linear(8, 32)

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


def test_conditioning_wrapper_accepts_progress_and_state_inputs():
    config = HLMemoryConfig(
        target_protocol="memer_film_progress_two_pass",
        progress_condition_enabled=True,
        state_condition_enabled=True,
        progress_condition_dim=4,
        state_condition_dim=4,
        proprio_state_dim=3,
    )
    model = configure_conditioning_model(_TinyCausalLM(), config)

    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        **{
            HL_PROGRESS_INPUT_IDS_KEY: torch.tensor([[4, 5]]),
            HL_PROGRESS_ATTENTION_MASK_KEY: torch.tensor([[1, 1]]),
            HL_STATE_VALUES_KEY: torch.zeros((1, 2, 3)),
            HL_STATE_MASKS_KEY: torch.ones((1, 2, 3)),
            HL_CONDITION_STAGE_IDS_KEY: torch.tensor([0]),
        },
    )

    assert outputs.logits.shape == (1, 3, 32)


def test_stage_strengths_use_explicit_horizon_strength():
    strengths = _stage_strengths(
        torch.tensor([STAGE_PREDICT, STAGE_HORIZON, STAGE_CONFIRM]),
        batch_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        predict_strength=0.25,
        horizon_strength=0.5,
        confirm_strength=1.0,
    )

    assert strengths.tolist() == [0.25, 0.5, 1.0]
