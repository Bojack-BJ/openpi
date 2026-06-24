from types import SimpleNamespace

import pytest
import torch

from openpi.hl_memory.training_loss import compute_hl_target_loss
from openpi.hl_memory.training_loss import compute_hl_target_loss_with_metrics
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_ANCHOR_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_CANONICAL_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_EVENT_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_MASK_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_POSITION_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_UPDATE_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_VALID_POSITIONS_KEY


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids, logits_to_keep=None):
        del input_ids
        if logits_to_keep is not None:
            raise TypeError("logits_to_keep unsupported")
        return SimpleNamespace(logits=self._logits)


class _FixedAuxiliaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 4, 2))
        self.position_logits = torch.nn.Parameter(torch.zeros(1, 4))
        self.event_logits = torch.nn.Parameter(torch.zeros(1))
        self.update_logits = torch.nn.Parameter(torch.zeros(1, 3))

    def forward(self, input_ids, hl_keyframe_aux_anchor_positions=None, logits_to_keep=None):
        del input_ids, hl_keyframe_aux_anchor_positions
        if logits_to_keep is not None:
            raise TypeError("logits_to_keep unsupported")
        return SimpleNamespace(
            logits=self.logits,
            hl_keyframe_aux_position_logits=self.position_logits,
            hl_keyframe_aux_event_logits=self.event_logits,
            hl_keyframe_aux_update_logits=self.update_logits,
        )


def test_two_pass_loss_averages_each_stage_before_weighting():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor(
        [
            [-100, 0, 0, 0],
            [-100, -100, -100, 1],
        ]
    )
    inputs = {
        "input_ids": torch.zeros_like(labels),
        "labels": labels,
        "_hl_stage_ids": torch.tensor([0, 1]),
    }
    expected_predict = torch.nn.functional.cross_entropy(logits[0, :3], torch.tensor([0, 0, 0]))
    expected_confirm = torch.nn.functional.cross_entropy(logits[1, 2:3], torch.tensor([1]))

    loss = compute_hl_target_loss(
        _FixedLogitModel(logits),
        inputs,
        two_pass_predict_weight=1.0,
        two_pass_confirm_weight=2.0,
    )

    assert loss.item() == pytest.approx(((expected_predict + 2 * expected_confirm) / 3).item())


def test_keyframe_auxiliary_losses_are_added_to_language_loss():
    labels = torch.tensor([[-100, 0, 0, 0]])
    inputs = {
        "input_ids": torch.zeros_like(labels),
        "labels": labels,
        KEYFRAME_AUX_ANCHOR_POSITIONS_KEY: torch.tensor([0]),
        KEYFRAME_AUX_MASK_KEY: torch.tensor([True]),
        KEYFRAME_AUX_POSITION_TARGETS_KEY: torch.tensor([[0.0, 0.0, 0.25, 0.75]]),
        KEYFRAME_AUX_EVENT_TARGETS_KEY: torch.tensor([1.0]),
        KEYFRAME_AUX_UPDATE_TARGETS_KEY: torch.tensor([1]),
        KEYFRAME_AUX_CANONICAL_POSITIONS_KEY: torch.tensor([3.0]),
        KEYFRAME_AUX_VALID_POSITIONS_KEY: torch.tensor([[True, True, True, True]]),
    }

    loss, metrics = compute_hl_target_loss_with_metrics(
        _FixedAuxiliaryModel(),
        inputs,
        keyframe_aux_position_weight=1.0,
        keyframe_aux_event_weight=1.0,
        keyframe_aux_timing_weight=1.0,
        keyframe_aux_update_weight=1.0,
    )

    assert loss.item() > metrics["loss_language"].item()
    assert metrics["loss_aux_position"].item() > 0.0
    assert metrics["loss_aux_event"].item() > 0.0
    assert metrics["loss_aux_timing"].item() > 0.0
    assert metrics["loss_aux_update"].item() > 0.0
