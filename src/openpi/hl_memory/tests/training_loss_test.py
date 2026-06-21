from types import SimpleNamespace

import pytest
import torch

from openpi.hl_memory.training_loss import compute_hl_target_loss


class _FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, input_ids, logits_to_keep=None):
        del input_ids
        if logits_to_keep is not None:
            raise TypeError("logits_to_keep unsupported")
        return SimpleNamespace(logits=self._logits)


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
