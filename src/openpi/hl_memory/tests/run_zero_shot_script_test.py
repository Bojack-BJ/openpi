import json

from scripts.hl_memory.run_hl_memory_zero_shot import _apply_known_prior_rollout_state
from scripts.hl_memory.run_hl_memory_zero_shot import _write_json_atomic
from openpi.hl_memory.schema import HLMemoryPrediction


def _prediction(*, objective: str, progress: float = 0.0, should_advance: bool = False) -> HLMemoryPrediction:
    return HLMemoryPrediction(
        updated_language_memory="Task progress: none",
        current_objective=objective,
        current_subtask=objective,
        keyframe_candidate_positions=(),
        phase=objective,
        subtask_progress=progress,
        should_advance_objective=should_advance,
    )


def test_known_prior_next_step_can_require_completion_evidence():
    steps = ("approach cabinet", "grasp cabinet handle", "open cabinet")
    prediction, next_index, match = _apply_known_prior_rollout_state(
        _prediction(objective="grasp cabinet handle"),
        known_prior_steps=steps,
        current_index=0,
        advance_threshold=0.7,
        match_threshold=0.62,
        max_advance_steps=3,
        next_step_require_completion=True,
        safe_skip_mode=True,
    )

    assert next_index == 0
    assert prediction.current_objective == steps[0]
    assert match["advance_reason"] == "matched_next_prior_step_waiting_for_completion"


def test_known_prior_next_step_advances_with_completion_evidence():
    steps = ("approach cabinet", "grasp cabinet handle", "open cabinet")
    prediction, next_index, match = _apply_known_prior_rollout_state(
        _prediction(objective="grasp cabinet handle", progress=0.7),
        known_prior_steps=steps,
        current_index=0,
        advance_threshold=0.7,
        match_threshold=0.62,
        max_advance_steps=3,
        next_step_require_completion=True,
        safe_skip_mode=True,
    )

    assert next_index == 1
    assert prediction.current_objective == steps[1]
    assert match["advance_reason"] == "matched_next_prior_step"


def test_known_prior_safe_skip_does_not_clamp_forward_without_completion():
    steps = ("approach cabinet", "grasp cabinet handle", "open cabinet")
    prediction, next_index, match = _apply_known_prior_rollout_state(
        _prediction(objective="open cabinet"),
        known_prior_steps=steps,
        current_index=0,
        advance_threshold=0.7,
        match_threshold=0.62,
        max_advance_steps=3,
        next_step_require_completion=True,
        safe_skip_mode=True,
        skip_match_threshold=0.95,
        skip_min_progress=0.8,
        skip_min_stall_steps=6,
    )

    assert next_index == 0
    assert prediction.current_objective == steps[0]
    assert match["advance_reason"] == "safe_skip_waiting_for_completion"


def test_write_json_atomic_replaces_live_summary(tmp_path):
    path = tmp_path / "summary.json"
    _write_json_atomic(path, {"steps": [{"step_index": 0}]})
    _write_json_atomic(path, {"steps": [{"step_index": 0}, {"step_index": 1}]})

    assert json.loads(path.read_text()) == {"steps": [{"step_index": 0}, {"step_index": 1}]}
    assert not (tmp_path / ".summary.json.tmp").exists()
