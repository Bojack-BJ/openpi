import json

from scripts.hl_memory.run_hl_memory_zero_shot import _apply_known_prior_rollout_state
from scripts.hl_memory.run_hl_memory_zero_shot import _parse_completed_event_log
from scripts.hl_memory.run_hl_memory_zero_shot import _protocol_language_memory_input
from scripts.hl_memory.run_hl_memory_zero_shot import _protocol_input_payload
from scripts.hl_memory.run_hl_memory_zero_shot import _protocol_prediction_payload
from scripts.hl_memory.run_hl_memory_zero_shot import _render_completed_event_log
from scripts.hl_memory.run_hl_memory_zero_shot import _update_keyframe_gated_rollout_state
from scripts.hl_memory.run_hl_memory_zero_shot import _write_json_atomic
from scripts.hl_memory.run_hl_memory_zero_shot import KeyframeGatedRolloutState
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.schema import HLMemoryPrediction


def _prediction(*, objective: str, progress: float = 0.0, should_advance: bool = False) -> HLMemoryPrediction:
    return HLMemoryPrediction(
        updated_language_memory="Task progress: none",
        current_objective=objective,
        current_subtask=objective,
        keyframe_candidate_positions=(),
        phase=objective,
        target_query="",
        goal_query="",
        subtask_progress=progress,
        should_advance_objective=should_advance,
    )


def _gated_prediction(
    *,
    objective: str = "place toast",
    completed_objective: str = "",
    keyframe_positions: tuple[int, ...] = (),
) -> HLMemoryPrediction:
    return HLMemoryPrediction(
        updated_language_memory="",
        current_objective=objective,
        current_subtask=objective,
        horizon_current_objective=objective,
        keyframe_candidate_positions=keyframe_positions,
        phase=objective,
        target_query="",
        goal_query="",
        completed_objective=completed_objective,
    )


def test_keyframe_gated_state_rejects_empty_completed_objective():
    state, update = _update_keyframe_gated_rollout_state(
        KeyframeGatedRolloutState(),
        prediction=_gated_prediction(keyframe_positions=(1,)),
        candidate_seconds=(1.0,),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert state.completed_events == ()
    assert update["accepted"] is False
    assert update["reason"] == "empty_completed_objective"


def test_keyframe_gated_state_rejects_missing_keyframes():
    state, update = _update_keyframe_gated_rollout_state(
        KeyframeGatedRolloutState(),
        prediction=_gated_prediction(completed_objective="place toast"),
        candidate_seconds=(),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert state.completed_events == ()
    assert update["accepted"] is False
    assert update["reason"] == "no_keyframe_candidates"


def test_keyframe_gated_state_accepts_completed_objective_with_keyframes():
    state, update = _update_keyframe_gated_rollout_state(
        KeyframeGatedRolloutState(),
        prediction=_gated_prediction(completed_objective="place toast", keyframe_positions=(1,)),
        candidate_seconds=(1.0,),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert state.completed_events == ("place toast",)
    assert state.accepted_keyframe_seconds == (1.0,)
    assert update["accepted"] is True


def test_keyframe_gated_state_rejects_near_duplicate_completed_objective():
    state = KeyframeGatedRolloutState(completed_events=("place toast",), accepted_keyframe_seconds=(1.0,))

    next_state, update = _update_keyframe_gated_rollout_state(
        state,
        prediction=_gated_prediction(completed_objective="place toast", keyframe_positions=(1,)),
        candidate_seconds=(1.5,),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert next_state == state
    assert update["accepted"] is False
    assert update["reason"] == "duplicate_completed_objective_near_existing_keyframe"


def test_keyframe_gated_state_accepts_far_duplicate_completed_objective():
    state = KeyframeGatedRolloutState(completed_events=("place toast",), accepted_keyframe_seconds=(1.0,))

    next_state, update = _update_keyframe_gated_rollout_state(
        state,
        prediction=_gated_prediction(completed_objective="place toast", keyframe_positions=(1,)),
        candidate_seconds=(4.0,),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert next_state.completed_events == ("place toast", "place toast")
    assert next_state.accepted_keyframe_seconds == (1.0, 4.0)
    assert update["accepted"] is True


def test_keyframe_gated_state_compacts_event_band_candidates_to_one_second():
    next_state, update = _update_keyframe_gated_rollout_state(
        KeyframeGatedRolloutState(),
        prediction=_gated_prediction(completed_objective="place toast", keyframe_positions=(1, 2, 3)),
        candidate_seconds=(1.0, 1.25, 1.5),
        memory_length=8,
        merge_distance_sec=2.0,
    )

    assert update["accepted"] is True
    assert update["representative_keyframe_second"] == 1.5
    assert next_state.accepted_keyframe_seconds == (1.5,)


def test_completed_event_log_preserves_repeated_events():
    memory = "Completed events: place toast; place toast; grasp lettuce."

    assert _parse_completed_event_log(memory) == ("place toast", "place toast", "grasp lettuce")


def test_completed_event_log_render_compacts_to_recent_events():
    events = tuple(f"event {idx}" for idx in range(10))

    rendered = _render_completed_event_log(events, max_events=3)

    assert rendered == "Recent completed events (last 3 of 10): event 7; event 8; event 9."


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


def test_known_prior_next_step_advances_after_consecutive_confirmations():
    steps = ("approach cabinet", "grasp cabinet handle", "open cabinet")
    prediction, next_index, match = _apply_known_prior_rollout_state(
        _prediction(objective="grasp cabinet handle"),
        known_prior_steps=steps,
        current_index=0,
        advance_threshold=0.7,
        match_threshold=0.62,
        max_advance_steps=3,
        next_step_require_completion=True,
        next_step_confirm_steps=2,
        safe_skip_mode=True,
    )

    assert next_index == 0
    assert match["next_step_confirmation_index"] == 1
    assert match["next_step_confirmation_steps"] == 1
    assert match["advance_reason"] == "matched_next_prior_step_waiting_for_completion"

    prediction, next_index, match = _apply_known_prior_rollout_state(
        _prediction(objective="grasp cabinet handle"),
        known_prior_steps=steps,
        current_index=0,
        advance_threshold=0.7,
        match_threshold=0.62,
        max_advance_steps=3,
        next_step_require_completion=True,
        next_step_confirm_steps=2,
        next_step_confirmation_index=int(match["next_step_confirmation_index"]),
        next_step_confirmation_steps=int(match["next_step_confirmation_steps"]),
        safe_skip_mode=True,
    )

    assert next_index == 1
    assert prediction.current_objective == steps[1]
    assert match["next_step_confirmed"]
    assert match["advance_reason"] == "matched_next_prior_step_confirmed"


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


def test_prev_stage_rollout_payload_contains_only_protocol_state():
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="insert the stick",
        language_memory="must not be exposed",
        updated_language_memory="must not be exposed",
        current_subtask="insert top-left",
        phase="insert top-left",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=(),
        recent_frame_indices=(),
        recent_valid_length=0,
        previous_stage_objective="grasp the stick",
    )

    payload = _protocol_input_payload(
        "objective_prev_stage",
        sample=sample,
        memory_seconds=(1.0,),
        recent_seconds=(2.0, 3.0),
        proprio_enabled=False,
    )

    assert payload["previous_stage_objective"] == "grasp the stick"
    assert "completed_subtasks_memory" not in payload
    assert "language_memory" not in payload


def test_keyframe_gated_rollout_payload_contains_event_log_only():
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="make sandwich",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="must not be exposed",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=(),
        recent_frame_indices=(),
        recent_valid_length=0,
    )

    payload = _protocol_input_payload(
        "keyframe_gated_memory",
        sample=sample,
        memory_seconds=(1.0,),
        recent_seconds=(2.0, 3.0),
        proprio_enabled=False,
    )

    assert payload["completed_event_log"] == "Completed events: grasp toast."
    assert "completed_subtasks_memory" not in payload
    assert "previous_stage_objective" not in payload


def test_keyframe_gated_memory_input_keeps_completed_event_log_verbatim():
    memory = "Completed events: grasp toast; place toast."

    assert _protocol_language_memory_input("keyframe_gated_memory", memory, "completed_only") == memory
    assert _protocol_language_memory_input("keyframe_gated_memory_two_pass", memory, "completed_only") == memory


def test_state_protocol_prediction_payload_is_protocol_specific():
    prediction = HLMemoryPrediction(
        updated_language_memory="completed memory",
        current_subtask="insert top-left",
        current_objective="insert top-left",
        horizon_current_objective="insert top-right",
        last_objective="grasp stick",
        previous_stage_objective="approach hole",
        keyframe_candidate_positions=(2,),
        phase="insert top-left",
        target_query="",
        goal_query="",
    )

    previous_stage = _protocol_prediction_payload("objective_prev_stage", prediction)
    last_objective = _protocol_prediction_payload("objective_last_objective", prediction)

    assert set(previous_stage) == {
        "current_objective",
        "horizon_current_objective",
        "keyframe_candidate_positions",
        "previous_stage_objective",
    }
    assert "updated_language_memory" not in previous_stage
    assert "last_objective" not in previous_stage
    assert set(last_objective) == {
        "current_objective",
        "horizon_current_objective",
        "keyframe_candidate_positions",
        "last_objective",
    }


def test_keyframe_gated_prediction_payload_is_protocol_specific():
    prediction = _gated_prediction(
        objective="place toast",
        completed_objective="place toast",
        keyframe_positions=(2,),
    )

    payload = _protocol_prediction_payload("keyframe_gated_memory", prediction)
    two_pass_payload = _protocol_prediction_payload("keyframe_gated_memory_two_pass", prediction)

    assert payload == {
        "current_objective": "place toast",
        "keyframe_candidate_positions": [2],
        "horizon_current_objective": "place toast",
        "completed_objective": "place toast",
    }
    assert two_pass_payload == payload
