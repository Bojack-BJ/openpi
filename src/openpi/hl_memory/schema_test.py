from openpi.hl_memory.schema import HLMemoryPrediction


def test_prediction_round_trip():
    prediction = HLMemoryPrediction(
        updated_language_memory="Completed subtasks: pick apple.",
        current_subtask="place apple",
        keyframe_candidate_positions=(1, 3),
        phase="place",
        target_query="apple",
        goal_query="basket",
    )

    parsed = HLMemoryPrediction.from_json(prediction.to_json())

    assert parsed == prediction


def test_prediction_parses_fenced_json():
    text = """```json
{"updated_language_memory":"m","current_subtask":"s","keyframe_candidate_positions":[2],"phase":"p","target_query":"t","goal_query":"g"}
```"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.keyframe_candidate_positions == (2,)


def test_prediction_drops_invalid_generated_positions():
    parsed = HLMemoryPrediction.from_json(
        '{"updated_language_memory":"m","current_subtask":"s",'
        '"keyframe_candidate_positions":[0,"bad",2,7,2],"phase":"p","target_query":"t","goal_query":"g"}'
    )

    assert parsed.keyframe_candidate_positions == (2, 7)
    assert parsed.with_recent_position_limit(2).keyframe_candidate_positions == (2,)


def test_prediction_defaults_missing_keyframe_positions_to_empty():
    parsed = HLMemoryPrediction.from_json(
        '{"updated_language_memory":"m","current_subtask":"s","phase":"p","target_query":"t","goal_query":"g"}'
    )

    assert parsed.keyframe_candidate_positions == ()
