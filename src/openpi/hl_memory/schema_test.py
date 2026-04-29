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


def test_prediction_parses_json_after_thinking_block():
    text = """
<think>
I will inspect the clips briefly and then answer.
</think>
{"updated_language_memory":"m","current_subtask":"s","keyframe_candidate_positions":[1],"phase":"p","target_query":"t","goal_query":"g"}
"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.current_subtask == "s"
    assert parsed.keyframe_candidate_positions == (1,)


def test_prediction_prefers_final_prediction_json():
    text = """
Reasoning with an irrelevant object {"debug": true}.
{"updated_language_memory":"old","current_subtask":"old","keyframe_candidate_positions":[],"phase":"old","target_query":"","goal_query":""}
Final answer:
{"updated_language_memory":"new","current_subtask":"new","keyframe_candidate_positions":[2],"phase":"new","target_query":"target","goal_query":"goal"}
"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.updated_language_memory == "new"
    assert parsed.current_subtask == "new"
    assert parsed.keyframe_candidate_positions == (2,)
