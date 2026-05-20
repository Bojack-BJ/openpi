from openpi.hl_memory.schema import HLMemoryPrediction


def test_prediction_round_trip():
    prediction = HLMemoryPrediction(
        updated_language_memory="",
        current_subtask="",
        keyframe_candidate_positions=(1, 3),
        phase="place",
        target_query="apple",
        goal_query="basket",
        task_progress="The apple has been picked.",
        current_objective="place apple",
        relevant_objects=("apple", "basket"),
        notes="none",
    )

    parsed = HLMemoryPrediction.from_json(prediction.to_json())

    assert parsed == prediction


def test_prediction_parses_fenced_json():
    text = """```json
{"task_progress":"m","current_objective":"s","relevant_objects":["t"],"notes":"n","keyframe_candidate_positions":[2],"phase":"p","target_query":"t","goal_query":"g"}
```"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.keyframe_candidate_positions == (2,)


def test_prediction_drops_invalid_generated_positions():
    parsed = HLMemoryPrediction.from_json(
        '{"task_progress":"m","current_objective":"s","relevant_objects":[],"notes":"n",'
        '"keyframe_candidate_positions":[0,"bad",2,7,2],"phase":"p","target_query":"t","goal_query":"g"}'
    )

    assert parsed.keyframe_candidate_positions == (2, 7)
    assert parsed.with_recent_position_limit(2).keyframe_candidate_positions == (2,)


def test_prediction_defaults_missing_keyframe_positions_to_empty():
    parsed = HLMemoryPrediction.from_json(
        '{"task_progress":"m","current_objective":"s","relevant_objects":[],"notes":"n","phase":"p","target_query":"t","goal_query":"g"}'
    )

    assert parsed.keyframe_candidate_positions == ()


def test_prediction_parses_json_after_thinking_block():
    text = """
<think>
I will inspect the clips briefly and then answer.
</think>
{"task_progress":"m","current_objective":"s","relevant_objects":[],"notes":"n","keyframe_candidate_positions":[1],"phase":"p","target_query":"t","goal_query":"g"}
"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.current_objective == "s"
    assert parsed.keyframe_candidate_positions == (1,)


def test_prediction_prefers_final_prediction_json():
    text = """
Reasoning with an irrelevant object {"debug": true}.
{"updated_language_memory":"old","current_subtask":"old","keyframe_candidate_positions":[],"phase":"old","target_query":"","goal_query":""}
Final answer:
{"task_progress":"new progress","current_objective":"new","relevant_objects":["target","goal"],"notes":"none","keyframe_candidate_positions":[2],"phase":"new","target_query":"target","goal_query":"goal"}
"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.task_progress == "new progress"
    assert parsed.current_objective == "new"
    assert parsed.keyframe_candidate_positions == (2,)


def test_prediction_parses_optional_sam_grounding_fields():
    parsed = HLMemoryPrediction.from_json(
        '{"task_progress":"m","current_objective":"s","relevant_objects":["cube"],"notes":"n","phase":"p",'
        '"target_query":"cube","goal_query":"bin","sam_text_prompt":"red cube",'
        '"sam_point_xy":{"x":12.4,"y":29.6},"target_bbox_xyxy":[1,2,30,40]}'
    )

    assert parsed.sam_text_prompt == "red cube"
    assert parsed.sam_point_xy == (12, 30)
    assert parsed.target_bbox_xyxy == (1, 2, 30, 40)
    assert HLMemoryPrediction.from_json(parsed.to_json()) == parsed


def test_prediction_repairs_unescaped_newlines_inside_json_strings():
    parsed = HLMemoryPrediction.from_json(
        '{"updated_language_memory":"Task progress: done\nCurrent objective: place apple\n'
        'Relevant objects: apple, basket\nNotes: none","current_subtask":"place apple",'
        '"keyframe_candidate_positions":[1],"phase":"place","target_query":"apple","goal_query":"basket"}'
    )

    assert parsed.current_objective == "place apple"
    assert parsed.task_progress == "done"
