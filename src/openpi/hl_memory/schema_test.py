from openpi.hl_memory.schema import HLMemoryPrediction


def test_prediction_round_trip():
    prediction = HLMemoryPrediction(
        updated_language_memory="Completed subtasks: pick apple.",
        current_subtask="place apple",
        keyframe_positions=(1, 3),
        phase="place",
        target_query="apple",
        goal_query="basket",
    )

    parsed = HLMemoryPrediction.from_json(prediction.to_json())

    assert parsed == prediction


def test_prediction_parses_fenced_json():
    text = """```json
{"updated_language_memory":"m","current_subtask":"s","keyframe_positions":[2],"phase":"p","target_query":"t","goal_query":"g"}
```"""

    parsed = HLMemoryPrediction.from_json(text)

    assert parsed.keyframe_positions == (2,)
