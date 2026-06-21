from scripts.hl_memory.coarsen_hl_annotations import _relabel_coarse_segment_end_keyframes


def test_relabel_coarse_segment_end_keyframes_clears_fine_labels():
    rows = [
        {"frame_index": 0, "current_objective": "grasp", "keyframe_label": True},
        {"frame_index": 5, "current_objective": "grasp", "keyframe_label": True},
        {"frame_index": 9, "current_objective": "grasp"},
        {"frame_index": 10, "current_objective": "place", "keyframe_label": True},
        {"frame_index": 15, "current_objective": "place"},
    ]

    _relabel_coarse_segment_end_keyframes(rows)

    assert [row["keyframe_label"] for row in rows] == [False, False, True, False, True]
    assert rows[0]["fine_keyframe_label"] is True
    assert rows[1]["fine_keyframe_label"] is True
    assert rows[3]["fine_keyframe_label"] is True
    assert rows[2]["coarse_keyframe_label_source"] == "coarse_segment_end"
    assert rows[4]["coarse_keyframe_label_source"] == "coarse_segment_end"
