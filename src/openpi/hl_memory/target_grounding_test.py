import pathlib

import numpy as np
from PIL import Image

from openpi.hl_memory.target_grounding import MaskSelectionPrediction
from openpi.hl_memory.target_grounding import TargetPointPrediction
from openpi.hl_memory.target_grounding import load_candidate_masks
from openpi.hl_memory.target_grounding import mask_to_bbox
from openpi.hl_memory.target_grounding import overlay_mask_candidates
from openpi.hl_memory.target_grounding import overlay_point


def test_target_point_prediction_parses_common_shapes():
    parsed = TargetPointPrediction.from_json(
        'prefix {"target_query":"red block","target_point":[12.2,34.8],"confidence":"0.7"}',
        image_size=(20, 30),
    )

    assert parsed.target_query == "red block"
    assert parsed.point_xy == (12, 29)
    assert parsed.confidence == 0.7


def test_target_point_prediction_parses_dict_point_after_thinking():
    parsed = TargetPointPrediction.from_json(
        """
<think>brief</think>
```json
{"target":"cube","point_xy":{"x":5,"y":7}}
```
"""
    )

    assert parsed.target_query == "cube"
    assert parsed.point_xy == (5, 7)


def test_mask_selection_prediction_accepts_aliases_and_checks_range():
    parsed = MaskSelectionPrediction.from_json('{"target_mask_id":"2","target_query":"cube"}', max_mask_id=3)

    assert parsed.selected_mask_id == 2
    assert parsed.target_query == "cube"


def test_load_candidate_masks_and_overlay(tmp_path: pathlib.Path):
    mask_a = np.zeros((24, 32), dtype=np.uint8)
    mask_a[3:10, 4:12] = 255
    mask_b = np.zeros((24, 32), dtype=np.uint8)
    mask_b[12:20, 18:30] = 255
    Image.fromarray(mask_a).save(tmp_path / "a.png")
    Image.fromarray(mask_b).save(tmp_path / "b.png")

    candidates = load_candidate_masks(tmp_path)
    image = Image.new("RGB", (32, 24), color=(20, 20, 20))
    overlay = overlay_mask_candidates(image, candidates, selected_mask_id=2)

    assert len(candidates) == 2
    assert candidates[0].bbox_xyxy == (4, 3, 12, 10)
    assert candidates[1].bbox_xyxy == (18, 12, 30, 20)
    assert overlay.size == image.size


def test_overlay_point_keeps_image_size():
    image = Image.new("RGB", (32, 24), color=(20, 20, 20))

    overlay = overlay_point(image, (999, -5), label="target")

    assert overlay.size == image.size


def test_mask_to_bbox_returns_empty_box_for_empty_mask():
    assert mask_to_bbox(np.zeros((4, 5), dtype=bool)) == (0, 0, 0, 0)
