# HL Memory GT Protocol

HL memory now separates the structured HL prediction from the four-line language memory consumed by the low-level policy.

## Structured Prediction

The model is trained to output one JSON object:

```json
{
  "task_progress": "The calculator has been picked up.",
  "current_objective": "move the calculator into the cabinet with the right hand",
  "relevant_objects": ["calculator", "cabinet"],
  "notes": "use the right hand",
  "keyframe_candidate_positions": [1, 8],
  "phase": "move calculator to cabinet",
  "target_query": "calculator",
  "goal_query": "cabinet"
}
```

`current_subtask` remains a legacy alias when reading old samples or old model outputs, but new GT should use `current_objective`.

## Rendered LL Memory

The downstream low-level policy receives a deterministic rendering:

```text
Task progress: The calculator has been picked up.
Current objective: move the calculator into the cabinet with the right hand
Relevant objects: calculator, cabinet
Notes: use the right hand
```

`task_progress` is historical state. It should accumulate completed milestones in compressed form. The current action belongs in `current_objective`, not `task_progress`.

## Offline LLM GT Normalization

Use a large text LLM to normalize rule-based annotations before exporting training samples:

```bash
PYTHONPATH=src python scripts/hl_memory/normalize_hl_annotations_with_llm.py \
  --input-jsonl /path/to/annotations.jsonl \
  --output-jsonl /path/to/annotations_llm_normalized.jsonl \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto
```

Then pass the normalized JSONL to the HL memory dataset export path in place of the rule-based annotations. The normalizer is offline-only: training and rollout should consume the cached JSONL, not call the LLM dynamically.

## Debug BBox

At inference time the VLM may also output:

```json
"target_bbox_xyxy": [x1, y1, x2, y2]
```

This field is optional and intended for debug visualization only. It is not required in GT.
