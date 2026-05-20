# HL Memory GT Protocol

HL memory separates the structured HL prediction from the deterministic four-line language memory consumed by the low-level policy. New data should treat `current_objective` as the executable LL objective and treat `task_progress` as historical state.

## Structured Prediction

The model is trained to output one JSON object. The required new-schema fields are:

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

Field semantics:

- `task_progress`: compressed history before or at this sample. It should mention completed milestones or stable state, for example `The calculator has been picked up.` It must not leak the full future task flow.
- `current_objective`: one short executable objective for the downstream low-level policy, for example `move the calculator into the cabinet with the right hand`.
- `relevant_objects`: object/location names that matter for this objective. Keep this short.
- `notes`: optional action-useful constraints, hand assignment, or spatial cues. Use `none` when there is no useful note.
- `keyframe_candidate_positions`: 1-indexed positions inside the recent clip that should be added to historical memory.
- `phase`, `target_query`, `goal_query`: compatibility and retrieval/debug fields. They should stay aligned with `current_objective`.

`current_subtask` remains a legacy alias when reading old samples or old model outputs, but new GT should use `current_objective`. Passive state such as `calculator is picked up` belongs in `task_progress`, not `current_objective`.

## Rendered LL Memory

The downstream low-level policy receives a deterministic rendering:

```text
Task progress: The calculator has been picked up.
Current objective: move the calculator into the cabinet with the right hand
Relevant objects: calculator, cabinet
Notes: use the right hand
```

The renderer is implemented in `src/openpi/hl_memory/schema.py` as `render_language_memory_fields(...)`. Exported samples still carry legacy `updated_language_memory` / `current_subtask` fields for compatibility, but those fields are derived from the structured fields in new data.

## Dataset Flow

Recommended flow:

```text
subtask_segments.json
  -> scripts/hl_memory/export_hl_annotations_from_subtasks.py
  -> annotations.jsonl
  -> optional scripts/hl_memory/normalize_hl_annotations_with_llm.py
  -> annotations_llm_normalized.jsonl
  -> scripts/hl_memory/export_hl_memory_dataset.py
  -> samples.jsonl
```

`scripts/hl_memory/export_hl_memory_dataset.py` reads `task_progress`, `current_objective`, `relevant_objects`, and `notes` from the annotation row when present. If those fields are absent, it falls back to deterministic rule-based rendering from `current_subtask`, `phase`, `target_query`, `goal_query`, and event metadata.

## Offline LLM GT Normalization

Use a large text LLM to normalize rule-based annotations before exporting training samples. This is useful when rule-based `current_subtask` strings are too state-like or when historical progress needs cleaner cumulative wording.

```bash
PYTHONPATH=src python scripts/hl_memory/normalize_hl_annotations_with_llm.py \
  --input-jsonl /path/to/annotations.jsonl \
  --output-jsonl /path/to/annotations_llm_normalized.jsonl \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto
```

Then pass the normalized JSONL to the HL memory dataset export path in place of the rule-based annotations. The normalizer is offline-only: training and rollout should consume the cached JSONL, not call the LLM dynamically.

```bash
PYTHONPATH=src python scripts/hl_memory/export_hl_memory_dataset.py \
  --source-config-name sponge_visual_guided_qwen3_5_2b_400m_touch \
  --annotations-jsonl /path/to/annotations_llm_normalized.jsonl \
  --output-dir /path/to/hl_memory/exported_train \
  --visual-mode raw \
  --overwrite
```

Normalizer output is intentionally cacheable. Use `--resume` to skip rows already written to the output JSONL. The prompt asks the LLM to output only:

```json
{
  "task_progress": "...",
  "current_objective": "...",
  "relevant_objects": ["..."],
  "notes": "...",
  "target_query": "...",
  "goal_query": "..."
}
```

Do not use an online LLM during train/eval/rollout. The only runtime VLM should be the trained HL model.

## Debug BBox

At inference time the VLM may also output:

```json
"target_bbox_xyxy": [x1, y1, x2, y2]
```

This field is optional and intended for debug visualization only. It is not required in GT and is not supervised by the exported dataset. `scripts/hl_memory/run_hl_memory_zero_shot.py --debug-dir ...` draws this box on the current frame when the VLM emits it.

## Prompt/Data Alignment Rules

- The train target JSON, eval parser, and rollout parser all use `HLMemoryPrediction` from `src/openpi/hl_memory/schema.py`.
- The prompt asks for `task_progress`, `current_objective`, `relevant_objects`, `notes`, `keyframe_candidate_positions`, `phase`, `target_query`, `goal_query`, optional SAM fields, and optional `target_bbox_xyxy`.
- `updated_language_memory` is a rendered compatibility field, not the primary training contract.
- `current_subtask` is accepted for old checkpoints/data, but new comparisons should prefer `current_objective`.
- If parsing fails at rollout, fallback does not reuse the previous objective as truth; it emits `continue the observed manipulation step` and keeps only safe historical progress.
