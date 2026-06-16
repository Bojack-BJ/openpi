# Coarse Subtask Protocol

The coarse-subtask pass reduces over-fragmented fine labels while preserving object grounding.

## Why Lookahead Exists

Fine labels often split one useful manipulation into visually ambiguous phases:

```text
approach object -> grasp object -> move above goal -> place object -> return hand
```

For HL-to-LL control, `approach cabinet handle` is usually not the useful low-level goal by itself; it is preparation for `grasp/open cabinet door`. Lookahead lets a short preparatory phase inherit the next stable executable intent.

This is a rule-based semantic merge. It does not pick whichever segment is longer.

## Merge Modes

| Mode | Rules | Recommended use |
| --- | --- | --- |
| `conservative` | `approach -> grasp/acquire`; `approach/grasp handle -> open/close/operate` | Default training data |
| `aggressive` | Conservative rules plus `grasp/transport -> place` when a near-future place/release appears | Ablation; can over-advance transport into placement |

Return/reset phases are preserved by default because they can still be useful executable hand objectives. Use `--merge-return-into-previous` only for a special ablation.

## Command

```bash
PYTHONPATH=src python scripts/hl_memory/coarsen_hl_annotations.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations_llm_normalized.jsonl \
  --output-name hl_annotations_llm_normalized_coarse.jsonl \
  --merge-mode conservative \
  --overwrite \
  --summary-json /root/Users/dataset/lerobot_home/subtask/batch_coarse_hl_annotations_summary.json
```

Aggressive ablation:

```bash
PYTHONPATH=src python scripts/hl_memory/coarsen_hl_annotations.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations_llm_normalized.jsonl \
  --output-name hl_annotations_llm_normalized_coarse_aggressive.jsonl \
  --merge-mode aggressive \
  --overwrite
```

## Export

```bash
PYTHONPATH=src python scripts/hl_memory/batch_export_hl_memory_dataset_from_subtasks.py \
  --subtask-root /root/Users/dataset/lerobot_home/subtask \
  --output-root /root/Users/dataset/hl_memory/subtask_coarse \
  --annotations-name hl_annotations_llm_normalized_coarse.jsonl \
  --workers 40 \
  --training-fps 20 \
  --frame-subsample 2 \
  --recent-sample-hz 2.0 \
  --frame-height 224 \
  --frame-width 456 \
  --proprio-enabled \
  --overwrite \
  --continue-on-error
```

Inspect a few generated rows before launching a full training run. The most important fields are `current_objective`, `fine_current_objective`, `coarse_action_type`, and `coarse_merge_reason`.
