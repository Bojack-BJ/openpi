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

After row-level coarsening, the default pass also merges compatible short adjacent runs up to `--short-run-merge-max-frames 20`. This is intentionally conservative:

- The merge requires overlapping object tokens and compatible action semantics.
- Same-action short fragments with equivalent object tokens prefer the shorter/general wording, e.g. `Fold packaging box sides` becomes `Fold packaging box`.
- Very short `transport -> place` pairs are collapsed into a single place intent, e.g. `Move to above the blue box` + short `Place large toy 2 into the blue box`.
- Very short terminal `release -> reset` pairs are collapsed into reset, e.g. short `Release shoebox` + `Move back to observation region`.
- Very short bottle-cap rotate/remove tails are collapsed into `Loosen and remove the bottle cap` or `... lid`.
- Reset/wait phases are preserved.
- Directional bottle-cap fragments that do not yet complete a remove sequence are still preserved.
- Fine/source labels are checked before merging, so acquisition labels are not collapsed backward into preparatory `move/approach` labels.

## Command

```bash
PYTHONPATH=src python scripts/hl_memory/coarsen_hl_annotations.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations_llm_normalized.jsonl \
  --output-name hl_annotations_llm_normalized_coarse.jsonl \
  --merge-mode conservative \
  --short-run-merge-max-frames 20 \
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
  --short-run-merge-max-frames 20 \
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

## Current Statistics

Latest full run on `/root/Users/dataset/lerobot_home/subtask` with `hl_annotations_llm_normalized_coarse.jsonl`:

| Metric | Value |
| --- | ---: |
| Coarse runs | 12,081 |
| Short-run p10 threshold | 23 frames |
| Runs at or below p10 | 1,345 |
| Segment duration min | 0 frames |
| Segment duration median | 62 frames |
| Segment duration p90 | 124 frames |
| Segment duration max | 312 frames |

Targeted checks after the latest pass:

- `20260309H068F`: short `rotate/twist -> remove` bottle-cap tail becomes `Loosen and remove the bottle cap`.
- `20260310H076A`: final short `Release shoebox` now merges into `Move back to observation region`, while earlier `Release left side of shoebox` stays separate.
- `20260319H107A`: short `Place large toy ... into the blue box` now absorbs the preceding `Move to above the blue box`.

The shortest remaining runs are now mostly true edge cases: reset/retract phases, isolated pull/open snippets, or directional bottle-cap fragments that do not yet complete a remove step.
