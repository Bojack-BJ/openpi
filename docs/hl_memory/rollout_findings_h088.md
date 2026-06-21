# H088 Coarse-Hole Step-800 Findings

## Evaluation Scope

- Task: `20260519H088Aa_LUMOSS_40`
- Checkpoint step: `800`
- Video inference: 10 episodes, 45 inference timestamps per episode
- Prompt condition: no language memory and no step prior
- Ground truth: `hl_annotations_coarse_hole.jsonl`, field `current_objective`
- Ground-truth taxonomy: eight objectives consisting of grasp, six location-specific insertion objectives, and return
- Semantic judge: Qwen3.5-27B score mode

## Results

| Setting | Offline exact | Video semantic | Main semantic errors |
|---|---:|---:|---|
| Baseline | 0.181 | 0.280 | wrong location 135, too early 116, wrong action 68 |
| Proprio per-frame | 0.147 | 0.229 | wrong location 168, wrong action 117, too early 62 |
| Proprio summary | 0.307 | 0.176 | wrong action 153, too early 112, wrong location 106 |
| Proprio per-frame + summary | 0.334 | 0.291 | wrong location 205, wrong action 93, too early 21 |

The ordinary location-agnostic coarse annotation has four objectives: grasp, insert, release, and return. This evaluation intentionally used the stricter `coarse_hole` annotation, which distinguishes top-left, top-right, middle-left, middle-right, bottom-left, and bottom-right.

## Why Hole Order Was Not Learned Reliably

The evaluated checkpoints were trained and evaluated without language memory and without step prior. The recent clip contains eight frames sampled at 2 Hz and covers approximately 3.5 seconds. Samples are independently shuffled during training and inference calls do not carry a recurrent hidden state.

Consequently, the model usually observes only the local grasp or insertion motion. It does not receive a persistent count of completed holes. Once the previous insertion leaves the recent window, the six insertion phases are visually similar and the current hole index becomes partially unobservable. The dominant failure is therefore a correct insertion action assigned to the wrong hole location.

Proprioception does not directly encode the sequence index. It provides current arm state and motion, but not which holes were previously completed. Per-frame proprio can also encourage trajectory memorization and amplify cross-session calibration variation.

## Dataset Defect Found

The exported file

`/root/Users/dataset/hl_memory/subtask/20260519H088Aa_LUMOSS_40_coarse_hole/train/samples.jsonl`

previously contained `Task progress: No completed subtask yet.` for every sample. The coarse annotation contains objective transitions but no `event_type=success` rows, while the exporter previously accumulated completed objectives only from explicit success events.

The exporter is corrected to:

1. Mark the previous explicit `current_objective` complete when the objective changes.
2. Derive continuous `subtask_progress` from each contiguous objective span when the annotation does not provide it.
3. Set `should_advance_objective=true` on the final sample of each objective span.

The corrected H088 train/val export must be used for future state-tracker training.

## Recommended State-Machine Architecture

Use an external state machine as the single authority:

```text
ordered step prior
        |
        v
current objective index + completed objectives
        |
        v
VLM receives current objective, compact completed state, recent RGB, optional proprio
        |
        v
VLM predicts only subtask_progress, should_advance_objective, keyframe candidates
        |
        v
debounced transition gate advances index by at most one
        |
        v
state machine deterministically regenerates objective and memory
```

The VLM must not freely generate the next objective or rewrite task history. Its echoed `current_objective` is ignored by runtime state. This prevents feedback excitation where a changed subtask rewrites memory and the rewritten memory immediately pushes another subtask change.

## Memory Maintenance Rules

- Persist only completed milestones and stable facts.
- Keep the active objective in state-machine fields, not in free-form model-owned history.
- Update completed history only after a debounced completion decision.
- Require multiple consecutive positive completion predictions or a high progress threshold before advancing.
- Advance by at most one step unless an explicit recovery policy is active.
- Never feed the model's proposed next objective directly back as authoritative input.
- Use keyframes as visual evidence, but let the state machine own ordering and counting.

## Next Experiments

1. Train `known_prior_tracker` on the corrected export.
2. Compare RGB-only, proprio-summary, and per-frame-plus-summary under the same state-machine protocol.
3. Report both strict coarse-hole accuracy and location-agnostic coarse action accuracy.
4. Tune transition debounce and progress thresholds on held-out episodes.

