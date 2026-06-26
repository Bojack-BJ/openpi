# Memory Shortcut Ablation Plan

This note tracks the failure mode where the HL VLM predicts `current_objective` from language memory, previous-stage state, or dataset order instead of grounding it in the recent video. The target behavior is **recent-primary, memory-as-prior**: recent observations should determine the current visible state; memory and task text should only disambiguate long-horizon order.

## Core Hypothesis

The model should first explain what is visible now, then infer the objective:

```json
{
  "visible_state": "...",
  "current_objective": "..."
}
```

- `visible_state`: short visual/proprioceptive state from recent frames only, for example `the left hand is holding toast above the plate`.
- `current_objective`: action objective inferred from `visible_state`, task instruction, and non-Markov memory prior.
- `language_memory`, completed-event log, previous-stage objective, and proprio are context priors, not direct answer sources.
- If memory conflicts with recent video, `visible_state` should remain stable and `current_objective` should prefer recent evidence unless the scene is genuinely ambiguous.

This is similar to a lightweight vision-CoT: force an intermediate visual grounding step before predicting the objective.

## Tested Findings

### K086A Coarse MEMER Eval

Qwen3.5-27B semantic judge on K086A coarse eval JSONL showed coarse settings are much more stable than fine settings, but proprio did not consistently improve accuracy.

| Code | Setting | Exact | Semantic | Note |
| --- | --- | ---: | ---: | --- |
| C1 | coarse baseline, no proprio | 0.486 | 0.505 | Strong baseline |
| C2 | coarse proprio per-frame | 0.495 | 0.552 | Best semantic score in this eval |
| C3 | coarse proprio summary | 0.457 | 0.505 | Similar semantic to baseline |
| C4 | coarse proprio per-frame + summary | 0.476 | 0.524 | Mild semantic gain |

Result folder:

```text
/root/Users/lixiaotong/HL_MEM_rollout/organized/K086A/coarse/_semantic_eval_27b_eval_jsonl/
```

### H088 Previous-Stage State Mismatch

The `objective_prev_stage_state` setting has very high offline eval accuracy but poor video rollout behavior.

| Checkpoint | Offline objective exact | Video exact | Runtime behavior |
| --- | ---: | ---: | --- |
| step 0400 | 0.999 | 0.133 | Predicts `Grasp the stick` for all 45 video steps |
| step 2000 | 0.977 | 0.133 | Same constant first-objective collapse |

Key paths:

```text
/root/Users/checkpoints/hl_memory/20260519H088Aa_LUMOSS_40_coarse_memer_objective_prev_stage_state/checkpoint-step-000400
/root/Users/lixiaotong/HL_MEM_rollout/organized/H088/coarse/state_context__prev_stage_state/step0400/eval_and_videos/
/root/Users/lixiaotong/HL_MEM_rollout/organized/H088/coarse/state_context__prev_stage_state/step2000/eval_and_videos/
```

Diagnosis:

- This is not primarily a step-2000 overfit issue; step-0400 has the same rollout collapse.
- Offline eval is inflated because it can consume teacher-forced state/context not available in the same form at runtime.
- Runtime `previous_stage_objective` can stay empty or stale, so the model has no robust visual transition trigger.
- The model has not learned to advance by recent observation alone; it learned a shortcut through state/memory/order context.

## Ablation Matrix

These settings should be run with the same data split, checkpoint family, train steps, eval protocol, and semantic judge. The goal is to measure whether `current_objective` is actually sensitive to recent video.

| Setting | Inputs | Expected if visually grounded | Failure signal |
| --- | --- | --- | --- |
| `recent_only` | Recent frames only | Above-chance objective when visual state is distinctive | Near-constant first objective |
| `recent_task` | Recent frames + task instruction | Better disambiguation than `recent_only` | Predicts task-order priors without visual evidence |
| `recent_task_proprio` | Recent + task + proprio | Improves hand/object state transitions | More proprio causes lower visual sensitivity |
| `recent_task_memory` | Recent + task + correct memory | Memory helps phase/order only | Model copies memory phase as answer |
| `recent_task_memory_proprio` | Full non-oracle context | Best robust setting if context is useful | Performs worse as context increases |
| `wrong_memory` | Recent + task + wrong memory | `visible_state` stable; objective mostly follows recent | Objective follows wrong memory |
| `shuffled_memory` | Recent + task + same-task wrong-time memory | Small degradation only near ambiguous transitions | Large degradation across all frames |
| `memory_only` | Task + memory, no recent frames | Low accuracy; negative control | High accuracy means label/order leakage |

The critical comparison is not only absolute accuracy. It is the delta between correct memory and perturbed memory:

```text
memory shortcut score = accuracy(correct memory) - accuracy(wrong/shuffled memory)
```

A large shortcut score with weak `recent_only` accuracy means the model is relying on memory as an answer source.

## Proposed Target Protocol

### Implemented MVP: `memer_film_progress_two_pass`

`memer_film_progress_two_pass` is the first shortcut-reduction implementation. It does not modify Qwen3.5 Gated DeltaNet internals. Instead it routes context through low-bandwidth FiLM conditions:

- Pass A current/keyframe prompt sees recent video + task text only; raw completed-event log is hidden from the prompt.
- Horizon uses an independent prompt and target JSON, so it cannot autoregressively read generated `current_objective` tokens.
- Progress FiLM encodes only runtime-maintainable state: completed-event log / maintained memory summary / last completed objective. It must not consume GT `current_subtask`, GT `phase`, GT `task_progress`, or GT current objective.
- State/proprio FiLM encodes continuous `recent_robot_states`; it is separate from the old proprio soft-token path and does not require textifying numbers in the prompt.
- Pass B receives candidate evidence frames plus stronger progress FiLM and predicts
  `new_completed_objective` plus the updated cumulative `task_progress`. The legacy
  `completed_objective` field remains an alias for old eval/runtime code, but it is
  not derived from `keyframe_label`.

Recommended training flags for this protocol:

```bash
--target-protocol memer_film_progress_two_pass \
--progress-condition-enabled \
--progress-condition-input-mode completed_only \
--progress-condition-dim 128 \
--progress-condition-dropout 0.3 \
--progress-condition-predict-strength 0.5 \
--progress-condition-confirm-strength 1.0 \
--state-condition-enabled \
--state-condition-mode film \
--state-condition-dim 128 \
--field-current-objective-loss-weight 1.5 \
--field-horizon-objective-loss-weight 1.2 \
--field-template-loss-weight 0.1 \
--field-keyframe-candidate-positions-loss-weight 0.1 \
--field-completed-objective-loss-weight 1.0
```

Use `--progress-condition-input-mode full` only as an ablation. It can still encode answer priors if the memory text contains current-like fields.

### Current FiLM Settings

The current Qwen3.5 two-pass experiments use the same output protocol and differ only in how non-visual context is routed. All three settings keep raw completed-event log / language memory out of the Pass A prompt for `current_objective`, `horizon_current_objective`, and `keyframe_candidate_positions`.

| Setting | Prompt input | FiLM / adapter input | Purpose |
| --- | --- | --- | --- |
| `recent_only` | Recent RGB clip + task instruction | none | Grounding baseline; tests whether recent video alone can recover the current phase. |
| `progress_film` | Recent RGB clip + task instruction | compact completed-only progress state | Tests whether low-bandwidth memory prior improves ordering without raw text shortcut. |
| `state_film` | Recent RGB clip + task instruction | continuous recent robot state / proprio vector | Tests whether low-level state helps visual grounding without textifying numbers. |

`progress_film` should use `--progress-condition-input-mode completed_only` by default. It may include runtime-maintainable completed objectives or maintained summary state, but must not consume GT `current_subtask`, GT `phase`, GT `task_progress`, or GT current objective.

`state_film` uses the state/proprio encoder as a continuous side channel. It should be compared against the older proprio-as-text/soft-token runs because the intended benefit is a shorter numeric path:

```text
state vector -> MLP -> FiLM -> field decoder
```

instead of:

```text
state numbers -> tokenizer/text tokens -> language model -> output tokens
```

The next natural ablation is `state_progress_film`, but it should be run after `recent_only`, `progress_film`, and `state_film` establish whether each side channel helps independently.

### Keyframe Prediction Path

The current preferred path is to let the VLM generate `keyframe_candidate_positions` in JSON and rebalance the JSON token loss:

- JSON key/template tokens such as `"keyframe_candidate_positions":`, punctuation, brackets, commas, and field names get low template weight.
- JSON value tokens inside each field get the field-specific weight.
- `keyframe_candidate_positions` value tokens can therefore be trained without over-weighting repeated template syntax.

This is preferable to relying on the first auxiliary head version as the main keyframe path. The auxiliary head is useful as a diagnostic, but it is intentionally shallow:

```text
final language hidden at anchor token -> LayerNorm -> Linear -> SiLU -> position/event/update heads
```

The anchor token is the token immediately before the assistant target starts. In code this is computed as `first supervised target token index - 1`. The hidden state comes from Qwen's final language-model norm, i.e. the final normalization layer of the text decoder stack before `lm_head`, not from the vision tower. This anchor sees the encoded prompt and video context, but it is still only one pooled text-position representation. If that representation does not preserve keyframe-local evidence, a small MLP head cannot recover it.

For this reason, the default training path should keep auxiliary keyframe losses off unless explicitly testing the aux head:

```bash
--keyframe-aux-position-loss-weight 0.0 \
--keyframe-aux-event-loss-weight 0.0 \
--keyframe-aux-timing-loss-weight 0.0 \
--keyframe-aux-update-loss-weight 0.0 \
--field-template-loss-weight 0.1 \
--field-keyframe-candidate-positions-loss-weight 0.1
```

If the aux head is revisited, it should be treated as a recall-oriented auxiliary regularizer, not a replacement for VLM JSON selection. Required improvements would be soft event targets / focal loss, stronger positive weighting, and top-K recall metrics instead of thresholding the event head into an empty prediction.

Add a target protocol variant that predicts visual grounding before the objective, for example:

```json
{
  "visible_state": "the right gripper is above the top-right hole while holding the stick",
  "current_objective": "place the stick into the top-right hole",
  "horizon_current_objective": "move back to the center",
  "keyframe_candidate_positions": []
}
```

Rules:

- `visible_state` must not include future actions, task order, or memory-only facts.
- `current_objective` may use memory/task context, but should be consistent with `visible_state`.
- `horizon_current_objective` is optional for this ablation; it should not be used to judge current objective grounding.
- If no clean visual state is available, output an uncertainty phrase instead of hallucinating an object relation.

## Metrics

Use both exact metrics and semantic judge metrics.

| Metric | Purpose |
| --- | --- |
| Objective exact / normalized match | Fast regression signal |
| Objective semantic accuracy | Allows equivalent wording |
| Visible-state semantic consistency | Checks visual grounding field |
| Memory override rate | Fraction of wrong/shuffled-memory samples where prediction follows memory |
| Transition lag / early switch | Detects delayed or premature objective changes |
| Per-step rollout accuracy | Measures runtime behavior, not teacher-forced eval |
| Perturbation sensitivity | Difference between correct, wrong, shuffled, and dropped memory |

For rollout, report per-episode curves and aggregate bar charts separately. Offline eval can be useful, but must be labeled as either teacher-forced context eval or runtime-context eval.

## Implementation Tasks

1. Add a target protocol such as `memer_visible_state_objective`.
2. Update prompt builder to explicitly state: recent observation is primary evidence; memory is long-term prior.
3. Export or generate `visible_state` GT. For v1, a coarse template can be derived from objective/state annotations; for stronger supervision, use a larger VLM/LLM teacher plus spot checks.
4. Add eval perturbation modes: `drop_memory`, `wrong_memory`, and `shuffled_memory`.
5. Add semantic judge support for both `visible_state` and `current_objective`.
6. Separate teacher-forced eval from runtime-context eval in result folders and markdown summaries.

## Decision Criteria

Prefer a setting only if it satisfies all of these:

- It improves or preserves video rollout accuracy, not only offline eval.
- It remains stable under wrong/shuffled memory.
- It changes objective when recent video clearly changes phase.
- It does not collapse to first objective or task-order memorization.
- It can run at the intended inference frequency without adding excessive latency.
