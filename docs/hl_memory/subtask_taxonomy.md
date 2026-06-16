# HL Memory Subtask Taxonomy Notes

This note summarizes the current `subtask_segments.json` segmentation style across the 87 usable LeRobot-home tasks and proposes a coarser atomic-action taxonomy for training/evaluation.

## Current Segmentation Statistics

The summary below was produced by `scripts/hl_memory/summarize_subtask_segments.py` over `/root/Users/dataset/lerobot_home/subtask`.

| Metric | Value |
| --- | ---: |
| Usable tasks | 87 |
| Total segments | 14,827 |
| Segment duration p10 / median / p90 | 17 / 51 / 107 frames |
| Segment duration min / max | 1 / 313 frames |
| Segments per episode p10 / median / p90 | 5 / 9 / 14 |
| Segments per episode min / max | 3 / 19 |

At 20 Hz, the median segment is about 2.55 seconds and p10 is about 0.85 seconds. With 1 Hz rollout probes, many fine-grained boundaries are under-observed or visually ambiguous.

## Current Atomic Categories

| Category | Count | Typical text |
| --- | ---: | --- |
| `grasp/hold` | 3,855 | hold bottle, grasp handle, pick up basket |
| `approach/pregrasp` | 2,916 | approach cabinet door, move to right end |
| `place/release` | 2,796 | place lid on table, put object into cabinet |
| `return/retreat` | 2,455 | move back to observation region |
| `transport/align` | 1,163 | move above plate, align object, carry object |
| `operate/articulate` | 1,045 | open door, close drawer, rotate lid |
| `other` | 597 | task-specific or underspecified actions |

The most frequent adjacent patterns are:

| Adjacent categories | Count | Interpretation |
| --- | ---: | --- |
| `approach/pregrasp -> grasp/hold` | 2,324 | Often one semantic action: acquire object/handle |
| `place/release -> return/retreat` | 1,669 | Return is often a reset action after the useful manipulation |
| `grasp/hold -> place/release` | 1,667 | Object transfer may skip an explicit move phase |
| `return/retreat -> approach/pregrasp` | 708 | Repeated object-by-object cycle |
| `grasp/hold -> transport/align` | 605 | Start of object transport |
| `grasp/hold -> operate/articulate` | 517 | Handle/cap acquisition before articulation |
| `operate/articulate -> return/retreat` | 427 | Articulation followed by reset |
| `transport/align -> place/release` | 384 | End of object transport |

## Problem With The Current Fine Labels

The labels often split one manipulation into phases such as:

```text
approach object -> hold/grasp object -> move above goal -> place object -> move back
```

This is useful for detailed annotation, but it is too strict for current HL VLM evaluation:

- A visually plausible prediction such as `place toast on plate` can be judged wrong if the GT phase is still `move above the plate`.
- `approach` versus `hold` is often ambiguous in fisheye RGB, especially when the hand is already near the object.
- `return to observation region` is frequent and short, but often not a useful LL-VLA objective.
- Prior-based rollout can appear to skip because several fine phases collapse into one visually continuous action.
- Exact or strict semantic eval under-scores predictions that are useful at the policy level but not phase-identical.

## Recommended Coarse Atomic Types

Use these as normalized targets for coarse evaluation or a future `coarse_objective` field. Keep the original fine `current_subtask` for debugging.

| Coarse type | Meaning | Example target |
| --- | --- | --- |
| `acquire_object` | Approach and grasp/pick an object or handle | `Pick up the left toast` |
| `transport_object` | Move a held object toward a goal region | `Move the toast above the plate` |
| `place_object` | Place/insert/stack/release an object at the goal | `Place the toast on the plate` |
| `operate_articulated_object` | Open/close/press/pull/push/rotate an articulated object | `Open the cabinet door` |
| `handover_or_transfer` | Transfer object between hands or to another agent | `Hand over the bottle to the left hand` |
| `reset_hand` | Return/retreat to observation or neutral region | `Move the right hand back` |
| `wait_or_verify` | State check or ambiguous static phase | `Verify the object is inside the cabinet` |

## Merge Rules To Try

Use rule-based coarse labels first; do not destroy fine labels.

| Fine pattern | Suggested coarse label | Rationale |
| --- | --- | --- |
| `approach/pregrasp + grasp/hold` | `acquire_object` | Boundary is often visually subtle; endpoint is object/handle acquired |
| `grasp/hold + transport/align` | `transport_object` | Holding at source and moving away are often one continuous policy phase |
| `transport/align + place/release` | `place_object` when goal contact is near | Reduces `move above` vs `place` timing ambiguity |
| `approach handle + grasp handle + open/close` | `operate_articulated_object` | LL objective is usually opening/closing, not the handle pregrasp itself |
| `place/release + return/retreat` | keep `place_object`; optionally drop `reset_hand` | Return is usually reset, not semantic task progress |
| consecutive same category/object phases | merge if duration is short or only wording changes | Avoids repeated `hold/hold` or `move/move` jitter |

## Current Coarsening Policy

`scripts/hl_memory/coarsen_hl_annotations.py` uses action-type classification plus rule-based lookahead. It is not a "pick the longer segment" heuristic. The current row is relabeled only when the future objective gives a more stable executable intent.

The script writes both coarse and fine fields:

| Field | Meaning |
| --- | --- |
| `coarse_action_type` | Fixed action class such as `acquire_object` or `operate_articulated_object` |
| `coarse_current_objective` | Natural-language coarse objective with object/hand/location text preserved |
| `coarse_merge_reason` | Why the row was changed, e.g. `approach_to_acquire_lookahead` |
| `coarse_source_objective` | Fine objective used as the source for the coarse decision |
| `fine_current_objective` | Original fine label before replacement |

The train-facing fields `current_objective/current_subtask/phase` are replaced with `coarse_current_objective` by default, but the output is not a bare class name. For example:

```json
{
  "fine_current_objective": "The left hand approaches the cabinet door handle.",
  "coarse_action_type": "operate_articulated_object",
  "coarse_current_objective": "Open the cabinet door.",
  "current_objective": "Open the cabinet door."
}
```

This preserves object grounding while reducing phase fragmentation.

Two merge modes are available:

| Mode | Rules | Use |
| --- | --- | --- |
| `conservative` | `approach -> grasp/acquire`; `approach/grasp handle -> open/close/operate` | Default. Reduces the most ambiguous phase boundaries without pulling all mid-motion rows into final placement. |
| `aggressive` | Conservative rules plus `grasp/transport -> place` when a near-future place/release objective appears | Ablation only. It reduces run count more, but can over-label intermediate transport as final placement. |

`horizon_current_objective` is classified but does not use lookahead merge. It is already a future label, and applying another future lookahead can double-advance the target.

After the row-level pass, `--short-run-merge-max-frames 20` merges only compatible adjacent short runs. The second pass is source-aware:

| Case | Behavior |
| --- | --- |
| Same action/object wording variants | Merge and prefer the shorter/general label, e.g. `Fold packaging box sides` -> `Fold packaging box` |
| Reset/wait/retract | Preserve |
| Bottle/cap rotate/twist | Preserve because direction and cap-removal order matter |
| Fine source is grasp/place but coarse text looks like move/approach | Preserve the acquisition/place label rather than merging backward into preparation |

Current conservative coarse statistics on the 87-task normalized annotations:

| Statistic | Fine normalized | Coarse normalized |
| --- | ---: | ---: |
| Runs | 14,792 | 12,618 |
| Segment duration min | 0 | 0 |
| Segment duration p10 | 16 | 18 |
| Segment duration median | 50 | 59 |
| Segment duration p90 | 106 | 123 |
| Short runs at p10 threshold | 2,521 at <=20 frames | 1,272 at <=18 frames |

The remaining shortest coarse runs are mostly reset/release boundary frames and directional bottle-cap fragments. These should be inspected before any more aggressive merge rule; they are not obviously safe to collapse automatically.

## Evaluation Recommendation

Report both fine and coarse metrics:

| Metric | Purpose |
| --- | --- |
| Fine semantic accuracy | Measures exact phase tracking against current labels |
| Coarse semantic accuracy | Measures whether the predicted LL objective is useful |
| Transition timing error | Separates early/late switching from wrong action/object |
| Reset-hand exclusion metric | Recomputes accuracy after ignoring optional `return/retreat` phases |
| Future-step hallucination rate | Measures whether prior/proprio causes jumping ahead |

The current Qwen3.5-27B judge should keep the strict label set, but add an optional coarse-normalization pass. Otherwise good policy-level outputs will continue to be counted as `wrong_action` when they differ only by a fine phase boundary.

## Next Implementation Step

Add an optional `coarse_objective` field during export:

```json
{
  "current_subtask": "The left hand moves to above the white plate",
  "current_objective": "Move the toast above the white plate",
  "coarse_objective": "Place the toast on the white plate",
  "coarse_action_type": "place_object"
}
```

Then compare:

| Training target | Eval target | Question |
| --- | --- | --- |
| fine `current_objective` | fine semantic | Can the model track detailed phases? |
| fine `current_objective` | coarse semantic | Are apparent errors mostly over-fine GT? |
| coarse `coarse_objective` | coarse semantic | Does coarser supervision improve rollout stability? |
| MEMER horizon objective | coarse semantic | Does horizon prediction naturally absorb phase ambiguity? |
