# HL Memory Schemas

This note describes the fields that matter for current MEMER-style and coarse-subtask training.

## Annotation Row

The base normalized annotation row keeps fine labels and optional horizon labels:

```json
{
  "episode_index": 0,
  "frame_index": 120,
  "current_objective": "The left hand grasps the cabinet door handle.",
  "current_subtask": "The left hand grasps the cabinet door handle.",
  "phase": "The left hand grasps the cabinet door handle.",
  "horizon_frame_index": 124,
  "horizon_current_objective": "The left hand opens the cabinet door.",
  "horizon_current_subtask": "The left hand opens the cabinet door.",
  "subtask_progress": 0.65,
  "should_advance_objective": false,
  "active_hand": "left",
  "target_query": "cabinet door handle",
  "goal_query": "cabinet interior",
  "relevant_objects": ["cabinet door", "door handle"]
}
```

## Coarse Annotation Row

`scripts/hl_memory/coarsen_hl_annotations.py` preserves the fine label and adds coarse fields:

```json
{
  "fine_current_objective": "The left hand grasps the cabinet door handle.",
  "fine_current_subtask": "The left hand grasps the cabinet door handle.",
  "coarse_action_type": "operate_articulated_object",
  "coarse_current_objective": "The left hand opens the cabinet door.",
  "coarse_current_subtask": "The left hand opens the cabinet door.",
  "coarse_phase": "The left hand opens the cabinet door.",
  "coarse_merge_reason": "articulated_target_lookahead",
  "coarse_source_objective": "The left hand grasps the cabinet door handle.",
  "coarse_horizon_action_type": "operate_articulated_object",
  "coarse_horizon_current_objective": "The left hand opens the cabinet door."
}
```

By default, train-facing fields are replaced:

```json
{
  "current_objective": "The left hand opens the cabinet door.",
  "current_subtask": "The left hand opens the cabinet door.",
  "phase": "The left hand opens the cabinet door."
}
```

The coarse target is not a bare class such as `operate_articulated_object`. The class is metadata; the objective remains natural language with object/hand/location grounding.

## Coarse Action Types

| Action type | Meaning | Example objective |
| --- | --- | --- |
| `acquire_object` | Approach and grasp/pick an object or handle | `Pick up the left slice of toast.` |
| `grasp_object` | Grasp/hold without a reliable future merge | `Grasp the cabinet handle.` |
| `transport_object` | Move or align a held object | `Move the toast above the plate.` |
| `place_object` | Place/insert/stack/release at a goal | `Place the toast on the plate.` |
| `operate_articulated_object` | Open/close/press/pull/push/rotate | `Open the cabinet door.` |
| `handover_or_transfer` | Transfer between hands or agents | `Hand over the bottle to the left hand.` |
| `reset_hand` | Return/retreat to neutral observation region | `Move the right hand back to the observation region.` |
| `wait_or_verify` | Static verification or waiting | `Verify the object is inside the cabinet.` |

## Target Protocols

| Protocol | Target content | Current recommendation |
| --- | --- | --- |
| `memer_objective` | Current/horizon objective plus keyframes, no progress/memory JSON | Main coarse-subtask baseline |
| `keyframe_gated_memory` | Current/horizon objective, recent keyframe candidates, and an instantaneous completed objective | Experimental typed-memory protocol |
| `keyframe_gated_memory_two_pass` | Pass A predicts current/horizon/keyframe proposal; Pass B confirms completed event only | Main keyframe-gated follow-up setting |
| `subtask_keyframe` | Current objective plus keyframes | Diagnostic baseline |
| `hl_v1` | Full JSON with memory/progress/advance fields | Useful later, not the current minimal baseline |

For coarse data, `memer_objective` reads the replaced `current_objective` / `horizon_current_objective`, so no training-code fork is needed.

## Keyframe-Gated Memory Semantics

The gated protocol separates an instantaneous completion event from historical
visual memory:

- `keyframe_label`: the current sample closes a coarse objective.
- `keyframe_candidate_positions`: recent-clip positions containing keyframe
  proposals. With the default `event_band` mode, a sample is positive if any
  recent frame falls in the transition band around a canonical keyframe.
- `keyframe_event_ids` / `keyframe_event_frame_indices`: optional compact event
  metadata. Multiple samples can point to the same canonical event; rollout
  memory stores the merged canonical event, not every event-band frame.
- `completed_objective`: non-empty only when `keyframe_label=true`.
- `historical_keyframe_seconds`: runtime state derived from previously accepted
  keyframe candidates.

Event-band supervision keeps `keyframe_label` as the strict canonical single
point, but trains candidate proposal with a wider band:

- `--keyframe-candidate-label-mode event_band`: default candidate supervision.
- `--keyframe-event-band-before-sec 1.0`
- `--keyframe-event-band-after-sec 0.5`
- `--keyframe-candidate-label-mode canonical`: strict legacy candidate target.

The current first-stage implementation uses one shared causal decoder, so every
generated field can still attend to every prompt input. The intended typed
attention routing is:

| Output | Primary context | Additional context |
| --- | --- | --- |
| Current/horizon objective | Recent observation window | Historical keyframes, completed-event log, instruction |
| Keyframe candidates | Recent observation window | Instruction |
| Completed-event/memory update | Historical keyframes plus accepted candidate | Completed-event log, instruction |

A normal 2D padding mask cannot enforce this routing. It requires a block/typed
causal mask over memory/recent/output token groups, or separate prediction
passes/heads. Historical keyframes provide non-empty long-term context on most
later timesteps and can make memory reconstruction less sensitive to one noisy
current-frame event, but explicit completion-event supervision is still needed.

`keyframe_gated_memory_two_pass` implements strict typed source routing without
changing Qwen internals:

- Pass A sees accepted historical keyframes plus the full recent observation
  window and outputs only
  `current_objective`, `horizon_current_objective`, and
  `keyframe_candidate_positions`.
- Pass B sees accepted historical keyframes, candidate routing metadata, and
  only the proposed candidate frames. Pass A current/horizon text is hidden
  from Pass B to avoid teacher-forced semantic leakage. All non-candidate
  recent frames are physically removed from the Pass B processor input. If
  Pass A proposes no candidate at inference, Pass B is skipped and completion
  is forced empty.
- Training expands each sample into two VLM examples, so each pass has its own
  prompt span, visual source set, and target span. Pass B training uses
  perturbed positive proposals plus deterministic false proposals on negative
  samples; rollout uses generated Pass A candidates. Pass A and Pass B losses
  are normalized per example before stage weighting.
- When proprio is enabled, Pass A receives the full recent state window while
  Pass B receives only states aligned to its candidate frames.
- This hard source exclusion is stronger than applying a 4D mask only to
  Qwen3.5 full-attention layers. Qwen3.5-4B mixes 24 linear-attention layers
  with 8 full-attention layers; its linear-attention implementation does not
  support an arbitrary query-key block mask. A naive 4D mask would therefore
  leak the forbidden recent tokens through most layers.
