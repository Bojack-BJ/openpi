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
| `subtask_keyframe` | Current objective plus keyframes | Diagnostic baseline |
| `hl_v1` | Full JSON with memory/progress/advance fields | Useful later, not the current minimal baseline |

For coarse data, `memer_objective` reads the replaced `current_objective` / `horizon_current_objective`, so no training-code fork is needed.
