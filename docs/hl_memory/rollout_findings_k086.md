# K086A Coarse 10-Episode Findings

## Evaluation Scope

- Task: `20260328K086A`
- Four coarse MEMER settings
- Ten video episodes per setting
- 520 labeled inference timestamps per setting
- Ground truth was replaced with coarse objectives before comparison

## Exact-Match Summary

| Setting | Exact match | Accuracy |
|---|---:|---:|
| Baseline | 263 / 520 | 0.506 |
| Proprio per-frame | 205 / 520 | 0.394 |
| Proprio summary | 207 / 520 | 0.398 |
| Proprio per-frame + summary | 200 / 520 | 0.385 |

The baseline is clearly stronger in this 10-episode comparison. Adding more proprio information did not monotonically improve performance.

## Interpretation

The coarse objective is often visually identifiable from object contact, transport, placement, and return motion. Current proprio injection adds many continuously varying values but does not directly encode semantic task state. It can therefore compete with the visual representation, overfit trajectory details, and become sensitive to session-dependent calibration or timing.

The likely failure is not that proprio is intrinsically useless. The current representation asks the VLM to infer semantic progress from raw or summarized state while also predicting the objective. A stronger design separates responsibilities:

- External task state owns the ordered objective.
- The VLM uses RGB and optional proprio only to estimate progress and completion.
- Proprio summary should emphasize motion state, gripper transition, displacement, and contact-relevant changes rather than every raw coordinate.

## Relation to H088

H088 exposes a stronger form of the same issue. When repeated objectives differ mainly by ordinal location, recent RGB and raw proprio do not provide a persistent sequence index. Explicit task state is required. K086A may be less sequence-ambiguous, but the state-machine protocol still provides a cleaner test of whether proprio helps completion detection.

## Recommended Follow-Up

1. Re-run the four settings with `known_prior_tracker`.
2. Keep the ordered plan and completed objective index external.
3. Compare no proprio, compact proprio summary, and per-frame-plus-summary.
4. Evaluate completion/advance quality separately from objective classification.
5. Use debounce so one noisy prediction cannot advance the state twice.

