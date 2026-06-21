# HL Memory Experiment Settings

This document tracks the main HL-memory data/training settings tried so far. Each row is one setting. Use it as a compact experiment ledger when comparing future runs.

See also [README.md](README.md) for the shorter schema/protocol notes and [subtask_taxonomy.md](subtask_taxonomy.md) for the 87-task segmentation analysis.

## Data And Training Settings

| Setting | Data sampling | Ground-truth labeling | GT format | Input content | Dropout | Observed behavior / issue | Notes / fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 boundary-only | Only subtask boundary/start, sometimes end | Directly from `subtask.json` / `subtask_segments.json` | Mostly `current_subtask` text | Recent frames + instruction | None | Learns boundary labels but not within-subtask progress; rollout can stay on one objective or switch abruptly | Smoke only; not enough for progress-aware rollout |
| S1 start/middle/end | One start, one middle, one end/success per segment | Rule-generated from segment start/end | `current_subtask`, `phase`, `event_type`, `event_text` | Recent frames + language memory + instruction | Usually none | Text-memory eval looked reasonable, but progress supervision was sparse and late-stage switch cues were under-covered | Keep as baseline; not final training distribution |
| S2 stride / time-based progress | Fixed stride or target-frame sampling inside segment | Rule-generated from segment timeline | `current_subtask` plus raw frame-ratio progress | Recent frames + memory + instruction | Optional language memory dropout | Long segments dominate; short segments may only contribute 0/1-like states; raw progress has arbitrary decimals | Do not use alone; add per-segment cap and progress quantization |
| S3 fixed-fraction progress | Fixed fractions per segment, e.g. `0.1,0.3,0.5,0.7,0.9` | Rule-generated from segment timeline | `current_subtask` plus progress | Recent frames + memory + instruction | Optional language memory dropout | More balanced across segment lengths than stride, but long visual variation can be under-sampled; short segments can round multiple fractions to the same frame | Deduplicate frames; useful when mixed with dynamic sampling |
| S4 mixed dynamic + late fractions | Target frames per segment + regular fractions + late fractions like `0.85,0.9,0.95` | Rule-generated from segment timeline | `current_subtask`, `subtask_progress`, `should_advance_objective` | Recent frames + memory + instruction | Optional language memory dropout | Covers early/mid/late states better, but raw progress floats still make outputs unstable | Recommended base sampler after adding progress quantization |
| S5 mixed + short-segment adaptation | S4 plus short-segment fractions such as `0.2,0.4,0.6,0.75,0.9`; short min-gap can be relaxed | Rule-generated from segment timeline | `current_subtask`, quantizable progress, advance signal | Recent frames + memory + instruction | Optional language memory dropout | Fixes very short segments that otherwise only show start/end; increases sample count | Current recommended sampler; keep `max-progress-samples-per-segment` bounded |
| S6 LLM-normalized labels | Usually S5 | Offline LLM normalizes rule-based annotations once, then cached JSONL is used | JSON with `task_progress`, `current_objective`, `subtask_progress`, `should_advance_objective`, `active_hand`, `relevant_objects`, `notes` | Recent frames + memory + instruction | Optional language memory dropout | Text fields are cleaner, but exact-match eval can still under-score semantically equivalent wording | Evaluate normalized/semantic match in addition to exact match |
| S7 quantized progress labels | S5 | LLM-normalized labels plus progress rounded to fixed quantum | Same as S6; `subtask_progress` is numeric `[0,1]` rounded to `0.05` bins | Recent frames + memory + instruction | Optional language memory dropout | Easier for VLM/LLM to learn than arbitrary floats; still needs enough near-transition samples | Current recommended GT format; do not encode progress as `"40%"` strings |
| S8 known-prior training | S5 | LLM-normalized + quantized labels | JSON target as S7; `step_prior` is input context, not supervised target | Recent frames + memory + instruction + ordered step prior | `language_memory_dropout` and `step_prior_dropout` | More controllable for known workflows; without dropout the model may over-rely on the plan and ignore vision | Current recommended main training setting for known-prior rollout |
| S9 RGB-only fixed dual-slot layout | Can combine with S5-S8 | Same as selected label protocol | Same as selected GT format | RGB-only frames in fixed `456x224` dual-slot canvas; single view left slot and black right slot; dual view left/right slots | Same as selected setting | Prevents mask leakage and train/rollout visual-layout mismatch | Current visual protocol; re-export old datasets to match it |
| S10 mask/overlay leakage | Any old sampler | Any labels | Any format | RGB plus mask/overlay/highlight channels | Any | HL can cheat by reading target masks; incompatible with future SAM prompting from HL | Deprecated; export excludes mask/overlay/highlight columns |

## Current Recommended Setting

| Axis | Recommendation |
| --- | --- |
| Data sampling | S5: mixed dynamic progress sampling + late fractions + short-segment adaptation |
| Ground-truth labeling | Offline LLM normalization from rule-generated annotations |
| GT format | JSON target with `task_progress`, `current_objective`, `subtask_progress`, `should_advance_objective`, `active_hand`, `relevant_objects`, `notes` |
| Progress format | Numeric `[0,1]`, quantized with `--subtask-progress-quantum 0.05` |
| Input content | RGB recent clip, RGB memory keyframes, language memory, instruction, optional ordered `step_prior` |
| Dropout | `language_memory_dropout=0.3`; when training known-prior, use `step_prior_dropout` as well |
| Visual layout | Fixed dual-slot `456x224`; Qwen video metadata fps is `training_fps / frame_subsample`, default `4Hz` |

## Current K086A Setting Matrix

The current controlled K086A sweep should use the same data split, checkpoint base, vision train mode, rollout session, and eval protocol unless the row explicitly changes one axis.

| Setting | Target protocol | Label granularity | Memory input | Step prior input | Proprio | Proprio mode | Extra token | Expected use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `memer_baseline` | `memer_objective` | fine or coarse | dropped | dropped | off | n/a | none | Main simplified visual baseline |
| `memer_proprio_per_frame` | `memer_objective` | coarse preferred | dropped | dropped | on | `per_frame` | none | Tests whether aligned state sequence helps |
| `memer_proprio_summary` | `memer_objective` | coarse preferred | dropped | dropped | on | `summary` | none | Tests state bottleneck robustness |
| `memer_proprio_per_frame_plus_summary` | `memer_objective` | coarse preferred | dropped | dropped | on | `per_frame_plus_summary` | none | Current strongest proprio candidate |
| `memer_memory_aux` | `memer_objective` or future memory-aux protocol | coarse preferred | dropped | dropped | optional | chosen best mode | none | Supervise memory prediction without feeding text memory |
| `memer_context_summary` | `memer_objective` | coarse preferred | dropped | dropped | optional | chosen best mode | full-context summary token | Tests learned compression before JSON decode |
| `memer_prior_context` | `memer_objective` | coarse preferred | dropped | weak prior with dropout | optional | chosen best mode | optional | Tests whether prior helps after coarse labels/proprio |
| `subtask_keyframe_baseline` | `subtask_keyframe` | fine | dropped | dropped | off/on | selected | none | Diagnostic only; currently lower priority |
| `objective_memory_state` | `objective_memory_state` | coarse preferred | completed-subtasks memory in/out | dropped | optional | selected | `updated_language_memory` | Tests whether model can consume and update compact history |
| `objective_last_objective` | `objective_last_objective` | coarse preferred | no state input | dropped | optional | selected | `last_objective` | Auxiliary-output ablation that forces previous-objective reconstruction |
| `objective_prev_stage` | `objective_prev_stage` | coarse preferred | previous distinct objective in/out | dropped | optional | selected | `previous_stage_objective` | Tests compact self-maintained stage state |
| `hl_v1_full` | `hl_v1` | fine | optional | optional | optional | selected | none | Full memory/progress/advance protocol, not the current debugging baseline |

Recommended immediate sweep:

| Priority | Run | Why |
| ---: | --- | --- |
| 1 | `coarse + memer_baseline` | Separates target granularity from architecture/input effects |
| 2 | `coarse + memer_proprio_per_frame` | Tests direct time-aligned state tokens |
| 3 | `coarse + memer_proprio_summary` | Tests whether state-only bottleneck is enough |
| 4 | `coarse + memer_proprio_per_frame_plus_summary` | Tests information-rich proprio input |
| 5 | `coarse + memer_memory_aux` | Forces visual/state-to-memory learning while keeping text memory out of input |
| 6 | `coarse + memer_context_summary` | Tests a learned full-input summary token |
| 7 | prior variants | Only after the pure-model settings are stable |
| 8 | `coarse + objective_memory_state` | Tests full completed-history context without supervising free-form memory output |
| 9 | `coarse + objective_last_objective` | Tests the lightest online state feedback loop |
| 10 | `coarse + objective_prev_stage` | Tests the upper bound from clean previous-stage context |

Do not treat prior-trained + known-prior rollout as the same metric as pure-model rollout. Known-prior override rewrites model outputs and can mask or introduce timing errors.

The three `objective_*` protocols are MEMER-style state-context ablations. They supervise the same target as
`memer_objective`, `{"current_objective": "...", "horizon_current_objective": "...",
"keyframe_candidate_positions": [...]}`, and differ only in what text state is provided in the prompt:

- `objective_memory_state`: consumes `language_memory` and predicts `updated_language_memory`.
- `objective_last_objective`: predicts `last_objective` as an auxiliary target and does not consume it as prompt input.
- `objective_prev_stage`: consumes `previous_stage_objective` and predicts the maintained `previous_stage_objective`.

Existing `samples.jsonl` can train these protocols: the loader backfills `last_objective` and
`previous_stage_objective` when old exports do not contain those fields. Re-export is still recommended after label
logic changes that affect `language_memory` itself, because loader backfill cannot repair stale memory strings stored in
old samples.

### State Context Injection And Prediction

The `objective_*` protocols do not add learned tokens or change the video inputs. They keep the MEMER current/horizon
objective target and add one state field to the JSON target. Depending on the ablation, the state is also supplied as
plain text context in the user prompt.

`objective_memory_state` consumes and predicts compact memory:

```text
Task instruction: ...
Completed-subtasks memory: ...
```

target adds:

```json
{"updated_language_memory": "..."}
```

`objective_last_objective` does not provide `last_objective` as prompt input. It predicts it as an auxiliary training
field, and rollout can ignore it:

```json
{"last_objective": "..."}
```

`objective_prev_stage` consumes and predicts the previous distinct objective:

```text
Task instruction: ...
Previous stage objective: ...
```

target adds:

```json
{"previous_stage_objective": "..."}
```

The prompt explicitly tells the VLM that supplied state text is a weak temporal cue and that the last valid recent frame
should override stale or contradictory state text. This keeps the ablation comparable to `memer_objective`: same videos,
same core MEMER target JSON, plus one state-maintenance output.

Data source by protocol:

| Protocol | Prompt context line | Extra target field | Training source | Rollout source |
| --- | --- | --- | --- | --- |
| `objective_memory_state` | `Completed-subtasks memory: ...` | `updated_language_memory` | teacher-forced `sample.language_memory` -> `sample.updated_language_memory` | runtime `language_memory`; next state uses predicted `updated_language_memory` |
| `objective_last_objective` | none | `last_objective` | teacher-forced previous sample objective in the same episode | auxiliary output only; rollout can ignore it |
| `objective_prev_stage` | `Previous stage objective: ...` | `previous_stage_objective` | oracle previous distinct objective in the same episode | self-maintained from predicted `previous_stage_objective`, with transition fallback |

Important limitation: `last_objective` and `previous_stage_objective` can be derived from old samples because they depend
only on `current_objective` order. `language_memory` cannot be repaired this way; if the stored memory text is stale, for
example every row says `No completed subtask yet`, the dataset must be re-exported.

Interpretation:

- `objective_memory_state` is the closest to a real stateful rollout if the runtime memory is maintained outside the
  VLM or by the model's `updated_language_memory`.
- `objective_last_objective` is an auxiliary-output ablation. It forces the model to reconstruct the previous objective
  during training, but the field is not needed by rollout.
- `objective_prev_stage` is a compact self-maintained state ablation. Training is still teacher-forced/oracle; rollout
  quality depends on whether the model can keep the predicted previous-stage field stable.

## Coarse Label Sidecar

The current fine subtasks are often too granular for 1 Hz rollout and strict semantic evaluation. Generate a coarse annotation JSONL next to existing normalized annotations:

```bash
PYTHONPATH=src python scripts/hl_memory/coarsen_hl_annotations.py \
  --annotation-root /root/Users/dataset/lerobot_home/subtask \
  --input-name hl_annotations_llm_normalized.jsonl \
  --output-name hl_annotations_llm_normalized_coarse.jsonl \
  --merge-mode conservative \
  --short-run-merge-max-frames 35 \
  --overwrite \
  --summary-json /root/Users/dataset/lerobot_home/subtask/batch_coarse_hl_annotations_summary.json
```

Then export HL datasets from the coarse labels:

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

The coarse JSONL preserves original fields as `fine_current_objective`, `fine_current_subtask`, and `fine_horizon_*`, then replaces train-facing `current_objective/current_subtask/horizon_*` with coarse objectives. Existing `--target-protocol memer_objective` can train directly on this data.

Default merge behavior:

| Fine pattern | Coarse behavior |
| --- | --- |
| approach -> grasp | approach rows are relabeled toward acquire/grasp |
| approach/grasp handle -> open/close | handle acquisition is relabeled toward the articulation objective |
| same-object short wording variants | compatible short runs are merged and prefer the shorter/general label, e.g. `Fold packaging box sides` -> `Fold packaging box` |
| short staging transport -> place | `move above/over <support>` is merged into the following place objective, e.g. `Move above plate` -> `Place leaf vegetable on steak` |
| bottle/cap rotate/twist | preserved; do not short-run merge directional operation fragments |
| return/retreat | preserved as executable reset objectives; add `--merge-return-into-previous` only for a special ablation |

Use `--merge-mode aggressive` for the extra `grasp/transport -> place` rule. It reduces phase fragmentation more aggressively, but can over-label intermediate transport as final placement. The default `conservative` mode is the recommended first run.

This is intentionally heuristic. Inspect a few generated JSONL rows before launching a full run. The output keeps object grounding in `current_objective`; `coarse_action_type` is metadata, not the text target.

Latest conservative coarse stats should be regenerated after changing the merge threshold or staging-placement rule. Older 20-frame runs left some `move above plate -> place ...` boundaries unmerged; the current recommended 35-frame pass is intended to remove those timing-only boundaries before MEMER training.

## Architecture Additions To Consider

| Addition | Status | Priority | Notes |
| --- | --- | ---: | --- |
| Proprio `per_frame` tokens | implemented | high | One state token per recent timestep |
| Proprio `summary` token | implemented | medium | State-only learned-query summary |
| Proprio `per_frame_plus_summary` | implemented | high | Current default |
| Full-context summary token | planned | medium | A learned token after RGB/text/state context; output attends to it |
| Memory auxiliary target | planned | high | Keep memory out of input but supervise `task_progress` output |
| Recurrent memory token | planned/off by default | low for now | Requires rollout-style state carry, reset rules, and careful train/inference matching |

Recurrent tokens are not the next debugging step. They add a hidden state path whose semantics depend on target protocol, proprio mode, and rollout reset behavior. Keep this disabled until coarse labels and pure-model MEMER/proprio settings are understood.

## Setting-Independent Issues

| Issue | Cause | Current handling |
| --- | --- | --- |
| Low exact-match eval despite semantically correct outputs | Text can be semantically equivalent but not string-identical | Track normalized/semantic metrics, not only exact match |
| Metadata showed Qwen2.5 for Qwen3.5 experiments | Old export metadata serialized default `HLMemoryConfig` even though export does not load a VLM | Export metadata now writes only data-related `export_config` |
| FSDP per rank still reports about 1.126B params | Frozen/tied `embed_tokens` and `lm_head` are replicated to avoid FSDP flatten shape bugs; most other params are sharded | Keep stable path unless memory becomes blocker; more aggressive sharding needs a separate experiment |
| Large grad accumulation makes each optimizer step slow | One displayed `it` is an optimizer step containing many micro-steps; FSDP does not use `no_sync()` to preserve memory | Prefer increasing per-rank batch if memory allows, e.g. `batch=2, accum=4` instead of `batch=1, accum=8` |
| Multi-node training can be slower than single-node | Cross-node FSDP communication overhead; TCP is slow and RoCE config may be unstable | Use single-node unless RDMA/NCCL `NET/IB/GDRDMA` is stable and step time improves clearly |
| RoCE is visible but NCCL fails before step 1 | NCCL sees `NET/IB/GDRDMA`, but control/RDMA address or GID configuration is inconsistent | For formal training, fall back to single-node/TCP; for debugging, test one HCA and platform-provided `NCCL_IB_GID_INDEX` |

## K086A Qualitative Rollout Notes

These observations are from single-session K086A rollout videos/summaries around step 1400-2000. Treat them as qualitative hypotheses, not final metrics.

| Family | Setting | Qualitative behavior | Interpretation | Follow-up |
| --- | --- | --- | --- | --- |
| MEMER objective | No prior, no proprio | Better than expected, but the first toast placement can lag by several rollout steps and later steps still show local instability | Simplified objective helps, but image-only temporal grounding is still imperfect | Keep as MEMER baseline |
| MEMER objective | Proprio, no prior | Best qualitative behavior in this batch; fewer unstable transitions than MEMER baseline | Proprio likely provides useful hand-state phase information beyond RGB | Validate with batch semantic eval |
| MEMER objective | Prior + proprio | Sometimes hallucinates future subtasks earlier than the video supports | Ordered plan can become a shortcut; prior helps constrain labels but may bias toward future steps | Compare with no-prior using pure-model rollout, not known-prior override |
| Subtask keyframe | No prior, no proprio | Reaches a coarse "roughly right" level; sometimes comparable to MEMER baseline on visible progress | Current-objective-only target is usable but less robust around transitions | Keep as subtask baseline |
| Subtask keyframe | Proprio, no prior | Some early transitions improve, but later can get stuck/repeat object-state phases | Proprio helps but target protocol still lacks horizon smoothing | Compare against MEMER proprio/no-prior |
| Subtask keyframe | Prior or prior + proprio | Video can look more stable, but prior/state-machine or plan context can mask raw model mistakes | Do not mix with pure model comparisons unless known-prior override is disabled | Report separately as system-level rollout |

Current working hypothesis:

- `memer_objective` is likely better than `subtask_keyframe` for this setting because it asks for a shorter, less entangled target and includes short-horizon intent.
- Proprio is likely beneficial because hand pose/state helps disambiguate phase changes that are subtle in fisheye RGB.
- Step prior currently has negative or mixed effect for pure model quality because it can create future-step shortcuts. It should be evaluated separately as prompt context and as a rule-based system component.

For batch evaluation, exact string match is not sufficient. Use a semantic judge, e.g. Qwen3.5-27B, to decide whether each predicted objective is equivalent to the GT subtask/objective at the same timestamp. Recommended metrics:

| Metric | Purpose |
| --- | --- |
| Semantic current-objective accuracy | Main metric for `subtask_keyframe` and MEMER current objective |
| Semantic horizon-objective accuracy | Main MEMER horizon metric |
| Transition timing error | Measures early/late switching rather than only per-frame accuracy |
| Future-step hallucination rate | Counts predictions that semantically match a later prior step before GT transition |
| Backtracking/jitter count | Counts A -> B -> A local instability |
| Keyframe precision/recall | Evaluates whether predicted keyframes are useful, not just objective text |

When comparing prior-trained checkpoints, run two protocols:

| Protocol | Command behavior | Meaning |
| --- | --- | --- |
| Pure model | Provide session/task plan as input, but do not enable `--known-prior-mode` | Measures the checkpoint's raw prediction ability under its training prompt distribution |
| System rollout | Enable `--known-prior-mode` for protocols that support it | Measures model plus rule state machine; not directly comparable to pure model results |

## Qwen3.5-27B Semantic Judge

Use `scripts/hl_memory/semantic_judge_hl_memory_predictions.py` to evaluate rollout summaries or eval prediction JSONL files with a semantic label judge instead of exact string match.

The default `--judge-method score` does not ask Qwen to freely generate JSON. It scores a fixed label set by log-likelihood and selects the best label:

`EQUIVALENT`, `TOO_EARLY`, `TOO_LATE`, `WRONG_ACTION`, `WRONG_OBJECT`, `WRONG_LOCATION`, `WRONG_HAND`, `UNDERSPECIFIED`, `UNRELATED`.

Example for one rollout summary:

```bash
cd /lumos-vePFS/suzhou/Users/lixiaotong/openpi
mkdir -p /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge

CUDA_VISIBLE_DEVICES=0 /root/Users/miniconda3/envs/pi0_suzhou/bin/python \
  scripts/hl_memory/semantic_judge_hl_memory_predictions.py \
  --input-json /root/Users/lixiaotong/HL_MEM_rollout/20260328K086A_proprio_no_prior_memer_002000/summary.json \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --torch-dtype bfloat16 \
  --batch-size 4 \
  --judge-method score \
  --output-json /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/20260328K086A_proprio_no_prior_memer_002000.semantic.json \
  --output-jsonl /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/20260328K086A_proprio_no_prior_memer_002000.semantic.jsonl \
  --output-md /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/20260328K086A_proprio_no_prior_memer_002000.semantic.md
```

Example for multiple summaries:

```bash
CUDA_VISIBLE_DEVICES=0 /root/Users/miniconda3/envs/pi0_suzhou/bin/python \
  scripts/hl_memory/semantic_judge_hl_memory_predictions.py \
  --input-glob '/root/Users/lixiaotong/HL_MEM_rollout/20260328K086A_*_[0-9]*/summary.json' \
  --model-path /root/Users/lixiaotong/Qwen3.5-27B \
  --device-map auto \
  --torch-dtype bfloat16 \
  --batch-size 4 \
  --judge-method score \
  --output-json /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/k086a_all.semantic.json \
  --output-jsonl /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/k086a_all.semantic.jsonl \
  --output-md /root/Users/lixiaotong/HL_MEM_rollout/semantic_judge/k086a_all.semantic.md
```

Notes:

- Use an idle machine/GPU. Qwen3.5-27B can occupy one 80GB GPU or be automatically split by `device_map=auto`.
- `score` mode is preferred over `generate` mode because it is deterministic and cannot produce malformed JSON.
- The judge is intentionally strict about action, object, location, hand, and phase. A low absolute score can still be useful if all settings are judged with the same protocol.
- Inspect the `.semantic.md` failure examples before treating a run-level number as final.

Initial K086A selected-run result, using Qwen3.5-27B score-mode judge on 260 rollout steps:

| Setting | Rows | Semantic accuracy | Top errors |
| --- | ---: | ---: | --- |
| `20260328K086A_proprio_no_prior_memer_002000` | 52 | 0.327 | `wrong_action`, `too_early`, `wrong_object` |
| `20260328K086A_prior_proprio_memer_001800` | 52 | 0.308 | `wrong_action`, `too_early`, `wrong_object` |
| `20260328K086A_prior_no_proprio_002000` | 52 | 0.231 | `wrong_action` |
| `20260328K086A_proprio_no_prior_001800` | 52 | 0.192 | `wrong_action`, `wrong_object`, `too_early` |
| `20260328K086A_prior_proprio_001400` | 52 | 0.077 | `wrong_action`, `wrong_object` |

This supports the current qualitative direction that `memer_objective` is stronger than `subtask_keyframe` on this session, and proprio helps in the MEMER protocol. The absolute numbers are lower than human visual impression because the judge treats phase mismatches such as `approach` vs `hold` or `move above plate` vs `place` as errors.
