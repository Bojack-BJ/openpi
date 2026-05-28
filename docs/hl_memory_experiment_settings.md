# HL Memory Experiment Settings

This document tracks the main HL-memory data/training settings tried so far. Each row is one setting. Use it as a compact experiment ledger when comparing future runs.

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

## Setting-Independent Issues

| Issue | Cause | Current handling |
| --- | --- | --- |
| Low exact-match eval despite semantically correct outputs | Text can be semantically equivalent but not string-identical | Track normalized/semantic metrics, not only exact match |
| Metadata showed Qwen2.5 for Qwen3.5 experiments | Old export metadata serialized default `HLMemoryConfig` even though export does not load a VLM | Export metadata now writes only data-related `export_config` |
| FSDP per rank still reports about 1.126B params | Frozen/tied `embed_tokens` and `lm_head` are replicated to avoid FSDP flatten shape bugs; most other params are sharded | Keep stable path unless memory becomes blocker; more aggressive sharding needs a separate experiment |
| Large grad accumulation makes each optimizer step slow | One displayed `it` is an optimizer step containing many micro-steps; FSDP does not use `no_sync()` to preserve memory | Prefer increasing per-rank batch if memory allows, e.g. `batch=2, accum=4` instead of `batch=1, accum=8` |
| Multi-node training can be slower than single-node | Cross-node FSDP communication overhead; TCP is slow and RoCE config may be unstable | Use single-node unless RDMA/NCCL `NET/IB/GDRDMA` is stable and step time improves clearly |
| RoCE is visible but NCCL fails before step 1 | NCCL sees `NET/IB/GDRDMA`, but control/RDMA address or GID configuration is inconsistent | For formal training, fall back to single-node/TCP; for debugging, test one HCA and platform-provided `NCCL_IB_GID_INDEX` |

