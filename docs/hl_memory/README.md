# HL Memory Docs

This folder contains the HL-memory pipeline and protocol notes split by topic.

| Document | Purpose |
| --- | --- |
| [overview.md](overview.md) | System purpose, main fields, code entry points |
| [pipeline.md](pipeline.md) | Raw sessions -> LeRobot -> raw annotations -> LLM normalization -> LL sidecar |
| [dataset_export.md](dataset_export.md) | HL `samples.jsonl` export, visual layout, metadata, proprio export |
| [training.md](training.md) | Train commands, target protocols, FSDP/LoRA/vision finetuning notes |
| [eval_and_rollout.md](eval_and_rollout.md) | Eval, zero-shot rollout, known-prior behavior, debug outputs |
| [data_formats.md](data_formats.md) | Annotation/sample/memory/LL guidance formats |
| [schemas.md](schemas.md) | Compact schema contract for current MEMER/coarse-subtask work |
| [gt_protocol.md](gt_protocol.md) | Structured GT fields and LLM normalization protocol |
| [subtask_taxonomy.md](subtask_taxonomy.md) | 87-task fine subtask statistics and coarse atomic taxonomy |
| [coarse_subtasks.md](coarse_subtasks.md) | Coarse label generation rules and commands |
| [experiment_settings.md](experiment_settings.md) | Experiment matrix, current K086A settings, semantic judge notes |
| [memory_shortcut_ablation.md](memory_shortcut_ablation.md) | Memory shortcut failure modes, tested findings, and recent-primary ablations |

Backward-compatible top-level stubs:

- `docs/hl_memory_v1.md`
- `docs/hl_memory_experiment_settings.md`
- `docs/hl_memory_gt_protocol.md`
- `docs/hl_memory_subtask_taxonomy.md`
