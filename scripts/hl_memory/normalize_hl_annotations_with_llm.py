from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import importlib.util
import json
import math
import pathlib
from typing import Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - lightweight local help/dry-run environments.
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


PROMPT_VERSION = "hl_gt_normalizer_v3_task_sidecar"


@dataclasses.dataclass(frozen=True)
class SegmentInfo:
    episode_index: int
    segment_index: int
    raw_subtask: str
    start_frame: int
    end_frame: int


@dataclasses.dataclass(frozen=True)
class SegmentTemplate:
    current_objective: str
    relevant_objects: list[str]
    notes: str
    target_query: str
    goal_query: str
    active_hand: str
    raw_response: str
    parse_error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize rule-based HL annotations into structured GT fields using an offline LLM. "
            "The output is a deterministic sidecar JSONL consumed by export_hl_memory_dataset.py."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=pathlib.Path, help="Rule-based annotations.jsonl.")
    parser.add_argument("--output-jsonl", required=True, type=pathlib.Path, help="Normalized annotations JSONL.")
    parser.add_argument("--model-path", default="/root/Users/lixiaotong/Qwen3.5-27B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--granularity", choices=["task", "segment", "row"], default="task")
    parser.add_argument(
        "--sidecar-json",
        type=pathlib.Path,
        default=None,
        help="Task-level normalized segment sidecar. Default: <output-dir>/hl_segments_llm_sidecar.json.",
    )
    parser.add_argument("--memory-summary-mode", choices=["llm", "code"], default="llm")
    parser.add_argument("--advance-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    done = _read_done(args.output_jsonl) if args.resume else set()
    grouped = _group_by_episode(rows)
    tokenizer, model = _load_model(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as stream:
        normalize_rows(rows, args=args, tokenizer=tokenizer, model=model, done=done, stream=stream)


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    done: set[tuple[int, int, str]],
    stream: Any,
) -> None:
    if args.granularity == "row":
        _normalize_rows_per_row(rows, args=args, tokenizer=tokenizer, model=model, done=done, stream=stream)
        return
    if args.granularity == "task":
        _normalize_rows_by_task_sidecar(rows, args=args, tokenizer=tokenizer, model=model, done=done, stream=stream)
        return
    _normalize_rows_by_segment(rows, args=args, tokenizer=tokenizer, model=model, done=done, stream=stream)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    rows.sort(key=lambda row: (int(row["episode_index"]), int(row["frame_index"]), str(row.get("current_subtask", ""))))
    return rows


def _read_done(path: pathlib.Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {_row_key(row) for row in _read_jsonl(path)}


def _row_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["episode_index"]), int(row["frame_index"]), str(row.get("current_subtask", ""))


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
    return dict(grouped)


def _normalize_rows_per_row(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    done: set[tuple[int, int, str]],
    stream: Any,
) -> None:
    grouped = _group_by_episode(rows)
    for row in tqdm(rows, desc="Normalize HL annotations", unit="row"):
        key = _row_key(row)
        if key in done:
            continue
        prompt = _build_prompt(row, grouped.get(int(row["episode_index"]), []))
        raw_response = _generate(tokenizer, model, prompt, max_new_tokens=args.max_new_tokens)
        normalized, parse_error = _parse_normalized_response(raw_response)
        output = dict(row)
        if normalized is not None:
            output.update(normalized)
        output["llm_gt"] = normalized
        output["llm_model"] = str(args.model_path)
        output["prompt_version"] = "hl_gt_normalizer_v1_row"
        output["raw_response"] = raw_response
        output["parse_error"] = parse_error
        stream.write(json.dumps(output, ensure_ascii=False) + "\n")
        stream.flush()


def _normalize_rows_by_segment(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    done: set[tuple[int, int, str]],
    stream: Any,
) -> None:
    grouped = _group_by_episode(rows)
    for episode_index, episode_rows in tqdm(grouped.items(), desc="Normalize episodes", unit="episode"):
        sorted_rows = sorted(episode_rows, key=lambda row: (int(row["frame_index"]), str(row.get("event_type", ""))))
        segments = _infer_segments(sorted_rows, episode_index=episode_index)
        if not segments:
            continue
        row_to_segment = _assign_rows_to_segments(sorted_rows, segments)
        segment_templates = _normalize_segments(
            segments,
            episode_rows=sorted_rows,
            args=args,
            tokenizer=tokenizer,
            model=model,
        )
        progress_summaries = _build_progress_summaries(
            segments,
            segment_templates,
            args=args,
            tokenizer=tokenizer,
            model=model,
        )
        for row in sorted_rows:
            key = _row_key(row)
            if key in done:
                continue
            segment = row_to_segment[id(row)]
            template = segment_templates[segment.segment_index]
            normalized = _expand_row_from_segment(
                row,
                segment=segment,
                template=template,
                task_progress=progress_summaries.get(segment.segment_index, "No completed subtask yet."),
                advance_threshold=float(args.advance_threshold),
            )
            output = dict(row)
            output.update(normalized)
            output["llm_gt"] = normalized
            output["llm_model"] = str(args.model_path)
            output["prompt_version"] = PROMPT_VERSION
            output["raw_response"] = {
                "segment": template.raw_response,
                "task_progress": progress_summaries.get(f"{segment.segment_index}:raw_response", ""),
            }
            output["parse_error"] = template.parse_error
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()


def _normalize_rows_by_task_sidecar(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    done: set[tuple[int, int, str]],
    stream: Any,
) -> None:
    sidecar_path = _resolve_sidecar_path(args)
    sidecar = None
    if bool(args.resume and not getattr(args, "overwrite", False)) and sidecar_path.exists():
        sidecar = _load_task_sidecar(sidecar_path)
    if sidecar is None:
        sidecar = _build_task_sidecar(rows, args=args, tokenizer=tokenizer, model=model)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    templates = _templates_from_sidecar(sidecar)
    progress_summaries = _progress_summaries_from_sidecar(sidecar)
    canonical_by_subtask = _canonical_index_by_subtask(sidecar)
    grouped = _group_by_episode(rows)
    for episode_index, episode_rows in tqdm(grouped.items(), desc="Expand task sidecar", unit="episode"):
        sorted_rows = sorted(episode_rows, key=lambda row: (int(row["frame_index"]), str(row.get("event_type", ""))))
        segments = _infer_segments(sorted_rows, episode_index=episode_index)
        if not segments:
            continue
        row_to_segment = _assign_rows_to_segments(sorted_rows, segments)
        for row in sorted_rows:
            key = _row_key(row)
            if key in done:
                continue
            segment = row_to_segment[id(row)]
            canonical_index = _resolve_canonical_segment_index(
                segment,
                canonical_by_subtask=canonical_by_subtask,
                template_count=len(templates),
            )
            template = templates[canonical_index]
            normalized = _expand_row_from_segment(
                row,
                segment=segment,
                template=template,
                task_progress=progress_summaries.get(canonical_index, "No completed subtask yet."),
                advance_threshold=float(args.advance_threshold),
            )
            output = dict(row)
            output.update(normalized)
            output["llm_gt"] = normalized
            output["llm_model"] = str(args.model_path)
            output["prompt_version"] = PROMPT_VERSION
            output["llm_sidecar_json"] = str(sidecar_path)
            output["raw_response"] = {
                "segment": template.raw_response,
                "task_progress": progress_summaries.get(f"{canonical_index}:raw_response", ""),
            }
            output["parse_error"] = template.parse_error
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()


def _resolve_sidecar_path(args: argparse.Namespace) -> pathlib.Path:
    sidecar_json = getattr(args, "sidecar_json", None)
    if sidecar_json is not None:
        return pathlib.Path(sidecar_json).expanduser().resolve()
    output_jsonl = pathlib.Path(args.output_jsonl).expanduser().resolve()
    return output_jsonl.with_name("hl_segments_llm_sidecar.json")


def _build_task_sidecar(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    task_segments = _infer_task_segments(rows)
    if not task_segments:
        return {
            "prompt_version": PROMPT_VERSION,
            "segments": [],
            "progress_summaries": {"0": "No completed subtask yet."},
        }
    templates = _normalize_segments(
        task_segments,
        episode_rows=rows,
        args=args,
        tokenizer=tokenizer,
        model=model,
    )
    progress_summaries = _build_progress_summaries(
        task_segments,
        templates,
        args=args,
        tokenizer=tokenizer,
        model=model,
    )
    return {
        "prompt_version": PROMPT_VERSION,
        "model_path": str(args.model_path),
        "memory_summary_mode": str(args.memory_summary_mode),
        "segments": [
            {
                "segment_index": segment.segment_index,
                "raw_subtask": segment.raw_subtask,
                "current_objective": templates[segment.segment_index].current_objective,
                "relevant_objects": templates[segment.segment_index].relevant_objects,
                "notes": templates[segment.segment_index].notes,
                "target_query": templates[segment.segment_index].target_query,
                "goal_query": templates[segment.segment_index].goal_query,
                "active_hand": templates[segment.segment_index].active_hand,
                "raw_response": templates[segment.segment_index].raw_response,
                "parse_error": templates[segment.segment_index].parse_error,
            }
            for segment in task_segments
        ],
        "progress_summaries": {str(key): value for key, value in progress_summaries.items()},
    }


def _load_task_sidecar(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None
    return data


def _templates_from_sidecar(sidecar: dict[str, Any]) -> dict[int, SegmentTemplate]:
    templates: dict[int, SegmentTemplate] = {}
    for item in sidecar.get("segments", []):
        if not isinstance(item, dict):
            continue
        index = int(item.get("segment_index", len(templates)))
        templates[index] = SegmentTemplate(
            current_objective=str(item.get("current_objective", item.get("raw_subtask", ""))).strip(),
            relevant_objects=_parse_objects(item.get("relevant_objects", [])),
            notes=str(item.get("notes", "none")).strip() or "none",
            target_query=str(item.get("target_query", "")).strip(),
            goal_query=str(item.get("goal_query", "")).strip(),
            active_hand=str(item.get("active_hand", "")).strip(),
            raw_response=str(item.get("raw_response", "")),
            parse_error=str(item.get("parse_error", "")).strip() or None,
        )
    if not templates:
        templates[0] = SegmentTemplate(
            current_objective="continue the observed manipulation step",
            relevant_objects=[],
            notes="none",
            target_query="",
            goal_query="",
            active_hand="",
            raw_response="",
        )
    return templates


def _progress_summaries_from_sidecar(sidecar: dict[str, Any]) -> dict[int | str, str]:
    raw = sidecar.get("progress_summaries", {})
    summaries: dict[int | str, str] = {0: "No completed subtask yet."}
    if isinstance(raw, dict):
        for key, value in raw.items():
            text = str(value).strip() or "No completed subtask yet."
            try:
                summaries[int(str(key))] = text
            except ValueError:
                summaries[str(key)] = text
    return summaries


def _canonical_index_by_subtask(sidecar: dict[str, Any]) -> dict[str, int | None]:
    seen: dict[str, int | None] = {}
    for item in sidecar.get("segments", []):
        if not isinstance(item, dict):
            continue
        raw_subtask = str(item.get("raw_subtask", "")).strip()
        if not raw_subtask:
            continue
        normalized = raw_subtask.lower()
        index = int(item.get("segment_index", 0))
        seen[normalized] = None if normalized in seen else index
    return seen


def _resolve_canonical_segment_index(
    segment: SegmentInfo,
    *,
    canonical_by_subtask: dict[str, int | None],
    template_count: int,
) -> int:
    by_subtask = canonical_by_subtask.get(segment.raw_subtask.lower())
    if by_subtask is not None:
        return by_subtask
    return min(segment.segment_index, max(template_count - 1, 0))


def _infer_segments(rows: list[dict[str, Any]], *, episode_index: int) -> list[SegmentInfo]:
    segments: list[SegmentInfo] = []
    last_subtask = ""
    for row in rows:
        subtask = str(row.get("current_subtask", "")).strip()
        if not subtask:
            continue
        frame_index = int(row["frame_index"])
        is_new_segment = subtask != last_subtask or str(row.get("event_type", "")) == "subtask_boundary"
        if not segments or is_new_segment:
            segments.append(
                SegmentInfo(
                    episode_index=episode_index,
                    segment_index=len(segments),
                    raw_subtask=subtask,
                    start_frame=frame_index,
                    end_frame=frame_index + 1,
                )
            )
            last_subtask = subtask
    if not segments:
        return []
    max_frame = max(int(row["frame_index"]) for row in rows)
    resolved: list[SegmentInfo] = []
    for index, segment in enumerate(segments):
        end_frame = segments[index + 1].start_frame if index + 1 < len(segments) else max_frame + 1
        resolved.append(dataclasses.replace(segment, end_frame=max(end_frame, segment.start_frame + 1)))
    return resolved


def _infer_task_segments(rows: list[dict[str, Any]]) -> list[SegmentInfo]:
    grouped = _group_by_episode(rows)
    candidate_sequences: list[list[str]] = []
    for episode_index, episode_rows in grouped.items():
        segments = _infer_segments(
            sorted(episode_rows, key=lambda row: (int(row["frame_index"]), str(row.get("event_type", "")))),
            episode_index=episode_index,
        )
        sequence = [segment.raw_subtask for segment in segments if segment.raw_subtask]
        if sequence:
            candidate_sequences.append(sequence)
    if not candidate_sequences:
        return []
    canonical = max(candidate_sequences, key=lambda sequence: (len(sequence), -candidate_sequences.index(sequence)))
    return [
        SegmentInfo(
            episode_index=-1,
            segment_index=index,
            raw_subtask=subtask,
            start_frame=index,
            end_frame=index + 1,
        )
        for index, subtask in enumerate(canonical)
    ]


def _assign_rows_to_segments(rows: list[dict[str, Any]], segments: list[SegmentInfo]) -> dict[int, SegmentInfo]:
    assigned: dict[int, SegmentInfo] = {}
    segment_index = 0
    for row in rows:
        frame_index = int(row["frame_index"])
        while segment_index + 1 < len(segments) and frame_index >= segments[segment_index + 1].start_frame:
            segment_index += 1
        assigned[id(row)] = segments[segment_index]
    return assigned


def _normalize_segments(
    segments: list[SegmentInfo],
    *,
    episode_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
) -> dict[int, SegmentTemplate]:
    templates: dict[int, SegmentTemplate] = {}
    context = _compact_segment_context(episode_rows)
    instruction = _first_nonempty(str(row.get("instruction", "")).strip() for row in episode_rows) or "unspecified"
    for segment in tqdm(segments, desc=f"Normalize segments e{segments[0].episode_index}", unit="segment", leave=False):
        prompt = _build_segment_prompt(segment, instruction=instruction, segment_context=context)
        raw_response = _generate(tokenizer, model, prompt, max_new_tokens=args.max_new_tokens)
        templates[segment.segment_index] = _parse_segment_response(raw_response, fallback_subtask=segment.raw_subtask)
    return templates


def _build_progress_summaries(
    segments: list[SegmentInfo],
    templates: dict[int, SegmentTemplate],
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
) -> dict[int | str, str]:
    summaries: dict[int | str, str] = {0: "No completed subtask yet."}
    for segment in tqdm(segments[1:], desc=f"Summarize progress e{segments[0].episode_index}", unit="prefix", leave=False):
        completed_templates = [templates[index] for index in range(segment.segment_index)]
        if args.memory_summary_mode == "code":
            summaries[segment.segment_index] = _code_progress_summary(completed_templates)
            continue
        prompt = _build_progress_summary_prompt(completed_templates)
        raw_response = _generate(tokenizer, model, prompt, max_new_tokens=min(int(args.max_new_tokens), 256))
        parsed, parse_error = _parse_progress_summary_response(raw_response)
        summaries[segment.segment_index] = parsed or _code_progress_summary(completed_templates)
        if parse_error:
            summaries[f"{segment.segment_index}:parse_error"] = parse_error
        summaries[f"{segment.segment_index}:raw_response"] = raw_response
    return summaries


def _load_model(args: argparse.Namespace):
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install torch and transformers in the active environment to run LLM GT normalization."
        ) from exc

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        raise RuntimeError("Could not find a compatible Hugging Face AutoModel class.")
    model_kwargs = {"trust_remote_code": True}
    device_map = str(args.device_map).strip()
    if device_map and device_map.lower() not in {"none", "null", "false"}:
        if importlib.util.find_spec("accelerate") is None:
            raise ModuleNotFoundError(
                "`--device-map` requires accelerate. Install it with `python -m pip install accelerate`, "
                "or rerun with `--device-map none` if the model fits on one device."
            )
        model_kwargs["device_map"] = device_map
    try:
        model = model_cls.from_pretrained(str(args.model_path), dtype=dtype, **model_kwargs)
    except TypeError:
        model = model_cls.from_pretrained(str(args.model_path), torch_dtype=dtype, **model_kwargs)
    model.eval()
    return tokenizer, model


def _build_prompt(row: dict[str, Any], episode_rows: list[dict[str, Any]]) -> str:
    frame_index = int(row["frame_index"])
    completed = _completed_subtasks_before(row, episode_rows)
    segment_context = _compact_segment_context(episode_rows)
    return (
        "You normalize robot high-level annotations into stable training labels.\n"
        "Return exactly one JSON object and no extra text.\n"
        "Definitions:\n"
        "- task_progress: compact history summary of completed milestones or stable state before/at this sample. "
        "It must not contain the current action unless that action is already completed.\n"
        "- current_objective: one executable primitive for the low-level robot policy at this sample. "
        "Use active hand/object when present. Do not use passive state text such as 'the object is picked up'.\n"
        "- relevant_objects: JSON list of short object/location phrases needed for the objective.\n"
        "- notes: one short caution/spatial fact, or 'none'.\n"
        "- target_query: manipulated object or object part, noun phrase only.\n"
        "- goal_query: target location/slot/container, noun phrase only, or empty string.\n"
        "Required keys: task_progress, current_objective, relevant_objects, notes, target_query, goal_query.\n"
        "Do not leak the full future task flow into task_progress or current_objective.\n\n"
        f"Task instruction: {str(row.get('instruction', '')).strip() or 'unspecified'}\n"
        f"Episode segment context:\n{segment_context}\n\n"
        f"Completed subtasks before this sample: {completed or 'none'}\n"
        "If completed subtasks are listed, summarize them in task_progress. "
        "If none are listed, task_progress should be 'No completed subtask yet.' unless a stable completed state is explicitly visible in the event text.\n"
        f"Current raw subtask: {str(row.get('current_subtask', '')).strip()}\n"
        f"Current raw phase: {str(row.get('phase', '')).strip()}\n"
        f"Event type: {str(row.get('event_type', 'none')).strip()}\n"
        f"Event text: {str(row.get('event_text', '')).strip() or 'none'}\n"
        f"Existing target hint: {str(row.get('target_query', '')).strip() or 'none'}\n"
        f"Existing goal hint: {str(row.get('goal_query', '')).strip() or 'none'}\n"
    )


def _build_segment_prompt(segment: SegmentInfo, *, instruction: str, segment_context: str) -> str:
    return (
        "Normalize one robot subtask segment into stable training labels.\n"
        "Return exactly one JSON object and no extra text.\n"
        "Definitions:\n"
        "- current_objective: one short executable primitive for the low-level robot policy.\n"
        "- relevant_objects: JSON list of short object/location phrases needed for the objective.\n"
        "- notes: one short caution/spatial fact, or 'none'.\n"
        "- target_query: manipulated object or object part, noun phrase only.\n"
        "- goal_query: target location/slot/container, noun phrase only, or empty string.\n"
        "- active_hand: left, right, both, or empty string if unspecified.\n"
        "Required keys: current_objective, relevant_objects, notes, target_query, goal_query, active_hand.\n"
        "Do not include completed history or future task flow.\n\n"
        f"Task instruction: {instruction}\n"
        f"Episode segment context:\n{segment_context}\n\n"
        f"Current raw subtask: {segment.raw_subtask}\n"
    )


def _build_progress_summary_prompt(completed_templates: list[SegmentTemplate]) -> str:
    completed = [
        {
            "current_objective": template.current_objective,
            "target_query": template.target_query,
            "goal_query": template.goal_query,
            "active_hand": template.active_hand,
        }
        for template in completed_templates
    ]
    return (
        "Summarize completed robot subtasks into one compact task_progress sentence for a low-level policy.\n"
        "Return exactly one JSON object and no extra text: {\"task_progress\":\"...\"}.\n"
        "Rules: describe only completed history, do not mention future steps, avoid a long list, keep action-useful state.\n\n"
        f"Completed normalized subtasks:\n{json.dumps(completed, ensure_ascii=False)}\n"
    )


def _parse_segment_response(text: str, *, fallback_subtask: str) -> SegmentTemplate:
    try:
        data = _extract_json(text)
        current_objective = str(data.get("current_objective", data.get("current_subtask", ""))).strip()
        if not current_objective:
            raise ValueError("missing current_objective")
        return SegmentTemplate(
            current_objective=current_objective,
            relevant_objects=_parse_objects(data.get("relevant_objects", [])),
            notes=str(data.get("notes", "none")).strip() or "none",
            target_query=str(data.get("target_query", "")).strip(),
            goal_query=str(data.get("goal_query", "")).strip(),
            active_hand=str(data.get("active_hand", "")).strip(),
            raw_response=text,
        )
    except Exception as exc:  # noqa: BLE001
        return SegmentTemplate(
            current_objective=fallback_subtask,
            relevant_objects=[],
            notes="none",
            target_query="",
            goal_query="",
            active_hand="",
            raw_response=text,
            parse_error=f"{type(exc).__name__}: {exc}",
        )


def _parse_progress_summary_response(text: str) -> tuple[str | None, str | None]:
    try:
        data = _extract_json(text)
    except ValueError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    task_progress = str(data.get("task_progress", "")).strip()
    if not task_progress or task_progress.lower() == "none":
        return "No completed subtask yet.", None
    return task_progress, None


def _expand_row_from_segment(
    row: dict[str, Any],
    *,
    segment: SegmentInfo,
    template: SegmentTemplate,
    task_progress: str,
    advance_threshold: float,
) -> dict[str, Any]:
    subtask_progress = _subtask_progress(row, segment)
    should_advance = bool(str(row.get("event_type", "")) == "success" or subtask_progress >= advance_threshold)
    relevant_objects = template.relevant_objects or _parse_objects([template.target_query, template.goal_query])
    return {
        "task_progress": task_progress or "No completed subtask yet.",
        "current_objective": template.current_objective,
        "relevant_objects": relevant_objects,
        "notes": template.notes,
        "target_query": template.target_query,
        "goal_query": template.goal_query,
        "active_hand": template.active_hand,
        "subtask_progress": subtask_progress,
        "should_advance_objective": should_advance,
    }


def _subtask_progress(row: dict[str, Any], segment: SegmentInfo) -> float:
    denom = max(segment.end_frame - segment.start_frame, 1)
    value = (int(row["frame_index"]) - segment.start_frame) / denom
    if str(row.get("event_type", "")) == "success":
        value = 1.0
    return float(min(max(value, 0.0), 1.0))


def _code_progress_summary(completed_templates: list[SegmentTemplate]) -> str:
    if not completed_templates:
        return "No completed subtask yet."
    recent = completed_templates[-3:]
    rendered = "; ".join(template.current_objective for template in recent if template.current_objective)
    if len(completed_templates) > len(recent):
        return f"Earlier setup steps are complete. Recently completed: {rendered}."
    return f"Completed: {rendered}."


def _first_nonempty(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _completed_subtasks_before(row: dict[str, Any], episode_rows: list[dict[str, Any]]) -> list[str]:
    """Infer completed subtasks before a row from segment order and explicit success events.

    Default exported annotations often contain only subtask_boundary/progress events. In that case, using only
    success events makes every task_progress empty. A later segment boundary implies earlier unique subtasks have
    completed, so use first-seen segment order as the fallback completion signal.
    """

    current_subtask = str(row.get("current_subtask", "")).strip()
    frame_index = int(row["frame_index"])
    current_segment_start = frame_index
    if current_subtask:
        matching_frames = [
            int(item["frame_index"])
            for item in episode_rows
            if str(item.get("current_subtask", "")).strip() == current_subtask
            and int(item["frame_index"]) <= frame_index
        ]
        if matching_frames:
            current_segment_start = min(matching_frames)

    completed: list[str] = []
    seen: set[str] = set()
    for item in sorted(episode_rows, key=lambda value: (int(value["frame_index"]), str(value.get("event_type", "")))):
        subtask = str(item.get("current_subtask", "")).strip()
        if not subtask or subtask in seen or subtask == current_subtask:
            continue
        item_frame = int(item["frame_index"])
        explicit_success = item_frame < frame_index and str(item.get("event_type", "")) == "success"
        earlier_segment = item_frame < current_segment_start
        if explicit_success or earlier_segment:
            completed.append(subtask)
            seen.add(subtask)
    return completed


def _compact_segment_context(rows: list[dict[str, Any]]) -> str:
    items: list[str] = []
    seen: set[str] = set()
    for row in rows:
        subtask = str(row.get("current_subtask", "")).strip()
        if not subtask or subtask in seen:
            continue
        seen.add(subtask)
        items.append(f"- {subtask}")
    return "\n".join(items[:40]) or "- unspecified"


def _generate(tokenizer, model, prompt: str, *, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You produce strict JSON for robot training labels."},
        {"role": "user", "content": prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is not None:
        try:
            rendered = apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            rendered = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    inputs = tokenizer([rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    suffix = output_ids[:, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(suffix[0], skip_special_tokens=True).strip()


def _parse_normalized_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = _extract_json(text)
    except ValueError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    task_progress = str(data.get("task_progress", "")).strip()
    if not task_progress or task_progress.lower() == "none":
        task_progress = "No completed subtask yet."
    normalized = {
        "task_progress": task_progress,
        "current_objective": str(data.get("current_objective", data.get("current_subtask", ""))).strip(),
        "relevant_objects": _parse_objects(data.get("relevant_objects", [])),
        "notes": str(data.get("notes", "none")).strip() or "none",
        "target_query": str(data.get("target_query", "")).strip(),
        "goal_query": str(data.get("goal_query", "")).strip(),
    }
    if not normalized["current_objective"]:
        return None, "ValueError: missing current_objective"
    return normalized, None


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found.")


def _parse_objects(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        raw_items = value
    else:
        raw_items = str(value).replace(";", ",").split(",")
    objects: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if not text or text.lower() == "none":
            continue
        if text.lower() not in {existing.lower() for existing in objects}:
            objects.append(text)
    return objects


if __name__ == "__main__":
    main()
