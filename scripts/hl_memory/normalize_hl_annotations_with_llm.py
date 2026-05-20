from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pathlib
from typing import Any

import torch
from tqdm.auto import tqdm


PROMPT_VERSION = "hl_gt_normalizer_v1"


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
            output["prompt_version"] = PROMPT_VERSION
            output["raw_response"] = raw_response
            output["parse_error"] = parse_error
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()


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


def _load_model(args: argparse.Namespace):
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install transformers in the active environment to run LLM GT normalization.") from exc

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
    model = model_cls.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _build_prompt(row: dict[str, Any], episode_rows: list[dict[str, Any]]) -> str:
    frame_index = int(row["frame_index"])
    completed = [
        str(item.get("current_subtask", "")).strip()
        for item in episode_rows
        if int(item["frame_index"]) < frame_index and str(item.get("event_type", "")) == "success"
    ]
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
        f"Current raw subtask: {str(row.get('current_subtask', '')).strip()}\n"
        f"Current raw phase: {str(row.get('phase', '')).strip()}\n"
        f"Event type: {str(row.get('event_type', 'none')).strip()}\n"
        f"Event text: {str(row.get('event_text', '')).strip() or 'none'}\n"
        f"Existing target hint: {str(row.get('target_query', '')).strip() or 'none'}\n"
        f"Existing goal hint: {str(row.get('goal_query', '')).strip() or 'none'}\n"
    )


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
    normalized = {
        "task_progress": str(data.get("task_progress", "")).strip() or "No completed subtask yet.",
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
