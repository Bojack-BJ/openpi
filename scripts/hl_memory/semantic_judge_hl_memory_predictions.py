from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
import glob
import json
from pathlib import Path
import re
from typing import Any

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for minimal envs.
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else []

JUDGE_LABELS = (
    "EQUIVALENT",
    "TOO_EARLY",
    "TOO_LATE",
    "WRONG_ACTION",
    "WRONG_OBJECT",
    "WRONG_LOCATION",
    "WRONG_HAND",
    "UNDERSPECIFIED",
    "UNRELATED",
)

LABEL_ALIASES = {
    "EQUIVALENT": "equivalent",
    "MATCH": "equivalent",
    "SAME": "equivalent",
    "TOO_EARLY": "too_early",
    "EARLY": "too_early",
    "FUTURE": "too_early",
    "TOO_LATE": "too_late",
    "LATE": "too_late",
    "PAST": "too_late",
    "WRONG_ACTION": "wrong_action",
    "WRONG_OBJECT": "wrong_object",
    "WRONG_LOCATION": "wrong_location",
    "WRONG_HAND": "wrong_hand",
    "UNDERSPECIFIED": "underspecified",
    "VAGUE": "underspecified",
    "UNRELATED": "unrelated",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use a text LLM such as Qwen3.5-27B to semantically judge HL-memory objective predictions. "
            "Supports rollout summary JSON/list files and eval prediction JSONL files."
        )
    )
    parser.add_argument("--input-json", type=Path, nargs="*", default=[], help="Rollout summary JSON files.")
    parser.add_argument("--input-jsonl", type=Path, nargs="*", default=[], help="Prediction JSONL files.")
    parser.add_argument("--input-glob", type=str, nargs="*", default=[], help="Extra glob(s) for JSON/JSONL inputs.")
    parser.add_argument("--model-path", type=Path, default=Path("/root/Users/lixiaotong/Qwen3.5-27B"))
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="HF device_map; use 'none' to call model.to(cuda/cpu).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--judge-method",
        choices=("score", "generate"),
        default="score",
        help="score ranks fixed labels by log-likelihood; generate asks the model to emit one label.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--judge-horizon", action="store_true", help="Also judge horizon objective when expected exists.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Extract rows and prompts but do not load the judge model.")
    args = parser.parse_args()

    paths = _resolve_input_paths(args)
    rows = _load_judge_rows(paths, judge_horizon=args.judge_horizon)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No judgeable rows found. Check that inputs contain GT/expected objective text.")

    if args.dry_run:
        judged_rows = [{**row, "judge": {"match": None, "error_type": "dry_run", "confidence": 0.0}} for row in rows]
    else:
        tokenizer, model = _load_model(args)
        judged_rows = _judge_rows(rows, tokenizer, model, args)

    summary = _summarize(judged_rows, paths=paths, model_path=args.model_path)
    rendered = json.dumps(summary, indent=2, ensure_ascii=True)
    print(rendered)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered + "\n")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as handle:
            for row in judged_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(summary, judged_rows) + "\n")


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths = [*args.input_json, *args.input_jsonl]
    for pattern in args.input_glob:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    if not unique:
        raise ValueError("Pass at least one --input-json, --input-jsonl, or --input-glob.")
    return unique


def _load_judge_rows(paths: Iterable[Path], *, judge_horizon: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open() as handle:
                for line_index, line in enumerate(handle):
                    if line.strip():
                        rows.extend(_rows_from_record(json.loads(line), source_path=path, line_index=line_index, judge_horizon=judge_horizon))
        else:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                records = data
                metadata: dict[str, Any] = {}
            elif isinstance(data, dict):
                records = data.get("steps") or data.get("predictions") or []
                metadata = data
            else:
                raise ValueError(f"Unsupported JSON root in {path}: {type(data).__name__}")
            for step_index, record in enumerate(records):
                rows.extend(
                    _rows_from_record(
                        record,
                        source_path=path,
                        line_index=step_index,
                        judge_horizon=judge_horizon,
                        metadata=metadata,
                    )
                )
    return rows


def _rows_from_record(
    record: dict[str, Any],
    *,
    source_path: Path,
    line_index: int,
    judge_horizon: bool,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = metadata or {}
    expected = _as_dict(record.get("expected"))
    prediction = _as_dict(record.get("prediction")) or _as_dict(record.get("model_prediction"))
    if not prediction:
        prediction = _maybe_parse_prediction(record.get("prediction")) or _maybe_parse_prediction(record.get("model_prediction"))
    if not prediction:
        return []
    task_id = str(record.get("task_id") or metadata.get("task_id") or _task_id_from_path(source_path) or "")
    base = {
        "source_path": str(source_path),
        "source_row": line_index,
        "task_id": task_id,
        "sample_id": record.get("sample_id"),
        "episode_index": record.get("episode_index"),
        "step_index": record.get("step_index"),
        "recent_end_sec": record.get("recent_end_sec"),
        "instruction": record.get("instruction") or metadata.get("instruction") or "",
        "known_prior_mode": record.get("known_prior_mode", metadata.get("known_prior_mode")),
        "proprio_enabled": record.get("proprio_enabled", metadata.get("proprio_enabled")),
        "target_protocol": metadata.get("target_protocol"),
    }

    gt_current = _first_nonempty(
        expected.get("current_objective"),
        expected.get("current_subtask"),
        record.get("ground_truth_objective"),
        record.get("ground_truth_subtask"),
        record.get("gt_current_objective"),
        record.get("gt_subtask"),
    )
    pred_current = _first_nonempty(prediction.get("current_objective"), prediction.get("current_subtask"))
    rows: list[dict[str, Any]] = []
    if gt_current and pred_current:
        rows.append(
            {
                **base,
                "field": "current_objective",
                "expected_text": gt_current,
                "predicted_text": pred_current,
                "horizon": False,
            }
        )

    if judge_horizon:
        gt_horizon = _first_nonempty(expected.get("horizon_current_objective"), record.get("ground_truth_horizon_objective"))
        pred_horizon = _first_nonempty(prediction.get("horizon_current_objective"))
        if gt_horizon and pred_horizon:
            rows.append(
                {
                    **base,
                    "field": "horizon_current_objective",
                    "expected_text": gt_horizon,
                    "predicted_text": pred_horizon,
                    "horizon": True,
                }
            )
    return rows


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_parse_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = _extract_json(text)
    except ValueError:
        return {"current_objective": text}
    return parsed


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "none":
            return text
    return ""


def _task_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"\d{8}[A-Z]\d{3}[A-Z]?", part):
            return part
    return None


def _load_model(args: argparse.Namespace):
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install torch and transformers to run semantic judge eval.") from exc

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    config_path = args.model_path / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config = {}
        arch_text = " ".join(str(item) for item in config.get("architectures", []))
        if "ConditionalGeneration" in arch_text and hasattr(transformers, "AutoModelForImageTextToText"):
            model_cls = getattr(transformers, "AutoModelForImageTextToText")
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        raise RuntimeError("Could not find a compatible Hugging Face AutoModel class.")
    kwargs = {"trust_remote_code": True}
    if str(args.device_map).lower() not in {"none", "null", "false", ""}:
        kwargs["device_map"] = args.device_map
    try:
        model = model_cls.from_pretrained(str(args.model_path), dtype=dtype, **kwargs)
    except TypeError:
        model = model_cls.from_pretrained(str(args.model_path), torch_dtype=dtype, **kwargs)
    if "device_map" not in kwargs:
        model.to(args.device)
    model.eval()
    return tokenizer, model


def _judge_rows(rows: list[dict[str, Any]], tokenizer: Any, model: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    judged: list[dict[str, Any]] = []
    for batch in tqdm(list(_chunks(rows, args.batch_size)), desc="Semantic judge", unit="batch"):
        prompts = [_build_prompt(row) for row in batch]
        if args.judge_method == "score":
            scored = _score_label_many(tokenizer, model, prompts)
            for row, score_result in zip(batch, scored, strict=True):
                label = str(score_result["label"])
                judge = _judge_from_label(label, confidence=float(score_result["confidence"]))
                judged.append({**row, "judge": judge, "raw_judge_output": label, "label_scores": score_result["scores"]})
        else:
            responses = _generate_many(tokenizer, model, prompts, max_new_tokens=args.max_new_tokens)
            for row, response in zip(batch, responses, strict=True):
                judge = _parse_judge_response(response)
                judged.append({**row, "judge": judge, "raw_judge_output": response})
    return judged


def _chunks(items: list[dict[str, Any]], size: int):
    if size <= 0:
        raise ValueError("--batch-size must be positive.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_prompt(row: dict[str, Any]) -> str:
    return (
        f"Task instruction: {row.get('instruction') or 'unspecified'}\n"
        f"Field: {row.get('field')}\n"
        f"Timestamp seconds: {row.get('recent_end_sec')}\n"
        f"Ground truth: {row['expected_text']}\n"
        f"Prediction: {row['predicted_text']}\n"
    )


def _generate_many(tokenizer: Any, model: Any, prompts: list[str], *, max_new_tokens: int) -> list[str]:
    import torch

    prompt_texts = [_format_judge_chat(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    prompt_token_length = int(inputs["input_ids"].shape[1])
    device = _model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    responses: list[str] = []
    for output in outputs:
        responses.append(tokenizer.decode(output[prompt_token_length:], skip_special_tokens=True).strip())
    return responses


def _score_label_many(tokenizer: Any, model: Any, prompts: list[str]) -> list[dict[str, Any]]:
    import torch

    base_prompts = [_format_judge_chat(tokenizer, prompt) for prompt in prompts]
    candidate_texts: list[str] = []
    candidate_meta: list[tuple[int, str, int]] = []
    for row_index, base_prompt in enumerate(base_prompts):
        base_len = len(tokenizer(base_prompt, add_special_tokens=False)["input_ids"])
        for label in JUDGE_LABELS:
            candidate_texts.append(f"{base_prompt}{label}")
            candidate_meta.append((row_index, label, base_len))

    inputs = tokenizer(candidate_texts, return_tensors="pt", padding=True)
    device = _model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

    per_row_scores: list[dict[str, float]] = [dict() for _ in prompts]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_width = int(input_ids.shape[1])
    for candidate_index, (row_index, label, base_len) in enumerate(candidate_meta):
        seq_len = int(attention_mask[candidate_index].sum().item())
        pad_offset = seq_width - seq_len if getattr(tokenizer, "padding_side", "right") == "left" else 0
        label_start = pad_offset + base_len
        label_end = pad_offset + seq_len
        token_log_probs: list[torch.Tensor] = []
        for token_pos in range(label_start, label_end):
            if token_pos <= 0:
                continue
            token_id = int(input_ids[candidate_index, token_pos].item())
            token_log_probs.append(log_probs[candidate_index, token_pos - 1, token_id])
        if not token_log_probs:
            score = float("-inf")
        else:
            score = float(torch.stack(token_log_probs).mean().item())
        per_row_scores[row_index][label] = score

    results: list[dict[str, Any]] = []
    for scores in per_row_scores:
        selected = max(scores.items(), key=lambda item: item[1])[0]
        score_tensor = torch.tensor([scores[label] for label in JUDGE_LABELS], dtype=torch.float32)
        confidence = float(torch.softmax(score_tensor, dim=0)[JUDGE_LABELS.index(selected)].item())
        results.append(
            {
                "label": selected,
                "confidence": confidence,
                "scores": {label: scores[label] for label in JUDGE_LABELS},
            }
        )
    return results


def _format_judge_chat(tokenizer: Any, prompt: str) -> str:
    system = (
        "You are a strict semantic judge for robot high-level objectives. "
        "Decide if the prediction means the same primitive as the ground truth at this timestamp. "
        "Ignore wording/synonyms, but be strict about action, object, location, hand, and phase. "
        "If it is a future step or past step, it is not a match. "
        "Return exactly one uppercase label and nothing else. Allowed labels: "
        "EQUIVALENT, TOO_EARLY, TOO_LATE, WRONG_ACTION, WRONG_OBJECT, WRONG_LOCATION, WRONG_HAND, "
        "UNDERSPECIFIED, UNRELATED."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{prompt}\nLABEL:"


def _model_input_device(model: Any):
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        for module in ("model.embed_tokens", "model", ""):
            device = hf_device_map.get(module)
            if device is not None and device != "disk":
                import torch

                return torch.device(device)
    return next(model.parameters()).device


def _parse_judge_response(text: str) -> dict[str, Any]:
    label = _parse_label_response(text)
    if label is not None:
        return _judge_from_label(label, confidence=1.0)
    try:
        data = _extract_json(text)
    except ValueError as exc:
        return {
            "match": False,
            "error_type": "parse_error",
            "confidence": 0.0,
            "rationale": f"{type(exc).__name__}: {exc}",
        }
    match_value = data.get("match", False)
    if isinstance(match_value, str):
        match = match_value.strip().lower() in {"true", "yes", "1", "equivalent", "match"}
    else:
        match = bool(match_value)
    error_type = str(data.get("error_type", "equivalent" if match else "unrelated")).strip() or "unrelated"
    if match:
        error_type = "equivalent"
    confidence = _safe_float(data.get("confidence"), default=0.0)
    rationale = str(data.get("rationale", "")).strip()
    return {
        "match": match,
        "error_type": error_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": rationale,
    }


def _parse_label_response(text: str) -> str | None:
    normalized = re.sub(r"[^A-Za-z_]+", " ", text).upper().strip()
    tokens = normalized.split()
    joined = "_".join(tokens[:2])
    for candidate in (normalized, joined, tokens[0] if tokens else ""):
        if candidate in LABEL_ALIASES:
            return LABEL_ALIASES[candidate]
    for token in tokens:
        if token in LABEL_ALIASES:
            return LABEL_ALIASES[token]
    return None


def _judge_from_label(label: str, *, confidence: float) -> dict[str, Any]:
    normalized = _parse_label_response(label) or str(label).strip().lower()
    return {
        "match": normalized == "equivalent",
        "error_type": normalized,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": "",
    }


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"no JSON object found in response: {text[:200]!r}") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge response JSON is not an object")
    return parsed


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _summarize(judged_rows: list[dict[str, Any]], *, paths: list[Path], model_path: Path) -> dict[str, Any]:
    by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judged_rows:
        by_field[str(row["field"])].append(row)
        by_source[str(row["source_path"])].append(row)
    return {
        "input_paths": [str(path) for path in paths],
        "judge_model_path": str(model_path),
        "num_rows": len(judged_rows),
        "overall": _aggregate(judged_rows),
        "by_field": {field: _aggregate(rows) for field, rows in sorted(by_field.items())},
        "by_source": {source: _aggregate(rows) for source, rows in sorted(by_source.items())},
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    matches = sum(1 for row in rows if bool(row.get("judge", {}).get("match")))
    error_counts = Counter(str(row.get("judge", {}).get("error_type", "missing")) for row in rows)
    mean_conf = sum(float(row.get("judge", {}).get("confidence", 0.0)) for row in rows) / max(total, 1)
    return {
        "total": total,
        "match_count": matches,
        "semantic_accuracy": matches / max(total, 1),
        "mean_confidence": mean_conf,
        "error_counts": dict(error_counts.most_common()),
    }


def _render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# HL Memory Semantic Judge Summary",
        "",
        f"- rows: `{summary['num_rows']}`",
        f"- judge model: `{summary['judge_model_path']}`",
        f"- overall semantic accuracy: `{summary['overall']['semantic_accuracy']:.3f}`",
        "",
        "| group | total | accuracy | top errors |",
        "| --- | ---: | ---: | --- |",
    ]
    for source, metrics in summary["by_source"].items():
        top_errors = ", ".join(f"{key}:{value}" for key, value in list(metrics["error_counts"].items())[:3])
        lines.append(f"| `{_source_display_name(source)}` | {metrics['total']} | {metrics['semantic_accuracy']:.3f} | {top_errors} |")
    failures = [row for row in rows if not bool(row.get("judge", {}).get("match"))]
    lines.extend(["", "## Failure Examples", ""])
    for row in failures[:30]:
        judge = row["judge"]
        lines.append(
            f"- `{_source_display_name(str(row['source_path']))}` step={row.get('step_index')} t={row.get('recent_end_sec')} "
            f"field={row.get('field')} error={judge.get('error_type')} conf={judge.get('confidence')}: "
            f"GT=`{row['expected_text']}` pred=`{row['predicted_text']}` reason={judge.get('rationale')}"
        )
    return "\n".join(lines)


def _source_display_name(source: str) -> str:
    path = Path(source)
    if path.name == "summary.json" and path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name


if __name__ == "__main__":
    main()
