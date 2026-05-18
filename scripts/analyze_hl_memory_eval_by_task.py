from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--memory-sim-threshold", type=float, default=0.8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions_jsonl.open() if line.strip()]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    summary = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "num_tasks": len(by_task),
        "num_samples": len(rows),
        "tasks": [],
    }
    lines = [
        "# HL Memory Full Evaluation Summary",
        "",
        f"- predictions: `{args.predictions_jsonl}`",
        f"- tasks: {len(by_task)}",
        f"- samples: {len(rows)}",
        f"- pass rule: `subtask_normalized_match == 1` and `phase_accuracy == 1`",
        "",
        "| task | pass | total | pass_rate | top3 reasons |",
        "|---|---:|---:|---:|---|",
    ]

    for task_id in sorted(by_task):
        task_rows = by_task[task_id]
        pass_count = sum(_is_pass(row) for row in task_rows)
        reason_counter: Counter[str] = Counter()
        examples: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in task_rows:
            for reason in _failure_reasons(row, memory_sim_threshold=args.memory_sim_threshold):
                reason_counter[reason] += 1
                if len(examples[reason]) < 3:
                    examples[reason].append(_compact_example(row))
        top3 = reason_counter.most_common(3)
        top_text = "<br>".join(f"{reason}: {count}" for reason, count in top3) or "-"
        lines.append(
            f"| {task_id} | {pass_count} | {len(task_rows)} | {pass_count / max(len(task_rows), 1):.3f} | {top_text} |"
        )
        summary["tasks"].append(
            {
                "task_id": task_id,
                "pass_count": pass_count,
                "total": len(task_rows),
                "pass_rate": pass_count / max(len(task_rows), 1),
                "top_reasons": [
                    {"reason": reason, "count": count, "examples": examples[reason]}
                    for reason, count in top3
                ],
            }
        )

    lines.extend(["", "## Top Reason Examples", ""])
    for task in summary["tasks"]:
        lines.append(f"### {task['task_id']}")
        for reason in task["top_reasons"]:
            lines.append(f"- {reason['reason']} ({reason['count']})")
            for example in reason["examples"]:
                lines.append(
                    "  - "
                    f"ep={example['episode_index']} step={example['step_index']} "
                    f"expected=`{example['expected_subtask']}` predicted=`{example['predicted_subtask']}` "
                    f"mem_sim={example['memory_similarity']:.3f}"
                )
        lines.append("")

    total_pass = sum(task["pass_count"] for task in summary["tasks"])
    summary["pass_count"] = total_pass
    summary["pass_rate"] = total_pass / max(len(rows), 1)
    lines.insert(5, f"- pass: {total_pass}/{len(rows)} ({summary['pass_rate']:.3f})")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({"output_md": str(args.output_md), "output_json": str(args.output_json) if args.output_json else None, "pass_rate": summary["pass_rate"]}, indent=2))


def _is_pass(row: dict) -> bool:
    metrics = row.get("metrics", {})
    return bool(metrics.get("subtask_normalized_match") == 1.0 and metrics.get("phase_accuracy") == 1.0)


def _failure_reasons(row: dict, *, memory_sim_threshold: float) -> list[str]:
    metrics = row.get("metrics", {})
    expected = row.get("expected", {})
    prediction = row.get("prediction", {})
    reasons: list[str] = []
    if metrics.get("subtask_normalized_match") != 1.0:
        reasons.append(_subtask_reason(str(expected.get("current_subtask", "")), str(prediction.get("current_subtask", ""))))
    if metrics.get("phase_accuracy") != 1.0:
        reasons.append("phase_mismatch")
    if metrics.get("event_accuracy") == 0.0:
        reasons.append("event_not_preserved_in_memory")
    if float(metrics.get("language_memory_similarity", 1.0)) < memory_sim_threshold:
        reasons.append("low_language_memory_similarity")
    if float(metrics.get("keyframe_precision", 1.0)) < 1.0 or float(metrics.get("keyframe_recall", 1.0)) < 1.0:
        reasons.append("keyframe_mismatch")
    return reasons


def _subtask_reason(expected: str, predicted: str) -> str:
    exp = expected.lower()
    pred = predicted.lower()
    if _hand(exp) != _hand(pred):
        return "subtask_mismatch_hand_or_actor"
    if _object_hint(exp) != _object_hint(pred):
        return "subtask_mismatch_object"
    if _verb_hint(exp) != _verb_hint(pred):
        return "subtask_mismatch_action_stage"
    return "subtask_mismatch_text_or_granularity"


def _hand(text: str) -> str:
    left = "left hand" in text
    right = "right hand" in text
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"


def _object_hint(text: str) -> str:
    for word in (
        "light bulb",
        "packaging box",
        "remote control",
        "cabinet",
        "cable",
        "box",
        "lid",
        "bottle",
        "cap",
        "block",
    ):
        if word in text:
            return word
    return "unknown"


def _verb_hint(text: str) -> str:
    for word in (
        "approach",
        "move close",
        "grasp",
        "pick",
        "hold",
        "push",
        "pull",
        "place",
        "return",
        "fold",
        "cover",
    ):
        if word in text:
            return word
    return "unknown"


def _compact_example(row: dict) -> dict[str, object]:
    metrics = row.get("metrics", {})
    return {
        "episode_index": row.get("episode_index"),
        "step_index": row.get("step_index"),
        "sample_id": row.get("sample_id"),
        "expected_subtask": row.get("expected", {}).get("current_subtask", ""),
        "predicted_subtask": row.get("prediction", {}).get("current_subtask", ""),
        "expected_phase": row.get("expected", {}).get("phase", ""),
        "predicted_phase": row.get("prediction", {}).get("phase", ""),
        "memory_similarity": float(metrics.get("language_memory_similarity", 0.0)),
    }


if __name__ == "__main__":
    main()
