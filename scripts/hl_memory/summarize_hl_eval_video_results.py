"""Summarize HL-memory eval/video result JSONs into tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections.abc import Mapping
from typing import Any


EVAL_METRICS = (
    "objective_normalized_match",
    "horizon_objective_normalized_match",
    "keyframe_f1",
    "completed_objective_precision",
    "completed_objective_recall",
    "language_memory_similarity",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--title", default="HL-memory eval/video summary")
    args = parser.parse_args()

    entries = json.loads(args.manifest_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = pathlib.Path(entry["path"])
        payload = json.loads(path.read_text())
        kind = entry["kind"]
        if kind == "eval":
            row = _summarize_eval(entry, payload)
        elif kind == "video":
            row = _summarize_video(entry, payload)
        else:
            raise ValueError(f"Unsupported result kind: {kind!r}")
        row["path"] = str(path)
        rows.append(row)

    _write_csv(args.output_dir / "summary_table.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.md").write_text(_render_markdown(args.title, rows), encoding="utf-8")
    _write_plots(args.output_dir, rows, title=args.title)


def _summarize_eval(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {}).get("sample_context", {})
    row = _base_row(entry)
    row.update({metric: _float_or_none(metrics.get(metric)) for metric in EVAL_METRICS})
    row["num_steps"] = _float_or_none(metrics.get("num_steps"))
    row["num_episodes"] = _float_or_none(metrics.get("num_episodes"))
    return row


def _summarize_video(entry: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    steps = payload.get("steps", [])
    accepted_count = 0
    parse_error_count = 0
    nonempty_completed_count = 0
    candidate_counts: list[int] = []
    completed_values: list[str] = []
    current_values: list[str] = []
    horizon_values: list[str] = []

    for step in steps:
        output = step.get("output", {}) if isinstance(step, Mapping) else {}
        diagnostics = step.get("diagnostics", {}) if isinstance(step, Mapping) else {}
        state_update = diagnostics.get("state_update", {}) if isinstance(diagnostics, Mapping) else {}
        candidates = output.get("keyframe_candidate_positions") or []
        completed = str(output.get("completed_objective") or "").strip()
        current = str(output.get("current_objective") or "").strip()
        horizon = str(output.get("horizon_current_objective") or "").strip()
        candidate_counts.append(len(candidates) if isinstance(candidates, list) else 0)
        if completed:
            nonempty_completed_count += 1
            completed_values.append(completed)
        if current:
            current_values.append(current)
        if horizon:
            horizon_values.append(horizon)
        if diagnostics.get("parse_error"):
            parse_error_count += 1
        if isinstance(state_update, Mapping) and state_update.get("accepted"):
            accepted_count += 1

    step_count = len(steps)
    row = _base_row(entry)
    row.update(
        {
            "video_steps": step_count,
            "accepted_event_count": accepted_count,
            "accepted_event_rate": accepted_count / step_count if step_count else 0.0,
            "parse_error_count": parse_error_count,
            "parse_error_rate": parse_error_count / step_count if step_count else 0.0,
            "nonempty_completed_count": nonempty_completed_count,
            "nonempty_completed_rate": nonempty_completed_count / step_count if step_count else 0.0,
            "avg_keyframe_candidates": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
            "unique_current_objectives": len(set(current_values)),
            "unique_horizon_objectives": len(set(horizon_values)),
            "unique_completed_objectives": len(set(completed_values)),
            "final_completed_event_log": payload.get("final_state", {}).get("completed_event_log", ""),
        }
    )
    return row


def _base_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": entry.get("task", ""),
        "granularity": entry.get("granularity", ""),
        "setting": entry.get("setting", ""),
        "kind": entry.get("kind", ""),
        "split": entry.get("split", ""),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(title: str, rows: list[dict[str, Any]]) -> str:
    eval_rows = [row for row in rows if row["kind"] == "eval"]
    video_rows = [row for row in rows if row["kind"] == "video"]
    lines = [f"# {title}", ""]
    lines += _markdown_table(
        "Eval metrics",
        eval_rows,
        [
            "task",
            "setting",
            "num_steps",
            "objective_normalized_match",
            "horizon_objective_normalized_match",
            "keyframe_f1",
            "completed_objective_precision",
            "completed_objective_recall",
            "language_memory_similarity",
        ],
    )
    lines += _markdown_table(
        "Video rollout metrics",
        video_rows,
        [
            "task",
            "setting",
            "video_steps",
            "accepted_event_count",
            "accepted_event_rate",
            "nonempty_completed_rate",
            "avg_keyframe_candidates",
            "parse_error_rate",
            "unique_current_objectives",
            "unique_horizon_objectives",
        ],
    )
    lines.extend(
        [
            "## Files",
            "",
            "- `summary_table.csv`: flat table for spreadsheets.",
            "- `summary.json`: machine-readable table.",
            "- `eval_metrics.png`: eval bar chart.",
            "- `video_metrics.png`: rollout bar chart.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        return lines + ["No rows.", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    return lines


def _write_plots(output_dir: pathlib.Path, rows: list[dict[str, Any]], *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    eval_rows = [row for row in rows if row["kind"] == "eval"]
    video_rows = [row for row in rows if row["kind"] == "video"]
    if eval_rows:
        _plot_grouped_bars(
            output_dir / "eval_metrics.png",
            eval_rows,
            ("objective_normalized_match", "horizon_objective_normalized_match", "keyframe_f1", "completed_objective_recall"),
            title=f"{title}: eval",
            ylabel="score",
        )
    if video_rows:
        _plot_grouped_bars(
            output_dir / "video_metrics.png",
            video_rows,
            ("accepted_event_rate", "nonempty_completed_rate", "avg_keyframe_candidates", "parse_error_rate"),
            title=f"{title}: video rollout",
            ylabel="rate / count",
        )


def _plot_grouped_bars(
    path: pathlib.Path,
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row['task']}\n{row['setting']}" for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5.5))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [_float_or_none(row.get(metric)) or 0.0 for row in rows]
        ax.bar([x + offset for x in x_positions], values, width=width, label=metric)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
