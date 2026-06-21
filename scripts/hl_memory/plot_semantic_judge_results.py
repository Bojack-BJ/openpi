from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
from typing import Any


ERROR_ORDER = [
    "equivalent",
    "wrong_action",
    "wrong_object",
    "too_early",
    "too_late",
    "wrong_location",
    "wrong_hand",
    "underspecified",
    "unrelated",
    "parse_error",
]

ERROR_COLORS = {
    "equivalent": "#2ca25f",
    "wrong_action": "#de2d26",
    "wrong_object": "#fd8d3c",
    "too_early": "#756bb1",
    "too_late": "#9e9ac8",
    "wrong_location": "#3182bd",
    "wrong_hand": "#6baed6",
    "underspecified": "#969696",
    "unrelated": "#252525",
    "parse_error": "#fdd0a2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HL-memory semantic judge JSON/JSONL results.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="HL Memory Semantic Judge")
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text())
    rows = _load_jsonl(args.rows_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _metrics_by_source(summary)
    _write_metrics_csv(metrics, args.output_dir / "semantic_metrics_by_setting.csv")
    if _matplotlib_available():
        _plot_accuracy(metrics, args.output_dir / "semantic_accuracy_by_setting.png", title=args.title)
        _plot_error_breakdown(metrics, args.output_dir / "semantic_error_breakdown_by_setting.png", title=args.title)
        _plot_timeline(rows, args.output_dir / "semantic_timeline_by_setting.png", title=args.title)
    _plot_accuracy_svg(metrics, args.output_dir / "semantic_accuracy_by_setting.svg", title=args.title)
    _plot_error_breakdown_svg(metrics, args.output_dir / "semantic_error_breakdown_by_setting.svg", title=args.title)
    _plot_timeline_svg(rows, args.output_dir / "semantic_timeline_by_setting.svg", title=args.title)


def _matplotlib_available() -> bool:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metrics_by_source(summary: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for source, data in summary.get("by_source", {}).items():
        item = {
            "source": source,
            "setting": _setting_name(source),
            "total": int(data.get("total", 0)),
            "match_count": int(data.get("match_count", 0)),
            "semantic_accuracy": float(data.get("semantic_accuracy", 0.0)),
            "mean_confidence": float(data.get("mean_confidence", 0.0)),
            "error_counts": dict(data.get("error_counts", {})),
        }
        metrics.append(item)
    metrics.sort(key=lambda item: item["semantic_accuracy"], reverse=True)
    return metrics


def _setting_name(source: str) -> str:
    path = Path(source)
    if path.name == "summary.json" and path.parent.name:
        return path.parent.name
    return path.stem


def _write_metrics_csv(metrics: list[dict[str, Any]], path: Path) -> None:
    error_keys = sorted({key for item in metrics for key in item["error_counts"]}, key=_error_sort_key)
    setting_codes = _setting_code_map(metrics)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting_code",
                "setting",
                "total",
                "match_count",
                "semantic_accuracy",
                "mean_confidence",
                *error_keys,
            ],
        )
        writer.writeheader()
        for item in metrics:
            row = {
                "setting_code": setting_codes[item["setting"]],
                "setting": item["setting"],
                "total": item["total"],
                "match_count": item["match_count"],
                "semantic_accuracy": item["semantic_accuracy"],
                "mean_confidence": item["mean_confidence"],
            }
            row.update({key: item["error_counts"].get(key, 0) for key in error_keys})
            writer.writerow(row)


def _plot_accuracy(metrics: list[dict[str, Any]], path: Path, *, title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    setting_codes = _setting_code_map(metrics)
    settings = [setting_codes[item["setting"]] for item in metrics]
    values = [item["semantic_accuracy"] for item in metrics]
    colors = [_setting_color(item["setting"]) for item in metrics]

    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=180)
    bars = ax.bar(settings, values, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylim(0.0, max(0.4, max(values, default=0.0) * 1.25))
    ax.set_ylabel("Semantic accuracy")
    ax.set_title(f"{title}: Semantic Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    handles = []
    for item in metrics:
        setting = item["setting"]
        handles.append(
            mpatches.Patch(
                color=_setting_color(setting),
                label=f"{setting_codes[setting]}: {_compact_setting_label(setting)}",
            )
        )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_error_breakdown(metrics: list[dict[str, Any]], path: Path, *, title: str) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    setting_codes = _setting_code_map(metrics)
    settings = [setting_codes[item["setting"]] for item in metrics]
    fig, ax = plt.subplots(figsize=(12.5, 7.8), dpi=180)
    bottoms = [0] * len(metrics)
    for error_type in _ordered_error_types(metrics):
        values = [item["error_counts"].get(error_type, 0) for item in metrics]
        if not any(values):
            continue
        ax.bar(
            settings,
            values,
            bottom=bottoms,
            label=error_type,
            color=ERROR_COLORS.get(error_type, "#bdbdbd"),
            edgecolor="white",
            linewidth=0.4,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_ylabel("Step count")
    ax.set_title(f"{title}: Error Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    error_legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncols=5,
        frameon=False,
        fontsize=8,
        title="Judge label",
    )
    ax.add_artist(error_legend)
    setting_handles = []
    for item in metrics:
        setting = item["setting"]
        setting_handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="none",
                label=f"{setting_codes[setting]}: {_compact_setting_label(setting)}",
            )
        )
    ax.legend(
        handles=setting_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=2,
        frameon=False,
        fontsize=8,
        title="Settings",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_setting_name(str(row.get("source_path", "")))].append(row)
    settings = sorted(grouped, key=lambda name: _setting_family_sort_key(name))
    setting_codes = _setting_code_map([{"setting": setting, "error_counts": {}, "total": 0} for setting in settings])
    max_step = max((int(row.get("step_index") or row.get("source_row") or 0) for row in rows), default=0)

    fig, ax = plt.subplots(figsize=(13, max(4.0, 0.65 * len(settings) + 2.5)), dpi=180)
    for y, setting in enumerate(settings):
        for row in grouped[setting]:
            step = int(row.get("step_index") or row.get("source_row") or 0)
            error_type = str(row.get("judge", {}).get("error_type", "missing"))
            ax.scatter(step, y, marker="s", s=75, color=ERROR_COLORS.get(error_type, "#bdbdbd"), edgecolor="white", linewidth=0.25)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels([setting_codes[setting] for setting in settings])
    ax.set_xlim(-1, max_step + 1)
    ax.set_xlabel("Rollout step")
    ax.set_title(f"{title}: Per-Step Judge Labels")
    ax.grid(axis="x", alpha=0.18)
    handles = [
        mpatches.Patch(color=ERROR_COLORS[label], label=label)
        for label in ERROR_ORDER
        if any(str(row.get("judge", {}).get("error_type")) == label for row in rows)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=5, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_accuracy_svg(metrics: list[dict[str, Any]], path: Path, *, title: str) -> None:
    width = 1100
    height = 680
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 250
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max(0.4, max((item["semantic_accuracy"] for item in metrics), default=0.0) * 1.25)
    bar_gap = 20
    bar_w = (chart_w - bar_gap * (len(metrics) - 1)) / max(len(metrics), 1)
    setting_codes = _setting_code_map(metrics)
    parts = [_svg_header(width, height), _svg_text(width / 2, 32, f"{title}: Semantic Accuracy", size=22, anchor="middle", weight="700")]
    _add_axes(parts, margin_left, margin_top, chart_w, chart_h, y_label="Semantic accuracy")
    for i in range(5):
        value = max_value * i / 4
        y = margin_top + chart_h - chart_h * value / max_value
        parts.append(_svg_text(margin_left - 10, y + 4, f"{value:.2f}", size=12, anchor="end"))
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_w}" y2="{y:.1f}" stroke="#e6e6e6"/>')
    for index, item in enumerate(metrics):
        x = margin_left + index * (bar_w + bar_gap)
        value = item["semantic_accuracy"]
        h = chart_h * value / max_value
        y = margin_top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{_setting_color(item["setting"])}" stroke="#222" stroke-width="0.8"/>')
        parts.append(_svg_text(x + bar_w / 2, y - 8, f"{value:.3f}", size=13, anchor="middle", weight="600"))
        parts.append(_svg_text(x + bar_w / 2, margin_top + chart_h + 28, setting_codes[item["setting"]], size=13, anchor="middle", weight="700"))
    _append_svg_setting_legend(parts, metrics, setting_codes, x=margin_left, y=height - 160, columns=2, column_width=490)
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def _plot_error_breakdown_svg(metrics: list[dict[str, Any]], path: Path, *, title: str) -> None:
    width = 1320
    height = 740
    margin_left = 80
    margin_top = 70
    margin_bottom = 265
    chart_w = 880
    chart_h = height - margin_top - margin_bottom
    max_total = max((item["total"] for item in metrics), default=1)
    bar_gap = 22
    bar_w = (chart_w - bar_gap * (len(metrics) - 1)) / max(len(metrics), 1)
    setting_codes = _setting_code_map(metrics)
    parts = [_svg_header(width, height), _svg_text(width / 2, 32, f"{title}: Error Breakdown", size=22, anchor="middle", weight="700")]
    _add_axes(parts, margin_left, margin_top, chart_w, chart_h, y_label="Step count")
    for index, item in enumerate(metrics):
        x = margin_left + index * (bar_w + bar_gap)
        bottom_y = margin_top + chart_h
        for error_type in _ordered_error_types(metrics):
            count = item["error_counts"].get(error_type, 0)
            if count <= 0:
                continue
            h = chart_h * count / max_total
            bottom_y -= h
            parts.append(f'<rect x="{x:.1f}" y="{bottom_y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{ERROR_COLORS.get(error_type, "#bdbdbd")}" stroke="white" stroke-width="0.6"/>')
        parts.append(_svg_text(x + bar_w / 2, margin_top + chart_h + 28, setting_codes[item["setting"]], size=13, anchor="middle", weight="700"))
    legend_x = margin_left + chart_w + 45
    legend_y = margin_top + 10
    parts.append(_svg_text(legend_x, legend_y - 22, "Judge label", size=13, weight="700"))
    for i, error_type in enumerate(_ordered_error_types(metrics)):
        y = legend_y + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{ERROR_COLORS.get(error_type, "#bdbdbd")}"/>')
        parts.append(_svg_text(legend_x + 22, y, error_type, size=12))
    _append_svg_setting_legend(parts, metrics, setting_codes, x=margin_left, y=height - 175, columns=2, column_width=500)
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def _plot_timeline_svg(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_setting_name(str(row.get("source_path", "")))].append(row)
    settings = sorted(grouped, key=lambda name: _setting_family_sort_key(name))
    setting_codes = _setting_code_map([{"setting": setting, "error_counts": {}, "total": 0} for setting in settings])
    max_step = max((int(row.get("step_index") or row.get("source_row") or 0) for row in rows), default=0)
    width = 1250
    row_h = 42
    margin_left = 95
    margin_right = 50
    margin_top = 72
    margin_bottom = 120
    chart_w = width - margin_left - margin_right
    height = margin_top + margin_bottom + row_h * max(len(settings), 1)
    parts = [_svg_header(width, height), _svg_text(width / 2, 32, f"{title}: Per-Step Judge Labels", size=22, anchor="middle", weight="700")]
    for step in range(0, max_step + 1, 5):
        x = margin_left + chart_w * step / max(max_step, 1)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top - 10}" x2="{x:.1f}" y2="{height - margin_bottom + 12}" stroke="#ececec"/>')
        parts.append(_svg_text(x, height - margin_bottom + 35, str(step), size=11, anchor="middle"))
    for y_index, setting in enumerate(settings):
        y = margin_top + y_index * row_h
        parts.append(_svg_text(margin_left - 14, y + 17, setting_codes[setting], size=13, anchor="end", weight="700"))
        parts.append(f'<line x1="{margin_left}" y1="{y + 16}" x2="{margin_left + chart_w}" y2="{y + 16}" stroke="#f2f2f2"/>')
        for row in grouped[setting]:
            step = int(row.get("step_index") or row.get("source_row") or 0)
            error_type = str(row.get("judge", {}).get("error_type", "missing"))
            x = margin_left + chart_w * step / max(max_step, 1)
            parts.append(f'<rect x="{x - 5:.1f}" y="{y + 7:.1f}" width="10" height="18" rx="2" fill="{ERROR_COLORS.get(error_type, "#bdbdbd")}" stroke="white" stroke-width="0.4"><title>{escape(setting)} step={step} {escape(error_type)}</title></rect>')
    legend_items = [label for label in ERROR_ORDER if any(str(row.get("judge", {}).get("error_type")) == label for row in rows)]
    legend_y = height - 58
    legend_x = margin_left
    for i, label in enumerate(legend_items):
        x = legend_x + (i % 5) * 172
        y = legend_y + (i // 5) * 24
        parts.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{ERROR_COLORS.get(label, "#bdbdbd")}"/>')
        parts.append(_svg_text(x + 22, y, label, size=12))
    parts.append(_svg_text(margin_left + chart_w / 2, height - margin_bottom + 60, "Rollout step", size=13, anchor="middle", weight="600"))
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'


def _add_axes(parts: list[str], x: int, y: int, width: int, height: int, *, y_label: str) -> None:
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#333"/>')
    parts.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#333"/>')
    parts.append(_svg_rotated_text(22, y + height / 2, y_label, -90, size=13, weight="600"))


def _svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#222">{escape(text)}</text>'


def _svg_rotated_text(x: float, y: float, text: str, angle: float, *, size: int = 12, weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate({angle:.1f} {x:.1f} {y:.1f})" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="end" fill="#222">{escape(text)}</text>'


def _ordered_error_types(metrics: list[dict[str, Any]]) -> list[str]:
    keys = {key for item in metrics for key in item["error_counts"]}
    return sorted(keys, key=_error_sort_key)


def _error_sort_key(error_type: str) -> tuple[int, str]:
    try:
        return (ERROR_ORDER.index(error_type), error_type)
    except ValueError:
        return (len(ERROR_ORDER), error_type)


def _short_label(setting: str) -> str:
    label = setting.replace("20260328K086A_", "")
    label = label.replace("_vision_fully", "")
    label = label.replace("_002000", "\n@2000").replace("_001800", "\n@1800").replace("_001400", "\n@1400")
    label = label.replace("_memer", "\nmemer")
    label = label.replace("_no_", "\nno_")
    label = label.replace("_", " ")
    return label


def _setting_code_map(metrics: list[dict[str, Any]]) -> dict[str, str]:
    counters = {"C": 0, "F": 0}
    codes: dict[str, str] = {}
    for item in metrics:
        setting = str(item["setting"])
        prefix = "C" if "coarse" in setting.lower() else "F"
        counters[prefix] += 1
        codes[setting] = f"{prefix}{counters[prefix]}"
    return codes


def _compact_setting_label(setting: str) -> str:
    label = setting.replace("20260328K086A_", "")
    label = label.replace("K086A_", "")
    replacements = {
        "coarse_memer_baseline": "coarse baseline",
        "coarse_memer_proprio_per_frame_plus_summary": "coarse proprio frame+summary",
        "coarse_memer_proprio_per_frame": "coarse proprio frame",
        "coarse_memer_proprio_summary": "coarse proprio summary",
        "prior_no_proprio_002000": "fine prior no proprio @2000",
        "prior_proprio_001400": "fine prior proprio @1400",
        "prior_proprio_memer_001800": "fine prior proprio memer @1800",
        "proprio_no_prior_001800": "fine proprio no prior @1800",
        "proprio_no_prior_memer_002000": "fine proprio no prior memer @2000",
    }
    return replacements.get(label, label.replace("_", " "))


def _append_svg_setting_legend(
    parts: list[str],
    metrics: list[dict[str, Any]],
    setting_codes: dict[str, str],
    *,
    x: int,
    y: int,
    columns: int,
    column_width: int,
) -> None:
    parts.append(_svg_text(x, y - 22, "Settings", size=13, weight="700"))
    for index, item in enumerate(metrics):
        setting = str(item["setting"])
        column = index % columns
        row = index // columns
        item_x = x + column * column_width
        item_y = y + row * 23
        code = setting_codes[setting]
        parts.append(_svg_text(item_x, item_y, f"{code}: {_compact_setting_label(setting)}", size=12))


def _svg_multiline_label(x: float, y: float, text: str, *, anchor: str = "start", size: int = 12) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [_svg_text(x, y + i * (size + 3), line, size=size, anchor=anchor) for i, line in enumerate(lines)]


def _setting_color(setting: str) -> str:
    if "memer" in setting and "proprio_no_prior" in setting:
        return "#1b9e77"
    if "memer" in setting:
        return "#66a61e"
    if "proprio_no_prior" in setting:
        return "#7570b3"
    if "prior_no_proprio" in setting:
        return "#d95f02"
    return "#e7298a"


def _setting_family_sort_key(setting: str) -> tuple[int, str]:
    if "proprio_no_prior_memer" in setting:
        return (0, setting)
    if "prior_proprio_memer" in setting:
        return (1, setting)
    if "prior_no_proprio" in setting:
        return (2, setting)
    if "proprio_no_prior" in setting:
        return (3, setting)
    return (4, setting)


if __name__ == "__main__":
    main()
