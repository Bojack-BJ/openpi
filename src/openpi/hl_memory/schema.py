from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import json
import re

_NEW_PREDICTION_KEYS = {
    "task_progress",
    "current_objective",
    "target_object",
    "target_slot",
    "relevant_objects",
    "notes",
    "new_completed_objective",
    "completed_objective",
}
_LEGACY_PREDICTION_KEYS = {
    "updated_language_memory",
    "current_subtask",
}


@dataclasses.dataclass(frozen=True)
class HLMemoryPrediction:
    updated_language_memory: str
    current_subtask: str
    keyframe_candidate_positions: tuple[int, ...]
    phase: str
    target_query: str
    goal_query: str
    target_object: str = ""
    target_slot: str = ""
    task_progress: str = ""
    current_objective: str = ""
    relevant_objects: tuple[str, ...] = ()
    notes: str = ""
    subtask_progress: float | None = None
    should_advance_objective: bool | None = None
    active_hand: str = ""
    horizon_current_objective: str = ""
    last_objective: str = ""
    previous_stage_objective: str = ""
    new_completed_objective: str = ""
    completed_objective: str = ""
    sam_text_prompt: str = ""
    sam_point_xy: tuple[int, int] | None = None
    target_bbox_xyxy: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        parsed_memory = _parse_ll_memory_fields(self.updated_language_memory)
        current_objective = (self.current_objective or self.current_subtask or parsed_memory.get("current objective", "")).strip()
        if not current_objective:
            raise ValueError("current_objective must be non-empty.")
        task_progress = _normalize_task_progress(
            self.task_progress or parsed_memory.get("task progress", "") or "No completed subtask yet."
        )
        notes = (self.notes or parsed_memory.get("notes", "") or "none").strip()
        relevant_objects = self.relevant_objects or _parse_relevant_objects(parsed_memory.get("relevant objects", ""))
        target_object = (self.target_object or self.target_query).strip()
        target_slot = (self.target_slot or self.goal_query).strip()
        if not relevant_objects:
            relevant_objects = tuple(
                value
                for value in (target_object, target_slot, self.target_query.strip(), self.goal_query.strip())
                if value and value.lower() != "none"
            )
        updated_language_memory = self.updated_language_memory.strip() or render_language_memory_fields(
            task_progress=task_progress,
            current_objective=current_objective,
            relevant_objects=relevant_objects,
            notes=notes,
        )
        current_subtask = self.current_subtask.strip() or current_objective
        phase = self.phase.strip() or current_objective
        new_completed_objective = (self.new_completed_objective or self.completed_objective).strip()
        completed_objective = (self.completed_objective or new_completed_objective).strip()
        object.__setattr__(self, "task_progress", task_progress)
        object.__setattr__(self, "current_objective", current_objective)
        object.__setattr__(self, "current_subtask", current_subtask)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "relevant_objects", tuple(str(item).strip() for item in relevant_objects if str(item).strip()))
        object.__setattr__(self, "target_object", target_object)
        object.__setattr__(self, "target_slot", target_slot)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "updated_language_memory", updated_language_memory)
        object.__setattr__(self, "new_completed_objective", new_completed_objective)
        object.__setattr__(self, "completed_objective", completed_objective)
        for position in self.keyframe_candidate_positions:
            if position <= 0:
                raise ValueError("keyframe_candidate_positions must be positive and 1-indexed.")

    def to_dict(self, *, include_legacy: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "task_progress": self.task_progress,
            "current_objective": self.current_objective,
            "relevant_objects": list(self.relevant_objects),
            "notes": self.notes,
            "keyframe_candidate_positions": list(self.keyframe_candidate_positions),
            "phase": self.phase,
            "target_query": self.target_query,
            "goal_query": self.goal_query,
        }
        if self.target_object:
            result["target_object"] = self.target_object
        if self.target_slot:
            result["target_slot"] = self.target_slot
        if include_legacy:
            result["updated_language_memory"] = self.updated_language_memory
            result["current_subtask"] = self.current_subtask
        if self.subtask_progress is not None:
            result["subtask_progress"] = float(self.subtask_progress)
        if self.should_advance_objective is not None:
            result["should_advance_objective"] = bool(self.should_advance_objective)
        if self.active_hand:
            result["active_hand"] = self.active_hand
        if self.horizon_current_objective:
            result["horizon_current_objective"] = self.horizon_current_objective
        if self.last_objective:
            result["last_objective"] = self.last_objective
        if self.previous_stage_objective:
            result["previous_stage_objective"] = self.previous_stage_objective
        if self.new_completed_objective:
            result["new_completed_objective"] = self.new_completed_objective
        if self.completed_objective:
            result["completed_objective"] = self.completed_objective
        if self.sam_text_prompt:
            result["sam_text_prompt"] = self.sam_text_prompt
        if self.sam_point_xy is not None:
            result["sam_point_xy"] = [int(self.sam_point_xy[0]), int(self.sam_point_xy[1])]
        if self.target_bbox_xyxy is not None:
            result["target_bbox_xyxy"] = [int(value) for value in self.target_bbox_xyxy]
        return result

    def to_runtime_schema_dict(self, *, include_grounding: bool = False) -> dict[str, object]:
        """Returns the compact prediction schema used by eval and rollout logs.

        This intentionally omits legacy compatibility fields such as
        updated_language_memory/current_subtask and unused query fields. The
        output is kept aligned with the current rollout contract:
        task progress in, objective/keyframe/completion state out.
        `include_grounding` keeps target fields present even when empty so
        debug logs can distinguish "not predicted" from "not part of schema".
        """
        result: dict[str, object] = {
            "task_progress": self.task_progress,
            "current_objective": self.current_objective,
            "horizon_current_objective": self.horizon_current_objective,
            "keyframe_candidate_positions": list(self.keyframe_candidate_positions),
            "new_completed_objective": self.new_completed_objective,
        }
        if include_grounding:
            result["target_object"] = self.target_object
            result["target_slot"] = self.target_slot
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(include_legacy=False), ensure_ascii=True, separators=(",", ":"))

    def with_recent_position_limit(self, recent_valid_length: int) -> "HLMemoryPrediction":
        """Drops keyframe positions that point outside the valid recent clip."""
        limit = max(int(recent_valid_length), 0)
        return dataclasses.replace(
            self,
            keyframe_candidate_positions=tuple(
                position for position in self.keyframe_candidate_positions if 1 <= position <= limit
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HLMemoryPrediction":
        raw_positions = data.get("keyframe_candidate_positions", data.get("keyframe_positions", []))
        if raw_positions is None:
            raw_positions = []
        if isinstance(raw_positions, str):
            raw_positions = _parse_keyframe_positions_text(raw_positions)
        if not isinstance(raw_positions, list):
            raise ValueError("keyframe_candidate_positions must be a list.")
        keyframe_candidate_positions: list[int] = []
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if position <= 0:
                continue
            if position not in keyframe_candidate_positions:
                keyframe_candidate_positions.append(position)
        updated_language_memory = str(data.get("updated_language_memory", "")).strip()
        parsed_memory = _parse_ll_memory_fields(updated_language_memory)
        current_objective = str(
            data.get(
                "current_objective",
                data.get("objective", data.get("current_subtask", parsed_memory.get("current objective", ""))),
            )
        ).strip()
        task_progress = _normalize_task_progress(data.get("task_progress", parsed_memory.get("task progress", "")))
        notes = str(data.get("notes", parsed_memory.get("notes", ""))).strip()
        if not current_objective and not updated_language_memory:
            raise ValueError("Prediction must include current_objective, current_subtask, or updated_language_memory.")
        target_query = str(data.get("target_query", "")).strip()
        goal_query = str(data.get("goal_query", "")).strip()
        target_object = str(data.get("target_object", target_query)).strip()
        target_slot = str(data.get("target_slot", goal_query)).strip()
        relevant_objects = _parse_relevant_objects(data.get("relevant_objects", parsed_memory.get("relevant objects", "")))
        return cls(
            updated_language_memory=updated_language_memory,
            current_subtask=str(data.get("current_subtask", current_objective)).strip(),
            keyframe_candidate_positions=tuple(keyframe_candidate_positions),
            phase=str(data.get("phase", current_objective)).strip(),
            target_query=target_query,
            goal_query=goal_query,
            target_object=target_object,
            target_slot=target_slot,
            task_progress=task_progress,
            current_objective=current_objective,
            relevant_objects=relevant_objects,
            notes=notes,
            subtask_progress=_parse_optional_float(data.get("subtask_progress")),
            should_advance_objective=_parse_optional_bool(data.get("should_advance_objective")),
            active_hand=str(data.get("active_hand", "")).strip(),
            horizon_current_objective=str(data.get("horizon_current_objective", "")).strip(),
            last_objective=str(data.get("last_objective", "")).strip(),
            previous_stage_objective=str(data.get("previous_stage_objective", "")).strip(),
            new_completed_objective=str(
                data.get("new_completed_objective", data.get("completed_objective", ""))
            ).strip(),
            completed_objective=str(data.get("completed_objective", "")).strip(),
            sam_text_prompt=str(data.get("sam_text_prompt", data.get("sam_prompt", ""))).strip(),
            sam_point_xy=_parse_optional_point(
                data.get("sam_point_xy")
                or data.get("point_xy")
                or data.get("target_point")
                or data.get("point_prompt")
            ),
            target_bbox_xyxy=_parse_optional_bbox(data.get("target_bbox_xyxy") or data.get("bbox_xyxy") or data.get("target_bbox")),
        )

    @classmethod
    def from_json(cls, text: str) -> "HLMemoryPrediction":
        return cls.from_dict(_extract_first_json_object(text))


def _extract_first_json_object(text: str) -> dict[str, object]:
    fallback: dict[str, object] | None = None
    for candidate_text in _candidate_json_texts(text):
        parsed_objects = list(_iter_json_objects(candidate_text))
        prediction_objects = [parsed for parsed in parsed_objects if _looks_like_prediction(parsed)]
        if prediction_objects:
            return prediction_objects[-1]
        if fallback is None and parsed_objects:
            fallback = parsed_objects[-1]

    if fallback is not None:
        return fallback

    raise ValueError("Could not parse a JSON object from model output.")


def _normalize_task_progress(value: object) -> str:
    if isinstance(value, list):
        items = [str(item).strip(" .") for item in value if str(item).strip(" .")]
        return "; ".join(items) + ("." if items else "")
    return str(value).strip()


def _parse_keyframe_positions_text(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [int(value) for value in parsed if _is_int_like(value)]
    if _is_int_like(parsed):
        return [int(parsed)]
    return [int(match) for match in re.findall(r"\d+", stripped)]


def _is_int_like(value: object) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _candidate_json_texts(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    if "</think>" in stripped:
        candidates.append(stripped.rsplit("</think>", maxsplit=1)[-1].strip())
    candidates.extend(_extract_fenced_blocks(stripped))
    candidates.append(_strip_fenced_block(stripped))
    candidates.append(stripped)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _iter_json_objects(text: str) -> Iterator[dict[str, object]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            repaired = _escape_control_newlines_inside_strings(text[index:])
            if repaired == text[index:]:
                continue
            try:
                parsed, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            yield parsed


def _looks_like_prediction(data: dict[str, object]) -> bool:
    return bool(_NEW_PREDICTION_KEYS & set(data) or _LEGACY_PREDICTION_KEYS & set(data))


def _parse_optional_point(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict) and {"x", "y"}.issubset(value):
        return int(round(float(value["x"]))), int(round(float(value["y"])))
    if isinstance(value, list | tuple) and len(value) >= 2:
        return int(round(float(value[0]))), int(round(float(value[1])))
    return None


def _parse_optional_bbox(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return (
                int(round(float(value["x1"]))),
                int(round(float(value["y1"]))),
                int(round(float(value["x2"]))),
                int(round(float(value["y2"]))),
            )
        if {"left", "top", "right", "bottom"}.issubset(value):
            return (
                int(round(float(value["left"]))),
                int(round(float(value["top"]))),
                int(round(float(value["right"]))),
                int(round(float(value["bottom"]))),
            )
    if isinstance(value, list | tuple) and len(value) >= 4:
        return tuple(int(round(float(item))) for item in value[:4])  # type: ignore[return-value]
    return None


def _parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_ll_memory_fields(memory: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in memory.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _parse_relevant_objects(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
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
    return tuple(objects)


def render_language_memory_fields(
    *,
    task_progress: str,
    current_objective: str,
    relevant_objects: tuple[str, ...] | list[str],
    notes: str,
) -> str:
    objects = ", ".join(str(item).strip() for item in relevant_objects if str(item).strip()) or "none"
    return "\n".join(
        [
            f"Task progress: {task_progress.strip() or 'No completed subtask yet.'}",
            f"Current objective: {current_objective.strip() or 'continue the task'}",
            f"Relevant objects: {objects}",
            f"Notes: {notes.strip() or 'none'}",
        ]
    )


def _escape_control_newlines_inside_strings(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    changed = False
    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            result.append(char)
            in_string = not in_string
            continue
        if in_string and char in {"\n", "\r"}:
            result.append("\\n")
            changed = True
            continue
        result.append(char)
    return "".join(result) if changed else text


def _extract_fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    in_block = False
    current: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_block:
                blocks.append("\n".join(current).strip())
                current = []
                in_block = False
            else:
                in_block = True
                current = []
            continue
        if in_block:
            current.append(line)
    return blocks


def _strip_fenced_block(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text
