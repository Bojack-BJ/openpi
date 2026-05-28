from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from openpi.hl_memory.schema import render_language_memory_fields

_ALLOWED_EVENT_TYPES = {"none", "subtask_boundary", "success", "failure", "progress", "discovery"}
DEFAULT_LANGUAGE_MEMORY = render_language_memory_fields(
    task_progress="No completed subtask yet.",
    current_objective="continue the task",
    relevant_objects=(),
    notes="none",
)


@dataclasses.dataclass(frozen=True)
class SubtaskAnnotation:
    episode_index: int
    frame_index: int
    current_subtask: str
    instruction: str = ""
    phase: str = ""
    target_query: str = ""
    goal_query: str = ""
    task_progress: str = ""
    current_objective: str = ""
    relevant_objects: tuple[str, ...] = ()
    notes: str = ""
    subtask_progress: float | None = None
    should_advance_objective: bool | None = None
    active_hand: str = ""
    event_type: str = "none"
    event_text: str = ""
    keyframe_label: bool | None = None
    horizon_frame_index: int | None = None
    horizon_current_objective: str = ""
    horizon_current_subtask: str = ""
    horizon_phase: str = ""

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        if not self.current_subtask.strip():
            raise ValueError("current_subtask must be non-empty.")
        if self.event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {self.event_type}")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SubtaskAnnotation":
        return cls(
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            current_subtask=str(data["current_subtask"]).strip(),
            instruction=str(data.get("instruction", "")).strip(),
            phase=str(data.get("phase", "")).strip(),
            target_query=str(data.get("target_query", "")).strip(),
            goal_query=str(data.get("goal_query", "")).strip(),
            task_progress=str(data.get("task_progress", "")).strip(),
            current_objective=str(data.get("current_objective", data.get("objective", ""))).strip(),
            relevant_objects=_parse_relevant_objects(data.get("relevant_objects", ())),
            notes=str(data.get("notes", "")).strip(),
            subtask_progress=_parse_optional_float(data.get("subtask_progress")),
            should_advance_objective=_parse_optional_bool(data.get("should_advance_objective")),
            active_hand=str(data.get("active_hand", "")).strip(),
            event_type=str(data.get("event_type", "none")).strip(),
            event_text=str(data.get("event_text", "")).strip(),
            keyframe_label=_parse_optional_bool(data.get("keyframe_label")),
            horizon_frame_index=_parse_optional_int(data.get("horizon_frame_index")),
            horizon_current_objective=str(data.get("horizon_current_objective", "")).strip(),
            horizon_current_subtask=str(data.get("horizon_current_subtask", "")).strip(),
            horizon_phase=str(data.get("horizon_phase", "")).strip(),
        )


@dataclasses.dataclass(frozen=True)
class TaskProgressState:
    completed_subtasks: tuple[str, ...] = ()
    failed_subtasks: tuple[str, ...] = ()
    recent_events: tuple[str, ...] = ()
    phase: str = ""
    target_query: str = ""
    goal_query: str = ""
    task_progress: str = ""
    current_objective: str = ""
    relevant_objects: tuple[str, ...] = ()
    notes: str = ""


def annotation_emits_keyframe(
    current: SubtaskAnnotation,
    previous: SubtaskAnnotation | None,
) -> bool:
    if current.event_type != "none":
        return True
    if current.event_text:
        return True
    if previous is None:
        return True
    return current.current_subtask != previous.current_subtask


def derive_keyframe_positions(
    annotations: Sequence[SubtaskAnnotation],
    current_index: int,
    recent_indices: Sequence[int],
) -> tuple[int, ...]:
    position_lookup = {int(frame_index): position + 1 for position, frame_index in enumerate(recent_indices)}
    positions: list[int] = []
    explicit_keyframe_labels = any(annotation.keyframe_label is not None for annotation in annotations)
    for annotation_index, annotation in enumerate(annotations[: current_index + 1]):
        if explicit_keyframe_labels:
            if annotation.keyframe_label is not True:
                continue
        else:
            previous = annotations[annotation_index - 1] if annotation_index > 0 else None
            if not annotation_emits_keyframe(annotation, previous):
                continue
        position = position_lookup.get(annotation.frame_index)
        if position is not None and position not in positions:
            positions.append(position)
    return tuple(positions)


def update_progress_state(
    state: TaskProgressState,
    annotation: SubtaskAnnotation,
) -> TaskProgressState:
    completed_subtasks = list(state.completed_subtasks)
    failed_subtasks = list(state.failed_subtasks)
    recent_events = list(state.recent_events)

    event_text = _event_text(annotation)
    if annotation.event_type == "success" and annotation.current_subtask not in completed_subtasks:
        completed_subtasks.append(annotation.current_subtask)
    if annotation.event_type == "failure" and annotation.current_subtask not in failed_subtasks:
        failed_subtasks.append(annotation.current_subtask)
    if event_text:
        recent_events.append(event_text)
    recent_events = recent_events[-4:]

    return TaskProgressState(
        completed_subtasks=tuple(completed_subtasks),
        failed_subtasks=tuple(failed_subtasks),
        recent_events=tuple(recent_events),
        phase=annotation.phase or annotation.current_subtask,
        target_query=annotation.target_query,
        goal_query=annotation.goal_query,
        task_progress=annotation.task_progress,
        current_objective=annotation.current_objective or annotation.phase or annotation.current_subtask,
        relevant_objects=annotation.relevant_objects
        or tuple(value for value in (annotation.target_query, annotation.goal_query) if value),
        notes=annotation.notes,
    )


def render_language_memory_fields_from_state(state: TaskProgressState) -> dict[str, object]:
    relevant_objects = state.relevant_objects or tuple(value for value in (state.target_query, state.goal_query) if value)
    return {
        "task_progress": state.task_progress or _render_task_progress(state),
        "current_objective": state.current_objective or state.phase or "continue the task",
        "relevant_objects": relevant_objects,
        "notes": state.notes or "none",
    }


def render_language_memory(state: TaskProgressState) -> str:
    fields = render_language_memory_fields_from_state(state)
    return render_language_memory_fields(
        task_progress=str(fields["task_progress"]),
        current_objective=str(fields["current_objective"]),
        relevant_objects=tuple(fields["relevant_objects"]),  # type: ignore[arg-type]
        notes=str(fields["notes"]),
    )


def expected_event_text(annotation: SubtaskAnnotation) -> str:
    return _event_text(annotation)


def _event_text(annotation: SubtaskAnnotation) -> str:
    if annotation.event_text:
        return annotation.event_text
    if annotation.event_type == "success":
        return f"Completed {annotation.current_subtask}."
    if annotation.event_type == "failure":
        return f"Failed {annotation.current_subtask}."
    if annotation.event_type == "subtask_boundary":
        return f"Started {annotation.current_subtask}."
    if annotation.event_type == "discovery" and annotation.target_query:
        return f"Observed target {annotation.target_query}."
    if annotation.event_type == "progress":
        return f"Progressed on {annotation.current_subtask}."
    return ""


def _render_task_progress(state: TaskProgressState) -> str:
    parts: list[str] = []
    if state.completed_subtasks:
        parts.append(f"Completed subtasks: {', '.join(state.completed_subtasks)}.")
    if state.failed_subtasks:
        parts.append(f"Failed subtasks: {', '.join(state.failed_subtasks)}.")
    if not parts:
        return "No completed subtask yet."
    return " ".join(parts)


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


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
