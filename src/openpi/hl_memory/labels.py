from __future__ import annotations

from collections.abc import Sequence
import dataclasses


_ALLOWED_EVENT_TYPES = {"none", "subtask_boundary", "success", "failure", "progress", "discovery"}
DEFAULT_LANGUAGE_MEMORY = "No progress has been recorded yet."


@dataclasses.dataclass(frozen=True)
class SubtaskAnnotation:
    episode_index: int
    frame_index: int
    current_subtask: str
    instruction: str = ""
    phase: str = ""
    target_query: str = ""
    goal_query: str = ""
    event_type: str = "none"
    event_text: str = ""

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
            event_type=str(data.get("event_type", "none")).strip(),
            event_text=str(data.get("event_text", "")).strip(),
        )


@dataclasses.dataclass(frozen=True)
class TaskProgressState:
    completed_subtasks: tuple[str, ...] = ()
    failed_subtasks: tuple[str, ...] = ()
    recent_events: tuple[str, ...] = ()
    phase: str = ""
    target_query: str = ""
    goal_query: str = ""


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
    for annotation_index, annotation in enumerate(annotations[: current_index + 1]):
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
    )


def render_language_memory(state: TaskProgressState) -> str:
    parts: list[str] = []
    if state.completed_subtasks:
        parts.append(f"Completed subtasks: {', '.join(state.completed_subtasks)}.")
    if state.failed_subtasks:
        parts.append(f"Failures: {', '.join(state.failed_subtasks)}.")
    if state.recent_events:
        normalized_events = " | ".join(event.rstrip(". ") for event in state.recent_events)
        parts.append(f"Recent events: {normalized_events}.")
    if state.phase:
        parts.append(f"Current phase: {state.phase}.")
    if state.target_query:
        parts.append(f"Target query: {state.target_query}.")
    if state.goal_query:
        parts.append(f"Goal query: {state.goal_query}.")
    if not parts:
        return DEFAULT_LANGUAGE_MEMORY
    return " ".join(parts)


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
