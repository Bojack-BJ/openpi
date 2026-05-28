from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.labels import SubtaskAnnotation
from openpi.hl_memory.labels import TaskProgressState
from openpi.hl_memory.labels import derive_keyframe_positions
from openpi.hl_memory.labels import render_language_memory
from openpi.hl_memory.labels import update_progress_state


def test_derive_keyframe_positions_uses_boundaries_and_events():
    annotations = [
        SubtaskAnnotation(episode_index=0, frame_index=0, current_subtask="search apple", event_type="subtask_boundary"),
        SubtaskAnnotation(episode_index=0, frame_index=5, current_subtask="pick apple", event_type="success"),
        SubtaskAnnotation(episode_index=0, frame_index=9, current_subtask="place apple"),
    ]

    positions = derive_keyframe_positions(annotations, 2, [0, 5, 9])

    assert positions == (1, 2, 3)


def test_derive_keyframe_positions_uses_explicit_keyframe_labels_when_present():
    annotations = [
        SubtaskAnnotation(
            episode_index=0,
            frame_index=0,
            current_subtask="approach apple",
            event_type="subtask_boundary",
            keyframe_label=False,
        ),
        SubtaskAnnotation(
            episode_index=0,
            frame_index=5,
            current_subtask="place apple",
            event_type="success",
            keyframe_label=True,
        ),
        SubtaskAnnotation(
            episode_index=0,
            frame_index=9,
            current_subtask="return hand",
            event_type="subtask_boundary",
            keyframe_label=False,
        ),
    ]

    positions = derive_keyframe_positions(annotations, 2, [0, 5, 9])

    assert positions == (2,)


def test_render_language_memory_defaults_when_no_progress():
    assert render_language_memory(TaskProgressState()) == DEFAULT_LANGUAGE_MEMORY
    assert "Task progress: No completed subtask yet." in DEFAULT_LANGUAGE_MEMORY
    assert "Current objective: continue the task" in DEFAULT_LANGUAGE_MEMORY


def test_render_language_memory_tracks_recent_state():
    state = TaskProgressState()
    state = update_progress_state(
        state,
        SubtaskAnnotation(
            episode_index=0,
            frame_index=0,
            current_subtask="pick apple",
            phase="pick",
            target_query="apple",
            goal_query="basket",
            event_type="subtask_boundary",
        ),
    )
    state = update_progress_state(
        state,
        SubtaskAnnotation(
            episode_index=0,
            frame_index=5,
            current_subtask="pick apple",
            phase="place",
            target_query="apple",
            goal_query="basket",
            event_type="success",
            event_text="Apple picked successfully.",
        ),
    )

    rendered = render_language_memory(state)

    assert "Task progress: Completed subtasks: pick apple." in rendered
    assert "Current objective: place" in rendered
    assert "Relevant objects: apple, basket" in rendered
    assert "Notes: none" in rendered
