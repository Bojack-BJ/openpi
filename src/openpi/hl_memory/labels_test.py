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


def test_render_language_memory_defaults_when_no_progress():
    assert render_language_memory(TaskProgressState()) == DEFAULT_LANGUAGE_MEMORY


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

    assert "Completed subtasks: pick apple." in rendered
    assert "Recent events: Started pick apple. | Apple picked successfully.." not in rendered
    assert "Recent events: Started pick apple | Apple picked successfully." in rendered
    assert "Current phase: place." in rendered
    assert "Target query: apple." in rendered
    assert "Goal query: basket." in rendered
