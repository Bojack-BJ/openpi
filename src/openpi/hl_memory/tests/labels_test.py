from openpi.hl_memory.labels import DEFAULT_LANGUAGE_MEMORY
from openpi.hl_memory.labels import SubtaskAnnotation
from openpi.hl_memory.labels import TaskProgressState
from openpi.hl_memory.labels import derive_event_band_keyframe_positions
from openpi.hl_memory.labels import derive_keyframe_event_bands
from openpi.hl_memory.labels import derive_keyframe_positions
from openpi.hl_memory.labels import render_completed_subtasks_memory
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


def test_derive_keyframe_event_bands_preserves_canonical_keyframe():
    annotations = [
        SubtaskAnnotation(
            episode_index=0,
            frame_index=20,
            current_subtask="place toast",
            current_objective="place toast",
            keyframe_label=True,
        ),
    ]

    bands = derive_keyframe_event_bands(
        annotations,
        training_fps=20.0,
        before_sec=1.0,
        after_sec=0.5,
    )

    assert len(bands) == 1
    assert bands[0].canonical_frame_index == 20
    assert bands[0].band_start_frame_index == 0
    assert bands[0].band_end_frame_index == 30
    assert bands[0].event_id == "event_0000"


def test_event_band_candidate_selects_recent_frame_closest_to_canonical():
    bands = derive_keyframe_event_bands(
        [
            SubtaskAnnotation(
                episode_index=0,
                frame_index=20,
                current_subtask="place toast",
                current_objective="place toast",
                keyframe_label=True,
            )
        ],
        training_fps=20.0,
        before_sec=1.0,
        after_sec=0.5,
    )

    positions, event_ids, frame_indices = derive_event_band_keyframe_positions(
        bands,
        recent_indices=[10, 18, 21, 30],
        upto_frame_index=21,
    )

    assert positions == (3,)
    assert event_ids == ("event_0000",)
    assert frame_indices == (20,)


def test_event_band_candidate_skips_future_band():
    bands = derive_keyframe_event_bands(
        [
            SubtaskAnnotation(
                episode_index=0,
                frame_index=100,
                current_subtask="place toast",
                current_objective="place toast",
                keyframe_label=True,
            )
        ],
        training_fps=20.0,
        before_sec=1.0,
        after_sec=0.5,
    )

    positions, event_ids, frame_indices = derive_event_band_keyframe_positions(
        bands,
        recent_indices=[40, 50, 60],
        upto_frame_index=60,
    )

    assert positions == ()
    assert event_ids == ()
    assert frame_indices == ()


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


def test_progress_state_marks_previous_objective_complete_on_objective_transition():
    state = update_progress_state(
        TaskProgressState(),
        SubtaskAnnotation(
            episode_index=0,
            frame_index=0,
            current_subtask="grasp stick",
            current_objective="Grasp the stick",
            event_type="subtask_boundary",
        ),
    )
    state = update_progress_state(
        state,
        SubtaskAnnotation(
            episode_index=0,
            frame_index=100,
            current_subtask="insert stick",
            current_objective="Move to and insert the stick into the top-left hole",
            event_type="subtask_boundary",
        ),
    )

    assert state.completed_subtasks == ("Grasp the stick",)
    assert state.current_objective == "Move to and insert the stick into the top-left hole"
    assert "Completed subtasks: Grasp the stick." in render_language_memory(state)


def test_progress_state_does_not_repeat_completion_within_same_objective():
    state = update_progress_state(
        TaskProgressState(),
        SubtaskAnnotation(
            episode_index=0,
            frame_index=0,
            current_subtask="grasp stick",
            current_objective="Grasp the stick",
        ),
    )
    state = update_progress_state(
        state,
        SubtaskAnnotation(
            episode_index=0,
            frame_index=2,
            current_subtask="grasp stick",
            current_objective="  grasp   the stick ",
            event_type="progress",
        ),
    )

    assert state.completed_subtasks == ()


def test_render_completed_subtasks_memory_excludes_active_and_llm_progress_fields():
    state = TaskProgressState(
        completed_subtasks=("Grasp left slice of toast", "Place toast on plate"),
        task_progress="The robot is currently approaching the steak.",
        current_objective="Grasp steak",
        relevant_objects=("steak",),
        notes="continue moving",
    )

    rendered = render_completed_subtasks_memory(state)

    assert rendered == (
        "Completed subtasks:\n"
        "1. Grasp left slice of toast\n"
        "2. Place toast on plate"
    )
    assert "approaching" not in rendered
    assert "Grasp steak" not in rendered


def test_render_completed_subtasks_memory_defaults_before_first_transition():
    assert render_completed_subtasks_memory(TaskProgressState()) == "No completed subtask yet."
