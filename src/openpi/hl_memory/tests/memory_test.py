from openpi.hl_memory.memory import EpisodicKeyframeMemory
from openpi.hl_memory.memory import build_recent_context_indices
from openpi.hl_memory.memory import cluster_candidate_indices
from openpi.hl_memory.memory import map_relative_positions_to_absolute


def test_build_recent_context_indices():
    indices = build_recent_context_indices(timestep=19, frame_subsample=5, recent_frames_length=4)

    assert indices == [4, 9, 14, 19]


def test_build_recent_context_indices_with_fixed_window():
    indices = build_recent_context_indices(
        timestep=40,
        frame_subsample=5,
        recent_frames_length=8,
        recent_window_frames=40,
    )

    assert indices == [0, 6, 11, 17, 23, 29, 34, 40]


def test_build_recent_context_indices_matches_default_two_hz_protocol():
    indices = build_recent_context_indices(
        timestep=70,
        frame_subsample=5,
        recent_frames_length=8,
        recent_window_frames=70,
    )

    assert indices == [0, 10, 20, 30, 40, 50, 60, 70]


def test_build_recent_context_indices_with_fixed_window_near_start():
    indices = build_recent_context_indices(
        timestep=20,
        frame_subsample=5,
        recent_frames_length=8,
        recent_window_frames=40,
    )

    assert indices == [0, 3, 6, 9, 11, 14, 17, 20]


def test_map_relative_positions_to_absolute():
    mapped, invalid = map_relative_positions_to_absolute([1, 3, 7], [10, 20, 30, 40])

    assert mapped == [10, 30]
    assert invalid == [7]


def test_cluster_candidate_indices_preserves_duplicate_votes():
    clustered = cluster_candidate_indices([4, 4, 5, 20, 21, 40], merge_distance=1)

    assert clustered == [4, 20, 40]


def test_visible_keyframes_excludes_recent_context():
    memory = EpisodicKeyframeMemory(memory_length=4, merge_distance=1)
    memory.add_candidates([5, 5, 6, 20, 40])

    assert memory.selected_indices() == [5, 20, 40]
    assert memory.visible_indices([20, 21, 22]) == [5, 40]
