from openpi.hl_memory.zero_shot import build_auto_memory_seconds
from openpi.hl_memory.zero_shot import build_recent_seconds
from openpi.hl_memory.zero_shot import build_rollout_end_seconds
from openpi.hl_memory.zero_shot import parse_seconds_argument
from openpi.hl_memory.zero_shot import update_rollout_memory_seconds


def test_parse_seconds_argument_sorts_and_dedupes():
    assert parse_seconds_argument("3,1,1,2.5") == [1.0, 2.5, 3.0]


def test_build_recent_seconds_defaults_from_video_end():
    assert build_recent_seconds(10.0, clip_length=4, recent_step_sec=2.0) == [4.0, 6.0, 8.0, 10.0]


def test_build_auto_memory_seconds_uses_prefix_before_recent():
    assert build_auto_memory_seconds(20.0, recent_seconds=[12.0, 13.0], clip_length=3) == [0.0, 5.9995, 11.999]


def test_build_rollout_end_seconds_includes_final_video_end():
    assert build_rollout_end_seconds(10.0, interval_sec=4.0, start_sec=2.0) == [2.0, 6.0, 10.0]


def test_update_rollout_memory_seconds_keeps_latest_unique_seconds():
    assert update_rollout_memory_seconds([1.0, 2.0], [2.0, 5.0, 7.0], memory_length=3) == (2.0, 5.0, 7.0)
