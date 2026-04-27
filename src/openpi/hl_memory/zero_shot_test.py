from openpi.hl_memory.zero_shot import build_auto_memory_seconds
from openpi.hl_memory.zero_shot import build_recent_seconds
from openpi.hl_memory.zero_shot import parse_seconds_argument


def test_parse_seconds_argument_sorts_and_dedupes():
    assert parse_seconds_argument("3,1,1,2.5") == [1.0, 2.5, 3.0]


def test_build_recent_seconds_defaults_from_video_end():
    assert build_recent_seconds(10.0, clip_length=4, recent_step_sec=2.0) == [4.0, 6.0, 8.0, 10.0]


def test_build_auto_memory_seconds_uses_prefix_before_recent():
    assert build_auto_memory_seconds(20.0, recent_seconds=[12.0, 13.0], clip_length=3) == [0.0, 5.9995, 11.999]
