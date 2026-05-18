import pathlib

from openpi.hl_memory.crosstask import CrossTaskSegment
from openpi.hl_memory.crosstask import build_subtask_annotations
from openpi.hl_memory.crosstask import read_segments
from openpi.hl_memory.crosstask import read_task_info
from openpi.hl_memory.crosstask import read_video_records


def test_read_task_info(tmp_path: pathlib.Path):
    path = tmp_path / "tasks_primary.txt"
    path.write_text(
        "\n".join(
            [
                "100",
                "make tea",
                "http://example.com/task",
                "2",
                "boil water,steep tea",
                "",
            ]
        )
        + "\n"
    )

    tasks = read_task_info(path)

    assert tasks["100"].title == "make tea"
    assert tasks["100"].steps == ("boil water", "steep tea")


def test_read_video_records(tmp_path: pathlib.Path):
    path = tmp_path / "videos.csv"
    path.write_text("100,abc123,http://example.com/video\n")

    records = read_video_records(path)

    assert records[0].task_id == "100"
    assert records[0].video_id == "abc123"


def test_read_segments_and_build_annotations(tmp_path: pathlib.Path):
    path = tmp_path / "100_abc123.csv"
    path.write_text("1,0.0,1.9\n2,2.0,4.0\n")
    segments = read_segments(path)
    task = read_task_info(
        tmp_path / "tasks_primary.txt"
    ) if False else None  # keep parser import exercised in this module
    del task

    class DummyTask:
        task_id = "100"
        steps = ("boil water", "steep tea")
        title = "make tea"

    annotations = build_subtask_annotations(
        episode_index=3,
        task=DummyTask(),
        segments=segments,
    )

    assert segments == [
        CrossTaskSegment(step_index=0, start_sec=0.0, end_sec=1.9),
        CrossTaskSegment(step_index=1, start_sec=2.0, end_sec=4.0),
    ]
    assert annotations[0].current_subtask == "boil water"
    assert annotations[0].event_type == "subtask_boundary"
    assert annotations[1].event_type == "success"
    assert annotations[2].current_subtask == "steep tea"
