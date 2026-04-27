import numpy as np
from PIL import Image

from openpi.hl_memory.frame_composer import compose_context_panel
from openpi.hl_memory.frame_composer import compose_observation_frame


def test_compose_observation_frame_returns_rgb_image():
    image = compose_observation_frame(
        {
            "cam_b": np.zeros((16, 16, 3), dtype=np.uint8),
            "cam_a": np.ones((16, 16, 3), dtype=np.float32),
        },
        frame_height=32,
        frame_width=32,
    )

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.width == 72
    assert image.height == 32


def test_compose_context_panel_stacks_memory_and_recent_sections():
    memory_frames = [Image.new("RGB", (32, 32), color=(255, 0, 0))]
    recent_frames = [Image.new("RGB", (32, 32), color=(0, 255, 0))]

    image = compose_context_panel(
        memory_frames,
        recent_frames,
        frame_height=32,
        frame_width=32,
        columns=2,
        gap=4,
    )

    assert image.mode == "RGB"
    assert image.width == 68
    assert image.height == 68
