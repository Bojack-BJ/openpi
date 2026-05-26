from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.hf_adapter import Qwen25HLAdapter


def test_qwen_video_metadata_uses_configured_effective_fps():
    adapter = Qwen25HLAdapter(HLMemoryConfig(training_fps=20.0, frame_subsample=5))

    metadata = adapter._prepare_video_metadata([[object(), object(), object()]])

    assert metadata == [
        {
            "total_num_frames": 3,
            "fps": 4.0,
            "duration": 0.5,
            "frames_indices": [0, 1, 2],
        }
    ]
