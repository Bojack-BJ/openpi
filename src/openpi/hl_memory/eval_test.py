from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.eval import evaluate_ablation_modes
from openpi.hl_memory.schema import HLMemoryPrediction


def test_evaluate_ablation_modes_tracks_sequence_accuracy():
    samples = [
        ExportedHLMemorySample(
            sample_id="ep0_step0",
            episode_index=0,
            step_index=0,
            frame_index=0,
            instruction="sort fruit",
            language_memory="No progress has been recorded yet.",
            updated_language_memory="Current phase: pick.",
            current_subtask="pick apple",
            phase="pick",
            target_query="apple",
            goal_query="basket",
            keyframe_candidate_positions=(1,),
            memory_frame_paths=(),
            memory_frame_indices=(),
            memory_valid_length=0,
            recent_frame_paths=("frames/ep0_0.png",),
            recent_frame_indices=(0,),
            recent_valid_length=1,
        ),
        ExportedHLMemorySample(
            sample_id="ep0_step1",
            episode_index=0,
            step_index=1,
            frame_index=5,
            instruction="sort fruit",
            language_memory="Current phase: pick.",
            updated_language_memory="Completed subtasks: pick apple. Current phase: place.",
            current_subtask="place apple",
            phase="place",
            target_query="apple",
            goal_query="basket",
            keyframe_candidate_positions=(1,),
            memory_frame_paths=("frames/ep0_0.png",),
            memory_frame_indices=(0,),
            memory_valid_length=1,
            recent_frame_paths=("frames/ep0_5.png",),
            recent_frame_indices=(5,),
            recent_valid_length=1,
            event_type="success",
            event_text="Completed pick apple.",
        ),
    ]

    def predict(sample: ExportedHLMemorySample) -> HLMemoryPrediction:
        return sample.target_prediction()

    metrics = evaluate_ablation_modes(samples, HLMemoryConfig(), predict)

    assert metrics["full"]["subtask_exact_match"] == 1.0
    assert metrics["full"]["episode_sequence_accuracy"] == 1.0
