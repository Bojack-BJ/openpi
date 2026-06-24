import dataclasses
import inspect
import json

from PIL import Image
import pytest

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.conditioning import render_progress_condition_text
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.hf_adapter import HLVLMGeneration
from openpi.hl_memory.hf_adapter import Qwen25HLAdapter
from openpi.hl_memory.hf_adapter import Qwen3VLHLAdapter
from openpi.hl_memory.hf_adapter import create_hf_adapter
from openpi.hl_memory.hf_adapter import _candidate_evidence_clips
from openpi.hl_memory.hf_adapter import _build_batched_field_ids
from openpi.hl_memory.hf_adapter import _keyframe_auxiliary_targets_for_sample
from openpi.hl_memory.hf_adapter import _parse_completed_objective_text
from openpi.hl_memory.proprio import PROPRIO_FRAME_TOKEN
from openpi.hl_memory.proprio import PROPRIO_SUMMARY_TOKEN
from openpi.hl_memory.proprio import build_proprio_batch
from openpi.hl_memory.proprio import render_proprio_token_text
from openpi.hl_memory.schema import HLMemoryPrediction


def test_create_hf_adapter_routes_qwen3_vl_backend():
    adapter = create_hf_adapter(HLMemoryConfig(vlm_backend="qwen3_vl"))

    assert isinstance(adapter, Qwen3VLHLAdapter)


class _CharOffsetTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        assert not add_special_tokens
        encoded = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded


def test_qwen_video_metadata_uses_configured_recent_sample_rate():
    adapter = Qwen25HLAdapter(HLMemoryConfig(training_fps=20.0, frame_subsample=5, recent_sample_hz=4.0))

    metadata = adapter._prepare_video_metadata([[object(), object(), object()]])

    assert metadata == [
        {
            "total_num_frames": 3,
            "fps": 4.0,
            "duration": 0.5,
            "frames_indices": [0, 1, 2],
        }
    ]


def test_memer_objective_target_text_is_minimal_and_uses_current_and_horizon_labels():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="memer_objective"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="memory",
        updated_language_memory="updated",
        current_subtask="current step",
        phase="current phase",
        target_query="target",
        goal_query="goal",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="current objective",
        horizon_current_objective="horizon objective",
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "current objective",
        "horizon_current_objective": "horizon objective",
        "keyframe_candidate_positions": [2],
    }


def test_field_ids_allow_chat_template_suffix_tokens():
    import torch

    target_text = '{"current_objective":"pick","horizon_current_objective":"place"}'
    target_ids = [ord(char) for char in target_text]
    labels = torch.tensor([[-100, *target_ids, 999]], dtype=torch.long)

    field_ids = _build_batched_field_ids(
        tokenizer=_CharOffsetTokenizer(),
        labels=labels,
        target_texts=(target_text,),
    )

    assert field_ids is not None
    assert field_ids.shape == labels.shape
    assert field_ids[0, 0].item() == -1
    assert field_ids[0, -1].item() == -1
    assert bool((field_ids[0, 1:-1] != -1).any())


def test_field_ids_skip_when_supervised_tokens_do_not_match_target_text(caplog):
    import logging
    import torch

    target_text = '{"current_objective":"pick"}'
    target_ids = [ord(char) for char in target_text]
    target_ids[3] += 1
    labels = torch.tensor([[-100, *target_ids]], dtype=torch.long)

    with caplog.at_level(logging.WARNING):
        field_ids = _build_batched_field_ids(
            tokenizer=_CharOffsetTokenizer(),
            labels=labels,
            target_texts=(target_text,),
        )

    assert field_ids is None
    assert "supervised target tokens do not match target_text" in caplog.text


def test_subtask_keyframe_target_text_is_minimal_and_uses_current_objective():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="subtask_keyframe"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="memory",
        updated_language_memory="updated",
        current_subtask="current step",
        phase="current phase",
        target_query="target",
        goal_query="goal",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="current objective",
        horizon_current_objective="horizon objective",
        horizon_current_subtask="horizon subtask",
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "current objective",
        "keyframe_candidate_positions": [2],
    }


def test_keyframe_gated_memory_target_text_uses_completed_objective():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="updated",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="place toast",
        horizon_current_objective="grasp steak",
        keyframe_label=True,
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "place toast",
        "horizon_current_objective": "grasp steak",
        "keyframe_candidate_positions": [2],
        "completed_objective": "place toast",
    }


def test_keyframe_auxiliary_targets_align_proposal_with_history_update():
    config = HLMemoryConfig(
        keyframe_aux_enabled=True,
        recent_frames_length=4,
        training_fps=20.0,
        keyframe_aux_timing_sigma_sec=0.5,
    )
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=20,
        instruction="task",
        language_memory="",
        updated_language_memory="",
        current_subtask="place",
        phase="place",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(3,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("0.png", "1.png", "2.png", "3.png"),
        recent_frame_indices=(5, 10, 15, 20),
        recent_valid_length=4,
        keyframe_event_ids=("episode:0:event:1",),
        keyframe_event_frame_indices=(20,),
    )

    add_target = _keyframe_auxiliary_targets_for_sample(sample, config=config, enabled=True)
    duplicate_target = _keyframe_auxiliary_targets_for_sample(
        dataclasses.replace(sample, memory_frame_indices=(20,)),
        config=config,
        enabled=True,
    )

    assert add_target["event_target"] == 1.0
    assert add_target["update_target"] == 1
    assert duplicate_target["update_target"] == 2
    assert add_target["canonical_position"] == 3.0
    assert max(range(4), key=lambda index: add_target["position_targets"][index]) == 3


def test_typed_mask_target_matches_single_pass_keyframe_gated_target():
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="updated",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="place toast",
        horizon_current_objective="grasp steak",
        keyframe_label=True,
    )
    single_pass = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory"))
    typed_mask = Qwen25HLAdapter(
        HLMemoryConfig(vlm_backend="qwen2_5_vl", target_protocol="keyframe_gated_memory_typed_mask")
    )

    assert json.loads(typed_mask.build_target_text(sample)) == json.loads(single_pass.build_target_text(sample))


def test_keyframe_gated_system_prompt_assigns_evidence_by_output_role():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory"))

    prompt = adapter.build_system_prompt()

    assert "recent observation clip as the primary visual evidence" in prompt
    assert "historical memory only as non-Markovian context" in prompt
    assert "Select keyframe candidates only from the recent observation clip" in prompt
    assert "visually confirms completion" in prompt


def test_two_pass_system_prompt_separates_proposal_and_confirmation():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory_two_pass"))

    prompt = adapter.build_system_prompt()

    assert "In Pass A" in prompt
    assert "propose recent keyframe evidence" in prompt
    assert "In Pass B" in prompt
    assert "only from the proposed candidate evidence" in prompt


def test_keyframe_gated_memory_prompt_uses_completed_event_log_without_free_memory_target():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="must not be prompted as target",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="place toast",
    )
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(object(),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    prompt = adapter.build_prompt(sample, clips)

    assert "Completed-event log: Completed events: grasp toast." in prompt
    assert "completed_objective" in prompt
    assert "updated_language_memory" in prompt
    assert "must not be prompted as target" not in prompt


def test_keyframe_gated_two_pass_targets_are_typed():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory_two_pass"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="must not be supervised",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="place toast",
        horizon_current_objective="grasp steak",
        keyframe_label=True,
    )

    pass_a = json.loads(adapter._build_two_pass_target_text(sample, stage="predict"))
    pass_b = json.loads(adapter._build_two_pass_target_text(sample, stage="confirm"))

    assert pass_a == {
        "current_objective": "place toast",
        "horizon_current_objective": "grasp steak",
        "keyframe_candidate_positions": [2],
    }
    assert pass_b == {"completed_objective": "place toast"}


def test_film_progress_two_pass_splits_current_and_horizon_targets():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="memer_film_progress_two_pass"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Completed events: grasp toast.",
        updated_language_memory="must not be supervised",
        current_subtask="place toast",
        phase="place toast",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="place toast",
        horizon_current_objective="grasp steak",
        keyframe_label=True,
    )

    current = json.loads(adapter._build_two_pass_target_text(sample, stage="predict"))
    horizon = json.loads(adapter._build_two_pass_target_text(sample, stage="horizon"))
    confirm = json.loads(adapter._build_two_pass_target_text(sample, stage="confirm"))

    assert current == {"current_objective": "place toast", "keyframe_candidate_positions": [2]}
    assert horizon == {"horizon_current_objective": "grasp steak"}
    assert confirm == {"completed_objective": "place toast"}


def test_film_progress_pass_a_prompt_hides_raw_memory_text():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="memer_film_progress_two_pass"))
    sample = _minimal_sample()
    sample = dataclasses.replace(
        sample,
        language_memory="Completed events: leaked completed text.\nCurrent objective: leaked current.",
    )
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(Image.new("RGB", (8, 6)),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    prompt = adapter._build_keyframe_gated_two_pass_predict_prompt(sample, clips)

    assert "leaked completed text" not in prompt
    assert "leaked current" not in prompt
    assert "learned low-bandwidth condition" in prompt


def test_progress_condition_completed_only_drops_current_like_fields():
    config = HLMemoryConfig(
        target_protocol="memer_film_progress_two_pass",
        progress_condition_enabled=True,
        progress_condition_input_mode="completed_only",
    )
    sample = dataclasses.replace(
        _minimal_sample(),
        language_memory=(
            "Task progress: toast has been placed.\n"
            "Current objective: leaked current objective.\n"
            "Phase: leaked phase.\n"
            "Completed events: toast placed."
        ),
        current_objective="gt current must not leak",
        phase="gt phase must not leak",
        task_progress="gt task progress must not leak",
    )

    text = render_progress_condition_text(sample, config)

    assert "toast has been placed" in text
    assert "toast placed" in text
    assert "leaked current objective" not in text
    assert "leaked phase" not in text
    assert "gt current must not leak" not in text
    assert "gt task progress must not leak" not in text


def test_two_pass_confirm_clip_contains_only_candidate_frames():
    frames = tuple(Image.new("RGB", (8, 6), color=(index, 0, 0)) for index in range(4))
    clips = LoadedVideoClips(
        memory_frames=(Image.new("RGB", (8, 6)),),
        recent_frames=frames,
        memory_valid_length=1,
        recent_valid_length=4,
    )

    routed = _candidate_evidence_clips(clips, (4, 2, 2, 9))

    assert routed.memory_frames == clips.memory_frames
    assert routed.recent_valid_length == 2
    assert routed.recent_frames == (frames[1], frames[3])


def test_two_pass_confirm_clip_uses_black_placeholder_for_no_candidates():
    current = Image.new("RGB", (8, 6), color=(255, 255, 255))
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(current,),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    routed = _candidate_evidence_clips(clips, ())

    assert routed.recent_valid_length == 0
    assert len(routed.recent_frames) == 1
    assert routed.recent_frames[0].size == (8, 6)
    assert routed.recent_frames[0].getpixel((0, 0)) == (0, 0, 0)


def test_two_pass_skips_confirm_generation_when_pass_a_has_no_candidates():
    class StubAdapter(Qwen25HLAdapter):
        stages: list[str]

        def __init__(self):
            super().__init__(HLMemoryConfig(target_protocol="keyframe_gated_memory_two_pass"))
            self.stages = []

        def _generate_two_pass_stage(self, loaded, sample, clips, *, stage, pass_a_prediction, device):
            del loaded, sample, clips, pass_a_prediction, device
            self.stages.append(stage)
            return HLVLMGeneration(
                prediction=HLMemoryPrediction(
                    updated_language_memory="",
                    current_subtask="current",
                    keyframe_candidate_positions=(),
                    phase="current",
                    target_query="",
                    goal_query="",
                    current_objective="current",
                    completed_objective="must be cleared",
                ),
                raw_output='{"current_objective":"current","keyframe_candidate_positions":[]}',
            )

    adapter = StubAdapter()
    sample = _minimal_sample()
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(Image.new("RGB", (8, 6)),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    generation = adapter._generate_two_pass_prediction(object(), sample, clips, device="cpu")

    assert adapter.stages == ["predict"]
    assert generation.prediction.completed_objective == ""
    assert json.loads(generation.raw_output)["pass_b"] is None


def test_two_pass_completed_objective_parser_distinguishes_invalid_json_from_empty_negative():
    assert _parse_completed_objective_text('{"completed_objective":""}') == ""
    with pytest.raises(ValueError, match="No JSON"):
        _parse_completed_objective_text("not json")
    with pytest.raises(ValueError, match="exactly"):
        _parse_completed_objective_text('{"completed_objective":"","extra":1}')
    with pytest.raises(ValueError, match="must be a string"):
        _parse_completed_objective_text('{"completed_objective":false}')


def test_two_pass_confirm_sample_routes_only_candidate_aligned_proprio():
    adapter = Qwen25HLAdapter(
        HLMemoryConfig(
            target_protocol="keyframe_gated_memory_two_pass",
            proprio_enabled=True,
            proprio_token_mode="per_frame_plus_summary",
        )
    )
    sample = dataclasses.replace(
        _minimal_sample(recent_length=3),
        recent_robot_states=((0.0,) * 14, (1.0,) * 14, (2.0,) * 14),
        recent_robot_state_masks=((1.0,) * 14,) * 3,
    )
    prediction = sample.target_prediction(
        target_protocol="keyframe_gated_memory_two_pass",
        keyframe_candidate_label_mode="event_band",
    )
    prediction = dataclasses.replace(prediction, keyframe_candidate_positions=(2,))

    routed = adapter._two_pass_stage_sample(
        sample,
        stage="confirm",
        pass_a_prediction=prediction,
    )

    assert routed.recent_valid_length == 1
    assert routed.recent_robot_states == ((1.0,) * 14,)
    assert render_proprio_token_text(routed, adapter.config).count(PROPRIO_FRAME_TOKEN) == 1


def test_two_pass_confirm_prompt_hides_pass_a_semantic_targets():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="keyframe_gated_memory_two_pass"))
    sample = _minimal_sample()
    prediction = HLMemoryPrediction(
        updated_language_memory="",
        current_subtask="leaked current objective",
        keyframe_candidate_positions=(1,),
        phase="leaked current objective",
        target_query="",
        goal_query="",
        current_objective="leaked current objective",
        horizon_current_objective="leaked horizon objective",
    )
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(Image.new("RGB", (8, 6)),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    prompt = adapter._build_keyframe_gated_two_pass_confirm_prompt(
        sample,
        clips,
        pass_a_prediction=prediction,
    )

    assert "leaked current objective" not in prompt
    assert "leaked horizon objective" not in prompt
    assert '"keyframe_candidate_positions":[1]' in prompt


def test_two_pass_training_uses_false_proposal_frames_for_negative_confirmation():
    adapter = Qwen25HLAdapter(
        HLMemoryConfig(
            target_protocol="keyframe_gated_memory_two_pass",
            two_pass_training_proposal_noise_probability=0.0,
        )
    )
    sample = _minimal_sample(recent_length=3)
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=tuple(Image.new("RGB", (8, 6)) for _ in range(3)),
        memory_valid_length=0,
        recent_valid_length=3,
    )

    prediction = adapter._training_pass_a_prediction(sample, clips)
    routed = adapter._two_pass_stage_clips(
        sample,
        clips,
        stage="confirm",
        pass_a_prediction=prediction,
    )

    assert len(prediction.keyframe_candidate_positions) == 1
    assert routed.recent_valid_length == 1
    assert prediction.completed_objective == ""


def test_two_pass_validation_keeps_clean_gt_routing():
    adapter = Qwen25HLAdapter(
        HLMemoryConfig(
            target_protocol="keyframe_gated_memory_two_pass",
            two_pass_training_proposal_noise_probability=1.0,
        )
    )
    sample = _minimal_sample(recent_length=3)
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=tuple(Image.new("RGB", (8, 6)) for _ in range(3)),
        memory_valid_length=0,
        recent_valid_length=3,
    )

    expanded = adapter._encoded_samples_for_training(
        [sample],
        [clips],
        apply_proposal_noise=False,
    )

    assert len(expanded) == 2
    assert expanded[1].recent_valid_length == 0


def test_qwen_batch_encoders_accept_two_pass_training_noise_switch():
    prompt_signature = inspect.signature(Qwen25HLAdapter._encode_batch_prompt_only)
    target_signature = inspect.signature(Qwen25HLAdapter._encode_batch_prompt_and_target)

    assert "apply_two_pass_training_noise" in prompt_signature.parameters
    assert "apply_two_pass_training_noise" in target_signature.parameters


def test_non_two_pass_message_rendering_does_not_reference_stage():
    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return json.dumps(messages)

    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="memer_objective"))
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(Image.new("RGB", (8, 6)),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    rendered = adapter._render_messages(FakeProcessor(), _minimal_sample(), clips, include_target=False)

    assert "Recent observation clip" in rendered


def _minimal_sample(*, recent_length: int = 1) -> ExportedHLMemorySample:
    return ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=recent_length - 1,
        instruction="task",
        language_memory="",
        updated_language_memory="",
        current_subtask="current",
        phase="current",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=tuple(f"recent-{index}.png" for index in range(recent_length)),
        recent_frame_indices=tuple(range(recent_length)),
        recent_valid_length=recent_length,
        current_objective="current",
    )


def test_known_prior_tracker_target_text_only_supervises_tracker_outputs():
    adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol="known_prior_tracker"))
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Task progress: none\nCurrent objective: current objective",
        updated_language_memory="updated",
        current_subtask="current objective",
        phase="current objective",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="current objective",
        subtask_progress=0.6,
        should_advance_objective=False,
    )

    payload = json.loads(adapter.build_target_text(sample))

    assert payload == {
        "current_objective": "current objective",
        "subtask_progress": 0.6,
        "should_advance_objective": False,
        "keyframe_candidate_positions": [2],
    }


def test_state_context_objective_protocol_targets_are_memer_style():
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Task progress: grasp done\nCurrent objective: insert top-left",
        updated_language_memory="updated",
        current_subtask="insert top-left",
        phase="insert top-left",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="insert top-left",
        horizon_current_objective="insert top-right",
        last_objective="grasp stick",
        previous_stage_objective="grasp stick",
        subtask_progress=0.6,
        should_advance_objective=False,
    )

    expected_state_fields = {
        "objective_memory_state": {"updated_language_memory": "updated"},
        "objective_last_objective": {"last_objective": "grasp stick"},
        "objective_prev_stage": {"previous_stage_objective": "grasp stick"},
    }
    for protocol in ("objective_memory_state", "objective_last_objective", "objective_prev_stage"):
        adapter = Qwen25HLAdapter(HLMemoryConfig(target_protocol=protocol))

        payload = json.loads(adapter.build_target_text(sample))

        assert payload == {
            "current_objective": "insert top-left",
            "horizon_current_objective": "insert top-right",
            "keyframe_candidate_positions": [2],
            **expected_state_fields[protocol],
        }


def test_state_context_objective_prompts_use_selected_context_only():
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="Task progress: grasp done\nCurrent objective: insert top-left",
        updated_language_memory="updated",
        current_subtask="insert top-left",
        phase="insert top-left",
        target_query="",
        goal_query="",
        keyframe_candidate_positions=(2,),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent.png",),
        recent_frame_indices=(0,),
        recent_valid_length=1,
        current_objective="insert top-left",
        last_objective="last inferred objective",
        previous_stage_objective="previous distinct objective",
    )
    clips = LoadedVideoClips(
        memory_frames=(),
        recent_frames=(object(),),
        memory_valid_length=0,
        recent_valid_length=1,
    )

    memory_prompt = Qwen25HLAdapter(HLMemoryConfig(target_protocol="objective_memory_state")).build_prompt(sample, clips)
    last_prompt = Qwen25HLAdapter(HLMemoryConfig(target_protocol="objective_last_objective")).build_prompt(sample, clips)
    prev_prompt = Qwen25HLAdapter(HLMemoryConfig(target_protocol="objective_prev_stage")).build_prompt(sample, clips)

    assert "Completed-subtasks memory: Task progress: grasp done" in memory_prompt
    assert "last inferred objective" not in last_prompt
    assert "last_objective" in last_prompt
    assert "Previous stage objective: previous distinct objective" in prev_prompt
    assert "horizon_current_objective" in memory_prompt
    assert "last inferred objective" not in memory_prompt
    assert "previous distinct objective" not in last_prompt


def test_proprio_per_frame_plus_summary_renders_expected_token_count():
    config = HLMemoryConfig(proprio_enabled=True, proprio_token_mode="per_frame_plus_summary")
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="memory",
        updated_language_memory="updated",
        current_subtask="current step",
        phase="current phase",
        target_query="target",
        goal_query="goal",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent0.png", "recent1.png"),
        recent_frame_indices=(0, 1),
        recent_valid_length=2,
        recent_robot_states=((0.0,) * 14, (1.0,) * 14),
        recent_robot_state_masks=((1.0,) * 14, (1.0,) * 14),
    )

    rendered = render_proprio_token_text(sample, config)

    assert rendered.count(PROPRIO_SUMMARY_TOKEN) == 1
    assert rendered.count(PROPRIO_FRAME_TOKEN) == 2


def test_proprio_batch_uses_sample_recent_state_shape():
    import torch

    config = HLMemoryConfig(proprio_enabled=True, proprio_token_mode="per_frame", proprio_state_dim=14)
    sample = ExportedHLMemorySample(
        sample_id="sample",
        episode_index=0,
        step_index=0,
        frame_index=0,
        instruction="task",
        language_memory="memory",
        updated_language_memory="updated",
        current_subtask="current step",
        phase="current phase",
        target_query="target",
        goal_query="goal",
        keyframe_candidate_positions=(),
        memory_frame_paths=(),
        memory_frame_indices=(),
        memory_valid_length=0,
        recent_frame_paths=("recent0.png", "recent1.png"),
        recent_frame_indices=(0, 1),
        recent_valid_length=2,
        recent_robot_states=((0.0,) * 14, (1.0,) * 14),
        recent_robot_state_masks=((1.0,) * 7 + (0.0,) * 7, (1.0,) * 14),
    )

    states, masks = build_proprio_batch([sample], config, device=torch.device("cpu"))

    assert tuple(states.shape) == (1, 2, 14)
    assert tuple(masks.shape) == (1, 2, 14)
    assert masks[0, 0, 7:].sum().item() == 0.0
