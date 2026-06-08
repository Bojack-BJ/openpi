from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import torch

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.schema import render_language_memory_fields


def create_hf_adapter(config: HLMemoryConfig) -> "BaseHLVLMAdapter":
    if config.vlm_backend == "paligemma":
        raise NotImplementedError(
            "HL memory V1 now expects ordered video-style clips and no longer supports `paligemma`. "
            "Use `qwen2_5_vl`."
        )
    if config.vlm_backend == "qwen2_5_vl":
        return Qwen25HLAdapter(config)
    if config.vlm_backend == "qwen3_5_vl":
        return Qwen35HLAdapter(config)
    raise ValueError(f"Unsupported HL VLM backend: {config.vlm_backend}")


@dataclasses.dataclass(frozen=True)
class LoadedHLVLM:
    processor: Any
    model: Any


@dataclasses.dataclass(frozen=True)
class HLVLMGeneration:
    prediction: HLMemoryPrediction
    raw_output: str
    parse_error: str | None = None


class BaseHLVLMAdapter:
    def __init__(self, config: HLMemoryConfig):
        self.config = config

    def load(self, *, model_path: str | None = None, device: str | torch.device = "cpu") -> LoadedHLVLM:
        started_at = time.perf_counter()
        transformers = _import_transformers()
        resolved_device = torch.device(device)
        torch_dtype = self._resolve_torch_dtype()
        if resolved_device.type == "cpu" and torch_dtype != torch.float32:
            torch_dtype = torch.float32
        elif resolved_device.type == "cuda" and torch_dtype == torch.bfloat16 and not _cuda_supports_bfloat16():
            logging.warning(
                "CUDA device does not report bfloat16 support; using float16 for HL VLM inference. "
                "Pass `--precision float16` to request this explicitly."
            )
            torch_dtype = torch.float16
        pretrained_path = model_path or self.config.resolved_model_id
        peft_adapter_path = _as_peft_adapter_path(pretrained_path)
        logging.info("[stage] loading processor from %s", pretrained_path)
        processor = transformers.AutoProcessor.from_pretrained(pretrained_path, trust_remote_code=True)
        logging.info("[stage] processor loaded in %.1fs", time.perf_counter() - started_at)
        model_started_at = time.perf_counter()
        logging.info("[stage] loading model weights from %s dtype=%s", pretrained_path, torch_dtype)
        if peft_adapter_path is None:
            model = self._load_model(transformers, pretrained_path, torch_dtype=torch_dtype)
        else:
            model = self._load_peft_model(transformers, peft_adapter_path, torch_dtype=torch_dtype)
        logging.info("[stage] model weights loaded in %.1fs", time.perf_counter() - model_started_at)
        if self._uses_parallel_model_loading():
            logging.info(
                "Loaded HL VLM with parallel_mode=%s; skipping explicit model.to(%s).",
                self.config.parallel_mode,
                device,
            )
        else:
            move_started_at = time.perf_counter()
            logging.info("[stage] moving model to %s", device)
            model.to(device)
            logging.info("[stage] moved model to %s in %.1fs", device, time.perf_counter() - move_started_at)
        logging.info("[stage] switching model to eval mode")
        model.eval()
        logging.info("[stage] HL VLM load complete in %.1fs", time.perf_counter() - started_at)
        return LoadedHLVLM(processor=processor, model=model)

    def prepare_training_inputs(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> Mapping[str, torch.Tensor]:
        prompt_inputs = self._encode_prompt_only(loaded.processor, sample, clips)
        full_inputs = self._encode_prompt_and_target(loaded.processor, sample, clips)
        labels = full_inputs["input_ids"].clone()
        prompt_length = min(int(prompt_inputs["input_ids"].shape[-1]), int(labels.shape[-1]))
        labels[..., :prompt_length] = -100
        input_device = self._resolve_input_device(loaded.model, device)
        tensors = {
            key: value.to(input_device)
            for key, value in full_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tensors["labels"] = labels.to(input_device)
        return tensors

    def prepare_training_batch_inputs(
        self,
        loaded: LoadedHLVLM,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
        *,
        device: str | torch.device,
    ) -> Mapping[str, torch.Tensor]:
        if len(samples) != len(clips):
            raise ValueError(f"Expected one clip set per sample, got {len(samples)} samples and {len(clips)} clips.")
        if not samples:
            raise ValueError("Cannot prepare an empty HL training batch.")
        prompt_inputs = self._encode_batch_prompt_only(loaded.processor, samples, clips)
        full_inputs = self._encode_batch_prompt_and_target(loaded.processor, samples, clips)
        labels = _build_batched_labels(
            prompt_input_ids=prompt_inputs["input_ids"],
            prompt_attention_mask=prompt_inputs.get("attention_mask"),
            full_input_ids=full_inputs["input_ids"],
            full_attention_mask=full_inputs.get("attention_mask"),
        )
        input_device = self._resolve_input_device(loaded.model, device)
        tensors = {
            key: value.to(input_device)
            for key, value in full_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tensors["labels"] = labels.to(input_device)
        return tensors

    def predict(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLMemoryPrediction:
        return self.generate_prediction(loaded, sample, clips, device=device).prediction

    def predict_batch(
        self,
        loaded: LoadedHLVLM,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
        *,
        device: str | torch.device,
    ) -> list[HLMemoryPrediction]:
        return [generation.prediction for generation in self.generate_batch_predictions(loaded, samples, clips, device=device)]

    def generate_prediction(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLVLMGeneration:
        inputs = self._encode_prompt_only(loaded.processor, sample, clips)
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        generation_kwargs: dict[str, Any] = {}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        generated_ids = loaded.model.generate(
            **generation_inputs,
            max_new_tokens=self._generation_max_new_tokens(),
            do_sample=False,
            **generation_kwargs,
        )
        generated_suffix = generated_ids[:, input_ids.shape[-1] :]
        decoded = tokenizer.decode(generated_suffix[0], skip_special_tokens=True)
        try:
            prediction = HLMemoryPrediction.from_json(decoded).with_recent_position_limit(clips.recent_valid_length)
            return HLVLMGeneration(prediction=prediction, raw_output=decoded)
        except ValueError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "HL VLM output could not be parsed as JSON; using fallback prediction. raw_output=%r",
                decoded[:1000],
            )
            return HLVLMGeneration(
                prediction=self._fallback_prediction(sample),
                raw_output=decoded,
                parse_error=parse_error,
            )

    def generate_batch_predictions(
        self,
        loaded: LoadedHLVLM,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
        *,
        device: str | torch.device,
    ) -> list[HLVLMGeneration]:
        if len(samples) != len(clips):
            raise ValueError(f"Expected one clip set per sample, got {len(samples)} samples and {len(clips)} clips.")
        if not samples:
            return []
        inputs = self._encode_batch_prompt_only(loaded.processor, samples, clips)
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        generation_kwargs: dict[str, Any] = {}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        generated_ids = loaded.model.generate(
            **generation_inputs,
            max_new_tokens=self._generation_max_new_tokens(),
            do_sample=False,
            **generation_kwargs,
        )
        generated_suffix = generated_ids[:, input_ids.shape[-1] :]
        decoded_outputs = tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)
        generations: list[HLVLMGeneration] = []
        for sample, clip, decoded in zip(samples, clips, decoded_outputs, strict=True):
            try:
                prediction = HLMemoryPrediction.from_json(decoded).with_recent_position_limit(clip.recent_valid_length)
                generations.append(HLVLMGeneration(prediction=prediction, raw_output=decoded))
            except ValueError as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                logging.warning(
                    "HL VLM batch output could not be parsed as JSON; using fallback prediction. raw_output=%r",
                    decoded[:1000],
                )
                generations.append(
                    HLVLMGeneration(
                        prediction=self._fallback_prediction(sample),
                        raw_output=decoded,
                        parse_error=parse_error,
                    )
                )
        return generations

    def generate_image_text_response(
        self,
        loaded: LoadedHLVLM,
        *,
        system_prompt: str,
        user_text: str,
        images: list[Any],
        device: str | torch.device,
        max_new_tokens: int | None = None,
    ) -> str:
        rendered = self._render_image_text_messages(
            loaded.processor,
            system_prompt=system_prompt,
            user_text=user_text,
            image_count=len(images),
        )
        inputs = self._encode_image_text_processor_inputs(loaded.processor, rendered, images)
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        generation_kwargs: dict[str, Any] = {}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        generated_ids = loaded.model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens or self._generation_max_new_tokens(),
            do_sample=False,
            **generation_kwargs,
        )
        generated_suffix = generated_ids[:, input_ids.shape[-1] :]
        return tokenizer.decode(generated_suffix[0], skip_special_tokens=True)

    def build_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        if self.config.target_protocol == "memer_objective":
            return self._build_memer_objective_prompt(sample, clips)
        if self.config.target_protocol == "subtask_keyframe":
            return self._build_subtask_keyframe_prompt(sample, clips)
        previous_memory = sample.language_memory.strip() or "No progress has been recorded yet."
        step_prior = _render_step_prior(sample.step_prior)
        current_width, current_height = _current_recent_frame_size(clips)
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief: at most "
            f"{self.config.thinking_budget_tokens} tokens, then finish with exactly one final JSON object. "
            "Do not include reasoning after the JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains selected historical keyframes: sparse frames of particular importance from "
            "earlier actions the robot has executed.\n"
            "The second clip contains the recent observation window: the most recent actions the robot has executed.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of "
            f"{self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of "
            f"{self.config.recent_frames_length}.\n"
            "In each clip, positions are ordered from oldest to newest.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position "
            f"{clips.recent_valid_length} is the current frame. If the model internally sees 0-indexed frames, "
            f"frame 0 is oldest and frame {max(clips.recent_valid_length - 1, 0)} is current.\n"
            "`current_objective`, `phase`, `target_query`, `goal_query`, and `target_bbox_xyxy` must describe "
            "the state/action at the last valid recent frame, not the most salient earlier action in the clip.\n"
            "Use earlier recent frames only as temporal context for deciding how the final frame was reached.\n"
            "If there are two concatenated views in each frame, the left view shows the robot's left hand and the right"
            "view shows the robot's right hand. Predict the current objective and other fields from the perspective of"
            "the active hand, which is the hand currently moving, manipulating, or positioned at a target, and name the"
            "active hand in the objective, e.g., `approach with left hand` or `place with right hand`. If both hands are active, "
            "describe both hands in the same `current_objective` field, e.g., `left hand place, right hand hold`.\n"
            "Use visual evidence first. The HL input frames are raw RGB observations; no segmentation mask, mask "
            "overlay, or target highlight is provided. Treat language memory as a compact progress hint, and correct it when it "
            "conflicts with the current visual evidence.\n"
            "Manipulation segmentation rules:\n"
            "- Choose the current subtask from the visible robot behavior, not from the first object mentioned in "
            "the instruction or previous memory.\n"
            "- If the recent clip spans multiple objects or phases, follow the temporal order and report the phase "
            "visible at the last valid recent frame. Do not report an earlier object phase, such as a square-block "
            "phase, if later frames show another object, such as the crescent block, being manipulated, released, "
            "or followed by hand return.\n"
            "- A low-level subtask should be one primitive: approach, pick/grasp, transport above target, "
            "place/release, retreat/return, or transition to the next object.\n"
            "- Identify the active hand and active object from recent frames: the hand moving toward an object, "
            "contacting it, holding it, or releasing it at a target. Use the observed active object as "
            "`target_query` even if the instruction or previous memory mentions a different object.\n"
            "- For multi-object tasks, mark an object completed only after it is visibly released at its target. "
            "Do not advance to the next object until release plus return/transition is visually supported.\n"
            "- If the final recent frame shows an object already released at its slot, do not keep `current_objective` "
            "as place/release for that object unless release is happening in that final frame. Advance to the visible "
            "post-release state: retreat/return if a hand is moving back, transition to the next object if one "
            "remains, "
            "or task complete/hold position if all objects are placed and the robot is idle.\n"
            "- Use the nominal plan, if included in the task instruction, only as a segmentation prior. If the "
            "recent video contradicts that plan, correct to the observed manipulation step.\n"
            "- `target_query` should name the manipulated object or object part; `goal_query` should name the target "
            "slot/bin/plate/placement area. Do not put the full task instruction in either field.\n"
            "SAM grounding output:\n"
            f"- The current frame size is width={current_width}, height={current_height}. Pixel coordinates use x "
            f"from 0 to {max(current_width - 1, 0)} left-to-right and y from 0 to {max(current_height - 1, 0)} "
            "top-to-bottom.\n"
            "- If the current target instance is visible in the last valid recent frame, include `sam_text_prompt` "
            "as a short SAM/Grounding prompt and `sam_point_xy` as one pixel point near the center of that exact "
            "instance or object part.\n"
            "- Ground only the task-relevant target instance for the current subtask, not every object of the same "
            "class. If no target instance is visible, set `sam_text_prompt` to an empty string and omit "
            "`sam_point_xy`.\n"
            "BBox debug output:\n"
            "- If the current target instance is visible in the last valid recent frame, optionally include "
            "`target_bbox_xyxy` as [x1, y1, x2, y2] in current-frame pixel coordinates. The box should cover only "
            "the task-relevant target instance. If not visible, omit the field.\n"
            "Predict the current low-level objective and nominate recent frames worth remembering after they leave "
            "the recent window.\n"
            "Return exactly one JSON object with keys "
            "`task_progress`, `current_objective`, `relevant_objects`, `notes`, `keyframe_candidate_positions`, "
            "`phase`, `target_query`, `goal_query`, and optionally `subtask_progress`, "
            "`should_advance_objective`, `active_hand`, `sam_text_prompt`, `sam_point_xy`, `target_bbox_xyxy`.\n"
            f"{thinking_instruction}"
            "`task_progress` is a compact history summary. It should accumulate completed milestones or stable "
            "state, compressed into one short sentence. Do not put the current action here unless it is already "
            "completed or stable.\n"
            "`current_objective` is the one short executable objective for the downstream low-level policy.\n"
            "`relevant_objects` is a short JSON list of object/location phrases needed for the current objective.\n"
            "`notes` is one short caution or spatial fact, or `none`.\n"
            "`subtask_progress`, if returned, is a scalar in [0, 1] estimating completion of the current objective.\n"
            "`should_advance_objective`, if returned, is true only when the current objective appears complete.\n"
            "`active_hand`, if returned, should be `left`, `right`, `both`, or an empty string.\n"
            "Do not include timestamps, frame numbers, bullet lists, JSON inside the string, or per-step debug logs "
            "inside any string field.\n"
            "If the current objective is unchanged, update `task_progress` only when new action-useful state "
            "is observed; otherwise keep it semantically unchanged and concise.\n"
            "If memory is getting long or repetitive, compress it.\n"
            "`current_objective` must be a short executable robot instruction, not a passive state such as "
            "`the calculator is picked up`; that belongs in `task_progress`.\n"
            "`target_query` and `goal_query` must be short grounding phrases or noun phrases, not questions.\n"
            "`keyframe_candidate_positions` must be 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            f"Positions {clips.recent_valid_length + 1} to {self.config.recent_frames_length} are padding/fallback "
            "frames "
            "and must never be returned.\n"
            "Only nominate recent frames that contain information likely needed for future decisions, such as "
            "object locations, counted events, completed actions, failed attempts, or important state transitions. "
            "Prefer the first or last visually informative frame of such an event. Do not nominate redundant frames.\n"
            "If the recent window contains no information that needs future recall, return an empty list for "
            "`keyframe_candidate_positions`.\n"
            "Do not include markdown, explanation text, or extra keys outside the JSON object.\n"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"{step_prior}"
            f"Previous language memory: {previous_memory}\n"
        )

    def build_system_prompt(self) -> str:
        if self.config.target_protocol in {"memer_objective", "subtask_keyframe"}:
            thinking_instruction = (
                f"If thinking is used, keep it under {self.config.thinking_budget_tokens} tokens and place exactly one "
                "valid JSON object after the thinking text."
                if self.config.enable_thinking
                else "Do not output chain-of-thought or analysis text. Output only valid JSON."
            )
            return (
                "You are a robot high-level objective classifier and visual memory selector. Use the ordered "
                "historical keyframes and recent observation clip to predict the short-horizon executable objective "
                f"for the robot. {thinking_instruction}"
            )
        thinking_instruction = (
            f"If thinking is used, keep it under {self.config.thinking_budget_tokens} tokens and place exactly one "
            "valid JSON object after the thinking text."
            if self.config.enable_thinking
            else "Do not output chain-of-thought or analysis text. Output only valid JSON."
        )
        return (
            "You are a robot high-level memory policy. Your job is to choose the next language subtask for a "
            "low-level robot controller and to select task-relevant keyframes for future memory. Be concise, "
            "ground decisions in the provided images, and output only valid JSON. The recent observation clip is "
            "a past-to-current sequence; the last valid recent frame is the current state that must determine "
            f"`current_objective` or `current_subtask`. {thinking_instruction}"
        )

    def build_target_text(self, sample: ExportedHLMemorySample) -> str:
        prediction = sample.target_prediction(target_protocol=self.config.target_protocol)
        if self.config.target_protocol == "memer_objective":
            return json.dumps(
                {
                    "current_objective": prediction.current_objective,
                    "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        if self.config.target_protocol == "subtask_keyframe":
            return json.dumps(
                {
                    "current_subtask": prediction.current_subtask,
                    "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        if not prediction.sam_text_prompt and sample.target_query:
            prediction = dataclasses.replace(prediction, sam_text_prompt=sample.target_query)
        return prediction.to_json()

    def _build_memer_objective_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        horizon_text = (
            "Predict the short-horizon objective a few frames after the current frame, using recent motion as context.\n"
            if sample.horizon_frame_index is not None
            else "Predict the current executable objective at the last valid recent frame.\n"
        )
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains selected historical keyframes from earlier in the episode.\n"
            "The second clip contains the recent observation window, ordered oldest to newest.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "If each frame has two concatenated views, the left slot is the left hand and the right slot is the right hand.\n"
            f"{horizon_text}"
            "Return exactly one JSON object with keys `current_objective` and `keyframe_candidate_positions`.\n"
            "`current_objective` must be one short executable robot instruction.\n"
            "`keyframe_candidate_positions` must be a JSON list of 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            "Select only recent frames that should be kept as long-term visual memory for future decisions; return [] if none.\n"
            "Do not include progress, language memory, notes, markdown, explanation text, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
        )

    def _build_subtask_keyframe_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        horizon_text = (
            "Predict the short-horizon subtask a few frames after the current frame, using recent motion as context.\n"
            if sample.horizon_frame_index is not None
            else "Predict the current low-level subtask at the last valid recent frame.\n"
        )
        previous_memory = sample.language_memory.strip()
        memory_text = f"Previous language memory: {previous_memory}\n" if previous_memory else ""
        step_prior = _render_step_prior(sample.step_prior)
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains selected historical keyframes from earlier in the episode.\n"
            "The second clip contains the recent observation window, ordered oldest to newest.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "If each frame has two concatenated views, the left slot is the left hand and the right slot is the right hand.\n"
            f"{horizon_text}"
            "Use visual evidence in the last valid recent frame as the primary signal. Language memory and step prior, "
            "if provided, may be stale and should only be used as weak context.\n"
            "Return exactly one JSON object with keys `current_subtask` and `keyframe_candidate_positions`.\n"
            "`current_subtask` must be one short low-level robot primitive such as approach, grasp, transport above target, "
            "place/release, return/retreat, or transition to the next object.\n"
            "`keyframe_candidate_positions` must be a JSON list of 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            "Select recent frames that should be kept as long-term visual memory for future decisions, especially completed "
            "actions, object release/contact state, object locations, failures, or important state transitions. Return [] if none.\n"
            "Do not include progress, should_advance_objective, updated_language_memory, notes, markdown, explanation text, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"{step_prior}"
            f"{memory_text}"
        )

    def _fallback_prediction(self, sample: ExportedHLMemorySample) -> HLMemoryPrediction:
        previous_fields = _parse_ll_memory_fields(sample.language_memory)
        previous_progress = previous_fields.get("task progress", "").strip()
        current_objective = "continue the observed manipulation step"
        memory = render_language_memory_fields(
            task_progress=previous_progress or "Use current visual observations.",
            current_objective=current_objective,
            relevant_objects=(),
            notes="Parse failed; use current visual observations.",
        )
        return HLMemoryPrediction(
            updated_language_memory=memory,
            current_subtask=current_objective,
            keyframe_candidate_positions=(),
            phase="unknown",
            target_query="",
            goal_query="",
            task_progress=previous_progress or "Use current visual observations.",
            current_objective=current_objective,
            relevant_objects=(),
            notes="Parse failed; use current visual observations.",
        )

    def _render_image_text_messages(
        self,
        processor: Any,
        *,
        system_prompt: str,
        user_text: str,
        image_count: int,
    ) -> str:
        user_content = [{"type": "image"} for _ in range(image_count)]
        user_content.append({"type": "text", "text": user_text})
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        tokenizer = getattr(processor, "tokenizer", processor)
        apply_chat_template = getattr(processor, "apply_chat_template", None) or getattr(
            tokenizer, "apply_chat_template"
        )
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            **self._chat_template_kwargs(),
        }
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            kwargs.pop("enable_thinking", None)
            return apply_chat_template(messages, **kwargs)

    def _encode_image_text_processor_inputs(
        self,
        processor: Any,
        rendered: str,
        images: list[Any],
    ) -> Mapping[str, Any]:
        try:
            return processor(
                text=[rendered],
                images=images,
                text_kwargs={"padding": True, "return_tensors": "pt"},
                images_kwargs={"return_tensors": "pt"},
            )
        except TypeError as exc:
            if not _is_structured_processor_kwargs_error(exc):
                raise
            return processor(
                text=[rendered],
                images=images,
                padding=True,
                return_tensors="pt",
            )

    def _resolve_torch_dtype(self) -> torch.dtype:
        if self.config.precision == "bfloat16":
            return torch.bfloat16
        if self.config.precision == "float16":
            return torch.float16
        return torch.float32

    def _generation_max_new_tokens(self) -> int:
        if self.config.enable_thinking:
            return max(self.config.max_new_tokens, self.config.thinking_max_new_tokens)
        return self.config.max_new_tokens

    def _uses_parallel_model_loading(self) -> bool:
        return self.config.parallel_mode in {"device_map", "tensor_parallel"}

    def _parallel_from_pretrained_kwargs(self) -> dict[str, Any]:
        if self.config.parallel_mode == "none":
            return {}
        if self.config.parallel_mode == "device_map":
            return {"device_map": self.config.device_map}
        if self.config.parallel_mode == "tensor_parallel":
            return {"tp_plan": self.config.tensor_parallel_plan}
        raise ValueError(f"Unsupported parallel_mode: {self.config.parallel_mode}")

    def _resolve_input_device(self, model: Any, requested_device: str | torch.device) -> torch.device:
        if self._uses_parallel_model_loading():
            model_device = getattr(model, "device", None)
            if model_device is not None:
                resolved = torch.device(model_device)
                if resolved.type != "meta":
                    return resolved
            for parameter in model.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        return torch.device(requested_device)

    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        raise NotImplementedError

    def _load_peft_model(self, transformers: Any, adapter_path: pathlib.Path, *, torch_dtype: torch.dtype) -> Any:
        try:
            from peft import PeftConfig
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Loading a LoRA HL checkpoint requires `peft` to be installed.") from exc

        peft_config = PeftConfig.from_pretrained(str(adapter_path))
        base_model_path = peft_config.base_model_name_or_path or self.config.resolved_model_id
        logging.info("[stage] detected LoRA adapter; loading base model from %s", base_model_path)
        base_model = self._load_model(transformers, base_model_path, torch_dtype=torch_dtype)
        return PeftModel.from_pretrained(base_model, str(adapter_path))

    def _encode_prompt_only(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def _encode_batch_prompt_only(
        self,
        processor: Any,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
    ) -> Mapping[str, Any]:
        return _collate_processor_outputs(
            [self._encode_prompt_only(processor, sample, clip) for sample, clip in zip(samples, clips, strict=True)]
        )

    def _encode_batch_prompt_and_target(
        self,
        processor: Any,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
    ) -> Mapping[str, Any]:
        return _collate_processor_outputs(
            [
                self._encode_prompt_and_target(processor, sample, clip)
                for sample, clip in zip(samples, clips, strict=True)
            ]
        )


class Qwen25HLAdapter(BaseHLVLMAdapter):
    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        return transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **self._parallel_from_pretrained_kwargs(),
        )

    def _encode_prompt_only(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, clips, include_target=False)
        return self._encode_processor_inputs(processor, rendered, clips)

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        rendered = self._render_messages(processor, sample, clips, include_target=True)
        return self._encode_processor_inputs(processor, rendered, clips)

    def _encode_processor_inputs(
        self,
        processor: Any,
        rendered: str,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        videos = self._prepare_videos(clips)
        video_metadata = self._prepare_video_metadata(videos)
        try:
            return processor(
                text=[rendered],
                videos=videos,
                text_kwargs={"padding": True, "return_tensors": "pt"},
                videos_kwargs={
                    "do_sample_frames": False,
                    "video_metadata": video_metadata,
                    "return_tensors": "pt",
                },
            )
        except TypeError as exc:
            if not _is_structured_processor_kwargs_error(exc):
                raise
            try:
                return processor(
                    text=[rendered],
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                    do_sample_frames=False,
                    video_metadata=video_metadata,
                )
            except TypeError as fallback_exc:
                if not _is_structured_processor_kwargs_error(fallback_exc):
                    raise
                return processor(
                    text=[rendered],
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                )

    def _encode_batch_prompt_only(
        self,
        processor: Any,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
    ) -> Mapping[str, Any]:
        rendered = [
            self._render_messages(processor, sample, clip, include_target=False)
            for sample, clip in zip(samples, clips, strict=True)
        ]
        return self._encode_batch_processor_inputs(processor, rendered, clips)

    def _encode_batch_prompt_and_target(
        self,
        processor: Any,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
    ) -> Mapping[str, Any]:
        rendered = [
            self._render_messages(processor, sample, clip, include_target=True)
            for sample, clip in zip(samples, clips, strict=True)
        ]
        return self._encode_batch_processor_inputs(processor, rendered, clips)

    def _encode_batch_processor_inputs(
        self,
        processor: Any,
        rendered: list[str],
        clips: list[LoadedVideoClips],
    ) -> Mapping[str, Any]:
        videos = [video for clip in clips for video in self._prepare_videos(clip)]
        video_metadata = self._prepare_video_metadata(videos)
        try:
            return processor(
                text=rendered,
                videos=videos,
                text_kwargs={"padding": True, "return_tensors": "pt"},
                videos_kwargs={
                    "do_sample_frames": False,
                    "video_metadata": video_metadata,
                    "return_tensors": "pt",
                },
            )
        except TypeError as exc:
            if not _is_structured_processor_kwargs_error(exc):
                raise
            try:
                return processor(
                    text=rendered,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                    do_sample_frames=False,
                    video_metadata=video_metadata,
                )
            except TypeError as fallback_exc:
                if not _is_structured_processor_kwargs_error(fallback_exc):
                    raise
                return processor(
                    text=rendered,
                    videos=videos,
                    padding=True,
                    return_tensors="pt",
                )

    def _render_messages(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        include_target: bool,
    ) -> str:
        user_content = [
            {"type": "text", "text": "Historical memory clip, ordered oldest to newest."},
            {"type": "video"},
            {
                "type": "text",
                "text": (
                    "Recent observation clip, ordered oldest to newest. "
                    "The last valid frame is the current state to predict."
                ),
            },
            {"type": "video"},
            {"type": "text", "text": self.build_prompt(sample, clips)},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.build_system_prompt()}]},
            {"role": "user", "content": user_content},
        ]
        if include_target:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.build_target_text(sample)}],
                }
            )
        tokenizer = getattr(processor, "tokenizer", processor)
        apply_chat_template = getattr(processor, "apply_chat_template", None) or getattr(
            tokenizer, "apply_chat_template"
        )
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": not include_target,
            **self._chat_template_kwargs(),
        }
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            kwargs.pop("enable_thinking", None)
            return apply_chat_template(messages, **kwargs)

    def _prepare_videos(self, clips: LoadedVideoClips) -> list[list[Any]]:
        return [
            list(clips.memory_frames),
            list(clips.recent_frames),
        ]

    def _prepare_video_metadata(self, videos: list[list[Any]]) -> list[dict[str, Any]]:
        fps = self.config.video_fps
        return [
            {
                "total_num_frames": len(video),
                "fps": fps,
                "duration": float(max(len(video) - 1, 0)) / fps,
                "frames_indices": list(range(len(video))),
            }
            for video in videos
        ]

    def _chat_template_kwargs(self) -> dict[str, Any]:
        return {}


class Qwen35HLAdapter(Qwen25HLAdapter):
    def _chat_template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": self.config.enable_thinking}

    def _load_model(self, transformers: Any, pretrained_path: str, *, torch_dtype: torch.dtype) -> Any:
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if model_cls is None:
            raise NotImplementedError(
                "Loading Qwen3.5 HL memory models requires a recent `transformers` build with "
                "`AutoModelForImageTextToText` support. Qwen's model card recommends installing "
                "Transformers from main for Qwen3.5."
            )
        try:
            return model_cls.from_pretrained(
                pretrained_path,
                dtype=torch_dtype,
                trust_remote_code=True,
                **self._parallel_from_pretrained_kwargs(),
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            return model_cls.from_pretrained(
                pretrained_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **self._parallel_from_pretrained_kwargs(),
            )


def _import_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The HL memory adapters require `transformers` to be installed in the active environment."
        ) from exc
    return transformers


def _as_peft_adapter_path(pretrained_path: str) -> pathlib.Path | None:
    path = pathlib.Path(pretrained_path)
    if path.exists() and (path / "adapter_config.json").exists():
        return path
    return None


def _cuda_supports_bfloat16() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except RuntimeError:
        return False


def _is_structured_processor_kwargs_error(exc: TypeError) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "text_kwargs",
            "videos_kwargs",
            "images_kwargs",
            "video_metadata",
            "do_sample_frames",
        )
    )


def _build_batched_labels(
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    labels = full_input_ids.clone()
    if full_attention_mask is not None:
        labels = labels.masked_fill(full_attention_mask == 0, -100)
    for row_index in range(full_input_ids.shape[0]):
        prompt_ids = _nonpad_token_ids(prompt_input_ids[row_index], None if prompt_attention_mask is None else prompt_attention_mask[row_index])
        if not prompt_ids:
            continue
        full_mask = None if full_attention_mask is None else full_attention_mask[row_index]
        full_positions = _nonpad_positions(full_input_ids[row_index], full_mask)
        full_ids = [int(full_input_ids[row_index, position]) for position in full_positions]
        prompt_length = len(prompt_ids)
        if full_ids[:prompt_length] == prompt_ids:
            mask_positions = full_positions[:prompt_length]
        else:
            start = _find_subsequence(full_ids, prompt_ids)
            if start < 0:
                start = 0
                prompt_length = min(prompt_length, len(full_positions))
            mask_positions = full_positions[start : start + prompt_length]
        if mask_positions:
            labels[row_index, torch.tensor(mask_positions, device=labels.device)] = -100
    return labels


def _nonpad_token_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[int]:
    positions = _nonpad_positions(input_ids, attention_mask)
    return [int(input_ids[position]) for position in positions]


def _nonpad_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> list[int]:
    if attention_mask is None:
        return list(range(int(input_ids.shape[0])))
    return [int(index) for index in torch.nonzero(attention_mask.detach().cpu(), as_tuple=False).flatten().tolist()]


def _find_subsequence(values: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return 0
    max_start = len(values) - len(subsequence)
    for start in range(max_start + 1):
        if values[start : start + len(subsequence)] == subsequence:
            return start
    return -1


def _collate_processor_outputs(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty processor output list.")
    collated: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in items))
    for key in keys:
        values = [item[key] for item in items if key in item]
        if len(values) != len(items):
            continue
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.cat(values, dim=0)
        else:
            collated[key] = values
    return collated


def _parse_ll_memory_fields(memory: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in memory.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _is_usable_fallback_objective(objective: str, instruction: str) -> bool:
    normalized_objective = " ".join(objective.lower().split())
    normalized_instruction = " ".join(instruction.lower().split())
    primary_instruction = instruction.strip().splitlines()[0] if instruction.strip() else ""
    normalized_primary_instruction = " ".join(primary_instruction.lower().split())
    if not normalized_objective:
        return False
    if normalized_objective in {"continue the task", "task started"}:
        return False
    return normalized_objective not in {normalized_instruction, normalized_primary_instruction}


def _render_step_prior(step_prior: tuple[str, ...] | list[str]) -> str:
    steps = [str(step).strip() for step in step_prior if str(step).strip()]
    if not steps:
        return ""
    rendered = "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    return (
        "Task step prior:\n"
        f"{rendered}\n"
        "Use the step prior as a nominal ordered segmentation plan. Prefer visual evidence from the last valid "
        "recent frame if the plan and video disagree.\n"
    )


def _current_recent_frame_size(clips: LoadedVideoClips) -> tuple[int, int]:
    if clips.recent_valid_length <= 0 or not clips.recent_frames:
        return 0, 0
    index = min(clips.recent_valid_length, len(clips.recent_frames)) - 1
    size = getattr(clips.recent_frames[index], "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 0, 0
