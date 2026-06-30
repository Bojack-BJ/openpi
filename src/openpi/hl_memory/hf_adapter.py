from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import logging
import math
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image

from openpi.hl_memory.config import HLMemoryConfig
from openpi.hl_memory.conditioning import build_conditioning_batch
from openpi.hl_memory.conditioning import configure_conditioning_model
from openpi.hl_memory.conditioning import HL_CONDITION_STAGE_IDS_KEY
from openpi.hl_memory.conditioning import STAGE_CONFIRM
from openpi.hl_memory.conditioning import STAGE_HORIZON
from openpi.hl_memory.conditioning import STAGE_PREDICT
from openpi.hl_memory.data import ExportedHLMemorySample
from openpi.hl_memory.data import LoadedVideoClips
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_ANCHOR_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_CANONICAL_POSITIONS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_EVENT_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_MASK_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_POSITION_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_UPDATE_TARGETS_KEY
from openpi.hl_memory.keyframe_auxiliary import KEYFRAME_AUX_VALID_POSITIONS_KEY
from openpi.hl_memory.proprio import configure_model_for_proprio
from openpi.hl_memory.proprio import load_proprio_state_if_available
from openpi.hl_memory.proprio import render_proprio_token_text
from openpi.hl_memory.proprio import set_model_proprio_batch
from openpi.hl_memory.schema import HLMemoryPrediction
from openpi.hl_memory.schema import render_language_memory_fields
from openpi.hl_memory.typed_attention import build_qwen25_typed_attention_mask
from openpi.hl_memory.training_loss import HL_FIELD_IDS_KEY
from openpi.hl_memory.training_loss import HL_FIELD_VALUE_MASK_KEY
from openpi.hl_memory.training_loss import HL_LOSS_FIELD_IDS_BY_NAME


_TYPED_MASK_PROTOCOL = "keyframe_gated_memory_typed_mask"
_FILM_PROGRESS_TWO_PASS_PROTOCOL = "memer_film_progress_two_pass"
_TWO_PASS_PROTOCOLS = {"keyframe_gated_memory_two_pass", _FILM_PROGRESS_TWO_PASS_PROTOCOL}


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
    if config.vlm_backend == "qwen3_vl":
        return Qwen3VLHLAdapter(config)
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
        processor_path = pretrained_path
        try:
            logging.info("[stage] loading processor from %s", processor_path)
            processor = transformers.AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        except ValueError as exc:
            fallback_processor_path = self.config.resolved_model_id
            if pathlib.Path(str(pretrained_path)).resolve() == pathlib.Path(str(fallback_processor_path)).resolve():
                raise
            logging.warning(
                "Could not load processor from %s (%s); falling back to base model %s.",
                processor_path,
                exc,
                fallback_processor_path,
            )
            processor_path = fallback_processor_path
            processor = transformers.AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only generation must left-pad batched prompts so the last
            # input position is a real token for every sample.
            tokenizer.padding_side = "left"
        logging.info("[stage] processor loaded in %.1fs", time.perf_counter() - started_at)
        model_started_at = time.perf_counter()
        logging.info("[stage] loading model weights from %s dtype=%s", pretrained_path, torch_dtype)
        if peft_adapter_path is None:
            model = self._load_model(transformers, pretrained_path, torch_dtype=torch_dtype)
        else:
            model = self._load_peft_model(transformers, peft_adapter_path, torch_dtype=torch_dtype)
        configure_model_for_proprio(model, processor, self.config)
        load_proprio_state_if_available(model, pathlib.Path(pretrained_path))
        if self.config.progress_condition_enabled or self.config.state_condition_enabled:
            model = configure_conditioning_model(model, self.config, checkpoint_dir=pathlib.Path(pretrained_path))
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
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return self.prepare_training_batch_inputs(loaded, [sample], [clips], device=device)
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
        field_annotations = _build_batched_field_annotations(
            tokenizer=getattr(loaded.processor, "tokenizer", loaded.processor),
            labels=labels,
            target_texts=(self.build_target_text(sample),),
        )
        if field_annotations is not None:
            field_ids, field_value_mask = field_annotations
            tensors[HL_FIELD_IDS_KEY] = field_ids.to(input_device)
            tensors[HL_FIELD_VALUE_MASK_KEY] = field_value_mask.to(input_device)
        self._add_keyframe_auxiliary_targets(tensors, [sample], labels=labels, device=input_device)
        self._apply_typed_training_mask(
            loaded,
            tensors,
            tokenizer=getattr(loaded.processor, "tokenizer", loaded.processor),
        )
        set_model_proprio_batch(
            loaded.model,
            self._encoded_samples_for_training([sample], [clips], apply_proposal_noise=loaded.model.training),
            self.config,
            device=input_device,
        )
        self._add_conditioning_inputs(
            tensors,
            loaded=loaded,
            samples=[sample],
            device=input_device,
        )
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
        apply_proposal_noise = loaded.model.training
        prompt_inputs = self._encode_batch_prompt_only(
            loaded.processor,
            samples,
            clips,
            apply_two_pass_training_noise=apply_proposal_noise,
        )
        full_inputs = self._encode_batch_prompt_and_target(
            loaded.processor,
            samples,
            clips,
            apply_two_pass_training_noise=apply_proposal_noise,
        )
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
        target_texts = self._training_target_texts(samples, clips, apply_two_pass_training_noise=apply_proposal_noise)
        field_annotations = _build_batched_field_annotations(
            tokenizer=getattr(loaded.processor, "tokenizer", loaded.processor),
            labels=labels,
            target_texts=target_texts,
        )
        if field_annotations is not None:
            field_ids, field_value_mask = field_annotations
            tensors[HL_FIELD_IDS_KEY] = field_ids.to(input_device)
            tensors[HL_FIELD_VALUE_MASK_KEY] = field_value_mask.to(input_device)
        self._add_keyframe_auxiliary_targets(tensors, samples, labels=labels, device=input_device)
        self._apply_typed_training_mask(
            loaded,
            tensors,
            tokenizer=getattr(loaded.processor, "tokenizer", loaded.processor),
        )
        set_model_proprio_batch(
            loaded.model,
            self._encoded_samples_for_training(
                samples,
                clips,
                apply_proposal_noise=apply_proposal_noise,
            ),
            self.config,
            device=input_device,
        )
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            stage_pattern = self._two_pass_stage_id_pattern()
            stage_ids = torch.tensor(
                [stage for _ in samples for stage in stage_pattern],
                dtype=torch.long,
                device=input_device,
            )
            tensors["_hl_stage_ids"] = stage_ids
        else:
            stage_ids = None
        self._add_conditioning_inputs(
            tensors,
            loaded=loaded,
            samples=self._encoded_samples_for_training(
                samples,
                clips,
                apply_proposal_noise=apply_proposal_noise,
            ),
            device=input_device,
            stage_ids=stage_ids,
        )
        return tensors

    def _add_conditioning_inputs(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        loaded: LoadedHLVLM,
        samples: list[ExportedHLMemorySample],
        device: torch.device,
        stage_ids: torch.Tensor | None = None,
    ) -> None:
        if not (self.config.progress_condition_enabled or self.config.state_condition_enabled):
            return
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        tensors.update(
            build_conditioning_batch(
                samples,
                self.config,
                tokenizer,
                device=device,
                stage_ids=stage_ids,
            )
        )

    def _add_keyframe_auxiliary_targets(
        self,
        tensors: dict[str, torch.Tensor],
        samples: list[ExportedHLMemorySample],
        *,
        labels: torch.Tensor,
        device: torch.device,
    ) -> None:
        if not self.config.keyframe_aux_enabled:
            return
        if self.config.target_protocol not in {
            "memer_objective",
            "memer_objective_grounding",
            "subtask_keyframe",
            "keyframe_gated_memory",
            "keyframe_gated_memory_typed_mask",
            "keyframe_gated_memory_two_pass",
            _FILM_PROGRESS_TWO_PASS_PROTOCOL,
        }:
            raise ValueError(
                "Keyframe auxiliary loss requires a protocol that predicts keyframe_candidate_positions."
            )
        rows: list[tuple[ExportedHLMemorySample, bool]] = []
        for sample in samples:
            rows.append((sample, True))
            if self.config.target_protocol == "keyframe_gated_memory_two_pass":
                rows.append((sample, False))
            elif self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                rows.append((sample, False))
                rows.append((sample, False))
        if len(rows) != labels.shape[0]:
            raise ValueError(
                f"Keyframe auxiliary row mismatch: built {len(rows)} rows for labels {tuple(labels.shape)}."
            )
        anchors = []
        for row_labels in labels:
            supervised = torch.nonzero(row_labels != -100, as_tuple=False).flatten()
            if supervised.numel() == 0:
                raise ValueError("Keyframe auxiliary loss requires at least one supervised target token per row.")
            anchors.append(max(0, int(supervised[0].item()) - 1))

        targets = [
            _keyframe_auxiliary_targets_for_sample(
                sample,
                config=self.config,
                enabled=enabled,
            )
            for sample, enabled in rows
        ]
        tensors[KEYFRAME_AUX_ANCHOR_POSITIONS_KEY] = torch.tensor(
            anchors,
            dtype=torch.long,
            device=device,
        )
        tensors[KEYFRAME_AUX_MASK_KEY] = torch.tensor(
            [target["enabled"] for target in targets],
            dtype=torch.bool,
            device=device,
        )
        tensors[KEYFRAME_AUX_POSITION_TARGETS_KEY] = torch.tensor(
            [target["position_targets"] for target in targets],
            dtype=torch.float32,
            device=device,
        )
        tensors[KEYFRAME_AUX_EVENT_TARGETS_KEY] = torch.tensor(
            [target["event_target"] for target in targets],
            dtype=torch.float32,
            device=device,
        )
        tensors[KEYFRAME_AUX_UPDATE_TARGETS_KEY] = torch.tensor(
            [target["update_target"] for target in targets],
            dtype=torch.long,
            device=device,
        )
        tensors[KEYFRAME_AUX_CANONICAL_POSITIONS_KEY] = torch.tensor(
            [target["canonical_position"] for target in targets],
            dtype=torch.float32,
            device=device,
        )
        tensors[KEYFRAME_AUX_VALID_POSITIONS_KEY] = torch.tensor(
            [target["valid_positions"] for target in targets],
            dtype=torch.bool,
            device=device,
        )

    def _training_target_texts(
        self,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
        *,
        apply_two_pass_training_noise: bool,
    ) -> tuple[str, ...]:
        del clips, apply_two_pass_training_noise
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            texts: list[str] = []
            for sample in samples:
                texts.append(self._build_two_pass_target_text(sample, stage="predict"))
                if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                    texts.append(self._build_two_pass_target_text(sample, stage="horizon"))
                texts.append(self._build_two_pass_target_text(sample, stage="confirm"))
            return tuple(texts)
        return tuple(self.build_target_text(sample) for sample in samples)

    def _two_pass_stage_id_pattern(self) -> tuple[int, ...]:
        if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
            return (STAGE_PREDICT, STAGE_HORIZON, STAGE_CONFIRM)
        return (STAGE_PREDICT, STAGE_CONFIRM)

    def _two_pass_stage_id(self, stage: str) -> int:
        if stage == "predict":
            return STAGE_PREDICT
        if stage == "confirm":
            return STAGE_CONFIRM
        if stage == "horizon":
            return STAGE_HORIZON
        raise ValueError(f"Unsupported two-pass stage: {stage!r}")

    def _apply_typed_training_mask(
        self,
        loaded: LoadedHLVLM,
        tensors: dict[str, torch.Tensor],
        *,
        tokenizer: Any,
    ) -> None:
        if self.config.target_protocol != _TYPED_MASK_PROTOCOL:
            return
        attention_mask = tensors.get("attention_mask")
        if attention_mask is None or attention_mask.ndim != 2:
            raise ValueError("Qwen2.5 typed masking requires the processor's 2D attention_mask.")
        tensors["position_ids"] = _compute_qwen25_position_ids(
            loaded.model,
            input_ids=tensors["input_ids"],
            attention_mask=attention_mask,
            model_inputs=tensors,
        )
        additive_mask, _ = build_qwen25_typed_attention_mask(
            input_ids=tensors["input_ids"],
            attention_mask=attention_mask,
            labels=tensors["labels"],
            tokenizer=tokenizer,
            dtype=_model_compute_dtype(loaded.model),
        )
        tensors["attention_mask"] = additive_mask

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
        if self.config.target_protocol == _TYPED_MASK_PROTOCOL:
            return self._generate_typed_mask_prediction(loaded, sample, clips, device=device)
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return self._generate_two_pass_prediction(loaded, sample, clips, device=device)
        inputs = self._encode_prompt_only(loaded.processor, sample, clips)
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        set_model_proprio_batch(loaded.model, [sample], self.config, device=input_device)
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
        if self.config.target_protocol == _TYPED_MASK_PROTOCOL:
            return [
                self.generate_prediction(loaded, sample, clip, device=device)
                for sample, clip in zip(samples, clips, strict=True)
            ]
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return [self.generate_prediction(loaded, sample, clip, device=device) for sample, clip in zip(samples, clips, strict=True)]
        inputs = self._encode_batch_prompt_only(loaded.processor, samples, clips)
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        set_model_proprio_batch(loaded.model, samples, self.config, device=input_device)
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

    def _generate_typed_mask_prediction(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLVLMGeneration:
        inputs = self._encode_prompt_only(loaded.processor, sample, clips)
        input_device = self._resolve_input_device(loaded.model, device)
        tokenizer = getattr(loaded.processor, "tokenizer", loaded.processor)
        prompt_input_ids = inputs["input_ids"].to(input_device)
        prompt_attention_mask = inputs["attention_mask"].to(input_device)
        target_starts = torch.full(
            (prompt_input_ids.shape[0],),
            prompt_input_ids.shape[-1],
            dtype=torch.long,
            device=input_device,
        )
        generated_ids = prompt_input_ids
        generated_attention_mask = prompt_attention_mask
        static_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
            and key not in {"input_ids", "attention_mask", "position_ids", "rope_deltas"}
        }
        set_model_proprio_batch(loaded.model, [sample], self.config, device=input_device)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(self._generation_max_new_tokens()):
            step_static_inputs = _extend_qwen25_sequence_inputs(
                static_inputs,
                prompt_length=prompt_input_ids.shape[-1],
                sequence_length=generated_ids.shape[-1],
            )
            position_ids = _compute_qwen25_position_ids(
                loaded.model,
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
                model_inputs=step_static_inputs,
            )
            additive_mask, _ = build_qwen25_typed_attention_mask(
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
                target_starts=target_starts,
                tokenizer=tokenizer,
                dtype=_model_compute_dtype(loaded.model),
            )
            outputs = loaded.model(
                input_ids=generated_ids,
                attention_mask=additive_mask,
                position_ids=position_ids,
                use_cache=False,
                **step_static_inputs,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            generated_attention_mask = torch.cat(
                [generated_attention_mask, torch.ones_like(next_token)],
                dim=1,
            )
            if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                break
        decoded = tokenizer.decode(
            generated_ids[0, prompt_input_ids.shape[-1] :],
            skip_special_tokens=True,
        )
        try:
            prediction = HLMemoryPrediction.from_json(decoded).with_recent_position_limit(clips.recent_valid_length)
            return HLVLMGeneration(prediction=prediction, raw_output=decoded)
        except ValueError as exc:
            logging.warning(
                "HL VLM typed-mask output could not be parsed as JSON; using fallback prediction. raw_output=%r",
                decoded[:1000],
            )
            return HLVLMGeneration(
                prediction=self._fallback_prediction(sample),
                raw_output=decoded,
                parse_error=f"{type(exc).__name__}: {exc}",
            )

    def _encoded_samples_for_training(
        self,
        samples: list[ExportedHLMemorySample],
        clips: list[LoadedVideoClips],
        *,
        apply_proposal_noise: bool,
    ) -> list[ExportedHLMemorySample]:
        if self.config.target_protocol not in _TWO_PASS_PROTOCOLS:
            return samples
        encoded_samples: list[ExportedHLMemorySample] = []
        for sample, clip in zip(samples, clips, strict=True):
            pass_a_prediction = (
                self._training_pass_a_prediction(sample, clip)
                if apply_proposal_noise
                else sample.target_prediction(
                    target_protocol=self.config.target_protocol,
                    keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
                )
            )
            encoded_samples.append(sample)
            if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                encoded_samples.append(
                    self._two_pass_stage_sample(
                        sample,
                        stage="horizon",
                        pass_a_prediction=None,
                    )
                )
            encoded_samples.append(
                self._two_pass_stage_sample(
                    sample,
                    stage="confirm",
                    pass_a_prediction=pass_a_prediction,
                )
            )
        return encoded_samples

    def _generate_two_pass_prediction(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        device: str | torch.device,
    ) -> HLVLMGeneration:
        pass_a = self._generate_two_pass_stage(
            loaded,
            sample,
            clips,
            stage="predict",
            pass_a_prediction=None,
            device=device,
        )
        pass_horizon = None
        if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
            pass_horizon = self._generate_two_pass_stage(
                loaded,
                sample,
                clips,
                stage="horizon",
                pass_a_prediction=None,
                device=device,
            )
            pass_a = HLVLMGeneration(
                prediction=dataclasses.replace(
                    pass_a.prediction,
                    horizon_current_objective=pass_horizon.prediction.horizon_current_objective,
                ),
                raw_output=pass_a.raw_output,
                parse_error=pass_a.parse_error or pass_horizon.parse_error,
            )
        if not pass_a.prediction.keyframe_candidate_positions:
            prediction = dataclasses.replace(pass_a.prediction, new_completed_objective="", completed_objective="")
            return HLVLMGeneration(
                prediction=prediction.with_recent_position_limit(clips.recent_valid_length),
                raw_output=json.dumps(
                    {"pass_a": pass_a.raw_output, "pass_horizon": None if pass_horizon is None else pass_horizon.raw_output, "pass_b": None},
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                parse_error=pass_a.parse_error,
            )
        pass_b = self._generate_two_pass_stage(
            loaded,
            sample,
            clips,
            stage="confirm",
            pass_a_prediction=pass_a.prediction,
            device=device,
        )
        prediction = dataclasses.replace(
            pass_a.prediction,
            task_progress=pass_b.prediction.task_progress,
            new_completed_objective=pass_b.prediction.new_completed_objective,
            completed_objective=pass_b.prediction.new_completed_objective,
        )
        return HLVLMGeneration(
            prediction=prediction.with_recent_position_limit(clips.recent_valid_length),
            raw_output=json.dumps(
                {"pass_a": pass_a.raw_output, "pass_horizon": None if pass_horizon is None else pass_horizon.raw_output, "pass_b": pass_b.raw_output},
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            parse_error=pass_a.parse_error or pass_b.parse_error,
        )

    def _generate_two_pass_stage(
        self,
        loaded: LoadedHLVLM,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        stage: str,
        pass_a_prediction: HLMemoryPrediction | None,
        device: str | torch.device,
    ) -> HLVLMGeneration:
        stage_clips = self._two_pass_stage_clips(
            sample,
            clips,
            stage=stage,
            pass_a_prediction=pass_a_prediction,
        )
        inputs = self._encode_two_pass_prompt_only(
            loaded.processor,
            sample,
            stage_clips,
            stage=stage,
            pass_a_prediction=pass_a_prediction,
        )
        stage_sample = self._two_pass_stage_sample(
            sample,
            stage=stage,
            pass_a_prediction=pass_a_prediction,
        )
        input_device = self._resolve_input_device(loaded.model, device)
        input_ids = inputs["input_ids"].to(input_device)
        generation_inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        set_model_proprio_batch(loaded.model, [stage_sample], self.config, device=input_device)
        generation_inputs.update(
            build_conditioning_batch(
                [stage_sample],
                self.config,
                getattr(loaded.processor, "tokenizer", loaded.processor),
                device=input_device,
                stage_ids=torch.tensor([self._two_pass_stage_id(stage)], dtype=torch.long, device=input_device),
            )
        )
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
        decoded = tokenizer.decode(generated_ids[:, input_ids.shape[-1] :][0], skip_special_tokens=True)
        if stage == "confirm":
            try:
                new_completed_objective, task_progress = _parse_completion_update_text(decoded)
            except ValueError as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                logging.warning(
                    "HL VLM two-pass stage B output could not be parsed as JSON; rejecting completion. raw_output=%r",
                    decoded[:1000],
                )
                return HLVLMGeneration(
                    prediction=dataclasses.replace(
                        pass_a_prediction or self._fallback_prediction(sample),
                        new_completed_objective="",
                        completed_objective="",
                    ),
                    raw_output=decoded,
                    parse_error=parse_error,
                )
            return HLVLMGeneration(
                prediction=dataclasses.replace(
                    pass_a_prediction or self._fallback_prediction(sample),
                    task_progress=task_progress,
                    new_completed_objective=new_completed_objective,
                    completed_objective=new_completed_objective,
                ),
                raw_output=decoded,
            )
        if stage == "horizon":
            try:
                horizon_objective = _parse_horizon_objective_text(decoded)
            except ValueError as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                logging.warning(
                    "HL VLM horizon stage output could not be parsed as JSON; using fallback horizon. raw_output=%r",
                    decoded[:1000],
                )
                fallback = self._fallback_prediction(sample)
                return HLVLMGeneration(
                    prediction=dataclasses.replace(
                        fallback,
                        horizon_current_objective=fallback.horizon_current_objective or fallback.current_objective,
                    ),
                    raw_output=decoded,
                    parse_error=parse_error,
                )
            fallback = self._fallback_prediction(sample)
            return HLVLMGeneration(
                prediction=dataclasses.replace(fallback, horizon_current_objective=horizon_objective),
                raw_output=decoded,
            )
        try:
            prediction = HLMemoryPrediction.from_json(decoded).with_recent_position_limit(clips.recent_valid_length)
            return HLVLMGeneration(prediction=prediction, raw_output=decoded)
        except ValueError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "HL VLM two-pass stage A output could not be parsed as JSON; using fallback prediction. raw_output=%r",
                decoded[:1000],
            )
            return HLVLMGeneration(
                prediction=self._fallback_prediction(sample),
                raw_output=decoded,
                parse_error=parse_error,
            )

    def _two_pass_stage_clips(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        stage: str,
        pass_a_prediction: HLMemoryPrediction | None,
    ) -> LoadedVideoClips:
        if stage in {"predict", "horizon"}:
            return clips
        if stage != "confirm":
            raise ValueError(f"Unsupported two-pass stage: {stage!r}")
        prediction = pass_a_prediction or sample.target_prediction(
            target_protocol=self.config.target_protocol,
            keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
        )
        return _candidate_evidence_clips(clips, prediction.keyframe_candidate_positions)

    def _two_pass_stage_sample(
        self,
        sample: ExportedHLMemorySample,
        *,
        stage: str,
        pass_a_prediction: HLMemoryPrediction | None,
    ) -> ExportedHLMemorySample:
        if stage in {"predict", "horizon"}:
            return sample
        if stage != "confirm":
            raise ValueError(f"Unsupported two-pass stage: {stage!r}")
        prediction = pass_a_prediction or sample.target_prediction(
            target_protocol=self.config.target_protocol,
            keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
        )
        indices = _valid_candidate_indices(
            prediction.keyframe_candidate_positions,
            _sample_recent_indexable_length(sample),
        )
        return dataclasses.replace(
            sample,
            recent_frame_paths=tuple(sample.recent_frame_paths[index] for index in indices),
            recent_frame_indices=tuple(sample.recent_frame_indices[index] for index in indices),
            recent_valid_length=len(indices),
            recent_robot_states=tuple(sample.recent_robot_states[index] for index in indices)
            if sample.recent_robot_states
            else (),
            recent_robot_state_masks=tuple(sample.recent_robot_state_masks[index] for index in indices)
            if sample.recent_robot_state_masks
            else (),
            recent_object_center_points=tuple(sample.recent_object_center_points[index] for index in indices)
            if sample.recent_object_center_points
            else (),
        )

    def _training_pass_a_prediction(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> HLMemoryPrediction:
        prediction = sample.target_prediction(
            target_protocol=self.config.target_protocol,
            keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
        )
        probability = self.config.two_pass_training_proposal_noise_probability
        if clips.recent_valid_length <= 0:
            return prediction
        positions = list(prediction.keyframe_candidate_positions)
        if not positions:
            false_position = 1 + int(
                _stable_unit_interval(f"{sample.sample_id}:false-positive") * clips.recent_valid_length
            )
            return dataclasses.replace(
                prediction,
                keyframe_candidate_positions=(min(false_position, clips.recent_valid_length),),
            )
        if probability <= 0.0:
            return prediction
        selector = _stable_unit_interval(f"{sample.sample_id}:two-pass-proposal")
        if selector >= probability:
            return prediction
        if positions:
            base = positions[0]
            direction = -1 if _stable_unit_interval(f"{sample.sample_id}:direction") < 0.5 else 1
            shifted = min(max(base + direction, 1), clips.recent_valid_length)
            positions = [shifted]
            if shifted != base and _stable_unit_interval(f"{sample.sample_id}:extra") < 0.5:
                positions.append(base)
        return dataclasses.replace(
            prediction,
            keyframe_candidate_positions=tuple(sorted(set(positions))),
        )

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
        proprio_prefix = render_proprio_token_text(sample, self.config)
        object_context = _render_object_context(sample, clips)
        if self.config.target_protocol == "known_prior_tracker":
            return proprio_prefix + object_context + self._build_known_prior_tracker_prompt(sample, clips)
        if self.config.target_protocol in {"keyframe_gated_memory", _TYPED_MASK_PROTOCOL}:
            return proprio_prefix + object_context + self._build_keyframe_gated_memory_prompt(sample, clips)
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return proprio_prefix + object_context + self._build_keyframe_gated_two_pass_predict_prompt(sample, clips)
        if self.config.target_protocol in {
            "objective_memory_state",
            "objective_last_objective",
            "objective_prev_stage",
        }:
            return proprio_prefix + object_context + self._build_state_context_objective_prompt(sample, clips)
        if self.config.target_protocol in {"memer_objective", "memer_objective_grounding"}:
            return proprio_prefix + object_context + self._build_memer_objective_prompt(sample, clips)
        if self.config.target_protocol == "subtask_keyframe":
            return proprio_prefix + object_context + self._build_subtask_keyframe_prompt(sample, clips)
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
            proprio_prefix
            + object_context
            +
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
        thinking_instruction = (
            f"If thinking is used, keep it under {self.config.thinking_budget_tokens} tokens and place exactly one "
            "valid JSON object after the thinking text."
            if self.config.enable_thinking
            else "Do not output chain-of-thought or analysis text. Output only valid JSON."
        )
        if self.config.target_protocol in {"keyframe_gated_memory", _TYPED_MASK_PROTOCOL}:
            return (
                "You are a robot high-level state and event-memory module. Use the recent observation clip as the "
                "primary visual evidence for current and horizon objectives. Use historical memory only as "
                "non-Markovian context, not as a substitute for the current visual state. Select keyframe candidates "
                "only from the recent observation clip. Commit a completed objective only when nominated recent "
                f"keyframe evidence visually confirms completion. {thinking_instruction}"
            )
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return (
                "You are a robot high-level two-pass state and event-memory module. In Pass A, predict current and "
                "horizon objectives and propose recent keyframe evidence from the recent observation clip. In the FiLM "
                "progress protocol, horizon is predicted by an independent prompt so it cannot read current target tokens. In Pass B, "
                "confirm a completed event only from the proposed candidate evidence. Use historical memory only as "
                f"global progress context. {thinking_instruction}"
            )
        if self.config.target_protocol in {
            "memer_objective",
            "memer_objective_grounding",
            "subtask_keyframe",
            "known_prior_tracker",
            "objective_memory_state",
            "objective_last_objective",
            "objective_prev_stage",
        }:
            return (
                "You are a robot high-level objective predictor. Use recent visual evidence as the primary source for "
                "the current executable objective, and use historical memory or task priors only as context for global "
                f"progress. Select keyframes only from the recent observation clip when requested. {thinking_instruction}"
            )
        return (
            "You are a robot high-level memory policy. Your job is to choose the next language subtask for a "
            "low-level robot controller and to select task-relevant keyframes for future memory. Be concise, "
            "ground decisions in the provided images, and output only valid JSON. The recent observation clip is "
            "a past-to-current sequence; the last valid recent frame is the current state that must determine "
            f"`current_objective` or `current_subtask`. {thinking_instruction}"
        )

    def build_target_text(self, sample: ExportedHLMemorySample) -> str:
        prediction = sample.target_prediction(
            target_protocol=self.config.target_protocol,
            keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
        )
        if self.config.target_protocol == "known_prior_tracker":
            return json.dumps(
                {
                    "current_objective": prediction.current_objective,
                    "subtask_progress": prediction.subtask_progress,
                    "should_advance_objective": prediction.should_advance_objective,
                    "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        if self.config.target_protocol in {"keyframe_gated_memory", _TYPED_MASK_PROTOCOL}:
            payload = {
                "current_objective": prediction.current_objective,
                "horizon_current_objective": prediction.horizon_current_objective,
                "target_object": prediction.target_object,
                "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                "new_completed_objective": prediction.new_completed_objective,
                "task_progress": prediction.task_progress,
            }
            if prediction.target_slot:
                payload["target_slot"] = prediction.target_slot
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return self._build_two_pass_target_text(sample, stage="predict")
        if self.config.target_protocol in {
            "objective_memory_state",
            "objective_last_objective",
            "objective_prev_stage",
        }:
            payload = {
                "current_objective": prediction.current_objective,
                "horizon_current_objective": prediction.horizon_current_objective,
                "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
            }
            if self.config.target_protocol == "objective_memory_state":
                payload["updated_language_memory"] = prediction.updated_language_memory
            elif self.config.target_protocol == "objective_last_objective":
                payload["last_objective"] = prediction.last_objective
            elif self.config.target_protocol == "objective_prev_stage":
                payload["previous_stage_objective"] = prediction.previous_stage_objective
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if self.config.target_protocol == "memer_objective":
            return json.dumps(
                {
                    "current_objective": prediction.current_objective,
                    "horizon_current_objective": prediction.horizon_current_objective,
                    "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        if self.config.target_protocol == "memer_objective_grounding":
            payload = {
                "current_objective": prediction.current_objective,
                "horizon_current_objective": prediction.horizon_current_objective,
                "target_object": prediction.target_object,
                "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
            }
            if prediction.target_slot:
                payload["target_slot"] = prediction.target_slot
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if self.config.target_protocol == "subtask_keyframe":
            payload = {
                "current_objective": prediction.current_objective,
                "target_object": prediction.target_object,
                "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
            }
            if prediction.target_slot:
                payload["target_slot"] = prediction.target_slot
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
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
            "Predict both the current executable objective at the last valid recent frame and the short-horizon "
            "objective a few frames after it, using recent motion as context.\n"
            if sample.horizon_frame_index is not None
            else "Predict the current executable objective at the last valid recent frame.\n"
        )
        if self.config.target_protocol == "memer_objective_grounding":
            output_schema = (
                "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
                "`target_object`, optionally `target_slot`, and `keyframe_candidate_positions`.\n"
                "`target_object` must name the manipulated object or object part grounding the current objective.\n"
                "`target_slot` is optional and should name the destination/slot only when it is visually or semantically clear.\n"
            )
        else:
            output_schema = (
                "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
                "and `keyframe_candidate_positions`.\n"
            )
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains selected historical keyframes from earlier in the episode.\n"
            "The second clip contains the recent observation window, ordered oldest to newest.\n"
            "Use the historical keyframes as long-term visual evidence for non-Markov progress, such as which object, "
            "hole, slot, or repeated insertion/place step has already been completed; do not ignore them when the "
            "recent window alone is ambiguous.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "If each frame has two concatenated views, the left slot is the left hand and the right slot is the right hand.\n"
            f"{horizon_text}"
            f"{output_schema}"
            "`current_objective` must be one short executable robot instruction.\n"
            "`horizon_current_objective` must be the short executable robot instruction expected a few frames later.\n"
            "`keyframe_candidate_positions` must be a JSON list of 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            "These positions are relative to the recent observation clip, not absolute frame ids and not historical "
            "memory clip positions.\n"
            "Select only recent frames that should be kept as long-term visual memory for future decisions; return [] if none.\n"
            "Do not include progress, language memory, notes, markdown, explanation text, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
        )

    def _build_keyframe_gated_memory_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        task_progress = (
            "No accepted completed event yet."
            if self.config.typed_mask_suppress_language_memory and self.config.target_protocol == _TYPED_MASK_PROTOCOL
            else sample.language_memory.strip() or "No accepted completed event yet."
        )
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains accepted historical keyframes from earlier in the episode.\n"
            "The second clip contains the recent observation window, ordered oldest to newest.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "If each frame has two concatenated views, the left slot is the left hand and the right slot is the right hand.\n"
            "The task-progress log below is the accepted completed-task state. It may be empty, but it should only "
            "provide global progress. Use the recent observation window as the primary evidence for current and "
            "short-horizon objectives.\n"
            "Predict the current executable objective at the last valid recent frame and the short-horizon objective "
            "a few frames after it.\n"
            "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
            "`target_object`, optionally `target_slot`, `keyframe_candidate_positions`, `new_completed_objective`, and `task_progress`.\n"
            "`current_objective` must be one short executable robot instruction suitable for the low-level VLA.\n"
            "`horizon_current_objective` must be the short executable robot instruction expected a few frames later.\n"
            "`target_object` must name the manipulated object or object part grounding the current objective.\n"
            "`target_slot` is optional and should name the destination/slot only when it is visually or semantically clear.\n"
            "`keyframe_candidate_positions` must be a JSON list of 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            "These positions are relative to the recent observation clip, not absolute frame ids and not historical "
            "memory clip positions. The runtime maps each position to the corresponding recent frame for state updates.\n"
            "`new_completed_objective` must be empty unless the recent window contains a visually confirmed newly "
            "completed task that should be committed to long-term memory. When non-empty, it must describe only that "
            "newly completed task represented by the nominated keyframe(s).\n"
            "`task_progress` must be the updated cumulative completed-task history after applying "
            "`new_completed_objective`. If no new task is completed, keep the previous task-progress state unchanged.\n"
            "Do not include updated_language_memory, completed_objective, notes, markdown, explanation text, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"Task progress: {task_progress}\n"
        )

    def _build_keyframe_gated_two_pass_predict_prompt(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        completed_event_log = sample.language_memory.strip() or "No accepted completed event yet."
        if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
            return (
                "[Pass A: visual current prediction]\n"
                "You receive accepted historical keyframes and a recent observation window.\n"
                "Use the recent observation window as the primary evidence for the current objective and keyframe "
                "proposal. Long-term progress is provided only through a learned low-bandwidth condition, not as "
                "raw text in this prompt.\n"
                f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
                f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
                f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
                "is the last/current valid frame.\n"
                "Return exactly one JSON object with keys `current_objective`, `target_object`, optionally `target_slot`, "
                "and `keyframe_candidate_positions`.\n"
                "`current_objective` must be one short executable robot instruction suitable for the low-level VLA.\n"
                "`target_object` must name the manipulated object or object part grounding the current objective.\n"
                "`target_slot` is optional and should name the destination/slot only when it is visually or semantically clear.\n"
                "`keyframe_candidate_positions` is a high-recall proposal list for recent frames that may represent an event. "
                f"Valid positions are integers from 1 to {clips.recent_valid_length} inclusive. Return [] if there is no event evidence.\n"
                "Do not output `horizon_current_objective`, `completed_objective`, memory text, progress, notes, markdown, or extra keys.\n"
                f"{thinking_instruction}"
                f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            )
        return (
            "[Pass A: visual prediction]\n"
            "You receive accepted historical keyframes and a recent observation window.\n"
            "Use recent visual evidence as the primary signal for the current and short-horizon objective. Use "
            "the completed-event log and historical keyframes only to resolve global progress when the task is "
            "non-Markovian.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, `target_object`, "
            "optionally `target_slot`, and `keyframe_candidate_positions`.\n"
            "`target_object` must name the manipulated object or object part grounding the current objective.\n"
            "`target_slot` is optional and should name the destination/slot only when it is visually or semantically clear.\n"
            "`keyframe_candidate_positions` is a high-recall proposal list for recent frames that may represent an event. "
            f"Valid positions are integers from 1 to {clips.recent_valid_length} inclusive. Return [] if there is no event evidence.\n"
            "These positions are relative to the recent observation clip, not absolute frame ids and not historical "
            "memory clip positions.\n"
            "Do not output `completed_objective`, memory text, progress, notes, markdown, or extra keys in Pass A.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"Completed-event log: {completed_event_log}\n"
        )

    def _build_film_progress_horizon_prompt(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        return (
            "[Pass A2: independent horizon prediction]\n"
            "You receive accepted historical keyframes and a recent observation window.\n"
            "Predict the short-horizon objective a few frames after the last valid recent frame. Do not rely on any "
            "generated current-objective text; infer the horizon directly from the recent visual sequence, task "
            "instruction, and learned low-bandwidth progress/state conditions.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "Return exactly one JSON object with the single key `horizon_current_objective`.\n"
            "`horizon_current_objective` must be one short executable robot instruction expected a few frames later.\n"
            "Do not output `current_objective`, keyframes, completed events, memory text, markdown, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
        )

    def _build_keyframe_gated_two_pass_confirm_prompt(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        pass_a_prediction: HLMemoryPrediction,
    ) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        completed_event_log = sample.language_memory.strip() or "No accepted completed event yet."
        completed_event_line = (
            ""
            if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL
            else f"Completed-event log: {completed_event_log}\n"
        )
        pass_a_payload = {
            "keyframe_candidate_positions": list(pass_a_prediction.keyframe_candidate_positions),
        }
        return (
            "[Pass B: completion confirmation]\n"
            "Confirm whether Pass A's proposed keyframe evidence should be committed as a completed long-term event.\n"
            "Typed source routing is active: the first clip contains accepted historical keyframes, and the second clip "
            "contains only Pass A's proposed candidate frames. Other recent frames are excluded from this pass.\n"
            "Use canonical/end-state evidence: contact/release/placement/operation must be visually completed, not merely "
            "approaching or in progress. If the evidence is ambiguous, output an empty completed objective.\n"
            f"The candidate evidence clip has {clips.recent_valid_length} valid proposed frames.\n"
            "Return exactly one JSON object with keys `new_completed_objective` and `task_progress`.\n"
            "`new_completed_objective` is the newly completed task shown by the candidate evidence; use an empty string "
            "for no newly completed task and never output null.\n"
            "`task_progress` is the updated cumulative completed-task history after applying `new_completed_objective`. "
            "If there is no new completion, keep the prior completed-task history unchanged.\n"
            "`new_completed_objective` must be empty unless the candidate evidence contains a visually confirmed completed "
            "event represented by the candidate keyframe(s). Do not invent the next step.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"{completed_event_line}"
            "Pass A routing metadata (semantic objective text is intentionally hidden to prevent teacher-forcing "
            "leakage into completion confirmation): "
            f"{json.dumps(pass_a_payload, ensure_ascii=True, separators=(',', ':'))}\n"
        )

    def _build_two_pass_target_text(
        self,
        sample: ExportedHLMemorySample,
        *,
        stage: str,
        pass_a_prediction: HLMemoryPrediction | None = None,
    ) -> str:
        prediction = sample.target_prediction(
            target_protocol=self.config.target_protocol,
            keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
        )
        if stage == "predict":
            if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                payload = {
                    "current_objective": prediction.current_objective,
                    "target_object": prediction.target_object,
                    "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
                }
                if prediction.target_slot:
                    payload["target_slot"] = prediction.target_slot
                return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            payload = {
                "current_objective": prediction.current_objective,
                "horizon_current_objective": prediction.horizon_current_objective,
                "target_object": prediction.target_object,
                "keyframe_candidate_positions": list(prediction.keyframe_candidate_positions),
            }
            if prediction.target_slot:
                payload["target_slot"] = prediction.target_slot
            return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        if stage == "horizon":
            return json.dumps(
                {"horizon_current_objective": prediction.horizon_current_objective},
                ensure_ascii=True,
                separators=(",", ":"),
            )
        if stage == "confirm":
            has_candidates = pass_a_prediction is None or bool(pass_a_prediction.keyframe_candidate_positions)
            new_completed_objective = prediction.new_completed_objective if has_candidates else ""
            task_progress = prediction.task_progress if has_candidates else _previous_task_progress(sample)
            return json.dumps(
                {
                    "new_completed_objective": new_completed_objective,
                    "task_progress": task_progress,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        raise ValueError(f"Unsupported two-pass stage: {stage!r}")

    def _build_known_prior_tracker_prompt(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> str:
        memory_fields = _parse_ll_memory_fields(sample.language_memory)
        task_progress = sample.task_progress.strip() or memory_fields.get("task progress", "").strip()
        current_objective = (
            sample.current_objective.strip()
            or memory_fields.get("current objective", "").strip()
            or sample.current_subtask.strip()
        )
        step_prior = _render_step_prior(sample.step_prior)
        thinking_instruction = (
            "Thinking is enabled. Keep private reasoning brief and finish with exactly one final JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Output only the final JSON object. /no_think\n"
        )
        return (
            "You receive an ordered recent observation clip and optional historical keyframes.\n"
            "An external state machine owns the ordered task state. Do not choose, skip, or rewrite its objective.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of "
            f"{self.config.recent_frames_length}, ordered oldest to newest.\n"
            f"Completed task state: {task_progress or 'No completed subtask yet.'}\n"
            f"Current authoritative objective: {current_objective or 'continue the current known-prior objective'}\n"
            f"{step_prior}"
            "Estimate visual completion of the authoritative objective at the last valid recent frame.\n"
            "Return exactly one JSON object with keys `current_objective`, `subtask_progress`, "
            "`should_advance_objective`, and `keyframe_candidate_positions`.\n"
            "`current_objective` must copy the authoritative objective exactly; it is included only for schema validation "
            "and is ignored by the runtime state machine.\n"
            "`subtask_progress` must be a scalar in [0, 1].\n"
            "`should_advance_objective` must be true only when visual evidence shows the current objective is complete.\n"
            "`keyframe_candidate_positions` must contain 1-indexed positions in the recent clip that preserve completion "
            "evidence; return [] when no frame should be retained.\n"
            "Do not generate task history, a next objective, language memory, notes, markdown, or extra keys.\n"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
        )

    def _build_state_context_objective_prompt(
        self,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> str:
        context_label, context_text, context_policy = _state_context_for_protocol(self.config.target_protocol, sample)
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        return (
            "You receive two ordered video clips.\n"
            "The first clip contains selected historical keyframes from earlier in the episode.\n"
            "The second clip contains the recent observation window, ordered oldest to newest.\n"
            f"The historical memory clip has {clips.memory_valid_length} valid frames out of {self.config.memory_length}.\n"
            f"The recent observation clip has {clips.recent_valid_length} valid frames out of {self.config.recent_frames_length}.\n"
            f"In the recent observation clip, position 1 is the oldest valid recent frame and position {clips.recent_valid_length} "
            "is the last/current valid frame.\n"
            "Predict both the current executable objective at the last valid recent frame and the short-horizon "
            "objective a few frames after it, using recent motion and state context.\n"
            "Use the recent visual evidence as the primary signal. The state text below is a weak temporal cue; it may be "
            "missing or stale at runtime, so correct it when the current frame contradicts it.\n"
            f"{context_policy}\n"
            f"{_state_output_instruction_for_protocol(self.config.target_protocol)}"
            "`current_objective` must be one short executable robot instruction suitable for the low-level VLA.\n"
            "`horizon_current_objective` must be the short executable robot instruction expected a few frames later.\n"
            "`keyframe_candidate_positions` must be a JSON list of 1-indexed positions inside the recent observation clip only.\n"
            f"Valid keyframe candidate positions are integers from 1 to {clips.recent_valid_length} inclusive.\n"
            "Select only recent frames that should be kept as long-term visual memory for future decisions; return [] if none.\n"
            f"{_state_extra_key_instruction_for_protocol(self.config.target_protocol)}"
            f"{thinking_instruction}"
            f"Task instruction: {sample.instruction.strip() or 'unspecified'}\n"
            f"{context_label}: {context_text or 'none'}\n"
        )

    def _build_subtask_keyframe_prompt(self, sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
        thinking_instruction = (
            "Thinking is enabled. If you produce private reasoning, keep it brief and finish with exactly one final "
            "JSON object.\n"
            if self.config.enable_thinking
            else "Thinking is disabled. Do not reason step by step; output only the final JSON object. /no_think\n"
        )
        objective_text = "Predict the current executable objective at the last valid recent frame.\n"
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
            f"{objective_text}"
            "Use visual evidence in the last valid recent frame as the primary signal. Language memory and step prior, "
            "if provided, may be stale and should only be used as weak context.\n"
            "Return exactly one JSON object with keys `current_objective`, `target_object`, optionally `target_slot`, "
            "and `keyframe_candidate_positions`.\n"
            "`current_objective` must be one short executable robot instruction suitable for the low-level VLA.\n"
            "`target_object` must name the manipulated object or object part grounding the current objective.\n"
            "`target_slot` is optional and should name the destination/slot only when it is visually or semantically clear.\n"
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
        *,
        apply_two_pass_training_noise: bool = False,
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
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return self._encode_batch_prompt_only(processor, [sample], [clips])
        rendered = self._render_messages(processor, sample, clips, include_target=False)
        return self._encode_processor_inputs(processor, rendered, clips)

    def _encode_prompt_and_target(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
    ) -> Mapping[str, Any]:
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            return self._encode_batch_prompt_and_target(processor, [sample], [clips])
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
        *,
        apply_two_pass_training_noise: bool = False,
    ) -> Mapping[str, Any]:
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            rendered: list[str] = []
            expanded_clips: list[LoadedVideoClips] = []
            for sample, clip in zip(samples, clips, strict=True):
                pass_a_prediction = (
                    self._training_pass_a_prediction(sample, clip)
                    if apply_two_pass_training_noise
                    else sample.target_prediction(
                        target_protocol=self.config.target_protocol,
                        keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
                    )
                )
                predict_clip = self._two_pass_stage_clips(
                    sample,
                    clip,
                    stage="predict",
                    pass_a_prediction=None,
                )
                confirm_clip = self._two_pass_stage_clips(
                    sample,
                    clip,
                    stage="confirm",
                    pass_a_prediction=pass_a_prediction,
                )
                rendered.append(
                    self._render_two_pass_messages(
                        processor,
                        sample,
                        predict_clip,
                        stage="predict",
                        include_target=False,
                    )
                )
                if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                    rendered.append(
                        self._render_two_pass_messages(
                            processor,
                            sample,
                            predict_clip,
                            stage="horizon",
                            include_target=False,
                        )
                    )
                rendered.append(
                    self._render_two_pass_messages(
                        processor,
                        sample,
                        confirm_clip,
                        stage="confirm",
                        include_target=False,
                        pass_a_prediction=pass_a_prediction,
                    )
                )
                if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                    expanded_clips.extend((predict_clip, predict_clip, confirm_clip))
                else:
                    expanded_clips.extend((predict_clip, confirm_clip))
            return self._encode_batch_processor_inputs(processor, rendered, expanded_clips)
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
        *,
        apply_two_pass_training_noise: bool = False,
    ) -> Mapping[str, Any]:
        if self.config.target_protocol in _TWO_PASS_PROTOCOLS:
            rendered: list[str] = []
            expanded_clips: list[LoadedVideoClips] = []
            for sample, clip in zip(samples, clips, strict=True):
                pass_a_prediction = (
                    self._training_pass_a_prediction(sample, clip)
                    if apply_two_pass_training_noise
                    else sample.target_prediction(
                        target_protocol=self.config.target_protocol,
                        keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
                    )
                )
                predict_clip = self._two_pass_stage_clips(
                    sample,
                    clip,
                    stage="predict",
                    pass_a_prediction=None,
                )
                confirm_clip = self._two_pass_stage_clips(
                    sample,
                    clip,
                    stage="confirm",
                    pass_a_prediction=pass_a_prediction,
                )
                rendered.append(
                    self._render_two_pass_messages(
                        processor,
                        sample,
                        predict_clip,
                        stage="predict",
                        include_target=True,
                    )
                )
                if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                    rendered.append(
                        self._render_two_pass_messages(
                            processor,
                            sample,
                            predict_clip,
                            stage="horizon",
                            include_target=True,
                        )
                    )
                rendered.append(
                    self._render_two_pass_messages(
                        processor,
                        sample,
                        confirm_clip,
                        stage="confirm",
                        include_target=True,
                        pass_a_prediction=pass_a_prediction,
                    )
                )
                if self.config.target_protocol == _FILM_PROGRESS_TWO_PASS_PROTOCOL:
                    expanded_clips.extend((predict_clip, predict_clip, confirm_clip))
                else:
                    expanded_clips.extend((predict_clip, confirm_clip))
            return self._encode_batch_processor_inputs(processor, rendered, expanded_clips)
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

    def _encode_two_pass_prompt_only(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        stage: str,
        pass_a_prediction: HLMemoryPrediction | None,
    ) -> Mapping[str, Any]:
        rendered = self._render_two_pass_messages(
            processor,
            sample,
            clips,
            stage=stage,
            include_target=False,
            pass_a_prediction=pass_a_prediction,
        )
        return self._encode_processor_inputs(processor, rendered, clips)

    def _render_two_pass_messages(
        self,
        processor: Any,
        sample: ExportedHLMemorySample,
        clips: LoadedVideoClips,
        *,
        stage: str,
        include_target: bool,
        pass_a_prediction: HLMemoryPrediction | None = None,
    ) -> str:
        if stage == "predict":
            stage_sample = sample
            prompt = self._build_keyframe_gated_two_pass_predict_prompt(stage_sample, clips)
            target = self._build_two_pass_target_text(stage_sample, stage="predict")
            clip_description = (
                "Recent observation clip, ordered oldest to newest. "
                "The last valid frame is the current state to predict."
            )
        elif stage == "horizon":
            stage_sample = sample
            prompt = self._build_film_progress_horizon_prompt(stage_sample, clips)
            target = self._build_two_pass_target_text(stage_sample, stage="horizon")
            clip_description = (
                "Recent observation clip, ordered oldest to newest. "
                "The last valid frame anchors the independent horizon prediction."
            )
        elif stage == "confirm":
            pass_a_prediction = pass_a_prediction or sample.target_prediction(
                target_protocol=self.config.target_protocol,
                keyframe_candidate_label_mode=self.config.keyframe_candidate_label_mode,
            )
            stage_sample = self._two_pass_stage_sample(
                sample,
                stage=stage,
                pass_a_prediction=pass_a_prediction,
            )
            prompt = self._build_keyframe_gated_two_pass_confirm_prompt(
                stage_sample,
                clips,
                pass_a_prediction=pass_a_prediction,
            )
            target = self._build_two_pass_target_text(
                sample,
                stage="confirm",
                pass_a_prediction=pass_a_prediction,
            )
            clip_description = "Candidate evidence clip containing only Pass A proposed event frames."
        else:
            raise ValueError(f"Unsupported two-pass stage: {stage!r}")
        proprio_prefix = render_proprio_token_text(stage_sample, self.config)
        user_content = [
            {"type": "text", "text": "Historical memory clip, ordered oldest to newest."},
            {"type": "video"},
            {
                "type": "text",
                "text": clip_description,
            },
            {"type": "video"},
            {"type": "text", "text": proprio_prefix + prompt},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self.build_system_prompt()}]},
            {"role": "user", "content": user_content},
        ]
        if include_target:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
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


class Qwen3VLHLAdapter(Qwen35HLAdapter):
    """HF adapter for the official Qwen3-VL family.

    Qwen3-VL uses the same high-level Hugging Face image-text auto model API
    as the Qwen3.5-VL checkpoints used by the existing adapter. Keep it as a
    separate class so backend-specific probes and mask support can be added
    without changing Qwen3.5 behavior.
    """

    pass


def _model_compute_dtype(model: torch.nn.Module) -> torch.dtype:
    config = getattr(model, "config", None)
    for attribute in ("torch_dtype", "dtype"):
        value = getattr(config, attribute, None)
        if isinstance(value, torch.dtype):
            return value
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    weight = getattr(embeddings, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.is_floating_point():
        return weight.dtype
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _compute_qwen25_position_ids(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    qwen_model = _find_qwen25_multimodal_model(model)
    kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": model_inputs.get("image_grid_thw"),
        "video_grid_thw": model_inputs.get("video_grid_thw"),
        "inputs_embeds": None,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "second_per_grid_ts": model_inputs.get("second_per_grid_ts"),
        "mm_token_type_ids": model_inputs.get("mm_token_type_ids"),
    }
    accepted = set(inspect.signature(qwen_model.compute_3d_position_ids).parameters)
    return qwen_model.compute_3d_position_ids(**{key: value for key, value in kwargs.items() if key in accepted})


def _find_qwen25_multimodal_model(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if callable(getattr(module, "compute_3d_position_ids", None)) and hasattr(module, "language_model"):
            return module
    raise ValueError("Could not locate the Qwen2.5-VL multimodal model needed to compute MRoPE position ids.")


def _extend_qwen25_sequence_inputs(
    model_inputs: Mapping[str, torch.Tensor],
    *,
    prompt_length: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    extended = dict(model_inputs)
    extra_length = sequence_length - prompt_length
    if extra_length <= 0:
        return extended
    mm_token_type_ids = extended.get("mm_token_type_ids")
    if mm_token_type_ids is not None and mm_token_type_ids.shape[-1] == prompt_length:
        padding = torch.zeros(
            (*mm_token_type_ids.shape[:-1], extra_length),
            dtype=mm_token_type_ids.dtype,
            device=mm_token_type_ids.device,
        )
        extended["mm_token_type_ids"] = torch.cat([mm_token_type_ids, padding], dim=-1)
    return extended


def _import_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The HL memory adapters require `transformers` to be installed in the active environment."
        ) from exc
    return transformers


def _parse_completed_objective_text(text: str) -> str:
    data = _extract_json_dict(text)
    if set(data) != {"completed_objective"}:
        raise ValueError("Pass B output must contain exactly the `completed_objective` key.")
    objective = data["completed_objective"]
    if objective is None:
        return ""
    if not isinstance(objective, str):
        raise ValueError("Pass B `completed_objective` must be a string.")
    return objective.strip()


def _parse_completion_update_text(text: str) -> tuple[str, str]:
    data = _extract_json_dict(text)
    keys = set(data)
    if keys == {"completed_objective"}:
        objective = _parse_completed_objective_text(text)
        return objective, ""
    if keys != {"new_completed_objective", "task_progress"}:
        raise ValueError(
            "Pass B output must contain exactly `new_completed_objective` and `task_progress`."
        )
    objective = data["new_completed_objective"]
    task_progress = data["task_progress"]
    if objective is None:
        objective = ""
    if not isinstance(objective, str):
        raise ValueError("Pass B `new_completed_objective` must be a string.")
    if isinstance(task_progress, list):
        items = [str(item).strip(" .") for item in task_progress if str(item).strip(" .")]
        task_progress = "; ".join(items) + ("." if items else "")
    elif not isinstance(task_progress, str):
        raise ValueError("Pass B `task_progress` must be a string or list of strings.")
    return objective.strip(), task_progress.strip() or "No completed subtask yet."


def _parse_horizon_objective_text(text: str) -> str:
    data = _extract_json_dict(text)
    if set(data) != {"horizon_current_objective"}:
        raise ValueError("Horizon output must contain exactly the `horizon_current_objective` key.")
    objective = data["horizon_current_objective"]
    if not isinstance(objective, str):
        raise ValueError("Horizon `horizon_current_objective` must be a string.")
    return objective.strip()


def _extract_json_dict(text: str) -> dict[str, object]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(text[start : index + 1])
                if not isinstance(data, dict):
                    raise ValueError("Decoded JSON is not an object.")
                return data
    raise ValueError("Unterminated JSON object.")


def _build_batched_field_ids(
    *,
    tokenizer: Any,
    labels: torch.Tensor,
    target_texts: tuple[str, ...],
) -> torch.Tensor | None:
    annotations = _build_batched_field_annotations(tokenizer=tokenizer, labels=labels, target_texts=target_texts)
    return None if annotations is None else annotations[0]


def _build_batched_field_annotations(
    *,
    tokenizer: Any,
    labels: torch.Tensor,
    target_texts: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if labels.shape[0] != len(target_texts):
        logging.warning(
            "Skipping HL field loss breakdown: labels rows=%d target_texts=%d.",
            labels.shape[0],
            len(target_texts),
        )
        return None
    rows: list[torch.Tensor] = []
    value_rows: list[torch.Tensor] = []
    for row_index, target_text in enumerate(target_texts):
        row_field_ids = torch.full((labels.shape[1],), -1, dtype=torch.long)
        row_value_mask = torch.zeros((labels.shape[1],), dtype=torch.bool)
        target_field_annotations = _target_text_field_ids_and_value_mask(tokenizer, target_text)
        if target_field_annotations is None:
            return None
        target_field_ids, target_value_mask = target_field_annotations
        supervised_positions = torch.nonzero(labels[row_index] != -100, as_tuple=False).flatten().cpu()
        target_token_ids = _target_text_token_ids(tokenizer, target_text)
        if target_token_ids is None:
            return None
        supervised_token_ids = labels[row_index, supervised_positions].detach().cpu().tolist()
        if supervised_token_ids[: len(target_token_ids)] != target_token_ids:
            logging.warning(
                "Skipping HL field loss breakdown; supervised target tokens do not match target_text. "
                "row=%d supervised_tokens=%d target_tokens=%d first_mismatch=%s",
                row_index,
                len(supervised_token_ids),
                len(target_token_ids),
                _first_mismatched_token_index(supervised_token_ids, target_token_ids),
            )
            return None
        if len(supervised_token_ids) > len(target_token_ids):
            logging.debug(
                "HL field loss breakdown ignores %d supervised chat-template suffix tokens for row=%d.",
                len(supervised_token_ids) - len(target_token_ids),
                row_index,
            )
        assign_count = min(int(supervised_positions.numel()), len(target_field_ids))
        if assign_count > 0:
            row_field_ids[supervised_positions[:assign_count]] = torch.as_tensor(
                target_field_ids[:assign_count],
                dtype=torch.long,
            )
            row_value_mask[supervised_positions[:assign_count]] = torch.as_tensor(
                target_value_mask[:assign_count],
                dtype=torch.bool,
            )
        rows.append(row_field_ids)
        value_rows.append(row_value_mask)
    return torch.stack(rows, dim=0), torch.stack(value_rows, dim=0)


def _target_text_token_ids(tokenizer: Any, target_text: str) -> list[int] | None:
    try:
        encoded = tokenizer(target_text, add_special_tokens=False)
    except Exception as exc:  # pragma: no cover - tokenizer-specific fallback.
        logging.warning("Skipping HL field loss breakdown; target_text tokenization failed: %s", exc)
        return None
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if token_ids is None:
        logging.warning("Skipping HL field loss breakdown; tokenizer did not return input_ids.")
        return None
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _first_mismatched_token_index(left: list[int], right: list[int]) -> int | str:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return "none"


def _keyframe_auxiliary_targets_for_sample(
    sample: ExportedHLMemorySample,
    *,
    config: HLMemoryConfig,
    enabled: bool,
) -> dict[str, Any]:
    num_positions = config.recent_frames_length
    valid_length = min(sample.recent_valid_length, len(sample.recent_frame_indices), num_positions)
    valid_positions = [index < valid_length for index in range(num_positions)]
    event_positive = bool(sample.keyframe_event_ids or sample.keyframe_candidate_positions)

    canonical_frame: int | None = None
    if sample.keyframe_event_frame_indices:
        canonical_frame = int(sample.keyframe_event_frame_indices[0])
    elif sample.keyframe_label is True:
        canonical_frame = int(sample.frame_index)

    canonical_position = -1.0
    if canonical_frame is not None and valid_length > 0:
        nearest_index = min(
            range(valid_length),
            key=lambda index: abs(int(sample.recent_frame_indices[index]) - canonical_frame),
        )
        canonical_position = float(nearest_index)

    position_targets = [0.0] * num_positions
    if event_positive and valid_length > 0:
        sigma_frames = max(config.keyframe_aux_timing_sigma_sec * config.training_fps, 1e-6)
        if canonical_frame is not None:
            weights = [
                math.exp(
                    -0.5
                    * ((float(sample.recent_frame_indices[index]) - canonical_frame) / sigma_frames) ** 2
                )
                for index in range(valid_length)
            ]
        else:
            candidate_indices = [
                position - 1
                for position in sample.keyframe_candidate_positions
                if 1 <= position <= valid_length
            ]
            weights = [
                max(
                    (
                        math.exp(-0.5 * ((index - candidate_index) / max(config.recent_sample_hz, 1.0)) ** 2)
                        for candidate_index in candidate_indices
                    ),
                    default=0.0,
                )
                for index in range(valid_length)
            ]
        total = sum(weights)
        if total > 0.0:
            position_targets[:valid_length] = [weight / total for weight in weights]

    update_target = 0  # reject
    if event_positive and canonical_frame is not None and canonical_frame <= sample.frame_index:
        update_target = 2 if canonical_frame in sample.memory_frame_indices else 1

    return {
        "enabled": bool(enabled),
        "position_targets": position_targets,
        "event_target": float(event_positive),
        "update_target": update_target,
        "canonical_position": canonical_position,
        "valid_positions": valid_positions,
    }


def _target_text_field_ids(tokenizer: Any, target_text: str) -> list[int] | None:
    annotations = _target_text_field_ids_and_value_mask(tokenizer, target_text)
    return None if annotations is None else annotations[0]


def _target_text_field_ids_and_value_mask(tokenizer: Any, target_text: str) -> tuple[list[int], list[bool]] | None:
    field_spans = _target_text_field_char_spans(target_text)
    if not field_spans:
        return [], []
    try:
        encoded = tokenizer(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception as exc:  # pragma: no cover - tokenizer-specific fallback.
        logging.warning("Skipping HL field loss breakdown; tokenizer offset mapping failed: %s", exc)
        return None
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else None
    if token_ids is None or offsets is None:
        logging.warning("Skipping HL field loss breakdown; tokenizer did not return input_ids/offset_mapping.")
        return None
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    field_ids: list[int] = []
    token_offsets: list[tuple[int, int]] = []
    for offset in offsets:
        if isinstance(offset, torch.Tensor):
            start, end = (int(value) for value in offset.tolist())
        else:
            start, end = int(offset[0]), int(offset[1])
        token_offsets.append((start, end))
        field_ids.append(_field_id_for_char_range(start, end, field_spans))
    value_spans = _target_text_field_value_char_spans(target_text)
    value_mask = [_char_range_overlaps_spans(start, end, value_spans) for start, end in token_offsets]
    return field_ids, value_mask


def _target_text_field_char_spans(target_text: str) -> list[tuple[int, int, int]]:
    starts: list[tuple[int, str]] = []
    for field_name in HL_LOSS_FIELD_IDS_BY_NAME:
        marker = f'"{field_name}":'
        index = target_text.find(marker)
        if index >= 0:
            starts.append((index, field_name))
    starts.sort()
    spans: list[tuple[int, int, int]] = []
    for position, (start, field_name) in enumerate(starts):
        end = starts[position + 1][0] - 1 if position + 1 < len(starts) else target_text.rfind("}")
        if end < start:
            end = len(target_text)
        spans.append((start, end, HL_LOSS_FIELD_IDS_BY_NAME[field_name]))
    return spans


def _target_text_field_value_char_spans(target_text: str) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    for field_start, field_end, field_id in _target_text_field_char_spans(target_text):
        colon = target_text.find(":", field_start, field_end)
        if colon < 0:
            continue
        value_start = colon + 1
        while value_start < field_end and target_text[value_start].isspace():
            value_start += 1
        value_end = field_end
        while value_end > value_start and target_text[value_end - 1] in " \n\r\t,":
            value_end -= 1
        if value_end > value_start:
            spans.append((value_start, value_end, field_id))
    return spans


def _field_id_for_char_range(start: int, end: int, spans: list[tuple[int, int, int]]) -> int:
    if end <= start:
        return -1
    for span_start, span_end, field_id in spans:
        if start < span_end and end > span_start:
            return field_id
    return -1


def _char_range_overlaps_spans(start: int, end: int, spans: list[tuple[int, int, int]]) -> bool:
    if end <= start:
        return False
    return any(start < span_end and end > span_start for span_start, span_end, _ in spans)


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


def _previous_task_progress(sample: ExportedHLMemorySample) -> str:
    fields = _parse_ll_memory_fields(sample.language_memory)
    return fields.get("task progress", sample.language_memory.strip()).strip() or "No completed subtask yet."


def _state_context_for_protocol(protocol: str, sample: ExportedHLMemorySample) -> tuple[str, str, str]:
    if protocol == "objective_memory_state":
        return (
            "Completed-subtasks memory",
            sample.language_memory.strip(),
            "This protocol tests whether the model can consume and update compact completed-task memory.",
        )
    if protocol == "objective_last_objective":
        return (
            "State context",
            "",
            "This protocol does not provide last_objective as input. It asks the model to predict it as an auxiliary "
            "training target so the model learns the relation between previous and current objectives.",
        )
    if protocol == "objective_prev_stage":
        return (
            "Previous stage objective",
            sample.previous_stage_objective.strip(),
            "This protocol tests whether the model can consume and maintain the previous distinct objective as compact "
            "stage-level state.",
        )
    raise ValueError(f"Unsupported state-context protocol: {protocol!r}")


def _state_output_instruction_for_protocol(protocol: str) -> str:
    if protocol == "objective_memory_state":
        return (
            "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
            "`updated_language_memory`, and `keyframe_candidate_positions`.\n"
            "`updated_language_memory` must update the completed-subtasks memory after observing the current frame.\n"
        )
    if protocol == "objective_last_objective":
        return (
            "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
            "`last_objective`, and `keyframe_candidate_positions`.\n"
            "`last_objective` is an auxiliary training field: infer the immediately previous objective before the "
            "current objective when possible. Runtime may ignore this field.\n"
        )
    if protocol == "objective_prev_stage":
        return (
            "Return exactly one JSON object with keys `current_objective`, `horizon_current_objective`, "
            "`previous_stage_objective`, and `keyframe_candidate_positions`.\n"
            "`previous_stage_objective` is the previous distinct objective before the current stage. Update it when "
            "the visible objective has advanced to a new stage.\n"
        )
    raise ValueError(f"Unsupported state-output protocol: {protocol!r}")


def _state_extra_key_instruction_for_protocol(protocol: str) -> str:
    allowed = {
        "objective_memory_state": "`updated_language_memory`",
        "objective_last_objective": "`last_objective`",
        "objective_prev_stage": "`previous_stage_objective`",
    }[protocol]
    return (
        "Do not include progress, should_advance_objective, notes, markdown, explanation text, or extra keys other "
        f"than {allowed}.\n"
    )


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


def _render_object_context(sample: ExportedHLMemorySample, clips: LoadedVideoClips) -> str:
    object_name = sample.object_name.strip()
    centers = sample.recent_object_center_points
    if not object_name and not centers:
        return ""
    valid_length = max(0, min(clips.recent_valid_length, len(centers) if centers else clips.recent_valid_length))
    rendered_centers: list[str] = []
    for index in range(valid_length):
        point = centers[index] if index < len(centers) else (None, None)
        x_value, y_value = point
        if x_value is None or y_value is None:
            rendered_centers.append(f"{index + 1}: missing")
        else:
            rendered_centers.append(f"{index + 1}: ({float(x_value):.3f}, {float(y_value):.3f})")
    center_text = "; ".join(rendered_centers) if rendered_centers else "not provided"
    name_text = object_name or "tracked object"
    return (
        "Object grounding context:\n"
        f"- Tracked object name: {name_text}\n"
        "- Recent object center points are aligned with the recent observation clip positions and normalized to "
        f"[0, 1] image coordinates: {center_text}.\n"
        "Use this context only as weak grounding for the visual object; do not copy these fields into the output JSON "
        "unless the target schema explicitly asks for them.\n"
    )


def _current_recent_frame_size(clips: LoadedVideoClips) -> tuple[int, int]:
    if clips.recent_valid_length <= 0 or not clips.recent_frames:
        return 0, 0
    index = min(clips.recent_valid_length, len(clips.recent_frames)) - 1
    size = getattr(clips.recent_frames[index], "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 0, 0


def _candidate_evidence_clips(
    clips: LoadedVideoClips,
    candidate_positions: tuple[int, ...],
) -> LoadedVideoClips:
    selected_indices = _valid_candidate_indices(
        candidate_positions,
        min(clips.recent_valid_length, len(clips.recent_frames)),
    )
    selected_frames = tuple(clips.recent_frames[index] for index in selected_indices)
    if not selected_frames:
        width, height = _current_recent_frame_size(clips)
        selected_frames = (Image.new("RGB", (max(width, 1), max(height, 1)), color=(0, 0, 0)),)
    return LoadedVideoClips(
        memory_frames=clips.memory_frames,
        recent_frames=selected_frames,
        memory_valid_length=clips.memory_valid_length,
        recent_valid_length=len(selected_indices),
    )


def _valid_candidate_indices(candidate_positions: tuple[int, ...], valid_length: int) -> list[int]:
    selected_indices = {
        int(position) - 1
        for position in candidate_positions
        if 0 <= int(position) - 1 < valid_length
    }
    return sorted(selected_indices)


def _sample_recent_indexable_length(sample: ExportedHLMemorySample) -> int:
    """Return the valid recent-frame prefix that can be safely sliced from sample metadata."""

    lengths = [
        max(int(sample.recent_valid_length), 0),
        len(sample.recent_frame_paths),
        len(sample.recent_frame_indices),
    ]
    if sample.recent_robot_states:
        lengths.append(len(sample.recent_robot_states))
    if sample.recent_robot_state_masks:
        lengths.append(len(sample.recent_robot_state_masks))
    if sample.recent_object_center_points:
        lengths.append(len(sample.recent_object_center_points))
    return min(lengths)


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)
