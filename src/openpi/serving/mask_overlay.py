from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image


MASK_OVERLAY_KEY = "__mask_overlay"


@dataclasses.dataclass(frozen=True)
class MaskOverlayResult:
    image: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    score: float | None
    bbox_xyxy: tuple[int, int, int, int] | None
    initialized: bool
    needs_reprompt: bool = False
    reprompt_reason: str | None = None
    selected_obj_id: int | None = None
    candidate_count: int | None = None


@dataclasses.dataclass(frozen=True)
class _VideoCandidate:
    obj_id: int
    mask: np.ndarray
    score: float | None
    bbox_xyxy: tuple[int, int, int, int] | None


class Sam3ImageMaskOverlay:
    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        alpha: float = 0.35,
        multimask_output: bool = True,
        sam3_path: str | pathlib.Path | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._alpha = float(alpha)
        self._multimask_output = bool(multimask_output)
        self._sam3_path = pathlib.Path(sam3_path) if sam3_path is not None else _default_sam3_path()
        self._model = None
        self._processor = None
        self._state = None
        self._last_logits = None
        self._last_mask: np.ndarray | None = None
        self._last_bbox: tuple[int, int, int, int] | None = None

    def preload(self) -> None:
        self._ensure_loaded()

    def reset(self) -> None:
        self._state = None
        self._last_logits = None
        self._last_mask = None
        self._last_bbox = None

    def apply(
        self,
        image: Any,
        *,
        points: Any = None,
        point_labels: Any = None,
        text: str | None = None,
        reset: bool = False,
        alpha: float | None = None,
    ) -> MaskOverlayResult:
        del text
        if reset:
            self.reset()

        image_np = _as_uint8_hwc(image)
        self._ensure_loaded()
        self._state = self._processor.set_image(Image.fromarray(image_np))

        points_np = _optional_points(points)
        labels_np = _optional_point_labels(point_labels, num_points=0 if points_np is None else len(points_np))
        if points_np is not None:
            masks, scores, logits = self._predict_with_points(points_np, labels_np)
        elif self._last_bbox is not None:
            masks, scores, logits = self._predict_with_previous_bbox()
        else:
            return MaskOverlayResult(
                image=image_np,
                mask=np.zeros(image_np.shape[:2], dtype=np.uint8),
                overlay=image_np,
                score=None,
                bbox_xyxy=None,
                initialized=False,
            )

        scores_np = _to_numpy(scores).reshape(-1)
        best_index = int(np.argmax(scores_np)) if len(scores_np) else 0
        mask = np.squeeze(_to_numpy(masks[best_index])) > 0
        if mask.ndim != 2:
            raise ValueError(f"SAM3 returned an unsupported mask shape: {mask.shape}")
        score = float(scores_np[best_index]) if len(scores_np) else None
        self._last_logits = logits[best_index]
        self._last_mask = mask
        self._last_bbox = _mask_bbox_xyxy(mask)
        overlay = _overlay_mask(image_np, mask, alpha=self._alpha if alpha is None else float(alpha))
        return MaskOverlayResult(
            image=image_np,
            mask=(mask.astype(np.uint8) * 255),
            overlay=overlay,
            score=score,
            bbox_xyxy=self._last_bbox,
            initialized=True,
        )

    def _predict_with_points(self, points: np.ndarray, labels: np.ndarray):
        return self._model.predict_inst(
            self._state,
            point_coords=points.astype(np.float32),
            point_labels=labels.astype(np.int32),
            multimask_output=self._multimask_output,
        )

    def _predict_with_previous_bbox(self):
        assert self._last_bbox is not None
        box = np.asarray(self._last_bbox, dtype=np.float32)
        mask_input = self._last_logits
        if mask_input is not None and getattr(mask_input, "ndim", None) == 2:
            mask_input = mask_input[None, :, :]
        try:
            return self._model.predict_inst(
                self._state,
                point_coords=None,
                point_labels=None,
                box=box,
                mask_input=mask_input,
                multimask_output=False,
            )
        except Exception:
            logging.exception("SAM3 mask-input tracking failed; falling back to previous bbox prompt only.")
            return self._model.predict_inst(
                self._state,
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if self._checkpoint_path is not None:
            checkpoint_path_str = str(self._checkpoint_path)
            checkpoint_path = pathlib.Path(checkpoint_path_str).expanduser()
            if "://" not in checkpoint_path_str and not checkpoint_path.exists():
                raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint_path}")
            self._checkpoint_path = str(checkpoint_path)

        if self._sam3_path.exists():
            sam3_path = str(self._sam3_path)
            if sam3_path not in sys.path:
                sys.path.insert(0, sam3_path)

        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_package_dir = pathlib.Path(sam3.__file__).resolve().parent
        bpe_path = sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        build_kwargs = {
            "checkpoint_path": self._checkpoint_path,
            "device": self._device,
            "enable_inst_interactivity": True,
        }
        if bpe_path.exists():
            build_kwargs["bpe_path"] = str(bpe_path)
        logging.info(
            "Loading SAM3 mask overlay model on %s from %s",
            self._device,
            self._checkpoint_path or "Hugging Face default checkpoint",
        )
        start_time = time.monotonic()
        self._model = build_sam3_image_model(**build_kwargs)
        self._processor = Sam3Processor(self._model)
        logging.info("SAM3 mask overlay model loaded in %.1fs", time.monotonic() - start_time)


class Sam3VideoWindowMaskOverlay(Sam3ImageMaskOverlay):
    """Online SAM3 video tracking using a short rolling frame window.

    SAM3's video predictor initializes from a fixed video resource rather than accepting
    appended frames. For online rollout, we rebuild a short video session from the recent
    frames, prompt frame 0 with the tracked object's bbox, and propagate to the latest
    frame.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        alpha: float = 0.35,
        multimask_output: bool = True,
        sam3_path: str | pathlib.Path | None = None,
        video_window_size: int = 8,
        video_version: str = "sam3",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            alpha=alpha,
            multimask_output=multimask_output,
            sam3_path=sam3_path,
        )
        if video_window_size < 2:
            raise ValueError("video_window_size must be at least 2")
        if video_version not in {"sam3", "sam3.1"}:
            raise ValueError(f"Unsupported SAM3 video version: {video_version!r}")
        self._video_window_size = int(video_window_size)
        self._video_version = video_version
        self._video_model = None
        self._video_frames: list[np.ndarray] = []
        self._video_frame_bboxes: list[tuple[int, int, int, int] | None] = []
        self._video_prompt_bbox: tuple[int, int, int, int] | None = None

    def preload(self) -> None:
        super().preload()
        self._ensure_video_loaded()

    def reset(self) -> None:
        super().reset()
        self._video_frames = []
        self._video_frame_bboxes = []
        self._video_prompt_bbox = None

    def apply(
        self,
        image: Any,
        *,
        points: Any = None,
        point_labels: Any = None,
        text: str | None = None,
        reset: bool = False,
        alpha: float | None = None,
    ) -> MaskOverlayResult:
        del text
        if reset:
            self.reset()

        image_np = _as_uint8_hwc(image)
        points_np = _optional_points(points)
        if points_np is not None:
            result = super().apply(
                image_np,
                points=points_np,
                point_labels=point_labels,
                reset=False,
                alpha=alpha,
            )
            self._video_frames = [image_np]
            self._video_frame_bboxes = [result.bbox_xyxy]
            self._video_prompt_bbox = result.bbox_xyxy
            self._ensure_video_loaded()
            return result

        if self._video_prompt_bbox is None or not self._video_frames:
            return MaskOverlayResult(
                image=image_np,
                mask=np.zeros(image_np.shape[:2], dtype=np.uint8),
                overlay=image_np,
                score=None,
                bbox_xyxy=None,
                initialized=False,
            )

        self._ensure_video_loaded()
        self._video_frames.append(image_np)
        self._video_frame_bboxes.append(None)
        self._trim_video_window()

        try:
            result = self._track_with_video_window(alpha=self._alpha if alpha is None else float(alpha))
        except Exception:
            logging.exception("SAM3 video-window tracking failed; falling back to image bbox tracking.")
            result = super().apply(image_np, reset=False, alpha=alpha)
            if result.initialized:
                self._video_frames = [image_np]
                self._video_frame_bboxes = [result.bbox_xyxy]
                self._video_prompt_bbox = result.bbox_xyxy
            return result

        return result

    def _ensure_video_loaded(self) -> None:
        if self._video_model is not None:
            return

        self._ensure_loaded()

        import sam3
        from sam3 import build_sam3_predictor

        sam3_package_dir = pathlib.Path(sam3.__file__).resolve().parent
        bpe_path = sam3_package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        build_kwargs = {
            "checkpoint_path": self._checkpoint_path,
            "version": self._video_version,
            "compile": False,
            "async_loading_frames": False,
        }
        if bpe_path.exists():
            build_kwargs["bpe_path"] = str(bpe_path)
        logging.info(
            "Loading SAM3 video mask tracker (%s) from %s",
            self._video_version,
            self._checkpoint_path or "Hugging Face default checkpoint",
        )
        start_time = time.monotonic()
        self._video_model = build_sam3_predictor(**build_kwargs)
        logging.info("SAM3 video mask tracker loaded in %.1fs", time.monotonic() - start_time)

    def _track_with_video_window(self, *, alpha: float) -> MaskOverlayResult:
        assert self._video_model is not None
        assert self._video_prompt_bbox is not None

        frames = [Image.fromarray(frame) for frame in self._video_frames]
        response = self._video_model.handle_request({"type": "start_session", "resource_path": frames})
        session_id = response["session_id"]
        masks_by_frame: dict[int, np.ndarray] = {}
        scores_by_frame: dict[int, float | None] = {}
        try:
            height, width = self._video_frames[0].shape[:2]
            self._video_model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "bounding_boxes": [
                        _bbox_xyxy_to_normalized_xywh(self._video_prompt_bbox, width, height)
                    ],
                    "bounding_box_labels": [1],
                    "rel_coordinates": True,
                }
            )
            for event in self._video_model.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": len(self._video_frames),
                }
            ):
                frame_index = event.get("frame_index")
                outputs = event.get("outputs", {})
                if frame_index is None or not isinstance(outputs, dict):
                    continue
                mask, score = _mask_from_video_outputs(outputs)
                if mask is not None:
                    masks_by_frame[int(frame_index)] = mask
                    scores_by_frame[int(frame_index)] = score
        finally:
            self._video_model.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": False,
                }
            )

        last_index = len(self._video_frames) - 1
        mask = masks_by_frame.get(last_index)
        if mask is None:
            return MaskOverlayResult(
                image=self._video_frames[-1],
                mask=np.zeros(self._video_frames[-1].shape[:2], dtype=np.uint8),
                overlay=self._video_frames[-1],
                score=None,
                bbox_xyxy=None,
                initialized=False,
            )

        frame_bboxes = []
        for frame_index, previous_bbox in enumerate(self._video_frame_bboxes):
            frame_mask = masks_by_frame.get(frame_index)
            frame_bboxes.append(_mask_bbox_xyxy(frame_mask) if frame_mask is not None else previous_bbox)
        self._video_frame_bboxes = frame_bboxes
        self._video_prompt_bbox = self._video_frame_bboxes[0] or self._video_prompt_bbox
        self._last_mask = mask
        self._last_bbox = _mask_bbox_xyxy(mask)
        self._last_logits = None

        image_np = self._video_frames[-1]
        overlay = _overlay_mask(image_np, mask, alpha=alpha)
        return MaskOverlayResult(
            image=image_np,
            mask=(mask.astype(np.uint8) * 255),
            overlay=overlay,
            score=scores_by_frame.get(last_index),
            bbox_xyxy=self._last_bbox,
            initialized=True,
        )

    def _trim_video_window(self) -> None:
        while len(self._video_frames) > self._video_window_size:
            self._video_frames.pop(0)
            self._video_frame_bboxes.pop(0)
        if self._video_frame_bboxes:
            self._video_prompt_bbox = self._video_frame_bboxes[0] or self._video_prompt_bbox


class Sam3TextSelectVideoMaskOverlay(Sam3VideoWindowMaskOverlay):
    """Text-detect all instances, select one by point/location, then continuity-select it."""

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        alpha: float = 0.35,
        multimask_output: bool = True,
        sam3_path: str | pathlib.Path | None = None,
        video_window_size: int = 8,
        video_version: str = "sam3",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            alpha=alpha,
            multimask_output=multimask_output,
            sam3_path=sam3_path,
            video_window_size=video_window_size,
            video_version=video_version,
        )
        self._text_prompt: str | None = None

    def reset(self) -> None:
        super().reset()
        self._text_prompt = None

    def apply(
        self,
        image: Any,
        *,
        points: Any = None,
        point_labels: Any = None,
        text: str | None = None,
        reset: bool = False,
        alpha: float | None = None,
    ) -> MaskOverlayResult:
        del point_labels
        if reset:
            self.reset()

        image_np = _as_uint8_hwc(image)
        text_prompt = _optional_text(text) or self._text_prompt
        points_np = _optional_points(points)
        tracking_alpha = self._alpha if alpha is None else float(alpha)

        if points_np is not None:
            if text_prompt is None:
                return _needs_reprompt_result(
                    image_np,
                    alpha=tracking_alpha,
                    reason="text_select_video requires a text prompt with the clicked point.",
                )
            self._ensure_video_loaded()
            self._text_prompt = text_prompt
            self._video_frames = [image_np]
            self._video_frame_bboxes = [None]
            self._video_prompt_bbox = None
            return self._track_with_text_selection(
                alpha=tracking_alpha,
                text_prompt=text_prompt,
                point_xy=tuple(points_np[0].tolist()),
            )

        if text_prompt is not None:
            self._text_prompt = text_prompt
        if self._text_prompt is None:
            return _needs_reprompt_result(
                image_np,
                alpha=tracking_alpha,
                reason="text_select_video is missing the text prompt.",
            )
        if self._last_bbox is None or not self._video_frames:
            return _needs_reprompt_result(
                image_np,
                alpha=tracking_alpha,
                reason="text_select_video is not initialized; click the target again.",
            )

        self._ensure_video_loaded()
        self._video_frames.append(image_np)
        self._video_frame_bboxes.append(None)
        self._trim_video_window()
        return self._track_with_text_selection(alpha=tracking_alpha, text_prompt=self._text_prompt)

    def _track_with_text_selection(
        self,
        *,
        alpha: float,
        text_prompt: str,
        point_xy: tuple[float, float] | None = None,
    ) -> MaskOverlayResult:
        assert self._video_model is not None

        frames = [Image.fromarray(frame) for frame in self._video_frames]
        response = self._video_model.handle_request({"type": "start_session", "resource_path": frames})
        session_id = response["session_id"]
        masks_by_frame: dict[int, np.ndarray] = {}
        scores_by_frame: dict[int, float | None] = {}
        bboxes_by_frame: dict[int, tuple[int, int, int, int] | None] = {}
        selected_obj_id: int | None = None
        candidate_count: int | None = None
        try:
            prompt_response = self._video_model.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": text_prompt,
                }
            )
            prompt_outputs = prompt_response.get("outputs", {})
            if not isinstance(prompt_outputs, dict):
                return _needs_reprompt_result(
                    self._video_frames[-1],
                    alpha=alpha,
                    reason=f"SAM3 text prompt did not return candidates for {text_prompt!r}.",
                )

            candidates = _video_candidates_from_outputs(prompt_outputs)
            candidate_count = len(candidates)
            anchor_bbox = self._video_frame_bboxes[0] or self._video_prompt_bbox or self._last_bbox
            selected = _select_video_candidate(candidates, point_xy=point_xy, bbox_xyxy=anchor_bbox)
            if selected is None:
                return _needs_reprompt_result(
                    self._video_frames[-1],
                    alpha=alpha,
                    reason=f"No SAM3 instance matched text={text_prompt!r}; click the target again.",
                    candidate_count=candidate_count,
                )

            selected_obj_id = selected.obj_id
            current_bbox = selected.bbox_xyxy or anchor_bbox
            masks_by_frame[0] = selected.mask
            scores_by_frame[0] = selected.score
            bboxes_by_frame[0] = selected.bbox_xyxy

            if len(self._video_frames) > 1:
                for event in self._video_model.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": len(self._video_frames),
                    }
                ):
                    frame_index = event.get("frame_index")
                    outputs = event.get("outputs", {})
                    if frame_index is None or not isinstance(outputs, dict):
                        continue
                    frame_index_int = int(frame_index)
                    if frame_index_int == 0:
                        continue
                    frame_candidates = _video_candidates_from_outputs(outputs)
                    candidate = _select_video_candidate(frame_candidates, bbox_xyxy=current_bbox)
                    if candidate is None:
                        continue
                    selected_obj_id = candidate.obj_id
                    current_bbox = candidate.bbox_xyxy or current_bbox
                    masks_by_frame[frame_index_int] = candidate.mask
                    scores_by_frame[frame_index_int] = candidate.score
                    bboxes_by_frame[frame_index_int] = candidate.bbox_xyxy
        finally:
            self._video_model.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": False,
                }
            )

        last_index = len(self._video_frames) - 1
        mask = masks_by_frame.get(last_index)
        if mask is None:
            return _needs_reprompt_result(
                self._video_frames[-1],
                alpha=alpha,
                reason=f"Tracked SAM3 instance id={selected_obj_id} disappeared; click the target again.",
                candidate_count=candidate_count,
                selected_obj_id=selected_obj_id,
            )

        frame_bboxes = []
        for frame_index, previous_bbox in enumerate(self._video_frame_bboxes):
            frame_bboxes.append(bboxes_by_frame.get(frame_index, previous_bbox))
        self._video_frame_bboxes = frame_bboxes
        self._video_prompt_bbox = self._video_frame_bboxes[0] or self._video_prompt_bbox
        self._last_mask = mask
        self._last_bbox = _mask_bbox_xyxy(mask)
        self._last_logits = None

        image_np = self._video_frames[-1]
        overlay = _overlay_mask(image_np, mask, alpha=alpha)
        return MaskOverlayResult(
            image=image_np,
            mask=(mask.astype(np.uint8) * 255),
            overlay=overlay,
            score=scores_by_frame.get(last_index),
            bbox_xyxy=self._last_bbox,
            initialized=True,
            selected_obj_id=selected_obj_id,
            candidate_count=candidate_count,
        )


class MaskOverlayPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        default_view: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        alpha: float = 0.35,
        multimask_output: bool = True,
        sam3_path: str | pathlib.Path | None = None,
        tracking_mode: str = "image",
        video_window_size: int = 8,
        video_version: str = "sam3",
    ) -> None:
        self._policy = policy
        self._default_view = default_view
        self._tracking_mode = tracking_mode
        if tracking_mode == "image":
            self._segmenter = Sam3ImageMaskOverlay(
                checkpoint_path=checkpoint_path,
                device=device,
                alpha=alpha,
                multimask_output=multimask_output,
                sam3_path=sam3_path,
            )
        elif tracking_mode == "video_window":
            self._segmenter = Sam3VideoWindowMaskOverlay(
                checkpoint_path=checkpoint_path,
                device=device,
                alpha=alpha,
                multimask_output=multimask_output,
                sam3_path=sam3_path,
                video_window_size=video_window_size,
                video_version=video_version,
            )
        elif tracking_mode == "text_select_video":
            self._segmenter = Sam3TextSelectVideoMaskOverlay(
                checkpoint_path=checkpoint_path,
                device=device,
                alpha=alpha,
                multimask_output=multimask_output,
                sam3_path=sam3_path,
                video_window_size=video_window_size,
                video_version=video_version,
            )
        else:
            raise ValueError(f"Unsupported mask overlay tracking mode: {tracking_mode!r}")

    def preload(self) -> None:
        self._segmenter.preload()

    def infer(self, obs: dict) -> dict:
        request = obs.get(MASK_OVERLAY_KEY)
        if not isinstance(request, dict) or not request.get("enabled", True):
            return self._policy.infer(_strip_mask_request(obs))

        policy_obs = _strip_mask_request(obs)
        image_dict = policy_obs.get("image")
        if not isinstance(image_dict, dict) or not image_dict:
            return self._policy.infer(policy_obs)

        view = str(request.get("view") or self._default_view or next(iter(image_dict)))
        if view not in image_dict:
            raise KeyError(f"Mask overlay view '{view}' is not present in obs['image']; available={sorted(image_dict)}")

        result = self._segmenter.apply(
            image_dict[view],
            points=request.get("points"),
            point_labels=request.get("point_labels"),
            text=request.get("text"),
            reset=bool(request.get("reset", False)),
            alpha=request.get("alpha"),
        )

        response_payload = _result_payload(
            result,
            view=view,
            tracking_mode=self._tracking_mode,
            include_image=bool(request.get("return_image", False)),
        )
        if bool(request.get("preview_only", False)):
            return {MASK_OVERLAY_KEY: response_payload}
        if result.needs_reprompt:
            return {MASK_OVERLAY_KEY: response_payload}

        image_dict = dict(image_dict)
        image_dict[view] = result.overlay
        policy_obs = dict(policy_obs)
        policy_obs["image"] = image_dict
        output = self._policy.infer(policy_obs)
        output[MASK_OVERLAY_KEY] = response_payload
        return output

    def reset(self) -> None:
        self._segmenter.reset()
        self._policy.reset()


def _default_sam3_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3] / "third_party" / "sam3"


def _strip_mask_request(obs: dict) -> dict:
    if MASK_OVERLAY_KEY not in obs:
        return obs
    return {key: value for key, value in obs.items() if key != MASK_OVERLAY_KEY}


def _result_payload(
    result: MaskOverlayResult,
    *,
    view: str,
    tracking_mode: str,
    include_image: bool = False,
) -> dict[str, Any]:
    payload = {
        "view": view,
        "tracking_mode": tracking_mode,
        "mask": result.mask,
        "overlay": result.overlay,
        "score": result.score,
        "bbox_xyxy": result.bbox_xyxy,
        "initialized": result.initialized,
        "needs_reprompt": result.needs_reprompt,
        "reprompt_reason": result.reprompt_reason,
        "selected_obj_id": result.selected_obj_id,
        "candidate_count": result.candidate_count,
    }
    if include_image:
        payload["image"] = result.image
    return payload


def _as_uint8_hwc(image: Any) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim != 3:
        raise ValueError(f"Expected HWC/CHW RGB image, got shape={image_np.shape}")
    if image_np.shape[0] in (3, 4) and image_np.shape[-1] not in (3, 4):
        image_np = np.moveaxis(image_np, 0, -1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    if image_np.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={image_np.shape}")
    if image_np.dtype == np.uint8:
        return np.ascontiguousarray(image_np)
    if np.issubdtype(image_np.dtype, np.floating):
        if image_np.min() >= -1.0 and image_np.max() <= 1.0:
            image_np = (image_np + 1.0) * 127.5
        elif image_np.min() >= 0.0 and image_np.max() <= 1.0:
            image_np = image_np * 255.0
    return np.ascontiguousarray(np.clip(image_np, 0, 255).astype(np.uint8))


def _optional_points(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    points_np = np.asarray(points, dtype=np.float32)
    if points_np.size == 0:
        return None
    points_np = points_np.reshape(-1, 2)
    return points_np


def _optional_point_labels(labels: Any, *, num_points: int) -> np.ndarray:
    if labels is None:
        return np.ones((num_points,), dtype=np.int32)
    labels_np = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(labels_np) != num_points:
        raise ValueError(f"Expected {num_points} point labels, got {len(labels_np)}")
    return labels_np


def _optional_text(text: Any) -> str | None:
    if text is None:
        return None
    text_str = str(text).strip()
    return text_str or None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _needs_reprompt_result(
    image: np.ndarray,
    *,
    alpha: float,
    reason: str,
    candidate_count: int | None = None,
    selected_obj_id: int | None = None,
) -> MaskOverlayResult:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return MaskOverlayResult(
        image=image,
        mask=mask,
        overlay=_overlay_mask(image, mask, alpha=alpha),
        score=None,
        bbox_xyxy=None,
        initialized=False,
        needs_reprompt=True,
        reprompt_reason=reason,
        selected_obj_id=selected_obj_id,
        candidate_count=candidate_count,
    )


def _video_candidates_from_outputs(outputs: dict[str, Any]) -> list[_VideoCandidate]:
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return []

    masks_np = _to_numpy(masks)
    if masks_np.size == 0:
        return []
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    num_masks = int(masks_np.shape[0])
    obj_ids = outputs.get("out_obj_ids")
    obj_ids_np = _to_numpy(obj_ids).reshape(-1) if obj_ids is not None else np.arange(num_masks)
    if len(obj_ids_np) != num_masks:
        obj_ids_np = np.arange(num_masks)

    scores = outputs.get("out_probs")
    scores_np = _to_numpy(scores).reshape(-1) if scores is not None else np.asarray([], dtype=np.float32)

    candidates: list[_VideoCandidate] = []
    for index in range(num_masks):
        mask = np.squeeze(masks_np[index]) > 0
        if mask.ndim != 2:
            raise ValueError(f"SAM3 video tracker returned an unsupported mask shape: {mask.shape}")
        mask = _largest_connected_component(mask)
        if not mask.any():
            continue
        score = float(scores_np[index]) if len(scores_np) == num_masks else None
        candidates.append(
            _VideoCandidate(
                obj_id=int(obj_ids_np[index]),
                mask=mask,
                score=score,
                bbox_xyxy=_mask_bbox_xyxy(mask),
            )
        )
    return candidates


def _select_video_candidate(
    candidates: list[_VideoCandidate],
    *,
    point_xy: tuple[float, float] | None = None,
    bbox_xyxy: tuple[int, int, int, int] | None = None,
) -> _VideoCandidate | None:
    if not candidates:
        return None

    if point_xy is not None:
        point_x, point_y = point_xy
        containing = [candidate for candidate in candidates if _mask_contains_point(candidate.mask, point_x, point_y)]
        if containing:
            return max(containing, key=_candidate_score)
        return min(candidates, key=lambda candidate: _point_distance_to_mask(candidate.mask, point_x, point_y))

    if bbox_xyxy is not None:
        best_by_iou = max(candidates, key=lambda candidate: _bbox_iou(candidate.bbox_xyxy, bbox_xyxy))
        if _bbox_iou(best_by_iou.bbox_xyxy, bbox_xyxy) > 0.0:
            return best_by_iou
        center_x = (bbox_xyxy[0] + bbox_xyxy[2]) * 0.5
        center_y = (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5
        return min(candidates, key=lambda candidate: _candidate_center_distance(candidate, center_x, center_y))

    return max(candidates, key=_candidate_score)


def _mask_from_video_outputs(outputs: dict[str, Any]) -> tuple[np.ndarray | None, float | None]:
    candidates = _video_candidates_from_outputs(outputs)
    if not candidates:
        return None, None
    candidate = max(candidates, key=_candidate_score)
    return candidate.mask, candidate.score


def _mask_bbox_xyxy(mask: np.ndarray, *, padding: int = 4) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape[:2]
    return (
        max(int(xs.min()) - padding, 0),
        max(int(ys.min()) - padding, 0),
        min(int(xs.max()) + padding, width - 1),
        min(int(ys.max()) + padding, height - 1),
    )


def _bbox_xyxy_to_normalized_xywh(
    bbox_xyxy: Sequence[int | float],
    width: int,
    height: int,
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    x0 = min(max(x0, 0.0), float(width - 1))
    y0 = min(max(y0, 0.0), float(height - 1))
    x1 = min(max(x1 + 1.0, x0 + 1.0), float(width))
    y1 = min(max(y1 + 1.0, y0 + 1.0), float(height))
    return [
        x0 / float(width),
        y0 / float(height),
        (x1 - x0) / float(width),
        (y1 - y0) / float(height),
    ]


def _candidate_score(candidate: _VideoCandidate) -> float:
    return float(candidate.score) if candidate.score is not None else 0.0


def _mask_contains_point(mask: np.ndarray, point_x: float, point_y: float) -> bool:
    x = int(round(point_x))
    y = int(round(point_y))
    height, width = mask.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    return bool(mask[y, x])


def _point_distance_to_mask(mask: np.ndarray, point_x: float, point_y: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("inf")
    return float(np.min((xs.astype(np.float32) - point_x) ** 2 + (ys.astype(np.float32) - point_y) ** 2))


def _candidate_center_distance(candidate: _VideoCandidate, point_x: float, point_y: float) -> float:
    if candidate.bbox_xyxy is None:
        return float("inf")
    center_x = (candidate.bbox_xyxy[0] + candidate.bbox_xyxy[2]) * 0.5
    center_y = (candidate.bbox_xyxy[1] + candidate.bbox_xyxy[3]) * 0.5
    return float((center_x - point_x) ** 2 + (center_y - point_y) ** 2)


def _bbox_iou(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter_width = max(ix1 - ix0 + 1, 0)
    inter_height = max(iy1 - iy0 + 1, 0)
    intersection = inter_width * inter_height
    area_a = max(ax1 - ax0 + 1, 0) * max(ay1 - ay0 + 1, 0)
    area_b = max(bx1 - bx0 + 1, 0) * max(by1 - by0 + 1, 0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if labels_count <= 1:
            return mask
        largest_label = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest_label
    except Exception:
        return mask


def _overlay_mask(image: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    overlay = np.array(image, copy=True).astype(np.float32)
    color = np.asarray([0, 255, 0], dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
