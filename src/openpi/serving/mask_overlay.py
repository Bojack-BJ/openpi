from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import time
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
        reset: bool = False,
        alpha: float | None = None,
    ) -> MaskOverlayResult:
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
    ) -> None:
        self._policy = policy
        self._default_view = default_view
        self._segmenter = Sam3ImageMaskOverlay(
            checkpoint_path=checkpoint_path,
            device=device,
            alpha=alpha,
            multimask_output=multimask_output,
            sam3_path=sam3_path,
        )

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
            reset=bool(request.get("reset", False)),
            alpha=request.get("alpha"),
        )

        response_payload = _result_payload(result, view=view)
        if bool(request.get("preview_only", False)):
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


def _result_payload(result: MaskOverlayResult, *, view: str) -> dict[str, Any]:
    return {
        "view": view,
        "mask": result.mask,
        "overlay": result.overlay,
        "score": result.score,
        "bbox_xyxy": result.bbox_xyxy,
        "initialized": result.initialized,
    }


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


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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


def _overlay_mask(image: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    overlay = np.array(image, copy=True).astype(np.float32)
    color = np.asarray([0, 255, 0], dtype=np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)
