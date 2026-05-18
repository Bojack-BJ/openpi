from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


@dataclasses.dataclass(frozen=True)
class TargetPointPrediction:
    target_query: str
    point_xy: tuple[int, int]
    confidence: float | None = None

    @classmethod
    def from_json(cls, text: str, *, image_size: tuple[int, int] | None = None) -> "TargetPointPrediction":
        data = _extract_prediction_object(text)
        raw_point = (
            data.get("point_xy")
            or data.get("target_point")
            or data.get("pixel_point")
            or data.get("point")
            or data.get("target_pixel")
        )
        if raw_point is None and {"x", "y"}.issubset(data):
            raw_point = [data["x"], data["y"]]
        point_xy = _parse_point(raw_point)
        if image_size is not None:
            point_xy = _clip_point(point_xy, image_size)
        return cls(
            target_query=str(data.get("target_query", data.get("target", ""))).strip(),
            point_xy=point_xy,
            confidence=_parse_optional_float(data.get("confidence")),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "target_query": self.target_query,
            "point_xy": [int(self.point_xy[0]), int(self.point_xy[1])],
        }
        if self.confidence is not None:
            result["confidence"] = float(self.confidence)
        return result


@dataclasses.dataclass(frozen=True)
class MaskSelectionPrediction:
    selected_mask_id: int
    target_query: str = ""
    confidence: float | None = None

    @classmethod
    def from_json(cls, text: str, *, max_mask_id: int | None = None) -> "MaskSelectionPrediction":
        data = _extract_prediction_object(text)
        raw_id = data.get("selected_mask_id", data.get("mask_id", data.get("target_mask_id")))
        if raw_id is None:
            raise ValueError("Mask selection output must include `selected_mask_id`, `mask_id`, or `target_mask_id`.")
        selected_mask_id = int(raw_id)
        if selected_mask_id <= 0:
            raise ValueError("selected_mask_id must be positive and 1-indexed.")
        if max_mask_id is not None and selected_mask_id > max_mask_id:
            raise ValueError(f"selected_mask_id {selected_mask_id} is out of range for {max_mask_id} candidates.")
        return cls(
            selected_mask_id=selected_mask_id,
            target_query=str(data.get("target_query", data.get("target", ""))).strip(),
            confidence=_parse_optional_float(data.get("confidence")),
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"selected_mask_id": int(self.selected_mask_id)}
        if self.target_query:
            result["target_query"] = self.target_query
        if self.confidence is not None:
            result["confidence"] = float(self.confidence)
        return result


@dataclasses.dataclass(frozen=True)
class CandidateMask:
    mask_id: int
    mask: np.ndarray
    source: pathlib.Path | None = None
    score: float | None = None

    @property
    def area(self) -> int:
        return int(np.asarray(self.mask, dtype=bool).sum())

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return mask_to_bbox(self.mask)


def build_target_point_prompt(
    *,
    instruction: str,
    current_subtask: str = "",
    language_memory: str = "",
    target_query: str = "",
    goal_query: str = "",
    image_size: tuple[int, int],
) -> str:
    width, height = image_size
    return (
        "Identify the single target object or object part that is relevant to the current robot subtask. "
        "Return one pixel point near the center of that exact target instance, not every object of the same class.\n"
        f"Image coordinate system: x is 0..{width - 1} from left to right, y is 0..{height - 1} from top to bottom.\n"
        "Use the instruction, current subtask, previous memory, and visual evidence to disambiguate same-class "
        "objects.\n"
        "Return exactly one JSON object with keys `target_query`, `point_xy`, and optional `confidence`.\n"
        "Do not include markdown, explanation text, or extra keys.\n"
        f"Task instruction: {instruction.strip() or 'unspecified'}\n"
        f"Current subtask: {current_subtask.strip() or 'unspecified'}\n"
        f"Previous language memory: {language_memory.strip() or 'none'}\n"
        f"Target query hint: {target_query.strip() or 'none'}\n"
        f"Goal query hint: {goal_query.strip() or 'none'}\n"
    )


def build_mask_selection_prompt(
    *,
    instruction: str,
    current_subtask: str = "",
    language_memory: str = "",
    target_query: str = "",
    goal_query: str = "",
    candidate_count: int,
) -> str:
    return (
        "You receive two images: first the original current frame, then the same frame with numbered candidate masks. "
        "Select the one candidate mask that corresponds to the target object or object part for the current robot "
        "subtask. "
        "The correct answer should identify one task-grounded instance, not all objects of the same class.\n"
        f"Valid mask ids are integers from 1 to {candidate_count}.\n"
        "Use the instruction, current subtask, previous memory, target query, goal query, and visual evidence to "
        "disambiguate same-class objects.\n"
        "Return exactly one JSON object with keys `selected_mask_id`, `target_query`, and optional `confidence`.\n"
        "Do not include markdown, explanation text, or extra keys.\n"
        f"Task instruction: {instruction.strip() or 'unspecified'}\n"
        f"Current subtask: {current_subtask.strip() or 'unspecified'}\n"
        f"Previous language memory: {language_memory.strip() or 'none'}\n"
        f"Target query hint: {target_query.strip() or 'none'}\n"
        f"Goal query hint: {goal_query.strip() or 'none'}\n"
    )


def load_candidate_masks(mask_dir: pathlib.Path | str, *, min_area: int = 1) -> list[CandidateMask]:
    mask_dir = pathlib.Path(mask_dir)
    paths = sorted(path for path in mask_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    candidates: list[CandidateMask] = []
    for path in paths:
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if int(mask.sum()) < min_area:
            continue
        candidates.append(CandidateMask(mask_id=len(candidates) + 1, mask=mask, source=path))
    return candidates


def save_mask(mask: np.ndarray, output_path: pathlib.Path | str) -> pathlib.Path:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output_path)
    return output_path


def overlay_point(
    image: Image.Image,
    point_xy: tuple[int, int],
    *,
    label: str = "target",
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    x, y = _clip_point(point_xy, overlay.size)
    radius = max(5, min(overlay.size) // 40)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 40, 40), outline=(255, 255, 255), width=3)
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 255, 255), width=2)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 255, 255), width=2)
    _draw_label(draw, (x + radius + 4, y - radius), label)
    return overlay


def overlay_mask_candidates(
    image: Image.Image,
    candidates: Iterable[CandidateMask],
    *,
    selected_mask_id: int | None = None,
    alpha: float = 0.42,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for candidate in candidates:
        color = _candidate_color(candidate.mask_id)
        if selected_mask_id is not None and candidate.mask_id == selected_mask_id:
            color = (255, 48, 48)
        mask = _resize_mask_to_image(candidate.mask, base.size)
        color_layer = Image.new("RGBA", base.size, (*color, int(255 * alpha)))
        overlay = Image.composite(color_layer, overlay, Image.fromarray(mask.astype(np.uint8) * 255))
        bbox = mask_to_bbox(mask)
        if bbox != (0, 0, 0, 0):
            width = 4 if selected_mask_id == candidate.mask_id else 2
            for offset in range(width):
                draw.rectangle(
                    (bbox[0] + offset, bbox[1] + offset, bbox[2] - offset, bbox[3] - offset),
                    outline=(*color, 255),
                )
            _draw_label(draw, (bbox[0] + 3, bbox[1] + 3), str(candidate.mask_id), fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def segment_from_point_with_sam(
    image: Image.Image,
    point_xy: tuple[int, int],
    *,
    checkpoint: pathlib.Path | str,
    model_type: str,
    device: str,
) -> np.ndarray:
    sam_model_registry, SamPredictor, _ = _import_segment_anything()
    image_np = np.asarray(image.convert("RGB"))
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_np)
    point = np.asarray([point_xy], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(point_coords=point, point_labels=labels, multimask_output=True)
    best_index = int(np.argmax(scores))
    return np.asarray(masks[best_index], dtype=bool)


def generate_sam_candidate_masks(
    image: Image.Image,
    *,
    checkpoint: pathlib.Path | str,
    model_type: str,
    device: str,
    max_candidates: int,
    min_area: int,
) -> list[CandidateMask]:
    sam_model_registry, _, SamAutomaticMaskGenerator = _import_segment_anything()
    image_np = np.asarray(image.convert("RGB"))
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    generator = SamAutomaticMaskGenerator(sam)
    raw_masks = generator.generate(image_np)
    raw_masks = sorted(
        raw_masks,
        key=lambda item: (float(item.get("predicted_iou", 0.0)), int(item.get("area", 0))),
        reverse=True,
    )
    candidates: list[CandidateMask] = []
    for raw_mask in raw_masks:
        mask = np.asarray(raw_mask["segmentation"], dtype=bool)
        area = int(raw_mask.get("area", mask.sum()))
        if area < min_area:
            continue
        candidates.append(
            CandidateMask(
                mask_id=len(candidates) + 1,
                mask=mask,
                score=_parse_optional_float(raw_mask.get("predicted_iou")),
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def _extract_prediction_object(text: str) -> dict[str, Any]:
    fallback: dict[str, Any] | None = None
    for candidate_text in _candidate_json_texts(text):
        for parsed in _iter_json_objects(candidate_text):
            fallback = parsed
    if fallback is not None:
        return fallback
    raise ValueError("Could not parse a JSON object from model output.")


def _candidate_json_texts(text: str) -> list[str]:
    stripped = text.strip()
    candidates = []
    if "</think>" in stripped:
        candidates.append(stripped.rsplit("</think>", maxsplit=1)[-1].strip())
    candidates.extend(_extract_fenced_blocks(stripped))
    candidates.append(_strip_fenced_block(stripped))
    candidates.append(stripped)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def _extract_fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    in_block = False
    current: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_block:
                blocks.append("\n".join(current).strip())
                current = []
                in_block = False
            else:
                current = []
                in_block = True
            continue
        if in_block:
            current.append(line)
    return blocks


def _strip_fenced_block(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


def _parse_point(value: object) -> tuple[int, int]:
    if isinstance(value, dict):
        if {"x", "y"}.issubset(value):
            return int(round(float(value["x"]))), int(round(float(value["y"])))
        raise ValueError("Point dict must include `x` and `y`.")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(round(float(value[0]))), int(round(float(value[1])))
    raise ValueError("Point must be a list `[x, y]` or dict `{'x': x, 'y': y}`.")


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip_point(point_xy: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    x = min(max(int(point_xy[0]), 0), max(width - 1, 0))
    y = min(max(int(point_xy[1]), 0), max(height - 1, 0))
    return x, y


def _resize_mask_to_image(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (image_size[1], image_size[0]):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _candidate_color(mask_id: int) -> tuple[int, int, int]:
    palette = (
        (59, 130, 246),
        (16, 185, 129),
        (245, 158, 11),
        (236, 72, 153),
        (14, 165, 233),
        (132, 204, 22),
        (249, 115, 22),
        (168, 85, 247),
    )
    return palette[(mask_id - 1) % len(palette)]


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int, int] | tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = _load_font(18)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    padding = 4
    bg_fill = (0, 0, 0, 190) if len(fill) == 4 else (0, 0, 0)
    draw.rectangle(
        (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
        fill=bg_fill,
    )
    draw.text((x, y), text, font=font, fill=fill)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _import_segment_anything():
    try:
        from segment_anything import SamAutomaticMaskGenerator
        from segment_anything import SamPredictor
        from segment_anything import sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM execution requires the optional `segment-anything` package. "
            "Install it and pass `--sam-checkpoint` to run mask generation."
        ) from exc
    return sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
