from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from PIL import Image
from PIL import ImageOps
import torch


def compose_observation_frame(
    images: Mapping[str, np.ndarray | torch.Tensor],
    *,
    frame_height: int,
    frame_width: int,
    max_images: int = 4,
) -> Image.Image:
    ordered_images = [images[key] for key in sorted(images)[:max_images]]
    pil_images = [_to_pil_image(image) for image in ordered_images]
    return _compose_grid(
        pil_images,
        cell_height=frame_height,
        cell_width=frame_width,
        columns=2,
        gap=8,
    )


def compose_context_panel(
    memory_frames: Sequence[Image.Image],
    recent_frames: Sequence[Image.Image],
    *,
    frame_height: int,
    frame_width: int,
    columns: int,
    gap: int,
) -> Image.Image:
    sections: list[Image.Image] = []
    if memory_frames:
        sections.append(
            _compose_grid(
                memory_frames,
                cell_height=frame_height,
                cell_width=frame_width,
                columns=columns,
                gap=gap,
            )
        )
    if recent_frames:
        sections.append(
            _compose_grid(
                recent_frames,
                cell_height=frame_height,
                cell_width=frame_width,
                columns=columns,
                gap=gap,
            )
        )
    if not sections:
        return Image.new("RGB", (frame_width, frame_height), color=(0, 0, 0))
    if len(sections) == 1:
        return sections[0]

    width = max(section.width for section in sections)
    height = sum(section.height for section in sections) + gap
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    offset_y = 0
    for section in sections:
        canvas.paste(section, (0, offset_y))
        offset_y += section.height + gap
    return canvas


def _compose_grid(
    images: Sequence[Image.Image],
    *,
    cell_height: int,
    cell_width: int,
    columns: int,
    gap: int,
) -> Image.Image:
    if not images:
        return Image.new("RGB", (cell_width, cell_height), color=(0, 0, 0))
    prepared = [_resize_with_pad(image, cell_width=cell_width, cell_height=cell_height) for image in images]
    rows = math.ceil(len(prepared) / columns)
    width = columns * cell_width + max(columns - 1, 0) * gap
    height = rows * cell_height + max(rows - 1, 0) * gap
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    for index, image in enumerate(prepared):
        row = index // columns
        column = index % columns
        x = column * (cell_width + gap)
        y = row * (cell_height + gap)
        canvas.paste(image, (x, y))
    return canvas


def _resize_with_pad(image: Image.Image, *, cell_width: int, cell_height: int) -> Image.Image:
    contained = ImageOps.contain(image, (cell_width, cell_height))
    canvas = Image.new("RGB", (cell_width, cell_height), color=(0, 0, 0))
    offset_x = (cell_width - contained.width) // 2
    offset_y = (cell_height - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    return canvas


def _to_pil_image(image: np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with rank 3, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            if array.min() >= -1.0 and array.max() <= 1.0:
                array = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            elif array.min() >= 0.0 and array.max() <= 1.0:
                array = (array * 255.0).clip(0, 255).astype(np.uint8)
            else:
                array = array.clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3], mode="RGB")
