import dataclasses
from typing import Any
from typing import Callable
from typing import Protocol

import torch
from torch import nn


@dataclasses.dataclass
class PrefixBatch:
    """Backend-owned representation of the prefix token block.

    The current PyTorch path only consumes `embeds`, `pad_masks`, and `att_masks`.
    `metadata` is reserved for backend-specific multimodal position/cache state such as
    Qwen's `image_grid_thw`, `rope_deltas`, or modality type ids.
    """

    embeds: torch.Tensor
    pad_masks: torch.Tensor
    att_masks: torch.Tensor
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PrefixCache:
    """Backend-owned prefix cache for suffix-only denoising steps."""

    pad_masks: torch.Tensor
    past_key_values: Any
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class CheckpointFn(Protocol):
    def __call__(self, func: Callable[..., torch.Tensor], *args, **kwargs) -> torch.Tensor: ...


class VLMWithExpertModel(nn.Module):
    """Shared interface for prefix/suffix VLM backbones used by `PI0Pytorch`."""

    @staticmethod
    def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        if att_masks.ndim != 2:
            raise ValueError(att_masks.ndim)
        if pad_masks.ndim != 2:
            raise ValueError(pad_masks.ndim)

        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    @staticmethod
    def prepare_attention_mask_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def build_prefix_batch(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        *,
        checkpoint_fn: CheckpointFn | None = None,
    ) -> PrefixBatch:
        if checkpoint_fn is None:
            checkpoint_fn = lambda func, *args, **kwargs: func(*args, **kwargs)

        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = checkpoint_fn(self.embed_image, img)
            batch_size, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks += [0] * num_img_embs

        def embed_language(lang_tokens):
            lang_emb = self.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * (lang_emb_dim**0.5)

        lang_emb = checkpoint_fn(embed_language, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))

        return PrefixBatch(
            embeds=embs,
            pad_masks=pad_masks,
            att_masks=att_masks,
            metadata=self.build_prefix_metadata(images, img_masks, lang_tokens, lang_masks),
        )

    def build_prefix_metadata(self, images, img_masks, lang_tokens, lang_masks) -> dict[str, Any]:
        del images, img_masks, lang_tokens, lang_masks
        return {}

    def build_prefix_cache_metadata(
        self,
        prefix_batch: PrefixBatch,
        *,
        prefix_att_2d_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_att_2d_masks_4d: torch.Tensor,
    ) -> dict[str, Any]:
        del prefix_att_2d_masks, prefix_position_ids, prefix_att_2d_masks_4d
        return dict(prefix_batch.metadata)

    def build_prefix_cache(self, prefix_batch: PrefixBatch) -> PrefixCache:
        prefix_att_2d_masks = self.make_att_2d_masks(prefix_batch.pad_masks, prefix_batch.att_masks)
        prefix_position_ids = torch.cumsum(prefix_batch.pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.prepare_attention_mask_4d(prefix_att_2d_masks)

        self.set_prefix_attention_implementation("eager")
        _, past_key_values = self.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_batch.embeds, None],
            use_cache=True,
        )

        return PrefixCache(
            pad_masks=prefix_batch.pad_masks,
            past_key_values=past_key_values,
            metadata=self.build_prefix_cache_metadata(
                prefix_batch,
                prefix_att_2d_masks=prefix_att_2d_masks,
                prefix_position_ids=prefix_position_ids,
                prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
            ),
        )
