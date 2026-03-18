import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert to HWC uint8 image."""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # 如果通道在第一维，转成 HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class FastUMIInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    use_depth_as_left_wrist: bool = False   # ★ 新增字段

    def __call__(self, data: dict) -> dict:
        mask_padding = (self.model_type == _model.ModelType.PI0)

        state     = transforms.pad_to_dim(data["state"], self.action_dim)
        base_rgb  = _parse_image(data["image"]["front"])
        zero_img  = np.zeros_like(base_rgb)

        # depth → left_wrist_0_rgb
        if self.use_depth_as_left_wrist:
            left_wrist = _parse_image(data["image"]["depth"])
            left_mask  = True
        else:
            left_wrist = zero_img
            left_mask  = not mask_padding

        right_wrist = zero_img
        right_mask  = not mask_padding

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb":        base_rgb,
                "left_wrist_0_rgb":  left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb":        True,
                "left_wrist_0_rgb":  left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "action" in data:
            inputs["actions"] = transforms.pad_to_dim(data["action"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class FastUMIOutputs(transforms.DataTransformFn):
    """把模型输出切回 FastUMI 原始动作维度。"""
    original_action_dim: int = 7  # FastUMI delta action 是 7 维

    def __call__(self, data: dict) -> dict:
        # 这里把 data["actions"] 当成二维 [H, D] 来处理：
        # 只保留前 original_action_dim 维
        actions = np.asarray(data["actions"])  # 例如 (H, D) 或 (1, H, D) 都可
        if actions.ndim == 3:
            # 如果还是三维 [B, H, D]，先 squeeze batch 维
            actions = actions.squeeze(0)      # 变成 (H, D)
        return {"action": actions[:, : self.original_action_dim]}