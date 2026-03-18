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
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _infer_bimanual(data: dict) -> bool:
    """自动判断单臂/双臂：优先看 image keys，其次看 state 维度。"""
    img = data.get("image", {})
    if isinstance(img, dict):
        if "robot_0" in img or "robot_1" in img:
            return True
        if "front" in img:
            return False

    st = np.asarray(data.get("state", []))
    if st.ndim >= 1 and st.shape[-1] >= 13:
        return True
    return False


@dataclasses.dataclass(frozen=True)
class FastUMIInputs(transforms.DataTransformFn):
    """
    兼容单臂/双臂的 OpenPI 输入
    """
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    bimanual: bool | None = None
    use_depth_as_left_wrist: bool = False

    def __call__(self, data: dict) -> dict:
        mask_padding = (self.model_type == _model.ModelType.PI0)

        state = np.asarray(data["state"])

        img_dict = data.get("image", {})
        if not isinstance(img_dict, dict):
            raise ValueError(f"Expected data['image'] to be dict, got {type(img_dict)}")

        is_bi = _infer_bimanual({"state": state, "image": img_dict}) if (self.bimanual is None) else self.bimanual

        if not is_bi:
            if "front" not in img_dict:
                raise KeyError("Single-arm expects image['front'].")
            base_rgb = _parse_image(img_dict["front"])
            zero_img = np.zeros_like(base_rgb)

            left_rgb, right_rgb = zero_img, zero_img
            base_mask = True
            left_mask = (not mask_padding)
            right_mask = (not mask_padding)

        else:
            if "robot_0" not in img_dict or "robot_1" not in img_dict:
                raise KeyError("Bimanual expects image['robot_0'] and image['robot_1'].")

            img0 = _parse_image(img_dict["robot_0"])
            img1 = _parse_image(img_dict["robot_1"])
            zero_img = np.zeros_like(img0)

            base_rgb = zero_img
            left_rgb = img0
            right_rgb = img1

            base_mask = (not mask_padding)
            left_mask = True
            right_mask = True

        if self.use_depth_as_left_wrist:
            if "depth" not in img_dict:
                raise KeyError("use_depth_as_left_wrist=True but image['depth'] is missing.")
            left_rgb = _parse_image(img_dict["depth"])
            left_mask = True

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_rgb,
                "left_wrist_0_rgb": left_rgb,
                "right_wrist_0_rgb": right_rgb,
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        elif "action" in data:
            inputs["actions"] = np.asarray(data["action"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FastUMIOutputs(transforms.DataTransformFn):
    """
    兼容单臂/双臂：
    - 单臂通常输出 7 维：[xyz, rpy, g]
    - 双臂通常输出 14 维：左7 + 右7
    """
    original_action_dim: int | None = None
    model_action_dim: int | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim == 3:
            actions = actions.squeeze(0)
        elif actions.ndim == 1:
            actions = actions[None, :]

        if self.original_action_dim is not None:
            out_dim = int(self.original_action_dim)
        else:
            dim_ref = int(self.model_action_dim) if (self.model_action_dim is not None) else int(actions.shape[-1])
            out_dim = 7 if dim_ref <= 10 else 14

        return {"action": actions[:, :out_dim]}