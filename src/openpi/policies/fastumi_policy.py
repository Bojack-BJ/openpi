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


def _mask_to_rgb_image(mask, reference_image: np.ndarray) -> np.ndarray:
    """Convert an optional 2D/1CH/3CH mask to an HWC uint8 RGB image."""
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        if mask_np.shape[-1] == 1:
            mask_np = mask_np[..., 0]
        elif mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] in (3, 4):
            mask_np = np.max(mask_np[..., :3], axis=-1)
        elif mask_np.shape[0] in (3, 4):
            mask_np = np.max(mask_np[:3], axis=0)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected mask rank 2 or single-channel/rgb rank 3, got shape={mask_np.shape}")
    if mask_np.shape != reference_image.shape[:2]:
        raise ValueError(f"Mask shape {mask_np.shape} does not match image shape {reference_image.shape[:2]}")

    if mask_np.dtype == np.bool_:
        mask_bool = mask_np
    elif np.issubdtype(mask_np.dtype, np.floating):
        mask_bool = mask_np > (0.0 if np.nanmax(mask_np) > 1.0 else 0.5)
    else:
        mask_bool = mask_np > 0

    mask_rgb = np.zeros_like(reference_image, dtype=np.uint8)
    mask_rgb[mask_bool] = 255
    return mask_rgb


def _optional_mask_image(data: dict, img_dict: dict, name: str, reference_image: np.ndarray) -> tuple[np.ndarray, bool]:
    mask_dict = data.get("mask", {})
    if isinstance(mask_dict, dict) and name in mask_dict:
        return _mask_to_rgb_image(mask_dict[name], reference_image), True

    for key in (f"{name}_mask", f"{name}_mask_rgb"):
        if key in img_dict:
            return _mask_to_rgb_image(img_dict[key], reference_image), True

    return np.zeros_like(reference_image, dtype=np.uint8), False


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
    include_mask_images: bool = False

    def __call__(self, data: dict) -> dict:
        mask_padding = (self.model_type == _model.ModelType.PI0) or self.include_mask_images

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
            base_mask_rgb, base_guidance_mask = _optional_mask_image(data, img_dict, "front", base_rgb)
            left_mask_rgb, left_guidance_mask = zero_img, False
            right_mask_rgb, right_guidance_mask = zero_img, False

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
            base_mask_rgb, base_guidance_mask = zero_img, False
            left_mask_rgb, left_guidance_mask = _optional_mask_image(data, img_dict, "robot_0", left_rgb)
            right_mask_rgb, right_guidance_mask = _optional_mask_image(data, img_dict, "robot_1", right_rgb)

        if self.use_depth_as_left_wrist:
            if "depth" not in img_dict:
                raise KeyError("use_depth_as_left_wrist=True but image['depth'] is missing.")
            left_rgb = _parse_image(img_dict["depth"])
            left_mask = True
            left_mask_rgb, left_guidance_mask = _optional_mask_image(data, img_dict, "depth", left_rgb)

        image = {
            "base_0_rgb": base_rgb,
            "left_wrist_0_rgb": left_rgb,
            "right_wrist_0_rgb": right_rgb,
        }
        image_mask = {
            "base_0_rgb": base_mask,
            "left_wrist_0_rgb": left_mask,
            "right_wrist_0_rgb": right_mask,
        }
        if self.include_mask_images:
            image.update(
                {
                    "base_0_mask_rgb": base_mask_rgb,
                    "left_wrist_0_mask_rgb": left_mask_rgb,
                    "right_wrist_0_mask_rgb": right_mask_rgb,
                }
            )
            image_mask.update(
                {
                    "base_0_mask_rgb": base_guidance_mask,
                    "left_wrist_0_mask_rgb": left_guidance_mask,
                    "right_wrist_0_mask_rgb": right_guidance_mask,
                }
            )

        inputs = {
            "state": state,
            "image": image,
            "image_mask": image_mask,
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
