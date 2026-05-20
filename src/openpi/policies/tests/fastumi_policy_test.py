import numpy as np

from openpi.models import model as _model
from openpi.policies import fastumi_policy


def test_single_arm_mask_images_use_front_mask_and_mask_missing_slots():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    transform = fastumi_policy.FastUMIInputs(
        action_dim=32,
        model_type=_model.ModelType.PI0,
        include_mask_images=True,
    )

    out = transform({"state": np.zeros(8), "image": {"front": image}, "mask": {"front": mask}})

    assert list(out["image"]) == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        "base_0_mask_rgb",
        "left_wrist_0_mask_rgb",
        "right_wrist_0_mask_rgb",
    ]
    assert out["image_mask"]["base_0_rgb"] is True
    assert out["image_mask"]["base_0_mask_rgb"] is True
    assert out["image_mask"]["left_wrist_0_mask_rgb"] is False
    assert np.all(out["image"]["base_0_mask_rgb"][1:3, 1:3] == 255)


def test_dual_arm_mask_images_accept_image_mask_keys():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    transform = fastumi_policy.FastUMIInputs(
        action_dim=32,
        model_type=_model.ModelType.PI0,
        include_mask_images=True,
    )

    out = transform(
        {
            "state": np.zeros(16),
            "image": {
                "robot_0": image,
                "robot_1": image,
                "robot_0_mask": mask,
            },
        }
    )

    assert out["image_mask"]["left_wrist_0_rgb"] is True
    assert out["image_mask"]["right_wrist_0_rgb"] is True
    assert out["image_mask"]["left_wrist_0_mask_rgb"] is True
    assert out["image_mask"]["right_wrist_0_mask_rgb"] is False
    assert np.all(out["image"]["left_wrist_0_mask_rgb"][0:2, 0:2] == 255)
