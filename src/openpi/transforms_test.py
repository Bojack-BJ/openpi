import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_prompt_preserve_raw_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer, preserve_raw_prompt=True)

    data = transform({"prompt": "Hello, world!"})

    assert data["raw_prompt"].item() == "Hello, world!"


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_inject_optional_guidance_fields_adds_empty_masks_and_subtask():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    transform = _transforms.InjectOptionalGuidanceFields(
        image_to_mask_paths={"observation.images.robot_0_image": "observation.masks.robot_0_mask"},
    )

    data = transform(
        {
            "observation.images.robot_0_image": image,
            "current_subtask": np.asarray("pick up the sponge"),
        }
    )

    assert data["observation.masks.robot_0_mask"].shape == (4, 5, 1)
    assert data["observation.masks.robot_0_mask"].dtype == np.uint8
    assert data["subtask"].item() == "pick up the sponge"


def test_compose_prompt_with_subtask_appends_optional_subtask():
    transform = _transforms.ComposePromptWithSubtask()

    data = transform({"prompt": np.asarray("clean the board"), "subtask": np.asarray("pick up the sponge")})

    assert data["prompt"] == "Overall instruction: clean the board\nCurrent subtask: pick up the sponge"
    assert "subtask" not in data


def test_compose_prompt_with_subtask_noops_without_subtask():
    transform = _transforms.ComposePromptWithSubtask()

    data = transform({"prompt": "clean the board", "subtask": ""})

    assert data["prompt"] == "clean the board"
    assert "subtask" not in data


def test_overlay_masks_on_images_overlays_mask_and_drops_mask():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4, 1), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 255
    transform = _transforms.OverlayMasksOnImages(
        image_to_mask_keys={"front": "front"},
        alpha=1.0,
        color=(10, 20, 30),
        draw_contour=False,
    )

    data = transform({"image": {"front": image}, "mask": {"front": mask}})

    assert np.all(data["image"]["front"][1:3, 1:3] == np.array([10, 20, 30], dtype=np.uint8))
    assert np.all(data["image"]["front"][0, 0] == 0)
    assert "mask" not in data


def test_overlay_masks_on_images_rejects_mismatched_mask_shape():
    transform = _transforms.OverlayMasksOnImages(image_to_mask_keys={"front": "front"})

    with pytest.raises(ValueError, match="does not match image shape"):
        transform(
            {
                "image": {"front": np.zeros((4, 4, 3), dtype=np.uint8)},
                "mask": {"front": np.zeros((3, 3), dtype=np.uint8)},
            }
        )


def test_subtask_from_segments_uses_half_open_frame_ranges():
    transform = _transforms.SubtaskFromSegments(
        {
            7: (
                (0, 10, "pick up the sponge"),
                (10, 20, "place the sponge"),
            )
        }
    )

    data = transform({"episode_index": np.asarray(7), "frame_index": np.asarray(10)})

    assert data["subtask"].item() == "place the sponge"
