from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import numpy as np
from openpi_client import image_tools
import torch
from openpi.models import tokenizer as _tokenizer
from openpi.shared import normalize as _normalize
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R

DataDict: TypeAlias = Any
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


def _tree_map(fn: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _tree_map(fn, tree[k], *(r[k] for r in rest)) for k in tree}
    if isinstance(tree, list):
        return [_tree_map(fn, tree[i], *(r[i] for r in rest)) for i in range(len(tree))]
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, tree[i], *(r[i] for r in rest)) for i in range(len(tree)))
    return fn(tree, *rest)

def _resolve_action_key(data: DataDict, action_key: str | None) -> str | None:
    if action_key is not None:
        return action_key if action_key in data else None
    if "actions" in data:
        return "actions"
    if "action" in data:
        return "action"
    return None


def _validate_all_true_mask(mask: Sequence[bool] | None, expected_dim: int, *, name: str) -> None:
    if mask is None:
        return
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[-1] != expected_dim:
        raise ValueError(f"{name}: mask len={mask.shape[-1]} != expected_dim={expected_dim}")
    if not np.all(mask):
        raise ValueError(
            f"{name}: partial mask is not supported for representation-changing transforms. "
            f"Use mask=None or an all-True mask."
        )


@dataclasses.dataclass(frozen=True)
class SliceActions(DataTransformFn):
    """
    从 padded model action 中取前 dim 个有效语义维度。
    """
    dim: int
    action_key: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data
        out = dict(data)
        out[ak] = np.asarray(out[ak])[..., : self.dim]
        return out

# ---------------------------
# common helpers
# ---------------------------

@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: Any

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return _tree_map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: Any | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: Any | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PromptTokenizer
    discrete_state_input: bool = False
    # Preserve the original prompt alongside tokenized text so backends such as Qwen can
    # optionally rebuild their own prompt tokens without changing the data pipeline.
    preserve_raw_prompt: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        output = {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
        if self.preserve_raw_prompt:
            output["raw_prompt"] = np.asarray(prompt)
        return output


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: Any) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> Any:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: Any) -> Any:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: Any, selector: Any, fn: Callable[[T, S], T], *, strict: bool = False
) -> Any:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: Any) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )

@dataclasses.dataclass(frozen=True)
class ToNumpy(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:

        def _convert(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            # 粗判 JAX 数组类型
            if hasattr(x, "__array__") and "jax" in str(type(x)).lower():
                return np.asarray(x)
            return x

        return _tree_map(_convert, data)


# ---------------------------
# relative_6d
# ---------------------------

def _rotmat_to_6d(Rm: np.ndarray) -> np.ndarray:
    c0 = Rm[..., :, 0]
    c1 = Rm[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def _rot6d_to_rotmat(rot_6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + eps)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-1)


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPose(DataTransformFn):
    """
    绝对 [xyz, quat, g] -> 相对 [dxyz(local), rot6d, g]
    说明：
    - 这是 representation-changing transform
    - 输出始终是“纯 10D 语义动作”
    - mask 仅保留接口兼容，只允许 None 或全 True
    """
    pos_slice: slice = dataclasses.field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = dataclasses.field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = dataclasses.field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 10, name="ChunkRel6DPose")

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d or self.grip_slice.stop > d):
            raise ValueError(f"ChunkRel6DPose: state dim={d}, slices={self.pos_slice, self.quat_slice, self.grip_slice}")

        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
        base_R_T = np.swapaxes(base_R, -1, -2)

        actions_t = actions_full
        if actions_t.ndim == state.ndim:
            actions_t = actions_t[..., None, :]

        act_pos_abs = actions_t[..., self.pos_slice]
        act_quat_abs = actions_t[..., self.quat_slice]
        grip = actions_t[..., self.grip_slice]

        delta_pos = act_pos_abs - base_pos[..., None, :]
        rel_pos = np.einsum("...ij,...tj->...ti", base_R_T, delta_pos)

        R_abs = R.from_quat(act_quat_abs.reshape(-1, 4)).as_matrix()
        R_abs = R_abs.reshape(act_quat_abs.shape[:-1] + (3, 3))
        R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)
        rel_rot_6d = _rotmat_to_6d(R_rel)

        rel10 = np.concatenate([rel_pos, rel_rot_6d, grip], axis=-1)

        out = rel10[..., 0, :] if orig_ndim == state.ndim else rel10
        new_data = dict(data)
        new_data[ak] = out
        return new_data


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverse(DataTransformFn):
    """
    相对 [dxyz(local), rot6d, g] -> 绝对 [xyz, quat, g]
    说明：
    - 期望输入是纯 10D 语义动作
    - 如果误传入 padded 动作，也会只读取前 10 维
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 10, name="ChunkRel6DPoseInverse")

        state = np.asarray(data["state"])
        actions_orig = np.asarray(data[ak])
        orig_ndim = actions_orig.ndim

        actions = actions_orig[..., :10]
        if actions.ndim == state.ndim:
            actions = actions[..., None, :]

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d or self.grip_slice.stop > d):
            raise ValueError(
                f"ChunkRel6DPoseInverse: state dim={d}, "
                f"slices={self.pos_slice, self.quat_slice, self.grip_slice}"
            )

        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

        rel_pos = actions[..., 0:3]
        rel_rot6d = actions[..., 3:9]
        grip = actions[..., 9:10]

        delta_world = np.einsum("...ij,...tj->...ti", base_R, rel_pos)
        abs_pos = base_pos[..., None, :] + delta_world

        R_rel = _rot6d_to_rotmat(rel_rot6d)
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

        abs_quat = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_quat()
        abs_quat = abs_quat.reshape(R_abs.shape[:-2] + (4,))

        abs_actions = np.concatenate([abs_pos, abs_quat, grip], axis=-1)

        out = abs_actions[..., 0, :] if orig_ndim == state.ndim else abs_actions
        new_data = dict(data)
        new_data[ak] = out
        return new_data


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverseRPY(DataTransformFn):
    """
    相对 [dxyz(local), rot6d, g] -> 绝对 [xyz, rpy, g]
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 10, name="ChunkRel6DPoseInverseRPY")

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        actions = actions_full[..., :10]
        if actions.ndim == state.ndim:
            actions = actions[..., None, :]

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d or self.grip_slice.stop > d):
            raise ValueError(f"ChunkRel6DPoseInverseRPY: state dim={d}, slices={self.pos_slice, self.quat_slice, self.grip_slice}")

        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

        rel_pos = actions[..., 0:3]
        rel_rot6d = actions[..., 3:9]
        grip = actions[..., 9:10]

        abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, rel_pos)

        R_rel = _rot6d_to_rotmat(rel_rot6d)
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

        abs_rpy = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_euler(self.euler_order, degrees=true)
        abs_rpy = abs_rpy.reshape(R_abs.shape[:-2] + (3,))

        abs_actions = np.concatenate([abs_pos, abs_rpy, grip], axis=-1)

        out = abs_actions[..., 0, :] if orig_ndim == state.ndim else abs_actions
        new_data = dict(data)
        new_data[ak] = out
        return new_data


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseBimanual(DataTransformFn):
    """
    双臂绝对 [L8 + R8] -> 双臂相对 [L10 + R10] = 20D
    输出始终是纯 20D 语义动作。
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    left_action_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_action_slice: slice = field(default_factory=lambda: slice(8, 16))

    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))

    action_key: str | None = None
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 20, name="ChunkRel6DPoseBimanual")

        state16 = np.asarray(data["state"])
        act_full = np.asarray(data[ak])
        orig_ndim = act_full.ndim
        has_time = (act_full.ndim == state16.ndim + 1)

        act_t = act_full if has_time else act_full[..., None, :]

        sL = state16[..., self.left_state_slice]
        sR = state16[..., self.right_state_slice]
        aL = act_t[..., :, self.left_action_slice]
        aR = act_t[..., :, self.right_action_slice]

        single = ChunkRel6DPose(
            pos_slice=self.pos_slice,
            quat_slice=self.quat_slice,
            grip_slice=self.grip_slice,
            action_key="action",
            mask=None,
        )

        dL = single({"state": sL, "action": aL})
        dR = single({"state": sR, "action": aR})
        relL = np.asarray(dL["action"])
        relR = np.asarray(dR["action"])

        rel20 = np.concatenate([relL, relR], axis=-1)

        out = rel20[..., 0, :] if orig_ndim == state16.ndim else rel20
        new_data = dict(data)
        new_data[ak] = out
        return new_data


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverseRPYBimanual(DataTransformFn):
    """
    双臂相对 20D -> 双臂绝对 14D（xyz+rpy+g）
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    action_key: str | None = None
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 20, name="ChunkRel6DPoseInverseRPYBimanual")

        state16 = np.asarray(data["state"])
        act_full = np.asarray(data[ak])
        orig_ndim = act_full.ndim

        act_full = act_full[..., :20]
        has_time = (act_full.ndim == state16.ndim + 1)
        act_t = act_full if has_time else act_full[..., None, :]

        sL = state16[..., self.left_state_slice]
        sR = state16[..., self.right_state_slice]

        aL = act_t[..., :, 0:10]
        aR = act_t[..., :, 10:20]

        inv_single = ChunkRel6DPoseInverseRPY(
            pos_slice=slice(0, 3),
            quat_slice=slice(3, 7),
            grip_slice=slice(7, 8),
            action_key="actions",
            euler_order=self.euler_order,
            mask=None,
        )

        dL = inv_single({"state": sL, "actions": aL})
        dR = inv_single({"state": sR, "actions": aR})
        outL = np.asarray(dL["actions"])
        outR = np.asarray(dR["actions"])

        out14 = np.concatenate([outL, outR], axis=-1)

        out = out14[..., 0, :] if orig_ndim == state16.ndim else out14
        new_data = dict(data)
        new_data[ak] = out
        return new_data


# ---------------------------
# delta_rpy
# ---------------------------

def absolute_to_delta_pose_rpy_chunk_base(
    base_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    degrees: bool = True,
    pos_slice: slice = slice(0, 3),
    quat_slice: slice = slice(3, 7),
) -> np.ndarray:
    base_qpos = np.asarray(base_qpos)
    target_qpos = np.asarray(target_qpos)

    pos0 = base_qpos[..., pos_slice]
    quat0 = base_qpos[..., quat_slice]
    pos1 = target_qpos[..., pos_slice]
    quat1 = target_qpos[..., quat_slice]

    R0 = R.from_quat(quat0.reshape(-1, 4)).as_matrix().reshape(quat0.shape[:-1] + (3, 3))
    R1 = R.from_quat(quat1.reshape(-1, 4)).as_matrix().reshape(quat1.shape[:-1] + (3, 3))

    R0_T = np.swapaxes(R0, -1, -2)

    dpos = np.einsum("...ij,...j->...i", R0_T, (pos1 - pos0))
    R_rel = np.einsum("...ij,...jk->...ik", R0_T, R1)

    drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=degrees)
    drot = drot.reshape(R_rel.shape[:-2] + (3,))

    return np.concatenate([dpos, drot], axis=-1)


@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaPoseRPY(DataTransformFn):
    """
    绝对 [xyz, quat, (g)] -> 相对 [dxyz, drpy, (dg or g)]

    输出：
    - include_gripper=False: [dxyz, drpy]                (6D)
    - include_gripper=True, gripper_as_delta=True:
        [dxyz, drpy, dg]                                 (7D)
    - include_gripper=True, gripper_as_delta=False:
        [dxyz, drpy, g]                                  (7D)
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    degrees: bool = True
    include_gripper: bool = False
    gripper_as_delta: bool = True
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        rel_dim = 7 if self.include_gripper else 6
        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, rel_dim, name="ChunkRelDeltaPoseRPY")

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d):
            raise ValueError(f"ChunkRelDeltaPoseRPY: state dim={d}, slices={self.pos_slice, self.quat_slice}")

        if self.include_gripper and self.grip_slice.stop > d:
            raise ValueError(f"ChunkRelDeltaPoseRPY: state dim={d} has no gripper slice={self.grip_slice}")

        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        actions_t = actions_full
        if actions_t.ndim == state.ndim:
            actions_t = actions_t[..., None, :]

        act_pos_abs = actions_t[..., self.pos_slice]
        act_quat_abs = actions_t[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
        base_R_T = np.swapaxes(base_R, -1, -2)

        delta_world = act_pos_abs - base_pos[..., None, :]
        dpos = np.einsum("...ij,...tj->...ti", base_R_T, delta_world)

        R_abs = R.from_quat(act_quat_abs.reshape(-1, 4)).as_matrix()
        R_abs = R_abs.reshape(act_quat_abs.shape[:-1] + (3, 3))
        R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)

        drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=self.degrees)
        drot = drot.reshape(R_rel.shape[:-2] + (3,))

        rel6 = np.concatenate([dpos, drot], axis=-1)

        if self.include_gripper:
            if actions_t.shape[-1] < self.grip_slice.stop:
                raise ValueError(
                    f"ChunkRelDeltaPoseRPY: actions dim={actions_t.shape[-1]} has no gripper slice={self.grip_slice}"
                )

            next_grip = actions_t[..., self.grip_slice]  # (...,T,1)

            if self.gripper_as_delta:
                base_grip = state[..., self.grip_slice]          # (...,1)
                grip_part = next_grip - base_grip[..., None, :]  # (...,T,1) = dg
            else:
                grip_part = next_grip                            # (...,T,1) = g

            rel = np.concatenate([rel6, grip_part], axis=-1)     # (...,T,7)
        else:
            rel = rel6

        out = rel[..., 0, :] if orig_ndim == state.ndim else rel
        new_data = dict(data)
        new_data[ak] = out
        return new_data

@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaPoseRPYInverse(DataTransformFn):
    """
    相对 [dxyz, drpy, (dg or g)] -> 绝对 [xyz, rpy, (g)]

    输出：
    - include_gripper=False: [x, y, z, roll, pitch, yaw]        (6D)
    - include_gripper=True : [x, y, z, roll, pitch, yaw, g]     (7D)
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    degrees: bool = True
    include_gripper: bool = False
    gripper_as_delta: bool = True
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        rel_dim = 7 if self.include_gripper else 6
        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, rel_dim, name="ChunkRelDeltaPoseRPYInverse")

        state = np.asarray(data["state"])
        rel_full = np.asarray(data[ak])
        orig_ndim = rel_full.ndim

        d = state.shape[-1]
        if self.include_gripper and self.grip_slice.stop > d:
            raise ValueError(f"ChunkRelDeltaPoseRPYInverse: state dim={d} has no gripper slice={self.grip_slice}")

        rel = rel_full[..., :rel_dim]
        if rel.ndim == state.ndim:
            rel = rel[..., None, :]

        if rel.shape[-1] < 6:
            raise ValueError(f"ChunkRelDeltaPoseRPYInverse: rel dim must be >=6, got {rel.shape[-1]}")

        rel6 = rel[..., 0:6]
        dpos = rel6[..., 0:3]
        drot = rel6[..., 3:6]

        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

        abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, dpos)

        R_rel = R.from_euler(self.euler_order, drot.reshape(-1, 3), degrees=self.degrees).as_matrix()
        R_rel = R_rel.reshape(drot.shape[:-1] + (3, 3))
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

        abs_rpy = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_euler(self.euler_order, degrees=self.degrees)
        abs_rpy = abs_rpy.reshape(R_abs.shape[:-2] + (3,))

        if self.include_gripper:
            if rel.shape[-1] < 7:
                raise ValueError(f"ChunkRelDeltaPoseRPYInverse: include_gripper=True but rel dim={rel.shape[-1]} < 7")

            grip_part = rel[..., 6:7]

            if self.gripper_as_delta:
                base_grip = state[..., self.grip_slice]            # (...,1)
                abs_grip = base_grip[..., None, :] + grip_part     # (...,T,1)
            else:
                abs_grip = grip_part                               # (...,T,1)

            abs_actions = np.concatenate([abs_pos, abs_rpy, abs_grip], axis=-1)
        else:
            abs_actions = np.concatenate([abs_pos, abs_rpy], axis=-1)

        out = abs_actions[..., 0, :] if orig_ndim == state.ndim else abs_actions
        new_data = dict(data)
        new_data[ak] = out
        return new_data

@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaRPYBimanual(DataTransformFn):
    """
    双臂绝对 [L8 + R8] -> 双臂相对 [L7 + R7] = 14D

    每臂 7 维为：
    - gripper_as_delta=True : [dxyz, drpy, dg]
    - gripper_as_delta=False: [dxyz, drpy, g]
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    left_action_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_action_slice: slice = field(default_factory=lambda: slice(8, 16))

    action_key: str | None = None
    degrees: bool = True
    gripper_as_delta: bool = True
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 14, name="ChunkRelDeltaRPYBimanual")

        state16 = np.asarray(data["state"])
        act_full = np.asarray(data[ak])
        orig_ndim = act_full.ndim

        has_time = (act_full.ndim == state16.ndim + 1)
        act_t = act_full if has_time else act_full[..., None, :]

        sL = state16[..., self.left_state_slice]
        sR = state16[..., self.right_state_slice]
        aL = act_t[..., :, self.left_action_slice]
        aR = act_t[..., :, self.right_action_slice]

        def _one_arm(base_state8: np.ndarray, arm_action8: np.ndarray) -> np.ndarray:
            base_pos = base_state8[..., 0:3]
            base_quat = base_state8[..., 3:7]
            base_grip = base_state8[..., 7:8]

            act_pos = arm_action8[..., 0:3]
            act_quat = arm_action8[..., 3:7]
            next_grip = arm_action8[..., 7:8]

            base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
            base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
            base_R_T = np.swapaxes(base_R, -1, -2)

            delta_world = act_pos - base_pos[..., None, :]
            dpos = np.einsum("...ij,...tj->...ti", base_R_T, delta_world)

            R_abs = R.from_quat(act_quat.reshape(-1, 4)).as_matrix()
            R_abs = R_abs.reshape(act_quat.shape[:-1] + (3, 3))
            R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)

            drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=self.degrees)
            drot = drot.reshape(R_rel.shape[:-2] + (3,))

            if self.gripper_as_delta:
                grip_part = next_grip - base_grip[..., None, :]   # dg
            else:
                grip_part = next_grip                              # g

            return np.concatenate([dpos, drot, grip_part], axis=-1)  # (...,T,7)

        relL = _one_arm(sL, aL)
        relR = _one_arm(sR, aR)
        rel14 = np.concatenate([relL, relR], axis=-1)

        out = rel14[..., 0, :] if orig_ndim == state16.ndim else rel14
        new_data = dict(data)
        new_data[ak] = out
        return new_data

@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaRPYInverseBimanual(DataTransformFn):
    """
    双臂相对 14D -> 双臂绝对 14D（xyz+rpy+g）

    输入每臂 7 维：
    - gripper_as_delta=True : [dx,dy,dz,droll,dpitch,dyaw,dg]
    - gripper_as_delta=False: [dx,dy,dz,droll,dpitch,dyaw,g]

    输出每臂 7 维：
    [x,y,z,roll,pitch,yaw,g]
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))

    action_key: str | None = None
    degrees: bool = True
    gripper_as_delta: bool = True
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = _resolve_action_key(data, self.action_key)
        if ak is None:
            return data

        _validate_all_true_mask(self.mask, 14, name="ChunkRelDeltaRPYInverseBimanual")

        state16 = np.asarray(data["state"])
        rel_full = np.asarray(data[ak])
        orig_ndim = rel_full.ndim

        rel_full = rel_full[..., :14]
        has_time = (rel_full.ndim == state16.ndim + 1)
        rel_t = rel_full if has_time else rel_full[..., None, :]  # (...,T,14)

        sL = state16[..., self.left_state_slice]
        sR = state16[..., self.right_state_slice]

        relL = rel_t[..., :, 0:7]
        relR = rel_t[..., :, 7:14]

        def _one_arm(base_state8: np.ndarray, rel7: np.ndarray) -> np.ndarray:
            base_pos = base_state8[..., 0:3]
            base_quat = base_state8[..., 3:7]
            base_grip = base_state8[..., 7:8]

            dpos = rel7[..., 0:3]
            drot = rel7[..., 3:6]
            grip_part = rel7[..., 6:7]

            base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
            base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

            abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, dpos)

            R_rel = R.from_euler(self.euler_order, drot.reshape(-1, 3), degrees=self.degrees).as_matrix()
            R_rel = R_rel.reshape(drot.shape[:-1] + (3, 3))
            R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

            abs_rpy = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_euler(self.euler_order, degrees=self.degrees)
            abs_rpy = abs_rpy.reshape(R_abs.shape[:-2] + (3,))

            if self.gripper_as_delta:
                abs_grip = base_grip[..., None, :] + grip_part
            else:
                abs_grip = grip_part

            return np.concatenate([abs_pos, abs_rpy, abs_grip], axis=-1)  # (...,T,7)

        outL = _one_arm(sL, relL)
        outR = _one_arm(sR, relR)

        out14 = np.concatenate([outL, outR], axis=-1)

        out = out14[..., 0, :] if orig_ndim == state16.ndim else out14
        new_data = dict(data)
        new_data[ak] = out
        return new_data
