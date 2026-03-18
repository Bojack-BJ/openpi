from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
import torch
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R

DataDict: TypeAlias = at.PyTree
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

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
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
    norm_stats: at.PyTree[NormStats] | None
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
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

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
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


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


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
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
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
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


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
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
                return np.asarray(jax.device_get(x))
            return x

        return jax.tree.map(_convert, data)


# ---------------------------
# realtive_6d
# ---------------------------


def _rotmat_to_6d(Rm: np.ndarray) -> np.ndarray:
    """把旋转矩阵 (..., 3, 3) 转成 6D 表示 (..., 6)，取前两列。"""
    c0 = Rm[..., :, 0]
    c1 = Rm[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPose(DataTransformFn):
    """
    把 xyz + quat(xyzw) + gripper 的绝对姿态，转换为
    以 chunk 第 0 帧为参考的 [Δxyz(在首帧坐标系下), 6D_rot, g] 表示。
    """
    pos_slice: slice = dataclasses.field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = dataclasses.field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = dataclasses.field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    # ✅ 新增：和 DeltaActions 一样的 mask（长度可小于真实 action_dim）
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d or self.grip_slice.stop > d):
            raise ValueError(f"ChunkRel6DPose: state dim={d}, slices={self.pos_slice, self.quat_slice, self.grip_slice}")

        # ---------- 1) 基准位姿 ----------
        base_pos = state[..., self.pos_slice]       # (...,3)
        base_quat = state[..., self.quat_slice]     # (...,4) xyzw

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
        base_R_T = np.swapaxes(base_R, -1, -2)

        # ---------- 2) 确保 actions 有时间维 ----------
        actions_t = actions_full
        if actions_t.ndim == state.ndim:  # (...,D) -> (...,1,D)
            actions_t = actions_t[..., None, :]

        act_pos_abs  = actions_t[..., self.pos_slice]    # (...,T,3)
        act_quat_abs = actions_t[..., self.quat_slice]   # (...,T,4)
        grip         = actions_t[..., self.grip_slice]   # (...,T,1)

        # ---------- 3) 平移 ----------
        delta_pos = act_pos_abs - base_pos[..., None, :]
        rel_pos = np.einsum("...ij,...tj->...ti", base_R_T, delta_pos)

        # ---------- 4) 旋转 ----------
        R_abs = R.from_quat(act_quat_abs.reshape(-1, 4)).as_matrix()
        R_abs = R_abs.reshape(act_quat_abs.shape[:-1] + (3, 3))
        R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)
        rel_rot_6d = _rotmat_to_6d(R_rel)

        rel10 = np.concatenate([rel_pos, rel_rot_6d, grip], axis=-1)  # (...,T,10)

        # ✅ mask=None：保持旧行为（直接变成 10D）
        if self.mask is None:
            out = rel10
            if orig_ndim == state.ndim:  # 原本无 time 维
                out = out[..., 0, :]     # (...,10)
            data[ak] = out
            return data

        # ✅ mask!=None：同 DeltaActions 语义：只改 actions[..., :dims]，不改总维度
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if dims != rel10.shape[-1]:
            raise ValueError(f"ChunkRel6DPose(mask): mask len={dims} must equal rel dim={rel10.shape[-1]}")

        if actions_t.shape[-1] < dims:
            raise ValueError(f"ChunkRel6DPose(mask): actions dim={actions_t.shape[-1]} < mask len={dims}")

        # 只写回前 dims
        actions_t[..., :dims] = np.where(mask, rel10, actions_t[..., :dims])

        # 还原回原本维度
        if orig_ndim == state.ndim:
            data[ak] = actions_t[..., 0, :]
        else:
            data[ak] = actions_t
        return data


def _rot6d_to_rotmat(rot_6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    6D → 3x3 旋转矩阵。

    rot_6d: (..., 6) = [a1(3), a2(3)]
    返回:  (..., 3, 3) 列向量为正交基 b1,b2,b3
    """
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    # b1 = normalize(a1)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)

    # 从 a2 中减去在 b1 方向上的分量，然后归一化得到 b2
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + eps)

    # b3 = b1 × b2  保证右手系
    b3 = np.cross(b1, b2, axis=-1)

    # 以列为基向量拼成矩阵
    Rm = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    return Rm

@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverse(DataTransformFn):
    """
    把相对 6D 动作 [Δxyz(局部), rot6d, g] 还原成绝对 xyz+quat+g。

    假设:
      state:  (..., D)  绝对 xyz+quat+g，作为基准位姿
      actions: (..., T, 10) 或 (..., 10)
               [Δx, Δy, Δz, rot6d(6), g]
    """

    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        # 决定 ak
        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state = np.asarray(data["state"])
        actions_orig = np.asarray(data[ak])
        orig_ndim = actions_orig.ndim

        # 如果没有 time 维，就补一个 T=1
        actions = actions_orig
        if actions.ndim == state.ndim:  # (..., D) → (..., 1, D)
            actions = actions[..., None, :]

        d = state.shape[-1]
        if (self.pos_slice.stop > d or
            self.quat_slice.stop > d or
            self.grip_slice.stop > d):
            raise ValueError(
                f"ChunkRel6DPoseInverse: state dim={d}, "
                f"slices={self.pos_slice, self.quat_slice, self.grip_slice}"
            )

        # ---------- 1. 基准位姿 ----------
        base_pos  = state[..., self.pos_slice]   # (..., 3)
        base_quat = state[..., self.quat_slice]  # (..., 4)

        base_quat_flat = base_quat.reshape(-1, 4)
        base_R_flat = R.from_quat(base_quat_flat).as_matrix()          # (N,3,3)
        base_R = base_R_flat.reshape(base_quat.shape[:-1] + (3, 3))    # (...,3,3)

        # ---------- 2. 拆相对动作 ----------
        rel_pos   = actions[..., 0:3]    # (..., T, 3)  局部坐标下 Δp
        rel_rot6d = actions[..., 3:9]    # (..., T, 6)
        grip      = actions[..., 9:10]   # (..., T, 1)

        # ---------- 3. 平移：p_abs = p0 + R0 * Δp_local ----------
        delta_world  = np.einsum("...ij,...tj->...ti", base_R, rel_pos)  # ✅ base_R 直接广播到 time 维
        base_pos_exp = base_pos[..., None, :]
        abs_pos      = base_pos_exp + delta_world

        # ---------- 4. 旋转：R_abs = R0 * R_rel ----------
        R_rel = _rot6d_to_rotmat(rel_rot6d)                 # (...,T,3,3)
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)  # (...,T,3,3)

        # 转回 quat (x,y,z,w)
        R_abs_flat   = R_abs.reshape(-1, 3, 3)
        abs_quat_flat = R.from_matrix(R_abs_flat).as_quat()           # (N*T,4)
        abs_quat      = abs_quat_flat.reshape(R_abs.shape[:-2] + (4,))# (...,T,4)

        # ---------- 5. 拼回绝对动作: [xyz, quat, g] → 8 维 ----------
        abs_actions = np.concatenate([abs_pos, abs_quat, grip], axis=-1)  # (...,T,8)

        # 如果原来没有 time 维，就再 squeeze 回去
        if orig_ndim == state.ndim:
            abs_actions = abs_actions[..., 0, :]   # (..., 8)

        data[ak] = abs_actions
        return data

@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverseRPY(DataTransformFn):
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None
    euler_order: str = "xyz"

    # ✅ 新增：mask（同 DeltaActions 语义：取 actions[..., :dims]）
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        # ✅ 如果给了 mask：只取前 dims（并把 mask=False 的维度置 0）
        if self.mask is not None:
            mask = np.asarray(self.mask)
            dims = mask.shape[-1]
            actions_full = actions_full[..., :dims]
            actions_full = np.where(mask, actions_full, 0.0)

        # 确保 time 维
        actions = actions_full
        if actions.ndim == state.ndim:
            actions = actions[..., None, :]

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d or self.grip_slice.stop > d):
            raise ValueError(f"ChunkRel6DPoseInverseRPY: state dim={d}, slices={self.pos_slice, self.quat_slice, self.grip_slice}")

        base_pos  = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]
        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

        rel_pos   = actions[..., 0:3]
        rel_rot6d = actions[..., 3:9]
        grip      = actions[..., 9:10]

        abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, rel_pos)

        R_rel = _rot6d_to_rotmat(rel_rot6d)
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

        abs_rpy = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_euler(self.euler_order, degrees=False)
        abs_rpy = abs_rpy.reshape(R_abs.shape[:-2] + (3,))

        abs_actions = np.concatenate([abs_pos, abs_rpy, grip], axis=-1)  # (...,T,7)

        if orig_ndim == state.ndim:
            abs_actions = abs_actions[..., 0, :]
        data[ak] = abs_actions
        return data

@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseBimanual(DataTransformFn):
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    left_action_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_action_slice: slice = field(default_factory=lambda: slice(8, 16))

    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))

    action_key: str = "action"
    mask: Sequence[bool] | None = None   # ✅ 新增

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data or self.action_key not in data:
            return data

        state16 = np.asarray(data["state"])
        act_full = np.asarray(data[self.action_key])
        orig_ndim = act_full.ndim
        has_time = (act_full.ndim == state16.ndim + 1)

        act_t = act_full if has_time else act_full[..., None, :]

        sL = state16[..., self.left_state_slice]
        sR = state16[..., self.right_state_slice]

        if has_time:
            aL = act_t[..., :, self.left_action_slice]
            aR = act_t[..., :, self.right_action_slice]
        else:
            aL = act_t[..., 0, self.left_action_slice]
            aR = act_t[..., 0, self.right_action_slice]

        single = ChunkRel6DPose(
            pos_slice=self.pos_slice,
            quat_slice=self.quat_slice,
            grip_slice=self.grip_slice,
            action_key="action",
            mask=None,  # 这里先产出纯 10D
        )

        dL = single({"state": sL, "action": aL})
        dR = single({"state": sR, "action": aR})
        relL = np.asarray(dL["action"])
        relR = np.asarray(dR["action"])

        rel20 = np.concatenate([relL, relR], axis=-1)  # (...,T,20) or (...,20)

        if self.mask is None:
            data[self.action_key] = rel20
            return data

        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if dims != rel20.shape[-1]:
            raise ValueError(f"ChunkRel6DPoseBimanual(mask): mask len={dims} != rel dim={rel20.shape[-1]}")
        if act_t.shape[-1] < dims:
            raise ValueError(f"ChunkRel6DPoseBimanual(mask): action dim={act_t.shape[-1]} < mask len={dims}")

        rel20_t = rel20 if has_time else rel20[..., None, :]
        act_t[..., :dims] = np.where(mask, rel20_t, act_t[..., :dims])

        data[self.action_key] = act_t if has_time else act_t[..., 0, :]
        return data


@dataclasses.dataclass(frozen=True)
class ChunkRel6DPoseInverseRPYBimanual(DataTransformFn):
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    action_key: str | None = None
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None   # ✅ 新增

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data or self.action_key not in data:
            return data

        state16 = np.asarray(data["state"])
        act_full = np.asarray(data[self.action_key])
        has_time = (act_full.ndim == state16.ndim + 1)

        if self.mask is not None:
            mask = np.asarray(self.mask)
            dims = mask.shape[-1]
            act_full = act_full[..., :dims]
            act_full = np.where(mask, act_full, 0.0)

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
            mask=None,  # 这里输入已经是 10D
        )

        dL = inv_single({"state": sL, "actions": aL})
        dR = inv_single({"state": sR, "actions": aR})
        outL = np.asarray(dL["actions"])
        outR = np.asarray(dR["actions"])

        out14 = np.concatenate([outL, outR], axis=-1)  # (...,T,14)

        if not has_time:
            out14 = out14[..., 0, :]
        data[self.action_key] = out14
        return data


# ---------------------------
# delta_rpy
# ---------------------------


import dataclasses
from dataclasses import field
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R

def absolute_to_delta_pose_rpy_chunk_base(
    base_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    degrees: bool = True,
    pos_slice: slice = slice(0, 3),
    quat_slice: slice = slice(3, 7),
) -> np.ndarray:
    """
    base_qpos:  (..., D)  含 xyz + quat(xyzw)
    target_qpos:(..., D)  含 xyz + quat(xyzw)
    return: (..., 6) [dx, dy, dz, droll, dpitch, dyaw] 其中 d* 为 xyz 欧拉角 (默认 degrees)
    语义等价你写的:
        T_rel = inv(T0) @ T1
        dpos = T_rel[:3, 3]
        drot = euler(T_rel[:3,:3])
    """
    base_qpos = np.asarray(base_qpos)
    target_qpos = np.asarray(target_qpos)

    pos0 = base_qpos[..., pos_slice]
    quat0 = base_qpos[..., quat_slice]
    pos1 = target_qpos[..., pos_slice]
    quat1 = target_qpos[..., quat_slice]

    R0 = R.from_quat(quat0.reshape(-1, 4)).as_matrix().reshape(quat0.shape[:-1] + (3, 3))
    R1 = R.from_quat(quat1.reshape(-1, 4)).as_matrix().reshape(quat1.shape[:-1] + (3, 3))

    R0_T = np.swapaxes(R0, -1, -2)

    # dpos in base frame
    dpos = np.einsum("...ij,...j->...i", R0_T, (pos1 - pos0))
    # R_rel = R0^T * R1
    R_rel = np.einsum("...ij,...jk->...ik", R0_T, R1)

    drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=degrees)
    drot = drot.reshape(R_rel.shape[:-2] + (3,))

    return np.concatenate([dpos, drot], axis=-1)


# ---------------------------
# DataTransform: chunk 内第一帧为基准的 Δpose(rpy)
# ---------------------------

@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaPoseRPY(DataTransformFn):
    """
    把绝对 actions 的 xyz+quat (可选带 grip) 转成
    chunk 基准（state）下的 [dx,dy,dz,droll,dpitch,dyaw]（默认 degrees=True）。
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    degrees: bool = True
    include_gripper: bool = False  # ✅ 默认严格按你说的 6 维

    # 可选：同 DeltaActions 语义，只替换 actions[..., :dims]
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state = np.asarray(data["state"])
        actions_full = np.asarray(data[ak])
        orig_ndim = actions_full.ndim

        d = state.shape[-1]
        if (self.pos_slice.stop > d or self.quat_slice.stop > d):
            raise ValueError(f"ChunkRelDeltaPoseRPY: state dim={d}, slices={self.pos_slice, self.quat_slice}")

        # base pose from state (chunk 第一帧)
        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        # ensure time dim
        actions_t = actions_full
        if actions_t.ndim == state.ndim:  # (...,D)->(...,1,D)
            actions_t = actions_t[..., None, :]

        act_pos_abs = actions_t[..., self.pos_slice]    # (...,T,3)
        act_quat_abs = actions_t[..., self.quat_slice]  # (...,T,4)

        # compute base_R^T
        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
        base_R_T = np.swapaxes(base_R, -1, -2)

        # dpos (base frame)
        delta_world = act_pos_abs - base_pos[..., None, :]
        dpos = np.einsum("...ij,...tj->...ti", base_R_T, delta_world)

        # R_rel = R0^T * R_abs
        R_abs = R.from_quat(act_quat_abs.reshape(-1, 4)).as_matrix()
        R_abs = R_abs.reshape(act_quat_abs.shape[:-1] + (3, 3))
        R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)

        drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=self.degrees)
        drot = drot.reshape(R_rel.shape[:-2] + (3,))

        rel6 = np.concatenate([dpos, drot], axis=-1)  # (...,T,6)

        if self.include_gripper:
            # grip 直接透传（不做 delta）
            if actions_t.shape[-1] >= self.grip_slice.stop:
                grip = actions_t[..., self.grip_slice]  # (...,T,1)
                rel = np.concatenate([rel6, grip], axis=-1)  # (...,T,7)
            else:
                rel = rel6
        else:
            rel = rel6

        # mask=None：直接把 actions 替换成 rel 表示（维度会变 6 或 7）
        if self.mask is None:
            out = rel
            if orig_ndim == state.ndim:  # 原本无 time 维
                out = out[..., 0, :]
            data[ak] = out
            return data

        # mask!=None：只改 actions[..., :dims]，不改总维度
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if dims != rel.shape[-1]:
            raise ValueError(f"ChunkRelDeltaPoseRPY(mask): mask len={dims} must equal rel dim={rel.shape[-1]}")
        if actions_t.shape[-1] < dims:
            raise ValueError(f"ChunkRelDeltaPoseRPY(mask): actions dim={actions_t.shape[-1]} < mask len={dims}")

        actions_t[..., :dims] = np.where(mask, rel, actions_t[..., :dims])

        if orig_ndim == state.ndim:
            data[ak] = actions_t[..., 0, :]
        else:
            data[ak] = actions_t
        return data


@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaPoseRPYInverse(DataTransformFn):
    """
    把 chunk 相对 [dx,dy,dz,droll,dpitch,dyaw] 还原成绝对 [x,y,z,qx,qy,qz,qw]（可选带 grip）。
    """
    pos_slice: slice = field(default_factory=lambda: slice(0, 3))
    quat_slice: slice = field(default_factory=lambda: slice(3, 7))
    grip_slice: slice = field(default_factory=lambda: slice(7, 8))
    action_key: str | None = None

    degrees: bool = True
    include_gripper: bool = False  # 和 forward 对齐：默认只处理 6 维

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state = np.asarray(data["state"])
        rel_full = np.asarray(data[ak])
        orig_ndim = rel_full.ndim

        # ensure time dim
        rel = rel_full
        if rel.ndim == state.ndim:
            rel = rel[..., None, :]  # (...,1,K)

        # split rel
        if rel.shape[-1] < 6:
            raise ValueError(f"ChunkRelDeltaPoseRPYInverse: rel dim must be >=6, got {rel.shape[-1]}")

        rel6 = rel[..., 0:6]  # (...,T,6)
        dpos = rel6[..., 0:3]
        drot = rel6[..., 3:6]

        # base pose
        base_pos = state[..., self.pos_slice]
        base_quat = state[..., self.quat_slice]

        base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
        base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

        # pos_abs = pos0 + R0 * dpos
        abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, dpos)

        # R_abs = R0 * R_rel
        R_rel = R.from_euler("xyz", drot.reshape(-1, 3), degrees=self.degrees).as_matrix()
        R_rel = R_rel.reshape(drot.shape[:-1] + (3, 3))
        R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

        abs_quat = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_quat().reshape(R_abs.shape[:-2] + (4,))

        if self.include_gripper and rel.shape[-1] >= 7:
            grip = rel[..., 6:7]
            abs_actions = np.concatenate([abs_pos, abs_quat, grip], axis=-1)  # (...,T,8)
        else:
            abs_actions = np.concatenate([abs_pos, abs_quat], axis=-1)  # (...,T,7)

        if orig_ndim == state.ndim:
            abs_actions = abs_actions[..., 0, :]
        data[ak] = abs_actions
        return data


@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaRPYBimanual(DataTransformFn):
    """
    双臂绝对动作 -> 以 chunk 第0帧 state 为基准的相对动作

    每臂:
        [x, y, z, qx, qy, qz, qw, g]
      ->[dx, dy, dz, droll, dpitch, dyaw, g]

    双臂:
        16维 -> 14维

    gripper 的行为与 ChunkRel6DPoseBimanual 一样：直接保留，不做 delta。
    mask 的语义也与 ChunkRel6DPoseBimanual 一样：
      - mask=None: 直接输出纯 14 维
      - mask!=None: 只覆盖 actions[..., :14] 中 mask=True 的部分，其余维度保持原值
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))
    left_action_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_action_slice: slice = field(default_factory=lambda: slice(8, 16))

    action_key: str | None = None
    degrees: bool = True
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state16 = np.asarray(data["state"])   # (...,16)
        act_full = np.asarray(data[ak])       # (...,16) or (...,T,16)
        orig_ndim = act_full.ndim

        has_time = (act_full.ndim == state16.ndim + 1)
        act_t = act_full if has_time else act_full[..., None, :]  # (...,T,16)

        sL = state16[..., self.left_state_slice]   # (...,8)
        sR = state16[..., self.right_state_slice]  # (...,8)

        aL = act_t[..., :, self.left_action_slice]   # (...,T,8)
        aR = act_t[..., :, self.right_action_slice]  # (...,T,8)

        def _one_arm(base_state8: np.ndarray, arm_action8: np.ndarray) -> np.ndarray:
            base_pos = base_state8[..., 0:3]      # (...,3)
            base_quat = base_state8[..., 3:7]     # (...,4)

            act_pos = arm_action8[..., 0:3]       # (...,T,3)
            act_quat = arm_action8[..., 3:7]      # (...,T,4)
            grip = arm_action8[..., 7:8]          # (...,T,1)

            base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
            base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))
            base_R_T = np.swapaxes(base_R, -1, -2)

            # dpos = R0^T (p1 - p0)
            delta_world = act_pos - base_pos[..., None, :]
            dpos = np.einsum("...ij,...tj->...ti", base_R_T, delta_world)

            # R_rel = R0^T R1
            R_abs = R.from_quat(act_quat.reshape(-1, 4)).as_matrix()
            R_abs = R_abs.reshape(act_quat.shape[:-1] + (3, 3))
            R_rel = np.einsum("...ij,...tjk->...tik", base_R_T, R_abs)

            drot = R.from_matrix(R_rel.reshape(-1, 3, 3)).as_euler("xyz", degrees=self.degrees)
            drot = drot.reshape(R_rel.shape[:-2] + (3,))

            # 每臂 7 维: dxyz + drpy + g
            return np.concatenate([dpos, drot, grip], axis=-1)  # (...,T,7)

        relL = _one_arm(sL, aL)  # (...,T,7)
        relR = _one_arm(sR, aR)  # (...,T,7)

        rel14 = np.concatenate([relL, relR], axis=-1)  # (...,T,14)

        if self.mask is None:
            data[ak] = rel14 if has_time else rel14[..., 0, :]
            return data

        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        if dims != rel14.shape[-1]:
            raise ValueError(f"ChunkRelDeltaRPYBimanual(mask): mask len={dims} != rel dim={rel14.shape[-1]}")
        if act_t.shape[-1] < dims:
            raise ValueError(f"ChunkRelDeltaRPYBimanual(mask): action dim={act_t.shape[-1]} < mask len={dims}")

        rel14_t = rel14 if has_time else rel14[..., None, :]
        act_t[..., :dims] = np.where(mask, rel14_t, act_t[..., :dims])

        data[ak] = act_t if has_time else act_t[..., 0, :]
        return data


@dataclasses.dataclass(frozen=True)
class ChunkRelDeltaRPYInverseBimanual(DataTransformFn):
    """
    双臂相对动作 -> 还原回绝对 xyz+quat+g

    每臂:
        [dx, dy, dz, droll, dpitch, dyaw, g]
      ->[x, y, z, qx, qy, qz, qw, g]

    双臂:
        14维 -> 16维

    说明:
    - gripper 直接保留回去，不做任何变换
    - mask 的用法沿用你单臂/双臂 inverse 的风格：
        如果给了 mask，则只读取前14维，并把 mask=False 的相对量置0后再做还原
    """
    left_state_slice: slice = field(default_factory=lambda: slice(0, 8))
    right_state_slice: slice = field(default_factory=lambda: slice(8, 16))

    action_key: str | None = None
    degrees: bool = True
    euler_order: str = "xyz"
    mask: Sequence[bool] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data

        ak = self.action_key
        if ak is None:
            if "actions" in data:
                ak = "actions"
            elif "action" in data:
                ak = "action"
            else:
                return data
        elif ak not in data:
            return data

        state16 = np.asarray(data["state"])   # (...,16)
        rel_full = np.asarray(data[ak])       # (...,14) or (...,T,14) or padded
        orig_ndim = rel_full.ndim

        if self.mask is not None:
            mask = np.asarray(self.mask)
            dims = mask.shape[-1]
            rel_full = rel_full[..., :dims]
            rel_full = np.where(mask, rel_full, 0.0)

        has_time = (rel_full.ndim == state16.ndim + 1)
        rel_t = rel_full if has_time else rel_full[..., None, :]  # (...,T,K)

        if rel_t.shape[-1] < 14:
            raise ValueError(f"ChunkRelDeltaRPYInverseBimanual: action dim={rel_t.shape[-1]} < 14")

        sL = state16[..., self.left_state_slice]   # (...,8)
        sR = state16[..., self.right_state_slice]  # (...,8)

        relL = rel_t[..., :, 0:7]     # (...,T,7)
        relR = rel_t[..., :, 7:14]    # (...,T,7)

        def _one_arm(base_state8: np.ndarray, rel7: np.ndarray) -> np.ndarray:
            base_pos = base_state8[..., 0:3]      # (...,3)
            base_quat = base_state8[..., 3:7]     # (...,4)

            dpos = rel7[..., 0:3]                 # (...,T,3)
            drot = rel7[..., 3:6]                 # (...,T,3)
            grip = rel7[..., 6:7]                 # (...,T,1)

            base_R = R.from_quat(base_quat.reshape(-1, 4)).as_matrix()
            base_R = base_R.reshape(base_quat.shape[:-1] + (3, 3))

            # p_abs = p0 + R0 * dpos
            abs_pos = base_pos[..., None, :] + np.einsum("...ij,...tj->...ti", base_R, dpos)

            # R_abs = R0 * R_rel
            R_rel = R.from_euler(self.euler_order, drot.reshape(-1, 3), degrees=self.degrees).as_matrix()
            R_rel = R_rel.reshape(drot.shape[:-1] + (3, 3))
            R_abs = np.einsum("...ij,...tjk->...tik", base_R, R_rel)

            abs_quat = R.from_matrix(R_abs.reshape(-1, 3, 3)).as_quat()
            abs_quat = abs_quat.reshape(R_abs.shape[:-2] + (4,))

            # 每臂 8 维: xyz + quat + g
            return np.concatenate([abs_pos, abs_quat, grip], axis=-1)  # (...,T,8)

        outL = _one_arm(sL, relL)  # (...,T,8)
        outR = _one_arm(sR, relR)  # (...,T,8)

        out16 = np.concatenate([outL, outR], axis=-1)  # (...,T,16)

        if orig_ndim == state16.ndim:
            out16 = out16[..., 0, :]

        data[ak] = out16
        return data