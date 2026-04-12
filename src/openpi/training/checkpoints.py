from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from collections.abc import Mapping
from typing import Any
from typing import Protocol

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
from flax import traverse_util

import openpi.models.vlm_backbone as _vlm_backbone
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: Any,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        try:
            restored = _restore_train_state_and_params(
                checkpoint_manager,
                train_state=train_state,
                params=params,
                step=step,
                legacy_root=False,
            )
        except ValueError as exc:
            if _looks_like_opt_state_structure_mismatch(exc):
                logging.warning(
                    "Checkpoint optimizer state structure is incompatible with current code; "
                    "restoring params only and reinitializing optimizer state."
                )
                return _restore_params_only_state(
                    checkpoint_manager,
                    state,
                    state_sharding,
                    params=params,
                    step=step,
                    legacy_root=False,
                )
            if not _looks_like_legacy_vlm_root_mismatch(exc, params):
                raise
            logging.warning(
                "Checkpoint uses legacy VLM root layout; restoring params only and reinitializing optimizer state."
            )
            return _restore_params_only_state(
                checkpoint_manager,
                state,
                state_sharding,
                params=params,
                step=step,
                legacy_root=True,
            )
        restored = {
            "train_state": _device_put_like_tree(restored["train_state"], state_sharding),
            "params": _device_put_like_tree(restored["params"], {"params": state_sharding.params}),
        }
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        # 新版 Orbax 有 CommitFutureAwaitingContractedSignals，就沿用原设计
        if hasattr(future, "CommitFutureAwaitingContractedSignals"):
            return [
                future.CommitFutureAwaitingContractedSignals(
                    asyncio.to_thread(self.save, directory, args),
                    name="callback_save",
                )
            ]
        # 旧版 Orbax：没有 CommitFutureAwaitingContractedSignals，就同步等待保存完成
        await asyncio.to_thread(self.save, directory, args)
        return []

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def _looks_like_legacy_vlm_root_mismatch(exc: ValueError, reference_params: Mapping[str, Any]) -> bool:
    msg = str(exc)
    if "tree structures do not match" not in msg:
        return False
    if _vlm_backbone.RUNTIME_VLM_ROOT not in reference_params:
        return False
    return _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT in msg


def _looks_like_opt_state_structure_mismatch(exc: ValueError) -> bool:
    msg = str(exc)
    return "tree structures do not match" in msg and "opt_state" in msg


def _remap_restored_params_item(
    restored_params_item: Mapping[str, Any], _reference_params: Mapping[str, Any]
) -> dict[str, Any]:
    if "params" not in restored_params_item:
        return dict(restored_params_item)

    remapped = dict(restored_params_item)
    remapped["params"] = _swap_vlm_root_in_tree(
        remapped["params"],
        source_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
        target_root=_vlm_backbone.RUNTIME_VLM_ROOT,
    )
    return remapped


def _swap_vlm_root_in_tree(tree: Any, *, source_root: str, target_root: str) -> Any:
    if isinstance(tree, training_utils.TrainState):
        return dataclasses.replace(
            tree,
            params=_swap_vlm_root_in_tree(tree.params, source_root=source_root, target_root=target_root),
            opt_state=_swap_vlm_root_in_tree(tree.opt_state, source_root=source_root, target_root=target_root),
            ema_params=_swap_vlm_root_in_tree(tree.ema_params, source_root=source_root, target_root=target_root),
        )
    if hasattr(tree, "to_pure_dict") and hasattr(tree, "replace_by_pure_dict"):
        remapped_pure = _swap_vlm_root_in_tree(tree.to_pure_dict(), source_root=source_root, target_root=target_root)
        remapped_tree = jax.tree_util.tree_map(lambda x: x, tree)
        remapped_tree.replace_by_pure_dict(remapped_pure)
        return remapped_tree
    if isinstance(tree, Mapping):
        remapped: dict[Any, Any] = {}
        for key, value in tree.items():
            remapped_key = target_root if key == source_root else key
            remapped[remapped_key] = _swap_vlm_root_in_tree(value, source_root=source_root, target_root=target_root)
        return remapped
    if isinstance(tree, list):
        return [_swap_vlm_root_in_tree(value, source_root=source_root, target_root=target_root) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_swap_vlm_root_in_tree(value, source_root=source_root, target_root=target_root) for value in tree)
    return tree


def _make_host_restore_args_tree(item: Any) -> Any:
    item_leaves, item_treedef = jax.tree_util.tree_flatten(item)
    restore_args_leaves = []
    for value in item_leaves:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            restore_args_leaves.append(ocp.RestoreArgs(restore_type=np.ndarray))
        else:
            restore_args_leaves.append(ocp.RestoreArgs())

    return jax.tree_util.tree_unflatten(item_treedef, restore_args_leaves)


def _device_put_like_tree(item: Any, sharding_tree: Any) -> Any:
    item_leaves, item_treedef = jax.tree_util.tree_flatten(item)
    try:
        sharding_leaves = item_treedef.flatten_up_to(sharding_tree)
    except ValueError as exc:
        raise ValueError("Sharding tree is not structurally compatible with restored item.") from exc

    placed_leaves = []
    for value, shard in zip(item_leaves, sharding_leaves, strict=True):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            placed_leaves.append(jax.device_put(value, shard))
        else:
            placed_leaves.append(value)

    return jax.tree_util.tree_unflatten(item_treedef, placed_leaves)


def _restore_train_state_and_params(
    checkpoint_manager: ocp.CheckpointManager,
    *,
    train_state: training_utils.TrainState,
    params: at.Params,
    step: int | None,
    legacy_root: bool,
) -> dict[str, Any]:
    if legacy_root:
        train_state = _swap_vlm_root_in_tree(
            train_state,
            source_root=_vlm_backbone.RUNTIME_VLM_ROOT,
            target_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
        )
        params = _swap_vlm_root_in_tree(
            params,
            source_root=_vlm_backbone.RUNTIME_VLM_ROOT,
            target_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
        )

    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeRestore(
                item=train_state,
                restore_args=_make_host_restore_args_tree(train_state),
            ),
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                restore_args={"params": _make_host_restore_args_tree(params)},
            ),
        ),
    )
    if legacy_root:
        restored = {
            "train_state": _swap_vlm_root_in_tree(
                restored["train_state"],
                source_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
                target_root=_vlm_backbone.RUNTIME_VLM_ROOT,
            ),
            "params": _remap_restored_params_item(restored["params"], params),
        }
    return restored


def _restore_params_only_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: Any,
    *,
    params: at.Params,
    step: int | None,
    legacy_root: bool,
) -> training_utils.TrainState:
    restore_step = _resolve_restore_step(checkpoint_manager, step)
    restored_params_pure = _restore_params_pure_dict(
        checkpoint_manager.directory / str(restore_step) / "params",
        legacy_root=legacy_root,
    )
    params_sharding = state_sharding.params.to_pure_dict() if hasattr(state_sharding.params, "to_pure_dict") else state_sharding.params
    restored_params_pure = _device_put_like_tree(restored_params_pure, params_sharding)
    restored_params_state = _rehydrate_params_like_reference(params, restored_params_pure)
    restored_state = _merge_params(state, {"params": restored_params_state})
    return dataclasses.replace(restored_state, step=np.asarray(restore_step, dtype=np.int32))


def _resolve_restore_step(checkpoint_manager: ocp.CheckpointManager, step: int | None) -> int:
    if step is not None:
        return int(step)
    steps = tuple(checkpoint_manager.all_steps())
    if not steps:
        return 0
    return int(max(steps))


def _restore_params_pure_dict(params_path: epath.Path | str, *, legacy_root: bool) -> Any:
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item),
            ),
        )["params"]

    flat_params = traverse_util.flatten_dict(restored)
    if flat_params and all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
        restored = traverse_util.unflatten_dict(flat_params)

    if legacy_root:
        restored = _swap_vlm_root_in_tree(
            restored,
            source_root=_vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT,
            target_root=_vlm_backbone.RUNTIME_VLM_ROOT,
        )
    return restored


def _rehydrate_params_like_reference(reference_params: Any, restored_params_pure: Any) -> Any:
    if hasattr(reference_params, "replace_by_pure_dict"):
        restored_params_state = jax.tree_util.tree_map(lambda x: x, reference_params)
        restored_params_state.replace_by_pure_dict(restored_params_pure)
        return restored_params_state
    return restored_params_pure
