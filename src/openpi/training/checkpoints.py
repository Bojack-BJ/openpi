from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Any
from collections.abc import Mapping
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

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
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        try:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
        except ValueError as exc:
            if not _looks_like_legacy_vlm_root_mismatch(exc, params):
                raise
            logging.info(
                "Retrying checkpoint restore with legacy VLM root remap compatibility for train_state and params."
            )
            restored = checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    train_state=ocp.args.PyTreeRestore(),
                    params=ocp.args.PyTreeRestore(),
                ),
            )
            restored = {
                "train_state": _restore_legacy_train_state_item(restored["train_state"], train_state),
                "params": _remap_restored_params_item(restored["params"], params),
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


def _remap_restored_params_item(
    restored_params_item: Mapping[str, Any], _reference_params: Mapping[str, Any]
) -> dict[str, Any]:
    if "params" not in restored_params_item:
        return dict(restored_params_item)

    remapped = dict(restored_params_item)
    remapped["params"] = _remap_legacy_vlm_roots_in_tree(remapped["params"])
    return remapped


def _restore_legacy_train_state_item(
    restored_train_state_item: Any, reference_train_state: training_utils.TrainState
) -> training_utils.TrainState:
    remapped = _remap_legacy_vlm_roots_in_tree(restored_train_state_item)
    return dataclasses.replace(
        reference_train_state,
        step=_get_tree_attr(remapped, "step", reference_train_state.step),
        params=_get_tree_attr(remapped, "params", reference_train_state.params),
        model_def=_get_tree_attr(remapped, "model_def", reference_train_state.model_def),
        opt_state=_get_tree_attr(remapped, "opt_state", reference_train_state.opt_state),
        ema_params=_get_tree_attr(remapped, "ema_params", reference_train_state.ema_params),
    )


def _get_tree_attr(tree: Any, key: str, default: Any) -> Any:
    if isinstance(tree, Mapping):
        return tree.get(key, default)
    return getattr(tree, key, default)


def _remap_legacy_vlm_roots_in_tree(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        remapped: dict[Any, Any] = {}
        for key, value in tree.items():
            remapped_key = _vlm_backbone.RUNTIME_VLM_ROOT if key == _vlm_backbone.LEGACY_VLM_CHECKPOINT_ROOT else key
            remapped[remapped_key] = _remap_legacy_vlm_roots_in_tree(value)
        return remapped
    if isinstance(tree, list):
        return [_remap_legacy_vlm_roots_in_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_remap_legacy_vlm_roots_in_tree(value) for value in tree)
    return tree
