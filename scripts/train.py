import dataclasses
import functools
import logging
import multiprocessing
import os
import platform
import sys
import time
from collections import Counter
from typing import Any

# Reduce TensorFlow/XLA startup log noise (e.g., repeated cuDNN/cuBLAS registration warnings).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

# Avoid probing unsupported backends (e.g., rocm/tpu) and reduce startup noise.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    run_id_path = ckpt_dir / "wandb_id.txt"
    if resuming and run_id_path.exists():
        run_id = run_id_path.read_text().strip()
        if run_id:
            logging.info("Resuming wandb run %s with resume='allow'", run_id)
            wandb.init(id=run_id, resume="allow", project=config.project_name)
        else:
            logging.warning("Found empty wandb run id in %s. Starting a new wandb run.", run_id_path)
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            run_id_path.write_text(wandb.run.id)
    else:
        if resuming:
            logging.warning("No wandb run id found in %s. Starting a new wandb run.", run_id_path)
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        run_id_path.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates overlapping weights. Returns a subset to merge into initialized params."""
    if isinstance(loader, _weight_loaders.NoOpWeightLoader):
        return {}

    logging.info("Loading initialization weights with loader: %s", loader)
    loaded_params = loader.load(params_shape)

    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    matched = {}
    skipped = 0
    for key_path, value in flat_loaded.items():
        if key_path not in flat_expected:
            skipped += 1
            continue

        expected = flat_expected[key_path]
        if isinstance(expected, jax.ShapeDtypeStruct):
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError(
                    f"Checkpoint shape mismatch at {'/'.join(key_path)}: "
                    f"expected {expected.shape}, got {value.shape}."
                )
            if np.dtype(value.dtype) != np.dtype(expected.dtype):
                raise ValueError(
                    f"Checkpoint dtype mismatch at {'/'.join(key_path)}: "
                    f"expected {expected.dtype}, got {value.dtype}."
                )

        # Skip ShapeDtypeStruct leaves; we only merge real loaded arrays.
        if not isinstance(value, jax.ShapeDtypeStruct):
            matched[key_path] = value

    if not matched:
        raise ValueError(
            "Loaded checkpoint has no parameters that match the current model. "
            "Check that `config.model` and `weight_loader` point to compatible architectures."
        )

    if skipped > 0:
        logging.warning(
            "Skipping %d checkpoint params that are not present in the current model (partial load).",
            skipped,
        )

    return traverse_util.unflatten_dict(matched)


def _select_matching_tree(reference_tree: at.PyTree, subset_tree: at.PyTree) -> at.PyTree:
    """Selects the subtree from `reference_tree` with the same leaves as `subset_tree`."""
    flat_reference = traverse_util.flatten_dict(reference_tree)
    flat_subset = traverse_util.flatten_dict(subset_tree)
    return traverse_util.unflatten_dict({key: flat_reference[key] for key in flat_subset})


def _to_wandb_image_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and np.nanmin(image) >= 0.0 and np.nanmax(image) <= 1.0:
            image = image * 255.0

    return np.clip(image, 0, 255).astype(np.uint8)


def _summarize_param_tree(params: at.PyTree) -> str:
    tree = params.to_pure_dict() if hasattr(params, "to_pure_dict") else params
    flat = traverse_util.flatten_dict(tree)

    tensor_count = 0
    total_elements = 0
    dtype_counts = Counter()
    for value in flat.values():
        if not hasattr(value, "shape"):
            continue
        tensor_count += 1
        total_elements += int(np.prod(value.shape))
        dtype_counts[str(value.dtype)] += 1

    dtype_summary = ", ".join(f"{dtype}:{count}" for dtype, count in sorted(dtype_counts.items()))
    return f"tensor_count={tensor_count}, total_elements={total_elements:,}, dtypes=[{dtype_summary}]"


def _elapsed_logger_process(label: str, start_time: float, interval_sec: float, stop_event: multiprocessing.synchronize.Event):
    while not stop_event.wait(interval_sec):
        elapsed = time.perf_counter() - start_time
        print(f"{time.strftime('%H:%M:%S')} [I] {label} still running after {elapsed:.1f} seconds.", flush=True)
        sys.stdout.flush()


def _start_elapsed_logger(
    label: str, *, interval_sec: float = 60.0
) -> tuple[multiprocessing.synchronize.Event, multiprocessing.Process]:
    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    start_time = time.perf_counter()
    process = ctx.Process(
        target=_elapsed_logger_process,
        args=(label, start_time, interval_sec, stop_event),
        name=f"{label.replace(' ', '_')}_timer",
        daemon=True,
    )
    process.start()
    return stop_event, process


def _configure_jax_compilation_cache(config: _config.TrainConfig) -> None:
    if config.jax_compilation_cache_dir is None:
        logging.info("JAX persistent compilation cache disabled.")
        return

    cache_dir = epath.Path(config.jax_compilation_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    logging.info("JAX persistent compilation cache enabled at %s", cache_dir)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=config.log_sharding)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    param_sharding = state_sharding.params.to_pure_dict()
    partial_param_sharding = _select_matching_tree(param_sharding, partial_params)
    partial_params = jax.device_put(partial_params, partial_param_sharding)

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=(replicated_sharding, partial_param_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    _configure_jax_compilation_cache(config)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(_to_wandb_image_uint8(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1)))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if config.log_train_state_details:
        logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    else:
        logging.info("Initialized train state summary: %s", _summarize_param_tree(train_state.params))

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        is_first_step = step == start_step
        if is_first_step:
            logging.info(
                "First train step started: compiling train_step JIT for current model/sharding (this may take minutes)."
            )
            compile_t0 = time.perf_counter()
            compile_log_stop, compile_log_thread = _start_elapsed_logger(
                "First train step compile+execute",
                interval_sec=60.0,
            )
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if is_first_step:
            # Ensure compile + first execution time is fully materialized for accurate logging.
            train_state = jax.block_until_ready(train_state)
            info = jax.tree.map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                info,
            )
            compile_log_stop.set()
            compile_log_thread.join(timeout=1.0)
            logging.info(
                "First train step compile+execute finished in %.1f seconds.",
                time.perf_counter() - compile_t0,
            )
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
