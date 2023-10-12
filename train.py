import itertools as it
from functools import partial
from glob import glob
from typing import NamedTuple

import fire
import flax.traverse_util as ftu
import jax
import jax.numpy as jnp
import jax.sharding as jsh
import jax.tree_util as jtu
import numpy as np
import optax
import orbax.checkpoint
import polars as pl
import rich.progress as rp
from flax.linen.dtypes import promote_dtype
from flax.training import orbax_utils, train_state

from istrm import Beatmap, Difficulty
from model import Mapper


def main(
    width=128,
    depth=6,
    model_dtype="bfloat16",
    meta_path="osu-dl/meta.jsonl",
    osz_paths="./osu-dl/beatmapsets/*.zip",
    peak_lr=1e-5,
    train_steps=1 << 16,
    warmup_steps=1024,
    ssm_factor=0.3,
    num_threads=32,
    seq_length=16384,
    batch_size=64,
    save_every=1024,
):
    import concurrent.futures as cft

    from istrm import Batch, batch_array_trees, bms_chunks, scan_maps

    model = Mapper(width, depth, dtype=getattr(jnp, model_dtype))

    def _batch(array):
        return array.reshape(batch_size, -1, *array.shape[1:])

    def _dp_map(array):
        return jax.sharding.PositionalSharding(jax.devices()).reshape(
            [-1] + [1] * (len(array.shape) - 1)
        )

    class TrainState(train_state.TrainState):
        loss: float

    def _optimizer():
        def _is_ssm(path, value):
            del value
            is_ssm_module = path[:-3:-1] == ("att", "filter_fn_0")
            is_ssm_param = path[-1] in ["B", "C", "log_step"] or "Lambda" in path[-1]
            return not (is_ssm_module and is_ssm_param)

        def _not_ssm(path, value):
            return not _is_ssm(path, value)

        is_ssm = partial(ftu.path_aware_map, _is_ssm)
        not_ssm = partial(ftu.path_aware_map, _not_ssm)
        lsched = optax.warmup_cosine_decay_schedule(
            0.0, peak_lr, warmup_steps, train_steps
        )
        lsched = optax.linear_onecycle_schedule(train_steps, peak_lr)
        msched = lambda step: 0.95 - 0.1 * (lsched(step) / peak_lr)
        optim = optax.chain(
            optax.multi_transform(
                {
                    "params": optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.inject_hyperparams(optax.scale_by_lion)(msched),
                        optax.additive_weight_decay(0.05, mask=not_ssm),
                        optax.masked(optax.scale(ssm_factor), is_ssm),
                        optax.inject_hyperparams(optax.scale)(lsched),
                        optax.scale(-1),
                    ),
                    "lmbdas": optax.chain(
                        optax.trace(0.9),
                        optax.add_decayed_weights(0.01),
                        optax.scale(0.01),
                    ),
                },
                {"lmbdas": "lmbdas", "params": "params"},
            ),
        )
        return optim

    @partial(jax.jit, out_shardings=jsh.PositionalSharding(jax.devices()).replicate())
    def _train_init(rng, inputs: Batch):
        params = model.init(rng, inputs.samples, inputs.seq_ids, *inputs.ratings)
        params["lmbdas"] = Beatmap(
            *it.repeat(0.0, len(Beatmap._fields) - 1),
            difficulties=Difficulty(*it.repeat(0.0, len(Difficulty._fields))),
        )
        tstate = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=_optimizer(),
            loss=0.0,
        )
        return tstate

    @jax.checkpoint
    def _objective(params, inputs: Batch):
        lmbdas = params.pop("lmbdas")
        output: Beatmap = model.apply(
            params, inputs.samples, inputs.seq_ids, *inputs.ratings
        )
        cce = optax.softmax_cross_entropy_with_integer_labels
        bce = optax.sigmoid_binary_cross_entropy
        l2e = optax.l2_loss
        field_err_fns = Beatmap(
            positions=l2e,
            is_new_combo=bce,
            is_new_curve=bce,
            is_new_timing=bce,
            num_repeats=l2e,
            hit_types=cce,
            slider_types=cce,
            difficulties=Difficulty(*it.repeat(l2e, len(Difficulty._fields))),
        )
        flat_err_fns = jtu.tree_leaves(field_err_fns)
        flat_targets = jtu.tree_leaves(inputs.beatmap)
        flat_lmbdas = jnp.stack(jtu.tree_leaves(lmbdas))
        flat_predictions = jtu.tree_leaves(output)
        err_terms = [
            l(x, y).mean()
            for l, x, y in zip(flat_err_fns, flat_predictions, flat_targets)
        ]
        loss = jnp.sum(jax.nn.softmax(flat_lmbdas) * jnp.stack(err_terms))
        params["lmbdas"] = lmbdas
        return loss.mean()

    @partial(
        jax.jit,
        donate_argnums=0,
        out_shardings=jsh.PositionalSharding(jax.devices()).replicate(),
    )
    def _train_step(state: TrainState, inputs) -> TrainState:
        loss, grad = jax.value_and_grad(_objective)(state.params, inputs)
        step = state.step
        state = state.apply_gradients(grads=grad)
        step_inc = state.step
        loss_avg = (state.loss * jnp.minimum(step, save_every) + loss) / jnp.minimum(
            step_inc, save_every + 1
        )
        state = state.replace(loss=loss_avg)
        return state

    lazy_maps = scan_maps(meta_path)
    ckpointer = orbax.checkpoint.PyTreeCheckpointer()
    with cft.ThreadPoolExecutor(num_threads) as pool, rp.Progress(
        "loss: {task.fields[loss]:.3g}",
        *rp.Progress.get_default_columns()[:-2],
        rp.MofNCompleteColumn(),
        rp.TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "training...",
            start=False,
            total=train_steps,
            loss=float("nan"),
        )
        loader = batch_array_trees(
            batch_size * seq_length,
            bms_chunks(it.cycle(glob(osz_paths)), lazy_maps, mapper_fn=pool.map),
        )
        loader = (jtu.tree_map(_batch, x) for x in loader)
        loader = (jax.device_put(x, jtu.tree_map(_dp_map, x)) for x in loader)
        loader = iter(loader)
        inputs = next(loader)
        tstate = _train_init(jax.random.PRNGKey(42), inputs)
        for inputs in it.islice(loader, train_steps):
            tstate = _train_step(tstate, inputs)
            progress.start_task(task)
            loss = jax.device_get(tstate.loss).astype(float)
            progress.update(task, advance=1, loss=loss)
            step = jax.device_get(tstate.step).astype(int)
            if step == train_steps or step % save_every == 0:
                ckpt_target = tstate.params
                ckpt_config = orbax_utils.save_args_from_target(ckpt_target)
                ckpointer.save(
                    "params.ckpt", ckpt_target, save_args=ckpt_config, force=True
                )
                del ckpt_target


if __name__ == "__main__":
    fire.Fire(main)
