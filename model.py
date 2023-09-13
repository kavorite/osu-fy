from functools import partial
from typing import Callable, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from flax.linen.dtypes import promote_dtype

from S5 import S5Operator


def _fourier_act(x):
    x = x.at[..., ::2].set(jnp.cos(x[..., ::2]))
    x = x.at[..., 1::2].set(jnp.sin(x[..., 1::2]))
    return x


class TauPool(nn.Module):
    tau_factor: int = 4
    dim_factor: int = 2
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        z = rearrange("... (n k) d -> ... n (k d)", k=self.tau_factor)
        return nn.Dense(self.dim_factor * x.shape[-1], dtype=self.dtype)(z)


class MLP(nn.Module):
    ranks: Sequence[int]
    act_fn: Callable[[jax.Array], jax.Array]
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        for d in self.ranks[:-1]:
            x = self.act_fn(nn.Dense(d, dtype=self.dtype)(x))
        x = nn.Dense(self.ranks[-1], dtype=self.dtype)(x)
        return x


class EncoderBlock(nn.Module):
    width: int
    depth: int
    order: int = 2
    expand_factor: int = 2
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.shift_proj = nn.Dense(self.width, dtype=self.dtype)
        self.scale_proj = nn.Dense(self.width, dtype=self.dtype)
        self.att = S5Operator(
            self.width,
            order=self.order,
            short_filter_order=None,
            n_layer=self.depth,
            dtype=self.dtype,
            filter_args={
                "conj_sym": False,
                "C_init": "complex_normal",
                "dt_min": 0.001,
                "dt_max": 0.1,
                "clip_eigs": True,
                "activation": "gelu",
                "dtype": self.dtype,
            },
        )

        self.ffn = MLP(
            [self.width * self.expand_factor, self.width],
            act_fn=jax.nn.gelu,
            name="ffn",
            dtype=self.dtype,
        )

    def nrm(self, z, r):
        shift = _fourier_act(self.shift_proj(r))
        scale = _fourier_act(self.scale_proj(r))
        return jax.nn.standardize(z, epsilon=self.epsilon) * scale + shift

    def __call__(self, u_r_s, carry):
        del carry
        u, r, s = u_r_s
        reset = jnp.diff(s, axis=-1, prepend=0) != 0
        train = True
        z = u + self.att(u, reset, train)
        y = self.nrm(z + self.ffn(z), r)
        return (y, r), None


# class Mapper(nn.Module):
#     tau_factors: Sequence[int] = [6, 2, 2, 2]  # 48KHz -> 1KHz
#     rate_depths: Sequence[int] = [1, 1, 3, 1]
#     final_width: int = 128
#
#     def setup(self):
#         assert len(self.tau_factors) == len(self.rate_depths)
#         stage_count = len(self.tau_factors)
#         self.total_depth = sum(self.rate_depths)
#         dim_factors = 2 ** np.linspace(
#             1, np.log2(self.final_width), stage_count, endpoint=True
#         )
#         dim_factors = (2**dim_factors).astype(int)
#         self.dim_factors = dim_factors
#         self.enc_stack = []
#         self.tau_stack = []
#         for tau_factor, dim_factor in zip(dim_factors, self.tau_factors):
#             encoder = EncoderBlock(self.width, depth, dtype=self.dtype)
#             self.enc_stack.append(encoder)
#             pooling = TauPool(tau_factor, tau_factor * dim_factor, dtype=self.dtype)
#             self.tau_stack.append(pooling)
#
#     def __call__(self, inputs):
#         for enc, tau in zip(self.enc_stack, self.tau_stack):
#
#         for tau_factor, dim_factor in zip(dim_factors, self.tau_factors):
#
#             stack += [encoder, pooling]
#        inputs["samples"]


class Difficulty(NamedTuple):
    "Difficulty ratings. All values lie on [0, 10]."
    circle_size: float
    approach_rate: float
    drain_rate: float
    overall_difficulty: float
    slider_multiplier: float
    slider_tick_rate: float


class Position(NamedTuple):
    "Play field position."
    x: jax.Array
    y: jax.Array


class Beatmap(NamedTuple):
    positions: jax.Array
    hit_types: jax.Array
    durations: jax.Array
    path_end_mask: jax.Array
    combo_end_mask: jax.Array
    difficulty: Difficulty


class Mapper(nn.Module):
    width: int = 128
    depth: int = 6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs) -> Beatmap:
        Dense = partial(nn.Dense, dtype=self.dtype)
        inputs = promote_dtype(inputs, dtype=self.dtype)
        u = rearrange(
            inputs["raw_audio"], "... (k d) -> ... k d", k=48
        )  # 48KHz -> 1KHz
        r = jnp.stack([inputs["difficulty_rating"], inputs["fav_score"]], axis=-1)
        u = jax.nn.standardize(u, axis=-1)
        u = Dense(self.width)(u)
        for _ in range(self.depth):
            (u, r), carry = EncoderBlock(self.width, self.depth, dtype=self.dtype)(u, r)
            del carry

        global_latent = jnp.sum(u * jax.lax.rsqrt(u.shape[-2]), axis=-2)
        difficulty: Difficulty = 10.0 * jax.nn.relu(
            Dense(len(Difficulty._fields))(global_latent).swapaxes(0, -1)
        )
        hit_types = Dense(7)(u)
        path_end_mask, combo_end_mask = Dense(3)(u).swapaxes(0, -1)
        positions = jax.nn.tanh(Dense(len(Position._fields))(u).swapaxes(0, -1))
        durations = jax.nn.relu(Dense(1)(u).squeeze(-1))
        return Beatmap(
            positions,
            hit_types,
            durations,
            path_end_mask,
            combo_end_mask,
            difficulty,
        )
