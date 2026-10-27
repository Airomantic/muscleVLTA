"""Multi-rate mixture-of-transformers (MoT) encoder for the reactive force / tactile memory.

Every modality (e.g. the 15 Hz force stream and the low-rate tactile stream) is a separate "expert": it has its own
input projection, layer norms, attention projections, MLP and output projection, while all tokens share one global
self-attention (Mixture-of-Transformers, Liang et al. 2024). Each token carries the frame index (time) at which its
information is available; attention is causal in time (a token only sees tokens with time <= its own time), so the
memory stays causal for the latency-masked cross-attention of the reactive action expert.

Parameter names must not end with ``_1`` / ``_2``: such parameters are treated as action / reactive expert weights
when the reactive expert is initialized from the action expert.
"""

from collections.abc import Sequence

from flax import linen as nn
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np


def _time_embedding(times: np.ndarray, width: int) -> jnp.ndarray:
    half = width // 2
    freqs = np.exp(-np.log(10_000.0) * np.arange(half) / max(half - 1, 1))
    angles = np.asarray(times, dtype=np.float32)[:, None] * freqs[None, :]
    return jnp.asarray(np.concatenate([np.sin(angles), np.cos(angles)], axis=-1), dtype=jnp.float32)


class _MoTBlock(nn.Module):
    modalities: tuple[str, ...]
    width: int
    num_heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, xs: Sequence[jnp.ndarray], mask: jnp.ndarray) -> list[jnp.ndarray]:
        head_dim = self.width // self.num_heads
        lengths = [x.shape[1] for x in xs]
        splits = np.cumsum(lengths)[:-1].tolist()

        qkv = []
        for name, x in zip(self.modalities, xs, strict=True):
            h = nn.LayerNorm(name=f"{name}_attn_norm")(x)
            qkv.append(nn.Dense(3 * self.width, name=f"{name}_qkv")(h))
        qkv = jnp.concatenate(qkv, axis=1)
        batch, length, _ = qkv.shape
        q, k, v = jnp.split(qkv.reshape(batch, length, 3, self.num_heads, head_dim), 3, axis=2)
        q, k, v = (t[:, :, 0] for t in (q, k, v))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(head_dim)
        logits = jnp.where(mask[None, None], logits, jnp.finfo(logits.dtype).min)
        attn = jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), v).reshape(batch, length, self.width)

        out = []
        for name, x, a in zip(self.modalities, xs, jnp.split(attn, splits, axis=1), strict=True):
            x = x + nn.Dense(self.width, name=f"{name}_attn_out")(a)  # noqa: PLW2901
            h = nn.LayerNorm(name=f"{name}_mlp_norm")(x)
            h = nn.gelu(nn.Dense(self.mlp_ratio * self.width, name=f"{name}_mlp_in")(h))
            out.append(x + nn.Dense(self.width, name=f"{name}_mlp_out")(h))
        return out


class _LinenMultiRateMoT(nn.Module):
    """Encodes per-modality token sequences [b, t_m, d_m] into memory tokens [b, sum(t_m), output_size]."""

    modalities: tuple[str, ...]
    # Frame index of every token of every modality (static, e.g. (0..14) for 15 Hz force, (0, 3, ..., 12) for tactile).
    times: tuple[tuple[int, ...], ...]
    input_sizes: tuple[int, ...]
    output_size: int
    width: int = 256
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, streams: Sequence[jnp.ndarray]) -> jnp.ndarray:
        all_times = np.concatenate([np.asarray(t, dtype=np.int32) for t in self.times])
        mask = jnp.asarray(all_times[None, :] <= all_times[:, None])
        xs = []
        for name, stream, times in zip(self.modalities, streams, self.times, strict=True):
            x = nn.Dense(self.width, name=f"{name}_in")(stream.astype(jnp.float32))
            x = x + self.param(f"{name}_type", nn.initializers.normal(0.02), (self.width,))
            xs.append(x + _time_embedding(np.asarray(times), self.width)[None])
        for index in range(self.depth):
            xs = _MoTBlock(self.modalities, self.width, self.num_heads, self.mlp_ratio, name=f"block{index}")(xs, mask)
        out = [
            nn.Dense(self.output_size, name=f"{name}_out")(nn.LayerNorm(name=f"{name}_final_norm")(x))
            for name, x in zip(self.modalities, xs, strict=True)
        ]
        return jnp.concatenate(out, axis=1).astype(jnp.bfloat16)

    def init_shapes(self):
        self([jnp.zeros((1, len(t), d)) for t, d in zip(self.times, self.input_sizes, strict=True)])


def tactile_token_times(action_horizon: int, stride: int, *, history: int = 0, anchor: int = 0) -> tuple[int, ...]:
    """Frame times of the low-rate tactile tokens: every t in [-history, action_horizon) with t = anchor (mod stride).

    The defaults give 0, stride, 2 * stride, ...; negative times are history frames before the current one (0).
    """
    return tuple(t for t in range(-history, action_horizon) if (t - anchor) % stride == 0)


def tactile_pooling_matrix(action_horizon: int, stride: int, *, history: int = 0, anchor: int = 0) -> np.ndarray:
    """[num_tokens, history + action_horizon] causal averaging over the frames -history .. action_horizon - 1.

    Token j averages the frames (time_{j-1}, time_j]; the first token averages every frame from -history on.
    """
    times = tactile_token_times(action_horizon, stride, history=history, anchor=anchor)
    matrix = np.zeros((len(times), history + action_horizon), dtype=np.float32)
    start = 0
    for index, time in enumerate(times):
        end = time + history + 1
        matrix[index, start:end] = 1.0 / (end - start)
        start = end
    return matrix


def create_multirate_mot(
    *,
    modalities: tuple[str, ...],
    times: tuple[tuple[int, ...], ...],
    input_sizes: tuple[int, ...],
    output_size: int,
    width: int,
    depth: int,
    num_heads: int,
):
    return nnx_bridge.ToNNX(
        _LinenMultiRateMoT(
            modalities=modalities,
            times=times,
            input_sizes=input_sizes,
            output_size=output_size,
            width=width,
            depth=depth,
            num_heads=num_heads,
        )
    )
