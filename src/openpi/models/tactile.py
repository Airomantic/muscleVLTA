"""Visuo-tactile image encoder used by the reactive force branch."""

from flax import linen as nn
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp

# (features, kernel size) of the stride-2 convolutions.
_CONV_LAYERS = ((16, 5), (32, 3), (64, 3), (64, 3))


class _LinenTactileEncoder(nn.Module):
    """Small CNN that embeds every tactile image of every sensor and time step independently.

    The convolution weights are shared across sensors; the per-sensor features are concatenated in a fixed sensor
    order so that the downstream force encoder can tell the fingers apart.

    Layer names must not end with ``_1`` / ``_2``: parameters with these suffixes are treated as action / reactive
    expert weights when the reactive expert is initialized from the action expert.
    """

    feature_dim: int

    @nn.compact
    def __call__(self, tactile):
        # tactile: [batch, time, sensors, height, width] -> [batch, time, sensors * feature_dim]
        batch, time, sensors, height, width = tactile.shape
        x = tactile.reshape(batch * time * sensors, height, width, 1).astype(jnp.float32)
        for index, (features, kernel) in enumerate(_CONV_LAYERS):
            x = nn.Conv(features, (kernel, kernel), strides=(2, 2), padding="SAME", name=f"conv{index}")(x)
            x = nn.gelu(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.feature_dim, name="proj")(x)
        x = nn.LayerNorm(name="norm")(x)
        return x.reshape(batch, time, sensors * self.feature_dim)


def create_tactile_encoder(feature_dim: int):
    return nnx_bridge.ToNNX(_LinenTactileEncoder(feature_dim=feature_dim))
