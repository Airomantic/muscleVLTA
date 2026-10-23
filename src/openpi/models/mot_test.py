import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import mot
from openpi.models import pi0


def test_tactile_pooling_is_causal_and_covers_all_frames():
    matrix = mot.tactile_pooling_matrix(action_horizon=7, stride=3)
    assert mot.tactile_token_times(7, 3) == (0, 3, 6)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert (matrix.sum(axis=0) > 0).all()
    for row, time in zip(matrix, mot.tactile_token_times(7, 3), strict=True):
        assert not row[time + 1 :].any()


def test_multirate_cross_attn_mask_respects_token_times():
    times = (0, 1, 2, 3, 0, 2)
    mask = np.asarray(pi0.make_multirate_cross_attn_mask(2, 4, times, latency=1)[0, 0])
    expected = np.array([[t <= i - 1 for t in times] for i in range(4)])
    np.testing.assert_array_equal(mask, expected)
    # Must agree with the single-rate LIFT mask for force-only memory.
    np.testing.assert_array_equal(
        pi0.make_multirate_cross_attn_mask(1, 4, range(4), latency=1),
        pi0.make_cross_attn_mask((1, 4, 4, 1)),
    )


def test_mot_memory_is_causal_in_time():
    times = ((0, 1, 2, 3), (0, 2))
    module = mot._LinenMultiRateMoT(
        modalities=("force", "tactile"), times=times, input_sizes=(5, 3), output_size=8, width=16, depth=2, num_heads=2
    )
    force = jax.random.normal(jax.random.key(0), (1, 4, 5))
    tactile = jax.random.normal(jax.random.key(1), (1, 2, 3))
    params = module.init(jax.random.key(2), [force, tactile])
    out = module.apply(params, [force, tactile])
    assert out.shape == (1, 6, 8)
    # Changing the last force frame and the tactile token at time 2 must not change tokens with time < 2.
    out_changed = module.apply(params, [force.at[:, 3].add(1.0), tactile.at[:, 1].add(1.0)])
    early = jnp.array([0, 1, 4])
    np.testing.assert_allclose(out[:, early].astype(jnp.float32), out_changed[:, early].astype(jnp.float32))
    assert not np.allclose(out[:, 3].astype(jnp.float32), out_changed[:, 3].astype(jnp.float32))
