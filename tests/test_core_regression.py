import jax.numpy as jnp

from lens.core import kink_core


def test_core_regression_values_bitwise():
    x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float32)
    y = kink_core(x, kink=0.0, left_slope=0.25, right_slope=1.5)
    expected = jnp.array([-0.5, -0.25, 0.0, 1.5, 3.0], dtype=jnp.float32)
    assert jnp.array_equal(y, expected)
