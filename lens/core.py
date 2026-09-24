import jax.numpy as jnp


def kink_core(x, kink=0.0, left_slope=0.1, right_slope=1.0):
    x = jnp.asarray(x)
    dx = x - kink
    return jnp.where(dx < 0, left_slope * dx, right_slope * dx)
