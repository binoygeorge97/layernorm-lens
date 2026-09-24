import jax.numpy as jnp

from plants.quadrotor import G, MASS, dynamics


def hover_equilibrium_error(state=None):
    if state is None:
        state = jnp.zeros((12,))
    control = jnp.array([MASS * G, 0.0, 0.0, 0.0])
    return jnp.linalg.norm(dynamics(state, control))
