import jax.numpy as jnp


def integrator_step(x, u, dt=0.1):
    return x + dt * u


def damped_oscillator_step(state, u, dt=0.01, omega=1.0, zeta=0.1):
    x, v = state
    a = -omega**2 * x - 2 * zeta * omega * v + u
    return jnp.array([x + dt * v, v + dt * a])
