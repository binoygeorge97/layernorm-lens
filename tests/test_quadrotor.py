import jax
import jax.numpy as jnp

from plants.quadrotor import G, MASS, dynamics


def test_hover_is_equilibrium():
    x = jnp.zeros((12,))
    u = jnp.array([MASS * G, 0.0, 0.0, 0.0])
    dx = dynamics(x, u)
    assert jnp.allclose(dx, jnp.zeros_like(dx), atol=1e-8)


def test_autodiff_matches_finite_difference_state_jacobian():
    x = jnp.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.02, -0.03, 0.01, 0.2, -0.1, 0.05])
    u = jnp.array([MASS * G + 0.5, 0.01, -0.02, 0.03])

    f = lambda s: dynamics(s, u)
    j_auto = jax.jacobian(f)(x)

    eps = 1e-3
    cols = []
    for i in range(x.shape[0]):
        e = jnp.zeros_like(x).at[i].set(1.0)
        cols.append((f(x + eps * e) - f(x - eps * e)) / (2 * eps))
    j_fd = jnp.stack(cols, axis=1)

    assert jnp.allclose(j_auto, j_fd, atol=1e-3, rtol=1e-3)
