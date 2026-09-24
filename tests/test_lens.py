import jax.numpy as jnp

from lens.geometry import lens_from_weights


def test_lens_formulas_match_direct_computation():
    w = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    b = jnp.float32(0.5)

    out = lens_from_weights(w, b)

    norm2 = jnp.dot(w, w)
    z_star = -(b / norm2) * w
    sigma = jnp.eye(3, dtype=jnp.float32) - jnp.outer(w, w) / norm2
    r_star = jnp.linalg.norm(z_star)
    coverage = jnp.trace(sigma) / 3.0
    sharpness = jnp.linalg.norm(w)

    assert jnp.allclose(out["z_star"], z_star)
    assert jnp.allclose(out["Sigma"], sigma)
    assert jnp.allclose(out["r_star"], r_star)
    assert jnp.allclose(out["coverage"], coverage)
    assert jnp.allclose(out["sharpness"], sharpness)
