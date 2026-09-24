import jax.numpy as jnp


def lens_from_weights(weight, bias=0.0):
    w = jnp.asarray(weight, dtype=jnp.float32)
    b = jnp.asarray(bias, dtype=jnp.float32)
    n = w.shape[0]
    norm2 = jnp.dot(w, w)
    eps = jnp.finfo(w.dtype).eps
    safe_norm2 = jnp.maximum(norm2, eps)

    z_star = -(b / safe_norm2) * w
    projector = jnp.eye(n, dtype=w.dtype) - jnp.outer(w, w) / safe_norm2
    r_star = jnp.linalg.norm(z_star)
    coverage = jnp.trace(projector) / n
    sharpness = jnp.sqrt(safe_norm2)

    return {
        "z_star": z_star,
        "Sigma": projector,
        "r_star": r_star,
        "coverage": coverage,
        "sharpness": sharpness,
    }
