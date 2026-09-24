from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class NormedLinearLayer:
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __call__(self, x, eps=1e-6):
        x = jnp.asarray(x)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        normed = (x - mean) / jnp.sqrt(var + eps)
        return normed @ self.weight.T + self.bias


def init_normed_linear_stack(key, widths, init="xavier"):
    layers = []
    keys = jax.random.split(key, len(widths) - 1)
    for k, (inp, out) in zip(keys, zip(widths[:-1], widths[1:])):
        if init == "normal":
            scale = 1.0 / jnp.sqrt(inp)
            w = scale * jax.random.normal(k, (out, inp))
        elif init == "xavier":
            lim = jnp.sqrt(6.0 / (inp + out))
            w = jax.random.uniform(k, (out, inp), minval=-lim, maxval=lim)
        else:
            raise ValueError(f"unknown init: {init}")
        b = jnp.zeros((out,))
        layers.append(NormedLinearLayer(w, b))
    return layers
