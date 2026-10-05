"""P-I surrogates (docs/plan.md): two architectures, two initialisations, float64 JAX.

Weights use the (out, in) layout, so the first layer is theory.md's E ∈ R^{H×k} and
`first_layer` hands (E, b) to lens/geometry.py directly.

Architectures (`arch`)
----------------------
- "prenorm": the pre-norm residual block of lens/core.py, generalised to vector
  outputs. h = E z + b; for each block j:
      h ← h + W2_j GELU(W1_j (γ_j LN(h) + β_j) + b1_j) + b2_j
  and y = Wo h + bo. LN without affine is theory.md's ĥ (ε inside the square root);
  γ, β are the block's own affine parameters, applied before the branch.
- "normedlinear": TD-MPC2's NormedLinear stack (tdmpc2 e9f59321 common/layers.py:
  Linear → LayerNorm(affine) → Mish). n_blocks NormedLinear layers (k → H, then H → H),
  then a plain Linear H → n_out.

Initialisations (`init`), the initial-lens check's conventions (experiments/initial_lens/
config.yaml):
- "zero_bias": every Linear weight ~ U(−1/√fan_in, 1/√fan_in) (PyTorch's default
  nn.Linear weight, kaiming_uniform with a = √5); every Linear bias = 0.
- "torch_default": PyTorch's default nn.Linear: weight and bias ~ U(−1/√fan_in, 1/√fan_in).
LayerNorm γ = 1, β = 0 in both (PyTorch's default).
"""

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ARCHS = ("prenorm", "normedlinear")
INITS = ("zero_bias", "torch_default")


@dataclasses.dataclass(frozen=True)
class SurrogateSpec:
    arch: str = "prenorm"
    n_blocks: int = 1
    H: int = 128
    k: int = 16
    n_out: int = 12
    eps: float = 1e-5
    init: str = "zero_bias"
    branch_width: int = 128     # prenorm only: the branch MLP's hidden width

    def __post_init__(self):
        if self.arch not in ARCHS or self.init not in INITS or self.n_blocks < 1:
            raise ValueError(f"bad spec {self}")


def _linear(rng, fan_in, fan_out, init):
    """The bias is always drawn (then zeroed for zero_bias), so both initialisations of
    the same seed share every weight: a paired comparison, as in the initial-lens check."""
    lim = 1.0 / np.sqrt(fan_in)
    W = rng.uniform(-lim, lim, (fan_out, fan_in))
    b = rng.uniform(-lim, lim, fan_out)
    if init == "zero_bias":
        b = np.zeros(fan_out)
    return jnp.asarray(W, jnp.float64), jnp.asarray(b, jnp.float64)


def init_params(spec, seed):
    """Parameters as a flat dict of float64 arrays; numpy.random.default_rng(seed)."""
    rng = np.random.default_rng(seed)
    p = {}
    ones, zeros = jnp.ones(spec.H, jnp.float64), jnp.zeros(spec.H, jnp.float64)
    if spec.arch == "prenorm":
        p["E"], p["b"] = _linear(rng, spec.k, spec.H, spec.init)
        for j in range(spec.n_blocks):
            p[f"g{j}"], p[f"be{j}"] = ones, zeros
            p[f"W1_{j}"], p[f"b1_{j}"] = _linear(rng, spec.H, spec.branch_width, spec.init)
            p[f"W2_{j}"], p[f"b2_{j}"] = _linear(rng, spec.branch_width, spec.H, spec.init)
        p["Wo"], p["bo"] = _linear(rng, spec.H, spec.n_out, spec.init)
    else:
        fan = spec.k
        for j in range(spec.n_blocks):
            p[f"W{j}"], p[f"b{j}"] = _linear(rng, fan, spec.H, spec.init)
            p[f"g{j}"], p[f"be{j}"] = ones, zeros
            fan = spec.H
        p["Wo"], p["bo"] = _linear(rng, spec.H, spec.n_out, spec.init)
    return p


def first_layer(spec, p):
    """(E, b) of the first Linear, whose LayerNorm is the lens (theory.md)."""
    return (p["E"], p["b"]) if spec.arch == "prenorm" else (p["W0"], p["b0"])


def layer_norm(h, eps):
    """theory.md's ĥ: centred, divided by √(mean square + ε); no affine."""
    r = h - jnp.mean(h, -1, keepdims=True)
    return r / jnp.sqrt(jnp.mean(r ** 2, -1, keepdims=True) + eps)


def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def trace(spec, p, z, upto=None):
    """Forward pass of one input z (k,), returning (y, [ĥ_1, ..., ĥ_n]): the output and
    each block's pre-affine LayerNorm output. With `upto=j`, only blocks 1..j run and the
    head reads their output (the residual stream after block j for "prenorm", block j's
    activation for "normedlinear"): the model truncated after block j."""
    hats = []
    n = spec.n_blocks if upto is None else int(upto)
    if not 1 <= n <= spec.n_blocks:
        raise ValueError(f"upto must be in [1, {spec.n_blocks}]")
    if spec.arch == "prenorm":
        h = p["E"] @ z + p["b"]
        for j in range(n):
            v = layer_norm(h, spec.eps)
            hats.append(v)
            a = jax.nn.gelu(p[f"W1_{j}"] @ (p[f"g{j}"] * v + p[f"be{j}"]) + p[f"b1_{j}"], approximate=False)
            h = h + p[f"W2_{j}"] @ a + p[f"b2_{j}"]
        return p["Wo"] @ h + p["bo"], hats
    x = z
    for j in range(n):
        v = layer_norm(p[f"W{j}"] @ x + p[f"b{j}"], spec.eps)
        hats.append(v)
        x = mish(p[f"g{j}"] * v + p[f"be{j}"])
    return p["Wo"] @ x + p["bo"], hats


def forward(spec, p, z):
    """y = surrogate(z) for one input (k,)."""
    return trace(spec, p, z)[0]


def batched(spec):
    """f(p, Z) for Z (n, k) -> (n, n_out)."""
    return jax.vmap(lambda p, z: forward(spec, p, z), in_axes=(None, 0))


def n_params(p):
    return int(sum(np.prod(v.shape) for v in p.values()))
