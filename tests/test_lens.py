"""lens/geometry.py against Theorem 1 and against lens/core.py."""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lens import core
from lens import geometry as geo

H, K = 128, 16
EPSS = [0.0, 1e-6, 1e-5, 1e-2, 1.0]  # 1e-6 is core.py's default
SEEDS = [0, 1, 2]
N_Z = 1000


def torch_default(seed, H=H, k=K, zero_bias=False):
    """nn.Linear(k, H) default init: weight and bias ~ U(-1/sqrt(k), 1/sqrt(k))."""
    rng = np.random.default_rng(seed)
    lim = 1.0 / np.sqrt(k)
    E = rng.uniform(-lim, lim, (H, k))
    b = np.zeros(H) if zero_bias else rng.uniform(-lim, lim, H)
    return E, b


def layernorm(h, eps):
    """LayerNorm without affine parameters, written independently of both modules."""
    r = h - h.mean(-1, keepdims=True)
    return r / np.sqrt((r**2).mean(-1, keepdims=True) + eps)


def sample_z(L, seed):
    """Points near z*, at the lens scale, far out, and generic."""
    rng = np.random.default_rng(1000 + seed)
    r = float(np.median(L.principal_widths))
    n = N_Z // 4
    parts = [L.z_star[None]]
    for scale in (1e-3 * r, r, 1e2 * r):
        parts.append(L.z_star + scale * rng.standard_normal((n, K)))
    parts.append(rng.standard_normal((N_Z - 3 * n - 1, K)))
    return np.concatenate(parts)


def core_params(E, b):
    """lens/core.py stores E transposed, (n_in, H)."""
    return dict(E=E.T, b=b, Wd=np.zeros((E.shape[0], 1)))


@pytest.mark.parametrize("eps", EPSS)
@pytest.mark.parametrize("seed", SEEDS)
def test_theorem1(seed, eps):
    E, b = torch_default(seed)
    L = geo.lens(E, b, eps)
    assert not L.degenerate
    z = sample_z(L, seed)
    lhs = layernorm(z @ E.T + b, eps)
    rhs = geo.gnomonic(L, z)
    assert np.abs(lhs - rhs).max() < 1e-12
    # and the same against core.py's LayerNorm
    lhs_core = np.asarray(core.norm_apply("layer", jnp.asarray(z @ E.T + b), eps))
    assert np.abs(lhs_core - rhs).max() < 1e-12


@pytest.mark.parametrize("seed", SEEDS)
def test_identities(seed):
    E, b = torch_default(seed)
    L = geo.lens(E, b, 1e-5)
    scale = np.linalg.norm(L.A) * L.norm_c_perp
    assert np.abs(L.A.T @ L.c_perp).max() < 1e-13 * scale          # A^T c_perp = 0
    assert np.allclose(L.c_perp, L.P @ (E @ L.z_star + b), rtol=0, atol=1e-14)
    z = sample_z(L, seed)
    u = geo.u_of_z(L, z)
    assert np.abs(u @ L.c_hat).max() < 1e-12 * (1 + np.abs(u).max())  # u perp c_hat
    assert np.allclose(geo.lens_distance(L, z), (u**2).sum(-1), rtol=1e-12, atol=0)
    # Sigma = delta0^2 M^{-1} has eigenvalues = principal widths squared
    ev = np.sort(np.linalg.eigvalsh(L.Sigma))
    assert np.allclose(ev, np.sort(L.principal_widths**2), rtol=1e-10, atol=0)
    # radius of Theorem 1 (the floor only shrinks it)
    nu2 = (u**2).sum(-1)
    rad = np.linalg.norm(geo.gnomonic(L, z), axis=-1)
    assert np.allclose(rad, np.sqrt(H * (1 + nu2) / (1 + L.kappa + nu2)), rtol=1e-12, atol=0)


@pytest.mark.parametrize("eps", EPSS)
@pytest.mark.parametrize("seed", SEEDS)
def test_r_eff_matches_core(seed, eps):
    """core.geometry's r_star = width(d) sqrt(1 + kappa_l) = r_eff(d), per line."""
    E, b = torch_default(seed)
    L = geo.lens(E, b, eps)
    rng = np.random.default_rng(2000 + seed)
    p = core_params(E, b)
    for i in range(20):
        d = rng.standard_normal(K)
        d /= np.linalg.norm(d)
        z0 = None if i < 10 else L.z_star + rng.standard_normal(K)  # through / away from z*
        ln = geo.line(L, d, z0)
        g = core.geometry(p, d, L.z_star if z0 is None else z0, H=H, eps=eps)
        r_eff = geo.r_eff(L, d, z0)
        assert r_eff == pytest.approx(g["r_star"], rel=1e-12, abs=0)
        assert geo.width(L, d, z0) * np.sqrt(1 + ln["kappa_l"]) == pytest.approx(g["r_star"], rel=1e-12, abs=0)
        assert ln["s_star"] == pytest.approx(g["s_star"], rel=1e-9, abs=1e-12)
        assert ln["norm_c_perp_l"] == pytest.approx(g["n_cperp"], rel=1e-12, abs=0)
        if z0 is None:  # convention 1: through z*, the line's c_perp is the lens's
            assert ln["norm_c_perp_l"] == pytest.approx(L.norm_c_perp, rel=1e-12, abs=0)
            assert ln["kappa_l"] == pytest.approx(L.kappa, rel=1e-12, abs=0)
            assert abs(ln["s_star"]) < 1e-12
    # principal widths are the widths along the principal directions
    for i in range(K):
        v = L.principal_dirs[:, i]
        assert geo.width(L, v) == pytest.approx(L.principal_widths[i], rel=1e-12, abs=0)
        assert geo.r_eff(L, v) == pytest.approx(L.principal_widths_eff[i], rel=1e-12, abs=0)


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1.0])
def test_zero_bias_degenerate(eps):
    """Remark 1: zero bias gives c_perp = 0 exactly; returned, not raised."""
    E, b = torch_default(7, zero_bias=True)
    L = geo.lens(E, b, eps)
    assert L.degenerate
    assert L.norm_c_perp == 0.0 and np.all(L.c_perp == 0.0)
    assert np.all(L.z_star == 0.0)
    assert L.kappa == np.inf and L.phi == 1.0
    assert np.all(L.principal_widths == 0.0)
    assert np.allclose(L.principal_widths_eff, np.sqrt(H * eps) / L.sing, rtol=1e-14, atol=0)
    d = np.ones(K) / np.sqrt(K)
    assert geo.width(L, d) == 0.0
    g = core.geometry(core_params(E, b), d, np.zeros(K), H=H, eps=eps)
    assert geo.r_eff(L, d) == pytest.approx(g["r_star"], rel=1e-12, abs=0)
    assert geo.r_eff(L, d) == pytest.approx(np.sqrt(H * eps) / np.linalg.norm(L.A @ d), rel=1e-14)
    with pytest.raises(ValueError):
        geo.u_of_z(L, np.zeros(K))
    # a constant bias is removed by P: also degenerate
    assert geo.lens(E, np.full(H, 0.3), eps).degenerate
    # a bias in span(E): c_perp is round-off only
    Lr = geo.lens(E, E @ np.linspace(-1, 1, K), eps)
    assert Lr.degenerate and np.allclose(Lr.z_star, -np.linspace(-1, 1, K), atol=1e-12)


def test_rank_deficient_raises():
    E, b = torch_default(8)
    E[:, 1] = 2.0 * E[:, 0]
    with pytest.raises(ValueError, match="rank"):
        geo.lens(E, b, 1e-5)
