"""plants/quadrotor.py: hover equilibrium, Jacobians against finite differences, sign
conventions, the hover linearisation's textbook structure and RK4's order."""

import math
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plants import quadrotor as q  # noqa: E402

P = q.QuadrotorParams()


def test_params_are_the_approved_values():
    assert (P.m, P.L, P.Jx, P.Jy, P.Jz) == (0.027, 0.0397, 1.4e-5, 1.4e-5, 2.17e-5)
    assert (P.kf, P.km, P.thrust_to_weight, P.g) == (3.16e-10, 7.94e-12, 2.25, 9.81)
    assert P.c == pytest.approx(7.94e-12 / 3.16e-10)
    assert P.f_max == pytest.approx(2.25 * 0.027 * 9.81 / 4)
    assert not P.drag
    with pytest.raises(Exception):
        P.m = 1.0  # frozen
    with pytest.raises(ValueError):
        q.QuadrotorParams(drag=True)  # no approved default drag coefficients


def test_hover_is_an_equilibrium():
    x0, u0 = q.hover_equilibrium(P)
    assert float(jnp.max(jnp.abs(q.f(x0, u0, P)))) <= 1e-12
    assert float(u0[0]) == pytest.approx(P.m * P.g / 4) and float(u0[0]) < P.f_max
    x1, u1 = q.hover_equilibrium(P, position=(1.0, -2.0, 3.0), yaw=0.7)
    assert float(jnp.max(jnp.abs(q.f(x1, u1, P)))) <= 1e-12
    assert float(jnp.max(jnp.abs(q.rk4_step(x0, u0, P) - x0))) <= 1e-12


def _fd_jacobians(x, u, h=1e-6):
    x, u = np.asarray(x, np.float64), np.asarray(u, np.float64)
    fx = lambda xx: np.asarray(q.f(xx, u, P))
    fu = lambda uu: np.asarray(q.f(x, uu, P))
    A = np.stack([(fx(x + h * e) - fx(x - h * e)) / (2 * h) for e in np.eye(12)], 1)
    hu = h * P.f_max
    B = np.stack([(fu(u + hu * e) - fu(u - hu * e)) / (2 * hu) for e in np.eye(4)], 1)
    return A, B


def _random_states(n=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-3, 3, 3),
                            np.deg2rad(rng.uniform(-60, 60, 2)), [rng.uniform(-np.pi, np.pi)],
                            rng.uniform(-5, 5, 3)])
        u = rng.uniform(0, P.f_max, 4)
        out.append((x, u))
    return out


@pytest.mark.parametrize("case", ["hover"] + list(range(20)))
def test_jacobians_match_central_differences(case):
    if case == "hover":
        x, u = q.hover_equilibrium(P)
    else:
        x, u = _random_states()[case]
    A, B = q.linearize(x, u, P)
    Afd, Bfd = _fd_jacobians(x, u)
    for J, Jfd in ((np.asarray(A), Afd), (np.asarray(B), Bfd)):
        scale = max(np.max(np.abs(J)), 1e-300)
        assert np.max(np.abs(J - Jfd)) <= 1e-7 * scale + 1e-9


def test_mixer_signs_and_geometry():
    M = np.asarray(q.mixer(P))
    a = P.L / np.sqrt(2)
    # cf2x.urdf prop positions: 0 (+a,-a), 1 (-a,-a), 2 (-a,+a), 3 (+a,+a); tau = r x (0,0,f)
    pos = np.array([[a, -a], [-a, -a], [-a, a], [a, a]])
    for i in range(4):
        r = np.array([pos[i, 0], pos[i, 1], 0.0])
        tau = np.cross(r, [0.0, 0.0, 1.0])
        assert np.allclose(M[1:3, i], tau[:2], rtol=1e-14, atol=0)
    assert np.allclose(M[0], 1.0)
    assert np.allclose(M[3], P.c * np.array([-1, 1, -1, 1]))  # gym-pybullet-drones CF2X signs
    assert abs(np.linalg.det(M)) > 0  # invertible: (T, tau) reachable


def test_sign_conventions():
    x0, u0 = q.hover_equilibrium(P)
    d = 0.01 * P.m * P.g / 4
    # more thrust on the left rotors (2, 3; +y side) rolls positive (left side up)
    xd = q.f(x0, u0 + jnp.array([0, 0, d, d]), P)
    assert xd[9] > 0 and abs(xd[10]) < 1e-12
    # more thrust on the rear rotors (1, 2) pitches positive (nose down about +y)
    xd = q.f(x0, u0 + jnp.array([0, d, d, 0]), P)
    assert xd[10] > 0 and abs(xd[9]) < 1e-12
    # positive pitch tilts thrust forward: +x acceleration; positive roll: -y acceleration
    xd = q.f(x0.at[7].set(0.1), u0, P)
    assert xd[3] > 0
    xd = q.f(x0.at[6].set(0.1), u0, P)
    assert xd[4] < 0
    # rotors 1 and 3 produce +z (yaw) torque
    xd = q.f(x0, u0 + jnp.array([0, d, 0, d]), P)
    assert xd[11] > 0
    # equal extra thrust on all rotors: pure climb
    xd = q.f(x0, u0 * 1.1, P)
    assert xd[5] == pytest.approx(0.1 * P.g) and np.allclose(np.asarray(xd)[[3, 4, 9, 10, 11]], 0, atol=1e-12)
    # yaw rotates the thrust direction about z: with psi = pi/2, positive pitch accelerates +y
    xd = q.f(x0.at[8].set(np.pi / 2).at[7].set(0.1), u0, P)
    assert xd[4] > 0 and abs(xd[3]) < 1e-12


def test_rotation_and_euler_rates():
    eta = jnp.array([0.3, -0.4, 1.1])
    R = q.rotation(eta)
    assert np.allclose(np.asarray(R.T @ R), np.eye(3), atol=1e-14) and float(jnp.linalg.det(R)) == pytest.approx(1)
    W, Winv = q.euler_rate_matrix(eta), q.euler_rate_matrix_inv(eta)
    assert np.allclose(np.asarray(W @ Winv), np.eye(3), atol=1e-14)
    # Rdot = R [omega]x with omega = W eta_dot (body rates)
    eta_dot = jnp.array([0.2, -0.5, 0.7])
    Rdot = jax.jvp(q.rotation, (eta,), (eta_dot,))[1]
    w = W @ eta_dot
    skew = jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    assert np.allclose(np.asarray(Rdot), np.asarray(R @ skew), atol=1e-14)


def test_hover_linearisation_structure():
    x0, u0 = q.hover_equilibrium(P)
    A, B = (np.asarray(m) for m in q.linearize(x0, u0, P))
    g, m = P.g, P.m
    Aw = np.zeros((12, 12))
    Aw[0:3, 3:6] = np.eye(3)        # p' = v
    Aw[3, 7] = g                    # vx' = g theta
    Aw[4, 6] = -g                   # vy' = -g phi
    Aw[6:9, 9:12] = np.eye(3)       # eta' = omega
    assert np.allclose(A, Aw, rtol=0, atol=1e-12)
    Bw = np.zeros((12, 4))
    Bw[5, :] = 1.0 / m              # vz' = T/m
    Bw[9:12, :] = np.diag(1.0 / np.array([P.Jx, P.Jy, P.Jz])) @ np.asarray(q.mixer(P))[1:]
    assert np.allclose(B, Bw, rtol=1e-12, atol=1e-12)
    # controllable at hover
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(12)])
    assert np.linalg.matrix_rank(C) == 12


def test_rk4_is_fourth_order():
    """Global error ~ dt^4: halving dt divides it by ~16. A smooth manoeuvre near hover
    (differential thrust of 5%, moderate rates), in the asymptotic regime for these dt."""
    x0, u0 = q.hover_equilibrium(P)
    x = x0.at[6:9].set(jnp.array([0.2, -0.1, 0.3])).at[3:6].set(jnp.array([0.5, -0.3, 0.2]))
    x = x.at[9:12].set(jnp.array([1.0, -0.8, 0.5]))
    u = u0 * jnp.array([1.05, 0.97, 1.02, 0.99])
    T = 0.2

    def integrate(dt):
        n = int(round(T / dt))
        step = jax.jit(lambda xx: q.rk4_step(xx, u, P, dt))
        xx = x
        for _ in range(n):
            xx = step(xx)
        return np.asarray(xx)

    ref = integrate(T / 2560)
    errs = [np.linalg.norm(integrate(T / n) - ref) for n in (20, 40, 80)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(14.0 < r < 18.0 for r in ratios), (errs, ratios)


def test_linear_drag_flag():
    Pd = q.QuadrotorParams(drag=True, drag_coeffs=(0.01, 0.02, 0.03))
    x0, u0 = q.hover_equilibrium(Pd)
    x = x0.at[3:6].set(jnp.array([1.0, 1.0, 1.0]))
    diff = np.asarray(q.f(x, u0, Pd) - q.f(x, u0, P))
    assert np.allclose(diff[3:6], -np.array([0.01, 0.02, 0.03]) / Pd.m, rtol=1e-12)
    assert np.allclose(np.delete(diff, [3, 4, 5]), 0, atol=1e-12)


def test_discrete_step_jacobian_consistent_with_continuous():
    x0, u0 = q.hover_equilibrium(P)
    A, B = (np.asarray(m) for m in q.linearize(x0, u0, P))
    Ad, Bd = (np.asarray(m) for m in q.linearize_step(x0, u0, P, 0.01))
    # RK4 of a linear system: Ad = sum_{k<=4} (A dt)^k / k!
    dt = 0.01
    Ad_series = sum(np.linalg.matrix_power(A * dt, k) / math.factorial(k) for k in range(5))
    assert np.allclose(Ad, Ad_series, atol=1e-13)
