"""The hover linearisation check (docs/plan.md): the surrogate's (A, B) at hover against
the simulator's, both by autodiff and in the same (physical) coordinates.

The surrogate models the scaled increment y = (x_{t+1} − x_t)/dt, so its Jacobian at
hover is compared with the true y-map's: A_y = (A_d − I)/dt, B_y = B_d/dt, (A_d, B_d)
the RK4 map's Jacobians. (A_y, B_y) is used as a continuous-time model ẋ ≈ A x + B u for
the LQR gain and the closed-loop spectrum (spectral abscissa).

Reported: relative Frobenius errors of A and B; sign agreement on the entries whose true
magnitude exceeds sign_rel_threshold · max|truth| (per matrix); the LQR gain from the
surrogate with fixed Q, R against the true gain; the closed-loop eigenvalues of the
surrogate's gain on the true linearisation (stable or not, spectral abscissa).

"No stabilising gain" (author, D16 (a)): the Riccati solve for the surrogate's (A, B)
fails, or its gain does not stabilise the surrogate's own (A, B). It is recorded, and
counts as a failure for H1 (`h1_fail`).

Trims (D16): steady level flight without drag. With no drag, any position, velocity and
yaw with level attitude, zero body rates and hover thrust u0 is a trim: velocity,
attitude and rates stay constant and only the position moves. The true y-map Jacobian
depends on the trim only through yaw (position does not enter the dynamics, and
velocity does not enter the accelerations), so `trim_check` reports, besides each trim's
error, the error in the surrogate's variation relative to hover:
‖(J_s(trim) − J_s(hover)) − (J_t(trim) − J_t(hover))‖_F / ‖J_t(hover)‖_F.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from control import lqr  # noqa: E402


def true_y_jacobians(plant, x=None, u=None):
    """(A_y, B_y) of y = (step(x, u) − x)/dt at (x, u), by default the plant's trim."""
    x = plant.x0 if x is None else x
    u = plant.u0 if u is None else u
    z0 = jnp.asarray(np.concatenate([np.asarray(x), np.asarray(u)]), jnp.float64)

    def y(z):
        x, u = z[:plant.n_x], z[plant.n_x:]
        return (plant.step(x, u) - x) / plant.dt
    J = np.asarray(jax.jacfwd(y)(z0))
    return J[:, :plant.n_x], J[:, plant.n_x:]


def physical_jacobian(J_std, scalers, n_x):
    """A surrogate Jacobian in standardised coordinates (n_y, n_z) back to physical units:
    J = diag(σ_y) J_std diag(1/σ_z); returns (A, B)."""
    J = np.asarray(J_std) * scalers["sd_y"][:, None] / scalers["sd_z"][None, :]
    return J[:, :n_x], J[:, n_x:]


def bryson(hx, hu):
    """Q = diag(1/h_x²), R = diag(1/h_u²) from the sampling half-widths."""
    return np.diag(1.0 / np.asarray(hx) ** 2), np.diag(1.0 / np.asarray(hu) ** 2)


def sign_agreement(M_s, M_t, rel_threshold):
    big = np.abs(M_t) > rel_threshold * np.max(np.abs(M_t))
    return float(np.mean(np.sign(M_s[big]) == np.sign(M_t[big]))), int(big.sum())


def hover_check(A_s, B_s, A_t, B_t, Q, R, sign_rel_threshold=1e-3):
    A_s, B_s, A_t, B_t = (np.asarray(m, np.float64) for m in (A_s, B_s, A_t, B_t))
    out = dict(rel_err_A=float(np.linalg.norm(A_s - A_t) / np.linalg.norm(A_t)),
               rel_err_B=float(np.linalg.norm(B_s - B_t) / np.linalg.norm(B_t)))
    out["sign_agree_A"], out["n_sign_A"] = sign_agreement(A_s, A_t, sign_rel_threshold)
    out["sign_agree_B"], out["n_sign_B"] = sign_agreement(B_s, B_t, sign_rel_threshold)
    K_t = lqr.lqr_continuous(A_t, B_t, Q, R)
    out["true_closed_loop_abscissa"] = lqr.spectral_abscissa(A_t - B_t @ K_t)
    try:
        K_s = lqr.lqr_continuous(A_s, B_s, Q, R)
        own = lqr.spectral_abscissa(A_s - B_s @ K_s)
        if not (np.all(np.isfinite(K_s)) and own < 0):
            raise ValueError(f"the gain does not stabilise the surrogate's own (A, B): abscissa {own}")
    except (np.linalg.LinAlgError, ValueError) as e:  # no stabilising LQR solution
        out.update(lqr_ok=False, no_stabilising_gain=True, lqr_error=str(e), stable=False)
        out["h1_fail"] = h1_fail(out)
        return out
    eig = lqr.closed_loop_eigs(A_t, B_t, K_s)
    out.update(lqr_ok=True, no_stabilising_gain=False, K_s=K_s, K_t=K_t,
               rel_err_K=float(np.linalg.norm(K_s - K_t) / np.linalg.norm(K_t)),
               surrogate_own_abscissa=own, closed_loop_eigs=eig, spectral_abscissa=float(np.max(eig.real)),
               stable=bool(np.max(eig.real) < 0))
    out["h1_fail"] = h1_fail(out)
    return out


def h1_fail(r):
    """H1 for one model at hover (p1 draft, D16): the linearisation is wrong in sign (any
    sign disagreement above the threshold in A or B) or in stability (no stabilising
    gain, or the surrogate's gain leaves the true closed loop unstable)."""
    return bool(r["sign_agree_A"] < 1.0 or r["sign_agree_B"] < 1.0 or r.get("no_stabilising_gain", False)
                or not r["stable"])


# --------------------------------------------------------------------------- #
# steady-flight trims                                                           #
# --------------------------------------------------------------------------- #

TRIM_FREE = (0, 1, 2, 3, 4, 5, 8)   # position, velocity, yaw (plants/quadrotor state order)


def sample_trims(plant, hx, n, seed):
    """n no-drag steady-flight trims inside the training box: position, velocity and yaw
    uniform in x0 ± hx (numpy.random.default_rng(seed)); roll, pitch and body rates 0;
    thrust u0. Returns X (n, n_x)."""
    rng = np.random.default_rng(seed)
    X = np.tile(np.asarray(plant.x0, np.float64), (n, 1))
    idx = np.asarray(TRIM_FREE)
    X[:, idx] += rng.uniform(-1.0, 1.0, (n, len(idx))) * np.asarray(hx)[idx]
    return X


def check_trims(plant, X, hx, tol=1e-12):
    """Max deviations for the trims X: (inside, step_dev). inside: every trim lies in the
    box x0 ± hx; step_dev: max over trims of |step(x, u0) − x − dt·(v, 0, …)|, which is
    zero (to rounding) for a trim (velocity, attitude and rates constant, position
    advancing by v·dt)."""
    X = np.asarray(X, np.float64)
    inside = bool(np.all(np.abs(X - plant.x0) <= np.asarray(hx) * (1 + tol)))
    u = jnp.asarray(plant.u0, jnp.float64)
    devs = []
    for x in X:
        dx = np.asarray(plant.step(jnp.asarray(x), u)) - x
        want = np.zeros_like(x)
        want[0:3] = plant.dt * x[3:6]
        devs.append(float(np.max(np.abs(dx - want))))
    return inside, float(max(devs))


def trim_truth(plant, X):
    """The true (A_y, B_y) at every trim (x, u0), stacked: (n, n_x, n_x), (n, n_x, n_u).
    The truth does not depend on the surrogate, so it is computed once."""
    u0 = np.asarray(plant.u0, np.float64)
    Z = jnp.asarray(np.hstack([np.asarray(X, np.float64), np.tile(u0, (len(X), 1))]), jnp.float64)

    def y(z):
        x, u = z[:plant.n_x], z[plant.n_x:]
        return (plant.step(x, u) - x) / plant.dt
    J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(y)))(Z))
    return J[:, :, :plant.n_x], J[:, :, plant.n_x:]


def trim_check(J_std_batch, scalers, plant, X, truth, Z_hover, lens_distance_fn, sign_rel_threshold=1e-3):
    """Per trim: the surrogate's (A, B) against the truth at that trim (`truth` from
    `trim_truth`), both in physical units: relative Frobenius errors, sign agreement, the
    trim's lens distance (`lens_distance_fn(Z)` on standardised trims), and the error in
    the surrogate's variation relative to hover (module docstring). J_std_batch(Z) gives
    the surrogate's standardised Jacobians at the rows of Z."""
    n_x = plant.n_x
    X = np.asarray(X, np.float64)
    Zt = (np.hstack([X, np.tile(np.asarray(plant.u0), (len(X), 1))]) - scalers["mu_z"]) / scalers["sd_z"]
    Js = np.asarray(J_std_batch(np.vstack([Zt, np.asarray(Z_hover)[None, :]])))
    J_phys = Js * scalers["sd_y"][None, :, None] / scalers["sd_z"][None, None, :]
    Jsh = J_phys[-1]
    A_th, B_th = true_y_jacobians(plant)
    Jth = np.hstack([A_th, B_th])
    dist = np.asarray(lens_distance_fn(Zt))
    rows = []
    for i in range(len(X)):
        A_s, B_s = J_phys[i][:, :n_x], J_phys[i][:, n_x:]
        A_t, B_t = truth[0][i], truth[1][i]
        Jt = np.hstack([A_t, B_t])
        dvar = (J_phys[i] - Jsh) - (Jt - Jth)
        rows.append(dict(trim=i, lens_distance=float(dist[i]),
                         rel_err_A=float(np.linalg.norm(A_s - A_t) / np.linalg.norm(A_t)),
                         rel_err_B=float(np.linalg.norm(B_s - B_t) / np.linalg.norm(B_t)),
                         sign_agree_A=sign_agreement(A_s, A_t, sign_rel_threshold)[0],
                         sign_agree_B=sign_agreement(B_s, B_t, sign_rel_threshold)[0],
                         variation_err=float(np.linalg.norm(dvar) / np.linalg.norm(Jth)),
                         true_variation=float(np.linalg.norm(Jt - Jth) / np.linalg.norm(Jth))))
    return rows
