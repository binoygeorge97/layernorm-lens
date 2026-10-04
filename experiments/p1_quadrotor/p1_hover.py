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


def true_y_jacobians(plant):
    """(A_y, B_y) of y = (step(x, u) − x)/dt at the plant's trim."""
    z0 = jnp.asarray(np.concatenate([plant.x0, plant.u0]), jnp.float64)

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
    except (np.linalg.LinAlgError, ValueError) as e:  # surrogate pair not stabilisable
        out.update(lqr_ok=False, lqr_error=str(e), stable=False)
        return out
    eig = lqr.closed_loop_eigs(A_t, B_t, K_s)
    out.update(lqr_ok=True, K_s=K_s, K_t=K_t, rel_err_K=float(np.linalg.norm(K_s - K_t) / np.linalg.norm(K_t)),
               closed_loop_eigs=eig, spectral_abscissa=float(np.max(eig.real)),
               stable=bool(np.max(eig.real) < 0))
    return out
