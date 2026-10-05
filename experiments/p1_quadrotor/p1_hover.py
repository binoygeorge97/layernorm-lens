"""The hover linearisation check (docs/plan.md; prereg/p1.md): the surrogate's (A, B) at
hover against the simulator's, both by autodiff and in the same (physical) coordinates.

The surrogate models the scaled increment y = (x_{t+1} − x_t)/dt. A := ∂y/∂x and
B := ∂y/∂u in physical units, taken the same way for the truth (the RK4 map,
A_y = (A_d − I)/dt, B_y = B_d/dt) and for the surrogate. (A, B) is used as a
continuous-time model ẋ ≈ A x + B u for the LQR gain and the closed-loop spectrum
(spectral abscissa).

Sign test (prereg/p1.md, author's change A1). RK4 adds O(dt) cross-terms to the y-map
that are zero in the physics (e.g. ∂y_p/∂θ ≈ g·dt/2). So the entries compared are those
that are physically meaningful: the mask is the set of entries of the continuous vector
field's Jacobians ∂f/∂x, ∂f/∂u (taken from the simulator's vector field, not the RK4 map)
with |∂f| > sign_rel_threshold · max|∂f|, per matrix. On the mask, the signs of the
surrogate's (A, B) are compared with those of the true (A, B). A precondition, checked
every time: on the mask, every true y-map entry has the sign of the vector field's
(`mask_sign_check`). The disagreement count on the unmasked set (entries with
|true y-map| > threshold · max|true y-map|, which includes the O(dt) integration terms)
is reported without a rule.

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


def _z(plant, x, u):
    x = plant.x0 if x is None else x
    u = plant.u0 if u is None else u
    return jnp.asarray(np.concatenate([np.asarray(x), np.asarray(u)]), jnp.float64)


def true_y_jacobians(plant, x=None, u=None):
    """(A, B) = (∂y/∂x, ∂y/∂u) of y = (step(x, u) − x)/dt at (x, u), by default the
    plant's trim (hover)."""
    def y(z):
        x, u = z[:plant.n_x], z[plant.n_x:]
        return (plant.step(x, u) - x) / plant.dt
    J = np.asarray(jax.jacfwd(y)(_z(plant, x, u)))
    return J[:, :plant.n_x], J[:, plant.n_x:]


def true_f_jacobians(plant, x=None, u=None):
    """(∂f/∂x, ∂f/∂u) of the continuous vector field ẋ = f(x, u) at (x, u), by default
    hover. The sign mask comes from these, not from the RK4 map."""
    if plant.f is None:
        raise ValueError("the plant has no vector field f: the sign mask needs ∂f (prereg/p1.md)")

    def f(z):
        return plant.f(z[:plant.n_x], z[plant.n_x:])
    J = np.asarray(jax.jacfwd(f)(_z(plant, x, u)))
    return J[:, :plant.n_x], J[:, plant.n_x:]


def physical_jacobian(J_std, scalers, n_x):
    """A surrogate Jacobian in standardised coordinates (n_y, n_z) back to physical units:
    J = diag(σ_y) J_std diag(1/σ_z); returns (A, B)."""
    J = np.asarray(J_std) * scalers["sd_y"][:, None] / scalers["sd_z"][None, :]
    return J[:, :n_x], J[:, n_x:]


def bryson(hx, hu):
    """Q = diag(1/h_x²), R = diag(1/h_u²) from the sampling half-widths."""
    return np.diag(1.0 / np.asarray(hx) ** 2), np.diag(1.0 / np.asarray(hu) ** 2)


def sign_mask(F, rel_threshold):
    """The physically meaningful entries: |F| > rel_threshold · max|F| (one matrix)."""
    F = np.asarray(F, np.float64)
    return np.abs(F) > rel_threshold * np.max(np.abs(F))


def masked_sign_agreement(M_s, M_t, mask):
    """(fraction of mask entries where sign(M_s) = sign(M_t), number of disagreements,
    mask size)."""
    agree = np.sign(np.asarray(M_s)[mask]) == np.sign(np.asarray(M_t)[mask])
    return float(np.mean(agree)), int(np.sum(~agree)), int(mask.sum())


def mask_sign_check(M_t, F, mask):
    """The precondition of the sign test: on the mask, every true y-map entry has the
    vector field's sign. Raises ValueError otherwise."""
    bad = np.sign(np.asarray(M_t)[mask]) != np.sign(np.asarray(F)[mask])
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} masked entries of the true y-map differ in sign from ∂f")


def unmasked_disagreements(M_s, M_t, rel_threshold):
    """Reported without a rule: sign disagreements on the entries with |M_t| > thr · max|M_t|
    (the true y-map's own large entries, O(dt) integration terms included)."""
    big = np.abs(M_t) > rel_threshold * np.max(np.abs(M_t))
    return int(np.sum(np.sign(np.asarray(M_s)[big]) != np.sign(np.asarray(M_t)[big]))), int(big.sum())


def hover_check(A_s, B_s, A_t, B_t, Q, R, F_x, F_u, sign_rel_threshold=1e-3):
    """The hover check for one surrogate (module docstring). F_x, F_u: the vector
    field's Jacobians at the same point, for the sign mask."""
    A_s, B_s, A_t, B_t = (np.asarray(m, np.float64) for m in (A_s, B_s, A_t, B_t))
    out = dict(rel_err_A=float(np.linalg.norm(A_s - A_t) / np.linalg.norm(A_t)),
               rel_err_B=float(np.linalg.norm(B_s - B_t) / np.linalg.norm(B_t)))
    for nm, Ms, Mt, F in (("A", A_s, A_t, F_x), ("B", B_s, B_t, F_u)):
        mask = sign_mask(F, sign_rel_threshold)
        mask_sign_check(Mt, F, mask)
        out[f"sign_agree_{nm}"], out[f"n_sign_disagree_{nm}"], out[f"n_sign_{nm}"] = masked_sign_agreement(Ms, Mt, mask)
        out[f"n_sign_disagree_unmasked_{nm}"], out[f"n_unmasked_{nm}"] = unmasked_disagreements(Ms, Mt, sign_rel_threshold)
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
    """H1 for one model at hover (prereg/p1.md): the linearisation is wrong in sign (any
    sign disagreement on the mask, in A or B) or in stability (no stabilising gain, or the
    surrogate's gain leaves the true closed loop unstable)."""
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
    """At every trim (x, u0): the true (A, B) of the y-map and the vector field's
    (∂f/∂x, ∂f/∂u), stacked: dict(A, B, F_x, F_u). Computed once (independent of the
    surrogate)."""
    u0 = np.asarray(plant.u0, np.float64)
    Z = jnp.asarray(np.hstack([np.asarray(X, np.float64), np.tile(u0, (len(X), 1))]), jnp.float64)
    n_x = plant.n_x

    def y(z):
        return (plant.step(z[:n_x], z[n_x:]) - z[:n_x]) / plant.dt

    def f(z):
        return plant.f(z[:n_x], z[n_x:])
    J = np.asarray(jax.jit(jax.vmap(jax.jacfwd(y)))(Z))
    F = np.asarray(jax.jit(jax.vmap(jax.jacfwd(f)))(Z))
    return dict(A=J[:, :, :n_x], B=J[:, :, n_x:], F_x=F[:, :, :n_x], F_u=F[:, :, n_x:])


def trim_check(J_std_batch, scalers, plant, X, truth, Z_hover, lens_distance_fn, sign_rel_threshold=1e-3):
    """Per trim: the surrogate's (A, B) against the truth at that trim (`truth` from
    `trim_truth`), both in physical units: relative Frobenius errors, sign agreement on
    that trim's vector-field mask (after `mask_sign_check`), the trim's lens distance
    (`lens_distance_fn(Z)` on standardised trims), and the error in the surrogate's
    variation relative to hover (module docstring). J_std_batch(Z) gives the surrogate's
    standardised Jacobians at the rows of Z."""
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
        A_t, B_t = truth["A"][i], truth["B"][i]
        Jt = np.hstack([A_t, B_t])
        dvar = (J_phys[i] - Jsh) - (Jt - Jth)
        row = dict(trim=i, lens_distance=float(dist[i]),
                   rel_err_A=float(np.linalg.norm(A_s - A_t) / np.linalg.norm(A_t)),
                   rel_err_B=float(np.linalg.norm(B_s - B_t) / np.linalg.norm(B_t)),
                   variation_err=float(np.linalg.norm(dvar) / np.linalg.norm(Jth)),
                   true_variation=float(np.linalg.norm(Jt - Jth) / np.linalg.norm(Jth)))
        for nm, Ms, Mt, F in (("A", A_s, A_t, truth["F_x"][i]), ("B", B_s, B_t, truth["F_u"][i])):
            mask = sign_mask(F, sign_rel_threshold)
            mask_sign_check(Mt, F, mask)
            row[f"sign_agree_{nm}"] = masked_sign_agreement(Ms, Mt, mask)[0]
        rows.append(row)
    return rows
