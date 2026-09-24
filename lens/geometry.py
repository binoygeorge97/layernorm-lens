"""The lens of the first LayerNorm after an affine map, from the weights alone.

Notation and definitions follow docs/theory.md exactly:

    h = E z + b,  E in R^{H x k},  P = I - 11^T/H,  A = P E  (rank k required)
    z*  = -(A^T A)^{-1} A^T P b
    c_perp = P b + A z*          (A^T c_perp = 0)
    c_hat = c_perp/|c_perp|,  kappa = H eps/|c_perp|^2,  phi = kappa/(1 + kappa)
    M = A^T A / H,  delta0^2 = |c_perp|^2 / H,  Sigma = delta0^2 M^{-1}
    u(z) = A (z - z*) / |c_perp|
    Theorem 1:  LN(h(z)) = sqrt(H) (c_hat + u) / sqrt(1 + kappa + |u|^2)

Widths (implementation convention 1; confirmed). For the line z0 + s d, with its
own offset c_perp_l and floor ratio kappa_l = H eps/|c_perp_l|^2:

    r*_l  = |c_perp_l| / |A d|                        (width, eps-free)
    r_eff = sqrt(|c_perp_l|^2 + H eps) / |A d| = r*_l sqrt(1 + kappa_l)

r_eff is lens/core.py's `r_star`: the width of the normalised output's profile
along the line. Principal widths are |c_perp| / s_i, s_i the singular values of A.

Degenerate case (Remark 1: zero bias gives c_perp = 0 exactly): the result is
returned with degenerate=True, r* = 0, kappa = inf, phi = 1, and the eps-limited
widths r_eff = sqrt(H eps)/s_i. Theorem 1 does not apply there, so u(z) raises.
Only rank A < k raises (Remark 1b/2: k >= H - 1, handled elsewhere).

Everything here is numpy float64; E and b may be JAX or torch-converted arrays.
"""

from dataclasses import dataclass

import numpy as np

# |c_perp| below this fraction of |b| is round-off: b lies in span(E, 1). Relative
# to |b|, not |P b|, because P itself leaves round-off of order 1e-16 |b|.
DEGENERATE_RTOL = 1e-12


@dataclass(frozen=True)
class Lens:
    H: int
    k: int
    eps: float
    P: np.ndarray            # (H, H) centring projector
    A: np.ndarray            # (H, k) = P E
    z_star: np.ndarray       # (k,)
    c_perp: np.ndarray       # (H,)
    norm_c_perp: float
    c_hat: np.ndarray        # (H,), NaN if degenerate
    kappa: float             # inf if degenerate
    phi: float
    sing: np.ndarray         # (k,) singular values of A, descending
    principal_dirs: np.ndarray  # (k, k) columns: right singular vectors of A
    principal_widths: np.ndarray      # |c_perp| / s_i   (0 if degenerate)
    principal_widths_eff: np.ndarray  # sqrt(|c_perp|^2 + H eps) / s_i
    M: np.ndarray            # (k, k) = A^T A / H
    delta0_sq: float         # |c_perp|^2 / H
    Sigma: np.ndarray        # (k, k) = delta0^2 M^{-1}  (0 if degenerate)
    degenerate: bool


def centring_projector(H):
    return np.eye(H) - np.ones((H, H)) / H


def lens(E, b, eps):
    """Lens of LN(E z + b). E: (H, k), b: (H,), eps: the LayerNorm epsilon."""
    E = np.asarray(E, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    H, k = E.shape
    if b.shape != (H,):
        raise ValueError(f"b has shape {b.shape}, expected ({H},)")
    eps = float(eps)
    P = centring_projector(H)
    A = P @ E
    Pb = P @ b
    _, sing, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(sing > sing[0] * max(H, k) * np.finfo(float).eps)) if sing[0] > 0 else 0
    if rank < k:
        raise ValueError(f"rank A = {rank} < k = {k}: degenerate lens (theory.md, "
                         "Remark 1b/2); not handled by lens().")
    z_star = np.linalg.lstsq(A, -Pb, rcond=None)[0]
    c_perp = Pb + A @ z_star
    ncp = float(np.linalg.norm(c_perp))
    degenerate = ncp == 0.0 or ncp <= DEGENERATE_RTOL * float(np.linalg.norm(b))
    M = A.T @ A / H
    widths_eff = np.sqrt(ncp**2 + H * eps) / sing
    if degenerate:
        ncp = 0.0
        c_perp = np.zeros(H)
        c_hat = np.full(H, np.nan)
        kappa, phi = np.inf, 1.0
        widths = np.zeros(k)
        widths_eff = np.sqrt(H * eps) / sing
        Sigma = np.zeros((k, k))
    else:
        c_hat = c_perp / ncp
        kappa = H * eps / ncp**2
        phi = kappa / (1.0 + kappa)
        widths = ncp / sing
        Sigma = (ncp**2 / H) * np.linalg.inv(M)
    return Lens(H=H, k=k, eps=eps, P=P, A=A, z_star=z_star, c_perp=c_perp,
                norm_c_perp=ncp, c_hat=c_hat, kappa=kappa, phi=phi, sing=sing,
                principal_dirs=Vt.T, principal_widths=widths,
                principal_widths_eff=widths_eff, M=M, delta0_sq=ncp**2 / H,
                Sigma=Sigma, degenerate=degenerate)


def line(L, d, z0=None):
    """Geometry of the line z0 + s d (Corollary 1). z0=None means through z*.

    Returns s_star (the line's closest approach, in units of s), c_perp_l,
    norm_c_perp_l, kappa_l, r_star (= |c_perp_l|/|A d|) and r_eff
    (= sqrt(|c_perp_l|^2 + H eps)/|A d|). For unit d these are input units.
    """
    d = np.asarray(d, dtype=np.float64)
    z0 = L.z_star if z0 is None else np.asarray(z0, dtype=np.float64)
    q = L.A @ d
    nq = float(np.linalg.norm(q))
    if nq == 0.0:
        raise ValueError("A d = 0: d must be nonzero (A has full column rank).")
    c = L.A @ (z0 - L.z_star) + L.c_perp  # = P (E z0 + b)
    s_star = -float(q @ c) / nq**2
    c_l = c + s_star * q
    ncl = float(np.linalg.norm(c_l))
    kappa_l = np.inf if ncl == 0.0 else L.H * L.eps / ncl**2
    return dict(s_star=s_star, c_perp_l=c_l, norm_c_perp_l=ncl, kappa_l=kappa_l,
                norm_Ad=nq, r_star=ncl / nq,
                r_eff=float(np.sqrt(ncl**2 + L.H * L.eps)) / nq)


def width(L, d, z0=None):
    """r*_l = |c_perp_l| / |A d| (convention 1 when z0 is None and |d| = 1)."""
    return line(L, d, z0)["r_star"]


def r_eff(L, d, z0=None):
    """sqrt(|c_perp_l|^2 + H eps) / |A d| = r*_l sqrt(1 + kappa_l); core.py's r_star."""
    return line(L, d, z0)["r_eff"]


def u_of_z(L, z):
    """u = A (z - z*) / |c_perp|, for z of shape (..., k). Undefined if degenerate."""
    if L.degenerate:
        raise ValueError("c_perp = 0: u (and Theorem 1) is undefined for a degenerate lens.")
    return (np.asarray(z, dtype=np.float64) - L.z_star) @ L.A.T / L.norm_c_perp


def lens_distance(L, z):
    """rho(z) = (z - z*)^T Sigma^{-1} (z - z*) = |u|^2 (convention 2); inf if degenerate."""
    dz = np.asarray(z, dtype=np.float64) - L.z_star
    Adz2 = np.sum((dz @ L.A.T) ** 2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Adz2 / L.norm_c_perp**2


def gnomonic(L, z):
    """Theorem 1's right-hand side: sqrt(H) (c_hat + u) / sqrt(1 + kappa + |u|^2)."""
    u = u_of_z(L, z)
    nu2 = np.sum(u**2, axis=-1, keepdims=True)
    return np.sqrt(L.H) * (L.c_hat + u) / np.sqrt(1.0 + L.kappa + nu2)
