"""LQR and closed-loop spectra (P-I hover linearisation check, docs/plan.md), numpy
float64 with scipy's Riccati solvers."""

import numpy as np
from scipy import linalg


def lqr_continuous(A, B, Q, R):
    """K for u = −K x minimising ∫ xᵀQx + uᵀRu for ẋ = A x + B u (continuous ARE)."""
    A, B, Q, R = (np.asarray(m, np.float64) for m in (A, B, Q, R))
    P = linalg.solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def lqr_discrete(A, B, Q, R):
    """K for u = −K x minimising Σ xᵀQx + uᵀRu for x⁺ = A x + B u (discrete ARE)."""
    A, B, Q, R = (np.asarray(m, np.float64) for m in (A, B, Q, R))
    P = linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def closed_loop_eigs(A, B, K):
    return np.linalg.eigvals(np.asarray(A) - np.asarray(B) @ np.asarray(K))


def spectral_abscissa(M):
    """max Re λ(M): continuous-time stability iff < 0."""
    return float(np.max(np.real(np.linalg.eigvals(np.asarray(M)))))


def spectral_radius(M):
    """max |λ(M)|: discrete-time stability iff < 1."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(M)))))
