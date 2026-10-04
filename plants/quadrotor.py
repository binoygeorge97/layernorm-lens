"""Quadrotor plant for P-I (docs/plan.md): 12 states, 4 inputs, JAX float64.

Parameters (approved by the author, 4 Oct 2026) are the Crazyflie 2.x values of
gym-pybullet-drones' `gym_pybullet_drones/assets/cf2x.urdf` (github.com/learnsyslab/
gym-pybullet-drones, last changed at 889ce4a5c068ae4d811df1442ceb4f4d6cdf43eb, unchanged
at `main` 7ebad1e on 3 Oct 2026; file SHA-256
81494018056df2995351da62b9365d2ef4e1512112feaf73e88f9203e20c884b):
mass 0.027 kg, arm 0.0397 m (X configuration), J = diag(1.4e-5, 1.4e-5, 2.17e-5) kg m²,
kf = 3.16e-10, km = 7.94e-12 (per RPM², as gym-pybullet-drones' BaseAviary uses them),
thrust-to-weight 2.25. Gravity is g = 9.81 m/s² here; gym-pybullet-drones uses 9.8.

Conventions
-----------
- World frame z-up, gravity (0, 0, −g). Body frame FLU (x forward, y left, z up);
  each rotor's thrust acts along body +z.
- State x = (p, v, η, ω) ∈ R¹²: position and velocity in the world frame; η = (φ, θ, ψ)
  ZYX Euler angles (roll, pitch, yaw), R(η) = Rz(ψ) Ry(θ) Rx(φ) maps body to world;
  ω = body angular velocity, with ω = W(η) η̇.
- Input u = (f0, f1, f2, f3), rotor thrusts in N, each in [0, f_max],
  f_max = thrust_to_weight · m g / 4. Rotor numbering and positions as the cf2x.urdf
  prop links: 0 front-right (+a, −a), 1 rear-right (−a, −a), 2 rear-left (−a, +a),
  3 front-left (+a, +a), a = L/√2.
- X-mixer (`mixer`): T = Σ f_i; τ = Σ r_i × (0, 0, f_i), i.e.
      τx = a (−f0 − f1 + f2 + f3),  τy = a (−f0 + f1 + f2 − f3),
  and the rotor drag torque τz = c (−f0 + f1 − f2 + f3), c = km/kf (in m), the same
  signs as gym-pybullet-drones' CF2X `_dynamics` (BaseAviary.py at 7ebad1e).

Dynamics (`f`):
    p' = v
    v' = R(η) (0, 0, T)/m − (0, 0, g)  [− (D/m) v with linear drag, off by default]
    η' = W(η)⁻¹ ω
    ω' = J⁻¹ (τ − ω × J ω)
No motor lag. The inputs are not clipped by `f`: actuator limits are the caller's
(and the controller's) responsibility; `u_max(params)` gives f_max.
"""

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

N_STATE, N_INPUT = 12, 4
STATE_NAMES = ("px", "py", "pz", "vx", "vy", "vz", "phi", "theta", "psi", "wx", "wy", "wz")
INPUT_NAMES = ("f0", "f1", "f2", "f3")


@dataclasses.dataclass(frozen=True)
class QuadrotorParams:
    """Physical parameters (SI). Defaults: the approved cf2x.urdf values, g = 9.81."""
    m: float = 0.027
    L: float = 0.0397
    Jx: float = 1.4e-5
    Jy: float = 1.4e-5
    Jz: float = 2.17e-5
    kf: float = 3.16e-10
    km: float = 7.94e-12
    thrust_to_weight: float = 2.25
    g: float = 9.81
    drag: bool = False                 # linear drag, off by default
    drag_coeffs: tuple = None          # (Dx, Dy, Dz) in N/(m/s); required if drag is on

    def __post_init__(self):
        if self.drag and self.drag_coeffs is None:
            raise ValueError("drag=True needs drag_coeffs (Dx, Dy, Dz); no default is approved.")

    @property
    def c(self):
        """Yaw torque per newton of rotor thrust, km/kf (m)."""
        return self.km / self.kf

    @property
    def f_max(self):
        return self.thrust_to_weight * self.m * self.g / 4.0

    @property
    def J(self):
        return jnp.diag(jnp.array([self.Jx, self.Jy, self.Jz], jnp.float64))


def u_max(params):
    return params.f_max


def mixer(params):
    """M (4 × 4) with (T, τx, τy, τz) = M u for rotor thrusts u (see the module docstring)."""
    a = params.L / jnp.sqrt(2.0)
    c = params.c
    return jnp.array([[1.0, 1.0, 1.0, 1.0],
                      [-a, -a, a, a],
                      [-a, a, a, -a],
                      [-c, c, -c, c]], jnp.float64)


def rotation(eta):
    """R(η) = Rz(ψ) Ry(θ) Rx(φ): body to world."""
    phi, theta, psi = eta[0], eta[1], eta[2]
    cf, sf = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
        [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
        [-st, ct * sf, ct * cf]])


def euler_rate_matrix(eta):
    """W(η) with ω = W(η) η̇ for ZYX Euler angles."""
    phi, theta = eta[0], eta[1]
    cf, sf = jnp.cos(phi), jnp.sin(phi)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[1.0, 0.0, -st],
                      [0.0, cf, sf * ct],
                      [0.0, -sf, cf * ct]])


def euler_rate_matrix_inv(eta):
    """W(η)⁻¹ with η̇ = W(η)⁻¹ ω (singular at θ = ±π/2)."""
    phi, theta = eta[0], eta[1]
    cf, sf = jnp.cos(phi), jnp.sin(phi)
    ct, tt = jnp.cos(theta), jnp.tan(theta)
    return jnp.array([[1.0, sf * tt, cf * tt],
                      [0.0, cf, -sf],
                      [0.0, sf / ct, cf / ct]])


def f(x, u, params=QuadrotorParams()):
    """ẋ = f(x, u): the continuous-time dynamics (pure; JAX-differentiable)."""
    x = jnp.asarray(x, jnp.float64)
    u = jnp.asarray(u, jnp.float64)
    v, eta, w = x[3:6], x[6:9], x[9:12]
    wrench = mixer(params) @ u
    T, tau = wrench[0], wrench[1:]
    acc = rotation(eta) @ jnp.array([0.0, 0.0, T]) / params.m - jnp.array([0.0, 0.0, params.g])
    if params.drag:
        acc = acc - jnp.asarray(params.drag_coeffs, jnp.float64) * v / params.m
    J = params.J
    Jinv = jnp.diag(1.0 / jnp.diag(J))
    wdot = Jinv @ (tau - jnp.cross(w, J @ w))
    return jnp.concatenate([v, acc, euler_rate_matrix_inv(eta) @ w, wdot])


def rk4_step(x, u, params=QuadrotorParams(), dt=0.01):
    """One classical fourth-order Runge-Kutta step with u held constant (zero-order hold)."""
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def hover_equilibrium(params=QuadrotorParams(), position=(0.0, 0.0, 0.0), yaw=0.0):
    """(x0, u0): at rest at `position` with yaw `yaw`, each rotor carrying m g / 4."""
    x0 = jnp.zeros(N_STATE, jnp.float64).at[0:3].set(jnp.asarray(position, jnp.float64)).at[8].set(yaw)
    u0 = jnp.full(N_INPUT, params.m * params.g / 4.0, jnp.float64)
    return x0, u0


def linearize(x, u, params=QuadrotorParams()):
    """(A, B) = (∂f/∂x, ∂f/∂u) at (x, u), by forward-mode autodiff."""
    A = jax.jacfwd(f, argnums=0)(x, u, params)
    B = jax.jacfwd(f, argnums=1)(x, u, params)
    return A, B


def linearize_step(x, u, params=QuadrotorParams(), dt=0.01):
    """Jacobians of the RK4 map x_{t+1} = rk4_step(x_t, u_t) with respect to x and u."""
    Ad = jax.jacfwd(rk4_step, argnums=0)(x, u, params, dt)
    Bd = jax.jacfwd(rk4_step, argnums=1)(x, u, params, dt)
    return Ad, Bd
