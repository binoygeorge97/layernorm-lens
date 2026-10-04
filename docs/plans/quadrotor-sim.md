# Plan: quadrotor simulator (queue item 2, session 3)

Author-approved queue of 4 Oct 2026; branch `quadrotor-sim`.

## Files

- `plants/quadrotor.py`: JAX float64 plant.
  - A frozen `QuadrotorParams` dataclass holding the approved cf2x.urdf values, with
    g = 9.81.
  - The X-mixer `mixer(params)`: (T, τx, τy, τz) = M f.
  - A pure `f(x, u, params)` and `rk4_step(x, u, params, dt=0.01)`.
  - `hover_equilibrium(params)` and `linearize(x, u, params)` (jacfwd).
- `tests/test_quadrotor.py` covers:
  - hover f = 0 to 1e-12;
  - jacfwd against central differences at hover and at 20 random states;
  - sign conventions and the mixer;
  - the hover linearisation's textbook structure;
  - RK4's fourth order.
- `docs/DECISIONS.md` (new): the choices the spec leaves open.
- `docs/STATE.md`: repository layout, task (e) status and the questions section.

## Conventions

- **Frames:** the world frame is z-up with gravity −z. The body frame is FLU (x
  forward, y left, z up), with thrust along body +z.
- **Attitude:** ZYX Euler angles η = (φ, θ, ψ), R = Rz(ψ) Ry(θ) Rx(φ) mapping body to
  world. The body rates are ω = W(η) η̇.
- **Rotors:** positions and numbering from the cf2x.urdf prop links (0 front-right,
  1 rear-right, 2 rear-left, 3 front-left, at ±L/√2). Yaw signs follow
  gym-pybullet-drones' CF2X `_dynamics` (rotors 0 and 2 give −z torque, rotors 1 and
  3 give +z).
- **Inputs:** u = rotor thrusts in N, in [0, f_max], with f_max = 2.25 m g / 4.

## Test design

- Central differences use step 1e-6 with a relative tolerance of about 1e-7.
- RK4's order is checked on a smooth, non-trivial trajectory, against a reference at
  dt/64. The error ratio for halving dt should be about 16.
