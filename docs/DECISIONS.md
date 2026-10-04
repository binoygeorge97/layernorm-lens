# Decision log

Choices that the specs leave open, made under the author's standing mandate (CLAUDE.md;
queue of 4 Oct 2026), for asynchronous review. Newest last. Each entry gives the date,
the task, the choice, the reasoning, and the alternatives rejected.

## 2026-10-04, queue item 2 (quadrotor simulator, `plants/quadrotor.py`)

1. **Frames.** The world frame is z-up with gravity −z; the body frame is FLU (x
   forward, y left, z up), with thrust along body +z.
   - Why: gym-pybullet-drones, the source of the approved parameters, uses a z-up
     world, so this keeps that source's sign conventions.
   - Rejected: NED/FRD, the aerospace convention, which would flip every sign
     relative to the source.

2. **Rotor numbering, positions and yaw-torque signs** come from the cf2x.urdf prop
   links and gym-pybullet-drones' CF2X `_dynamics`. Rotors 0–3 are front-right,
   rear-right, rear-left and front-left, at ±L/√2; rotors 0 and 2 give −z torque,
   1 and 3 give +z.
   - Why: the mixer is then exactly equivalent to the source, and a test checks it
     against the URDF positions.
   - Rejected: Bitcraze's M1–M4 numbering, and a "+" configuration.

3. **Inputs are not clipped inside `f`.** f_max is exposed as `params.f_max`, and
   limits are left to callers and controllers.
   - Why: clipping in `f` would put kinks into the dynamics and corrupt the exact
     Jacobians that P-I needs.
   - Rejected: clipping inside `f`, and a smooth saturation.

4. **Linear drag has no default coefficients.** `drag=True` requires explicit
   (Dx, Dy, Dz).
   - Why: no linear drag coefficient is in the approved parameter set, and the URDF's
     drag is rotor-speed-dependent.
   - Rejected: deriving a linear coefficient from the URDF's rotor drag at hover,
     which would give about 5.6e-3 N/(m/s) for xy. That would be a parameter beyond
     the approved set (see the questions in STATE.md).

5. **u is held constant over each RK4 step** (zero-order hold), and the Euler
   singularity at θ = ±π/2 is not handled.
   - Why: the sampling and tests keep |φ|, |θ| < 60°.
   - Rejected: quaternions. They would change the 12-state spec.

6. **`linearize_step` was added:** the Jacobians of the RK4 map, beside `linearize`'s
   continuous (A, B).
   - Why: plan.md wants discrete-map values in the appendix.

7. **The RK4 order test uses a near-hover manoeuvre** (5% differential thrust,
   moderate rates) with dt = T/20, T/40 and T/80 (T = 0.2 s), against a reference at
   T/2560.
   - Why: with Crazyflie inertias, random full-range thrusts give angular
     accelerations of order 100 rad/s². Those are not yet in RK4's asymptotic regime
     at dt = 0.01: the observed error ratios were 66 and then 12.6.
