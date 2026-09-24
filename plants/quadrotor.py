import jax.numpy as jnp

G = 9.81
MASS = 1.0
INERTIA = jnp.array([0.02, 0.02, 0.04])


def _rotation_matrix(roll, pitch, yaw):
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def dynamics(state, control, mass=MASS, inertia=INERTIA, g=G):
    x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r = state
    thrust, tau_x, tau_y, tau_z = control

    rot = _rotation_matrix(roll, pitch, yaw)
    thrust_world = rot @ jnp.array([0.0, 0.0, thrust]) / mass
    accel = thrust_world - jnp.array([0.0, 0.0, g])

    angle_dot = jnp.array([p, q, r])
    omega = jnp.array([p, q, r])
    tau = jnp.array([tau_x, tau_y, tau_z])
    omega_dot = (tau - jnp.cross(omega, inertia * omega)) / inertia

    return jnp.array([
        vx,
        vy,
        vz,
        accel[0],
        accel[1],
        accel[2],
        angle_dot[0],
        angle_dot[1],
        angle_dot[2],
        omega_dot[0],
        omega_dot[1],
        omega_dot[2],
    ])


def step(state, control, dt=0.01):
    return state + dt * dynamics(state, control)
