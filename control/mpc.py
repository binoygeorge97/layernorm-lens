import jax.numpy as jnp


def rollout(dynamics_fn, x0, controls, dt=0.01):
    xs = [x0]
    x = x0
    for u in controls:
        x = x + dt * dynamics_fn(x, u)
        xs.append(x)
    return jnp.stack(xs)


def shooting_mpc(dynamics_fn, x0, candidates, cost_fn, dt=0.01):
    best_cost = jnp.inf
    best_u = candidates[0][0]
    for seq in candidates:
        traj = rollout(dynamics_fn, x0, seq, dt=dt)
        c = cost_fn(traj, seq)
        if c < best_cost:
            best_cost = c
            best_u = seq[0]
    return best_u, best_cost
