"""P-I data: sampling around a trim, the scaled-increment target, standardisation,
ground-truth Jacobians and storage with a SHA-256 manifest.

Generic in the plant: a `Plant` is a discrete step (x, u) -> x_next with its trim
(x0, u0), input bounds and time step, so tests can use synthetic plants. `quadrotor_plant`
builds the P-I plant from plants/quadrotor.py.

Target (docs/plan.md): y = (x_{t+1} − x_t)/dt for z = (x_t, u_t), in standardised
coordinates Z = (z − μ_z)/σ_z, Y = (y − μ_y)/σ_y, with μ, σ per coordinate from the
training split. Ground truth: J = ∂y/∂z by forward-mode autodiff, in standardised
coordinates J_std = diag(1/σ_y) J diag(σ_z).
"""

import dataclasses
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from plants import quadrotor as quad  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Plant:
    step: object            # (x, u) -> x_next, JAX-differentiable
    x0: np.ndarray
    u0: np.ndarray
    u_lo: np.ndarray
    u_hi: np.ndarray
    dt: float
    f: object = None        # the continuous vector field (x, u) -> ẋ, if known (hover check's sign mask)

    @property
    def n_x(self):
        return len(self.x0)

    @property
    def n_u(self):
        return len(self.u0)


def quadrotor_plant(cfg_plant):
    params = quad.QuadrotorParams(g=float(cfg_plant["g"]), drag=bool(cfg_plant["drag"]))
    dt = float(cfg_plant["dt"])
    x0, u0 = quad.hover_equilibrium(params)
    return Plant(step=lambda x, u: quad.rk4_step(x, u, params, dt), x0=np.asarray(x0), u0=np.asarray(u0),
                 u_lo=np.zeros(4), u_hi=np.full(4, params.f_max), dt=dt,
                 f=lambda x, u: quad.f(x, u, params)), params


def quadrotor_half_widths(hw, u0):
    """Half-widths of the sampling box from the config (state order of plants/quadrotor)."""
    a, deg = np.ones(3), np.pi / 180.0
    hx = np.concatenate([hw["position"] * a, hw["velocity"] * a,
                         [hw["roll_pitch_deg"] * deg, hw["roll_pitch_deg"] * deg, hw["yaw_deg"] * deg],
                         hw["body_rate"] * a])
    hu = hw["thrust_frac"] * np.asarray(u0)
    return hx, hu


def sample(plant, hx, hu, n, rng):
    """i.i.d. uniform in the box x0 ± hx, u0 ± hu; inputs clipped to [u_lo, u_hi]."""
    x = plant.x0 + rng.uniform(-1.0, 1.0, (n, plant.n_x)) * hx
    u = np.clip(plant.u0 + rng.uniform(-1.0, 1.0, (n, plant.n_u)) * hu, plant.u_lo, plant.u_hi)
    return x, u


def target_fn(plant):
    """y(z) = (step(x, u) − x)/dt for z = (x, u)."""
    def y(z):
        x, u = z[:plant.n_x], z[plant.n_x:]
        return (plant.step(x, u) - x) / plant.dt
    return y


def targets_and_jacobians(plant, x, u, with_jac=True):
    z = jnp.asarray(np.concatenate([x, u], 1), jnp.float64)
    yf = target_fn(plant)
    y = np.asarray(jax.vmap(yf)(z))
    J = np.asarray(jax.vmap(jax.jacfwd(yf))(z)) if with_jac else None
    return np.asarray(z), y, J


def standardiser(z_train, y_train):
    return dict(mu_z=z_train.mean(0), sd_z=z_train.std(0), mu_y=y_train.mean(0), sd_y=y_train.std(0))


def to_std_jacobian(J, sc):
    """J_std = diag(1/σ_y) J diag(σ_z) for J (n, n_y, n_z)."""
    return J * sc["sd_z"][None, None, :] / sc["sd_y"][None, :, None]


def make_dataset(plant, hx, hu, sizes, seed):
    """Splits train, val, test (in that order from one generator). Returns a dict of
    splits, each with raw z, y and standardised Z, Y, J_std; and the scalers, computed
    on the training split."""
    rng = np.random.default_rng(seed)
    raw = {}
    for split in ("train", "val", "test"):
        x, u = sample(plant, hx, hu, int(sizes[split]), rng)
        raw[split] = targets_and_jacobians(plant, x, u)
    sc = standardiser(raw["train"][0], raw["train"][1])
    out = dict(scalers=sc)
    for split, (z, y, J) in raw.items():
        out[split] = dict(z=z, y=y, Z=(z - sc["mu_z"]) / sc["sd_z"], Y=(y - sc["mu_y"]) / sc["sd_y"],
                          J_std=to_std_jacobian(J, sc))
    z_trim = np.concatenate([plant.x0, plant.u0])
    out["trim"] = dict(z=z_trim, Z=(z_trim - sc["mu_z"]) / sc["sd_z"])
    return out


def save_dataset(ds, data_dir, manifest_path):
    """One .npz per split plus scalers.npz under data_dir (outside git); a manifest of
    file names, sizes and SHA-256s (provenance.write_manifest) at manifest_path."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "r6_tdmpc2"))
    import provenance as pv
    os.makedirs(data_dir, exist_ok=True)
    files = []
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.npz")
        np.savez(path, **ds[split])
        files.append(path)
    path = os.path.join(data_dir, "scalers.npz")
    np.savez(path, **ds["scalers"], trim_z=ds["trim"]["z"], trim_Z=ds["trim"]["Z"])
    files.append(path)
    return pv.write_manifest(manifest_path, files, data_dir)


def load_dataset(data_dir, manifest_path):
    """Load the splits after verifying every file against the manifest (SHA-256)."""
    sys.path.insert(0, os.path.join(ROOT, "experiments", "r6_tdmpc2"))
    import provenance as pv
    rows = {r["file"]: r for r in pv.read_manifest(manifest_path)}
    out = {}
    for name in ("train", "val", "test", "scalers"):
        path = os.path.join(data_dir, f"{name}.npz")
        pv.verify_sha256(path, rows[f"{name}.npz"]["sha256"], f"P-I data {name}.npz")
        d = np.load(path)
        out[name] = {k: d[k] for k in d.files}
    sc = out.pop("scalers")
    out["trim"] = dict(z=sc.pop("trim_z"), Z=sc.pop("trim_Z"))
    out["scalers"] = sc
    return out
