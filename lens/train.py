from typing import Callable, Dict, List, Tuple

import jax


def train(
    params,
    model_fn: Callable,
    loss_fn: Callable,
    train_data: Tuple,
    val_data: Tuple,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 20,
    lens_logger: Callable | None = None,
):
    best_params = params
    best_val = float("inf")
    stale = 0
    history: List[Dict[str, float]] = []

    def objective(p, batch):
        pred = model_fn(p, batch[0])
        return loss_fn(pred, batch[1])

    grad_fn = jax.grad(objective)

    for epoch in range(epochs):
        grads = grad_fn(params, train_data)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

        train_loss = float(objective(params, train_data))
        val_loss = float(objective(params, val_data))

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if lens_logger is not None:
            row.update(lens_logger(params))
        history.append(row)

        if val_loss < best_val:
            best_val = val_loss
            best_params = params
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return best_params, history
