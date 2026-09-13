"""Training loop.

Loss = masked MSE + lam * sum_c w_c * penalty_c

`w_c` comes from conditioning.py and sums to the same budget for every arm, so
`lam` is the only knob that changes how hard the physics pushes. Pick it on
validation, once, and freeze it -- tuning lam per arm after seeing results is
how a null quietly becomes a positive.

Early stopping watches masked validation NSE. When an outage is severe enough
that no validation labels survive, there's nothing to stop on, so we fall back
to a fixed epoch count.
"""

import numpy as np
import torch

from . import conditioning
from .constraints import build as build_constraints
from .data import FEATURES, BasinWindows
from .metrics import nse


def _to_torch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v if k == "loss_start" else torch.as_tensor(v, device=device)
    return out


def masked_mse(y_hat, y, valid, loss_start):
    """MSE over observed days in the scored tail only."""
    y_hat, y = y_hat[:, loss_start:], y[:, loss_start:]
    m = valid[:, loss_start:].float()
    total = m.sum()
    if total < 1:
        return y_hat.sum() * 0.0
    return (((y_hat - y) ** 2) * m).sum() / total


def evaluate(model, ds, split, device, stride=90):
    """Masked NSE on a split. No gradients, whole windows, scored tail only."""
    model.eval()
    b = _to_torch(ds.all_windows(split, stride), device)
    with torch.no_grad():
        pred = model(b["x"])
    ls = b["loss_start"]
    y = b["y"][:, ls:].cpu().numpy().ravel()
    p = pred[:, ls:].cpu().numpy().ravel()
    m = b["valid"][:, ls:].cpu().numpy().ravel()
    model.train()
    return nse(y, p, m)


def train(df, attrs, cfg, arm_cfg, mask=None, device="cpu", seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ds = BasinWindows(df, cfg, mask=mask)
    names = arm_cfg["loss"]["constraints"]
    penalties = build_constraints(names)
    weights = conditioning.weights_for(arm_cfg["loss"]["weighting"], names, attrs)
    lam = float(arm_cfg["loss"]["lam"])

    from .model import LSTMFlow
    model = LSTMFlow(len(FEATURES), cfg["model"]["hidden_dim"],
                     cfg["model"]["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])

    # no surviving validation labels -> nothing to early-stop on
    can_stop = ds.valid[ds.splits["val"]].sum() >= 60
    epochs = cfg["optim"]["max_epochs"] if can_stop else cfg["optim"]["fixed_epochs"]

    best, best_state, waited = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        for _ in range(cfg["optim"]["steps_per_epoch"]):
            b = _to_torch(ds.sample("train", cfg["optim"]["batch_size"], rng), device)
            y_hat = model(b["x"])
            loss = masked_mse(y_hat, b["y"], b["valid"], b["loss_start"])
            for name, pen in penalties.items():
                loss = loss + lam * weights.get(name, 0.0) * pen(y_hat, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if not can_stop:
            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  (fixed-epoch mode)")
            continue

        score = evaluate(model, ds, "val", device)
        if verbose:
            print(f"  epoch {epoch:3d}  loss {loss.item():.4f}  val NSE {score:.4f}")
        if np.isfinite(score) and score > best:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["optim"]["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ds, {"val_nse": best if np.isfinite(best) else None,
                       "epochs_run": epoch, "early_stopped": can_stop}
