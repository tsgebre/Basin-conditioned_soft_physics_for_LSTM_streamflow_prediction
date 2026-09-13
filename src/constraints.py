"""The soft-physics penalties.

Each one is a differentiable "this shouldn't happen" term. They all return a
scalar >= 0 and are added to the MSE, weighted. All of them score only the last
`loss_days` of the window -- `batch["loss_start"]` marks where that begins, so
the LSTM's warm-up period is never penalised.

"""

import torch
import torch.nn as nn

RAIN_MM = 0.1          # above this, call it a rain day
EPS = 1e-8
K = 3                  # response horizon in days, shared by the trend terms


def _masked_mean(x, mask):
    """Mean over True entries; 0 if nothing qualifies (not NaN -- it's a loss)."""
    m = mask.to(x.dtype)
    total = m.sum()
    if total < 1:
        return x.sum() * 0.0
    return (x * m).sum() / total


def _falls(y, ls, hi, k=K):
    """How much flow drops between t and t+k, clamped at 0. Shape (B, hi-ls+1)."""
    return torch.relu(-(y[:, ls + k : hi + 1 + k] - y[:, ls : hi + 1]))


class TrendAlignment(nn.Module):
    """Rain today -> flow shouldn't be lower in 3 days."""

    def forward(self, y, b):
        ls, hi = b["loss_start"], y.shape[1] - 1 - K
        if hi < ls:
            return y.sum() * 0.0
        return _masked_mean(_falls(y, ls, hi), b["prcp"][:, ls:hi + 1] > RAIN_MM)


class TempGatedTrend(nn.Module):
    """Same, but only when it's warm enough that rain isn't falling as snow."""

    def forward(self, y, b):
        ls, hi = b["loss_start"], y.shape[1] - 1 - K
        if hi < ls:
            return y.sum() * 0.0
        warm = (b["tmax"][:, ls:hi + 1] + b["tmin"][:, ls:hi + 1]) / 2 > 0.0
        return _masked_mean(_falls(y, ls, hi), (b["prcp"][:, ls:hi + 1] > RAIN_MM) & warm)


class MeltResponse(nn.Module):
    """Snow on the ground + above freezing + no rain -> melt should show up."""

    def forward(self, y, b):
        ls, hi = b["loss_start"], y.shape[1] - 1 - K
        if hi < ls:
            return y.sum() * 0.0
        s = slice(ls, hi + 1)
        cond = ((b["swe"][:, s] > 1.0)
                & ((b["tmax"][:, s] + b["tmin"][:, s]) / 2 > 0.0)
                & (b["prcp"][:, s] <= RAIN_MM))
        return _masked_mean(_falls(y, ls, hi), cond)


class AntecedentWetness(nn.Module):
    """A wet catchment responds harder to the same rain than a dry one."""

    def forward(self, y, b):
        ls, hi = b["loss_start"], y.shape[1] - 1 - K
        if hi < ls:
            return y.sum() * 0.0
        s = slice(ls, hi + 1)
        wet = b["api14"][:, s] > b["api14"][:, s].median()
        return _masked_mean(_falls(y, ls, hi), (b["prcp"][:, s] > RAIN_MM) & wet)


class AdaptiveLag(nn.Module):
    """Rain should show up SOMEWHERE in the next few days, not necessarily day 3.

    Takes the best of horizons 1..K_MAX, so a slow catchment isn't punished for
    being slow. Fixed kernel -- nothing here is learnable.
    """

    K_MAX = 5

    def forward(self, y, b):
        ls, hi = b["loss_start"], y.shape[1] - 1 - self.K_MAX
        if hi < ls:
            return y.sum() * 0.0
        rain = b["prcp"][:, ls:hi + 1] > RAIN_MM
        per_k = torch.stack([_falls(y, ls, hi, k) for k in range(1, self.K_MAX + 1)])
        return _masked_mean(per_k.min(dim=0).values, rain)


class VolumePlausibility(nn.Module):
    """You can't route out more water than rain + melt brought in.

    `beta` is the slack: > 1 because storage released from earlier months is
    real, and because our melt estimate is crude.
    """

    def forward(self, y, b):
        ls, T = b["loss_start"], y.shape[1]
        melt = torch.relu(b["swe"][:, ls - 1:T - 1] - b["swe"][:, ls:T])
        water_in = (b["prcp"][:, ls:T] + melt).sum(dim=1)
        bound = b.get("volume_beta", 1.5) * water_in
        water_out = torch.relu(y[:, ls:T]).sum(dim=1)
        return (torch.relu(water_out - bound) / (bound + EPS)).mean()


class NonnegMonotone(nn.Module):
    """Flow is never negative, and it doesn't rise on a dry day with no snow."""

    def forward(self, y, b):
        ls, T = b["loss_start"], y.shape[1]
        pen = torch.relu(-y[:, ls:T]).mean()
        hi = T - 1 - 1
        if hi < ls:
            return pen
        s = slice(ls, hi + 1)
        dry = (b["prcp"][:, s] <= RAIN_MM) & (b["swe"][:, s] <= 1.0)
        rises = torch.relu(y[:, ls + 1:hi + 2] - y[:, s])
        return pen + _masked_mean(rises, dry)


class Smoothness(nn.Module):
    """The generic control. Penalises jaggedness. Contains no hydrology at all,
    which is exactly why it's here: it's the strength-matched null."""

    def forward(self, y, b):
        ls = b["loss_start"]
        d = y[:, ls + 1:] - y[:, ls:-1]
        return (d ** 2).mean()


REGISTRY = {
    "trend_alignment": TrendAlignment,
    "temp_gated_trend": TempGatedTrend,
    "melt_response": MeltResponse,
    "antecedent_wetness": AntecedentWetness,
    "adaptive_lag": AdaptiveLag,
    "volume_plausibility": VolumePlausibility,
    "nonneg_monotone": NonnegMonotone,
    "smoothness": Smoothness,
}


def build(names):
    return {n: REGISTRY[n]() for n in names}
