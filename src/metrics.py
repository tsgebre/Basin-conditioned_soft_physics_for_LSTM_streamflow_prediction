"""NSE and KGE. Both skip missing days."""

import numpy as np

EPS = 1e-12


def nse(obs, sim, mask=None):
    """1 = perfect, 0 = no better than predicting the mean, negative = worse."""
    obs, sim = _valid(obs, sim, mask)
    denom = ((obs - obs.mean()) ** 2).sum()
    if denom < EPS:
        return float("nan")          # flat observations -- NSE is undefined
    return float(1.0 - ((sim - obs) ** 2).sum() / denom)


def kge(obs, sim, mask=None):
    """Gupta et al. 2009. Splits skill into correlation, variability, bias."""
    obs, sim = _valid(obs, sim, mask)
    if obs.std() < EPS or sim.std() < EPS:
        return float("nan")          # no variance to correlate
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / (obs.mean() + EPS))
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _valid(obs, sim, mask):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    keep = np.isfinite(obs) & np.isfinite(sim)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    return obs[keep], sim[keep]
