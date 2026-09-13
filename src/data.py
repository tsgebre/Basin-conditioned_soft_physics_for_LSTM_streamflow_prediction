"""Loading, windowing, and the outage masks.

Expected input: one CSV per basin, daily, named <basin_id>.csv, with columns

    date, prcp, tmax, tmin, srad, vp, dayl, swe, flow_mm

plus an attributes.csv with basin_id, frac_snow, aridity, max_water_content,
area_km2. Missing flow is NaN (or -999, which we convert).

Three derived features get added here rather than asked for: day-of-year sin/cos, a Hargreaves PET estimate, and a 14-day antecedent precipitation
index. They're cheap and every arm uses them.

The outage mask is the part worth reading. Real gauges don't fail one day at a time.
"""

import numpy as np
import pandas as pd

FEATURES = ["prcp", "tmax", "tmin", "srad", "vp", "dayl",
            "doy_sin", "doy_cos", "pet", "api14", "swe"]
MISSING = -999.0


# --------------------------------------------------------------------------
# outage masks
# --------------------------------------------------------------------------
def outage_mask(n_days, length_days, missing_frac, valid=None, seed=0):
    """True where the discharge label is hidden.

    Drops random contiguous blocks of `length_days` until `missing_frac` of the
    valid labels are gone. Blocks may overlap; the last one can overshoot, which
    is why the achieved fraction is returned rather than assumed.
    """
    rng = np.random.default_rng(seed)
    valid = np.ones(n_days, bool) if valid is None else np.asarray(valid, bool)
    masked = np.zeros(n_days, bool)
    target = int(round(missing_frac * valid.sum()))
    if target <= 0 or length_days < 1:
        return masked, 0.0
    hidden = 0
    for _ in range(100 * n_days):                 # generous cap, never hit in practice
        if hidden >= target:
            break
        start = int(rng.integers(0, max(1, n_days - length_days + 1)))
        block = slice(start, start + length_days)
        hidden += int(np.count_nonzero(valid[block] & ~masked[block]))
        masked[block] = True
    achieved = float(np.count_nonzero(masked & valid) / max(1, valid.sum()))
    return masked, achieved


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_basin(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    df["flow_mm"] = df["flow_mm"].replace(MISSING, np.nan)
    return add_derived(df)


def add_derived(df):
    doy = df.index.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    tmean = (df["tmax"] + df["tmin"]) / 2
    # Hargreaves: rough, but it only has to rank basins consistently
    df["pet"] = (0.0023 * (tmean + 17.8)
                 * np.sqrt((df["tmax"] - df["tmin"]).clip(lower=0))
                 * df["srad"].clip(lower=0) * 0.408).clip(lower=0)
    df["api14"] = df["prcp"].rolling(14, min_periods=1).sum()
    return df


class BasinWindows:
    """Windowed samples for one basin. Standardised on the training split only."""

    def __init__(self, df, cfg, mask=None):
        self.win = cfg["data"]["window_len"]
        self.loss_start = self.win - cfg["data"]["loss_days"]
        self.df = df

        tr0, tr1 = cfg["data"]["train"]
        train = df.loc[tr0:tr1]
        self.mu = train[FEATURES].mean()
        self.sd = train[FEATURES].std().replace(0, 1.0)

        self.x = ((df[FEATURES] - self.mu) / self.sd).to_numpy(np.float32)
        self.y = df["flow_mm"].to_numpy(np.float32)
        self.aux = {k: df[k].to_numpy(np.float32)
                    for k in ("prcp", "tmax", "tmin", "swe", "api14")}

        # a day is usable as a target if it's observed AND not hidden by the outage
        self.valid = np.isfinite(self.y)
        if mask is not None:
            self.valid &= ~np.asarray(mask, bool)

        self.splits = {k: self._range(df, cfg["data"][k]) for k in ("train", "val", "test")}

    def _range(self, df, window):
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        idx = np.flatnonzero((df.index >= lo) & (df.index <= hi))
        # a window ends at its last day, so the first legal end is win-1 in
        ends = idx[idx >= self.win - 1]
        if len(ends) == 0:
            raise ValueError(
                f"no {self.win}-day windows land in {window[0]}..{window[1]}; "
                f"the record runs {df.index[0].date()}..{df.index[-1].date()}. "
                "Either the CSV is too short or configs/base.yaml's windows "
                "don't match your data.")
        return ends

    def sample(self, split, size, rng):
        ends = self.splits[split]
        picks = rng.choice(ends, size=min(size, len(ends)), replace=len(ends) < size)
        return self._gather(picks)

    def all_windows(self, split, stride=90):
        return self._gather(self.splits[split][::stride])

    def _gather(self, ends):
        sl = [slice(e - self.win + 1, e + 1) for e in ends]
        batch = {
            "x": np.stack([self.x[s] for s in sl]),
            "y": np.stack([self.y[s] for s in sl]),
            "valid": np.stack([self.valid[s] for s in sl]),
            "loss_start": self.loss_start,
        }
        for k, v in self.aux.items():
            batch[k] = np.stack([v[s] for s in sl])
        return batch


# --------------------------------------------------------------------------
# synthetic basin, for the smoke test
# --------------------------------------------------------------------------
def synthetic_basin(n_days=12511, start="1980-10-01", seed=0):
    """Fake but hydrologically plausible: seasonal temperature, poisson rain,
    snow accumulating below freezing, flow as a leaky bucket. Enough to prove
    the code runs end to end without shipping anyone's data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range(start, periods=n_days, freq="D")   # covers the config windows
    tmean = 10 + 12 * np.sin(2 * np.pi * (t - 100) / 365.25) + rng.normal(0, 3, n_days)
    prcp = rng.gamma(0.6, 4.0, n_days) * (rng.random(n_days) < 0.35)

    swe, store, flow = np.zeros(n_days), 0.0, np.zeros(n_days)
    pack = 0.0
    for i in range(n_days):
        snow = prcp[i] if tmean[i] < 0 else 0.0
        rain = prcp[i] - snow
        melt = min(pack, max(0.0, 2.0 * tmean[i])) if tmean[i] > 0 else 0.0
        pack = pack + snow - melt
        swe[i] = pack
        store = store + rain + melt
        flow[i] = 0.12 * store
        store -= flow[i]

    df = pd.DataFrame({
        "prcp": prcp, "tmax": tmean + 5, "tmin": tmean - 5,
        "srad": 200 + 100 * np.sin(2 * np.pi * (t - 80) / 365.25),
        "vp": 500 + 300 * np.sin(2 * np.pi * (t - 80) / 365.25),
        "dayl": 40000 + 12000 * np.sin(2 * np.pi * (t - 80) / 365.25),
        "swe": swe, "flow_mm": flow + rng.normal(0, 0.05, n_days).clip(0),
    }, index=dates)
    df.index.name = "date"
    attrs = {"frac_snow": 0.35, "aridity": 0.9, "max_water_content": 0.5}
    return add_derived(df), attrs
