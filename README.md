# Basin-conditioned soft physics for LSTM streamflow prediction

Reference implementation for the paper. This is the training code, not the
study: no data, no result ledgers, no hash pinning, no orchestration. It is
here so you can read what was actually optimised and run it on your own data.

## What's here

```
configs/          hyperparameters. base.yaml + one file per arm.
src/data.py       windowed dataset, outage masks
src/model.py      the LSTM
src/constraints.py the seven soft-physics penalties
src/conditioning.py basin attributes -> per-constraint weights
src/train.py      training loop
src/metrics.py    NSE, KGE
train.py          CLI
```

## The idea in one paragraph

A single-layer LSTM predicts daily discharge from daily forcing. On top of the
usual masked mean-squared error we add differentiable penalties that encode
things a hydrologist expects to be true: rain should be followed by rising flow,
snowmelt should show up as flow, you can't route out more water than came in.
Each penalty gets a weight. The question the paper asks is whether those weights
should be **conditioned on the catchment** (snowy basins should care about melt;
arid basins shouldn't) rather than applied uniformly everywhere.

## The five arms

| arm | constraints | weights |
|---|---|---|
| `baseline` | none | — |
| `uniform_physics` | 7 physics penalties | equal |
| `conditioned_physics` | 7 physics penalties | from basin attributes |
| `uniform_generic` | smoothness only | equal, same budget |
| `conditioned_generic` | smoothness only | budget scaled per basin by attributes |

The generic arms exist so that any benefit can be attributed to the *content* of
the physics rather than to the mere presence of a regulariser.

`uniform_physics`, `conditioned_physics` and `uniform_generic` spread the same
total weight budget (7) across their constraints. `conditioned_generic`
conditions *strength* rather than allocation: its single smoothness weight is
7 times the basin's mean raw applicability across the seven physics
constraints, so it varies from basin to basin. Penalty strength is also set by
each arm's `lam`, which differs between arms (`conditioned_physics` 0.001,
`uniform_physics` 0.01, `uniform_generic` 0.01, `conditioned_generic` 0.1). So
the arms are not matched on overall penalty strength.

## Run it

```bash
pip install -r requirements.txt

# smoke test on synthetic data -- no CAMELS needed. --epochs caps the epoch
# count; each epoch is still 40 steps of a 256-unit LSTM over 365-day windows,
# so expect it to be slow on CPU.
python train.py --arm conditioned_physics --synthetic --epochs 5

# real run -- --data-dir holds <basin>.csv and attributes.csv (see below),
# not the raw CAMELS directory layout
python train.py --arm conditioned_physics --basin 01022500 \
                --data-dir /path/to/prepared_data --outage-length 30 --missing 0.5
```

`--outage-length` and `--missing` are the two knobs the paper sweeps: how long
each simulated sensor outage lasts (days) and what fraction of discharge labels
it removes. They mask the *target* only; forcing is never masked.

## Data you need to supply

In `--data-dir`: one CSV per basin named `<basin_id>.csv`, with a daily `date`
column and columns `prcp, tmax, tmin, srad, vp, dayl, swe, flow_mm`, plus an
`attributes.csv` with `basin_id, frac_snow, aridity, max_water_content,
area_km2`. `area_km2` is read but not used by this code. `src/data.py`
documents the exact expectations.

The paper used CAMELS-US, but the SWE column shipped with CAMELS is all zeros.
The paper replaced it with a degree-day SWE proxy (4.5 mm/°C/day); this code
does not compute that proxy and uses whatever is in your `swe` column. If you
pass CAMELS SWE as-is, the melt penalty never activates.

## What this deliberately leaves out

This is a simplified reference, not the screening pipeline, and it does not
reproduce the paper's numbers. Not included:

* the regional scope and its two learned-gate arms
* the SWE degree-day proxy
* the per-basin parameter conditioning: response-lag prior τ(b), wetness
  horizon H(b), and mass-balance bound β(b). Here the lag window is fixed at
  1-5 days, the wetness index is always 14 days, and the volume bound is a
  constant 1.5 × water in
* the frozen mask files and their SHA-256 manifest
* the pre-registration machinery and integrity checks
* the noise-floor calibration and bootstrap endpoint tests
* the altered-forcing evaluation
* the screening and descriptive readouts, and the figure-generation scripts
If you want to reproduce the *finding* rather than train a model, the paper is
the reference.

## Notes that will save you time

* `--missing 1.0` removes every discharge label. Training still runs, because the
  physics terms are the only gradient left, but there is nothing to early-stop
  on, so it falls back to a fixed epoch count. Expect poor skill. That cell is a
  descriptive limit case in the paper, not a comparison.
* The outage masks the train/validation period only. Test is left intact --
  otherwise you'd be measuring the mask, not the model.
* `lam` is per-arm and lives in the arm config. Choose it once on validation and
  leave it alone; re-tuning it after seeing test results is the easiest way to
  turn a null into a false positive.
* Constraints score only the last 90 days of each 365-day window. The first 275
  days are LSTM warm-up and are deliberately unpenalised.

## Citation

If you use this code, please cite it. `CITATION.cff` has the metadata; the
archived version's DOI is on its Zenodo record.

## License

MIT -- see `LICENSE`.
