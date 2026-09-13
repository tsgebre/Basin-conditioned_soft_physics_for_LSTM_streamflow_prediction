#!/usr/bin/env python3
"""CLI. Trains one arm on one basin under one outage setting.

    python train.py --arm conditioned_physics --synthetic
    python train.py --arm uniform_physics --basin 01022500 --data-dir ./camels \
                    --outage-length 30 --missing 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import load_basin, outage_mask, synthetic_basin
from src.metrics import kge, nse
from src.train import evaluate, train

ROOT = Path(__file__).resolve().parent


def load_config(arm):
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
    arm_cfg = yaml.safe_load((ROOT / "configs" / "arms" / f"{arm}.yaml").read_text())
    return cfg, arm_cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    choices=["baseline", "uniform_physics", "conditioned_physics",
                             "uniform_generic", "conditioned_generic"])
    ap.add_argument("--basin", help="basin id; looks for <data-dir>/<basin>.csv")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data -- no CAMELS needed")
    ap.add_argument("--outage-length", type=int, help="days per outage block")
    ap.add_argument("--missing", type=float, help="fraction of labels removed, 0-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, help="override max_epochs (handy for a quick look)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, help="write a small json result here")
    args = ap.parse_args()

    cfg, arm_cfg = load_config(args.arm)
    if args.outage_length is not None:
        cfg["outage"]["length_days"] = args.outage_length
    if args.missing is not None:
        cfg["outage"]["missing"] = args.missing
    if args.epochs is not None:
        cfg["optim"]["max_epochs"] = args.epochs
        cfg["optim"]["fixed_epochs"] = args.epochs

    if args.synthetic:
        df, attrs = synthetic_basin(seed=args.seed)
        basin = "synthetic"
    else:
        if not (args.basin and args.data_dir):
            ap.error("need --basin and --data-dir, or --synthetic")
        df = load_basin(args.data_dir / f"{args.basin}.csv")
        table = pd.read_csv(args.data_dir / "attributes.csv", dtype={"basin_id": str})
        attrs = table.set_index("basin_id").loc[args.basin].to_dict()
        basin = args.basin

    # mask the training/validation period only -- test stays untouched, or the
    # whole evaluation would be measuring the mask instead of the model
    mask = None
    if cfg["outage"]["missing"] > 0:
        trainval_end = pd.Timestamp(cfg["data"]["val"][1])
        n = len(df)
        in_trainval = np.asarray(df.index <= trainval_end)
        m, achieved = outage_mask(n, cfg["outage"]["length_days"],
                                  cfg["outage"]["missing"],
                                  valid=in_trainval & np.isfinite(df["flow_mm"].to_numpy()),
                                  seed=cfg["outage"]["seed"])
        mask = m & in_trainval
        print(f"outage: {cfg['outage']['length_days']}d blocks, "
              f"target {cfg['outage']['missing']:.0%}, achieved {achieved:.2%}")

    print(f"arm={args.arm} basin={basin} device={args.device}")
    model, ds, info = train(df, attrs, cfg, arm_cfg, mask=mask,
                            device=args.device, seed=args.seed)

    test_nse = evaluate(model, ds, "test", args.device)
    b = ds.all_windows("test", stride=90)
    ls = b["loss_start"]
    with torch.no_grad():
        pred = model(torch.as_tensor(b["x"], device=args.device))
    test_kge = kge(b["y"][:, ls:].ravel(),
                   pred[:, ls:].cpu().numpy().ravel(),
                   b["valid"][:, ls:].ravel())

    result = {"arm": args.arm, "basin": basin, "seed": args.seed,
              "outage": dict(cfg["outage"]), "lam": arm_cfg["loss"]["lam"],
              "val_nse": info["val_nse"], "test_nse": test_nse, "test_kge": test_kge,
              "epochs_run": info["epochs_run"], "early_stopped": bool(info["early_stopped"])}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
