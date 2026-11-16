"""Exploratory: does a neural-network stage 2 beat the linear stage 2 on identical fixtures?

Not a pre-registered test. Settings are chosen on the dev seasons (2022-24) only, then reported on
2024-26 and 2026-27 GW1-9, which earlier work has also used for development (brief §6.8). A gain here
would justify a pre-registered test, not adoption.

    python prototypes/run_neural_net.py            # writes results/neural_net/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import DEV_SEASONS, block_bootstrap, per_match_logloss, run_variant, summary  # noqa: E402
from tabet.neural import NNParams, run_nn_variant  # noqa: E402

OUT = ROOT / "results" / "neural_net"
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]                 # V1x inputs
XI = ["f_xi_position", "f_backline_opp"]                      # + confirmed-XI inputs = V2x
RICH = ADJ + XI + ["f_xgf", "f_xga_opp"] + [f"{p}_{c}" for p in ("s", "so") for c in
        ("shots_for", "shots_against", "corners_for", "corners_against", "fouls",
         "xg_per_shot_for", "xg_per_shot_against")]
PERIODS = {"2024-26": ("2024-25", "2025-26"), "2026-27 GW1-9": ("2026-27",)}
GRID = [NNParams(hidden=h, weight_decay=wd, epochs=300, seeds=3) for h in (8, 16) for wd in (1e-2, 1e-1)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_pickle(ROOT / "data/processed/forecast_rows.pkl")
    rows["cutoff_utc"] = pd.to_datetime(rows["cutoff_utc"], utc=True)
    rows["kickoff_utc"] = pd.to_datetime(rows["kickoff_utc"], utc=True)
    res = {"note": "exploratory, not pre-registered; settings chosen on 2022-24 only", "tuning": {}, "periods": {}}

    # 1. choose network settings on the dev seasons, per input set
    best = {}
    for name, cols in (("NN-V1x", ADJ), ("NN-V2x", ADJ + XI), ("NN-rich", RICH)):
        scores = {}
        for p in GRID:
            t = time.time()
            ll = per_match_logloss(run_nn_variant(rows, cols, DEV_SEASONS, p)).mean()
            scores[str(p)] = ll
            print(f"tune {name:8s} {p} dev log-loss {ll:.4f} ({time.time() - t:.0f}s)", flush=True)
        best[name] = min(GRID, key=lambda q: scores[str(q)])
        best[name] = NNParams(**{**best[name].__dict__, "seeds": 5})   # more seeds for the final runs
        res["tuning"][name] = {"scores": scores, "chosen": str(best[name])}
        print(f"  -> {name}: {best[name]}", flush=True)

    # 2. compare on later seasons, identical fixtures
    linear = {"V0": ([], 0.0), "V1x": (ADJ, 1.0), "V2x": (ADJ + XI, 100.0), "linear-rich": (RICH, 100.0)}
    for pname, seasons in PERIODS.items():
        preds = {k: run_variant(rows, c, l2, seasons) for k, (c, l2) in linear.items()}
        ids = preds["V0"]["match_id"]
        for name, cols in (("NN-V1x", ADJ), ("NN-V2x", ADJ + XI), ("NN-rich", RICH)):
            preds[name] = run_nn_variant(rows, cols, seasons, best[name])
        preds = {k: v.set_index("match_id").loc[ids].reset_index() for k, v in preds.items()}
        ko = pd.to_datetime(preds["V0"]["kickoff_utc"], utc=True).reset_index(drop=True)
        L = {k: per_match_logloss(v) for k, v in preds.items()}
        out = {"n": int(len(ids)), "summary": {k: summary(v) for k, v in preds.items()}, "contrasts": {}}
        for a, b in (("NN-V1x", "V1x"), ("NN-V2x", "V2x"), ("NN-rich", "linear-rich"), ("NN-rich", "V2x"),
                     ("V1x", "V0"), ("V2x", "V0")):
            m, lo, hi = block_bootstrap(L[a], L[b], ko, 14)
            out["contrasts"][f"{a} - {b}"] = {"mean": m, "ci95": [lo, hi]}
        res["periods"][pname] = out
        print(f"\n{pname} (n={len(ids)}), log-loss:")
        for k, s in out["summary"].items():
            print(f"  {k:12s} {s['log_loss']:.4f}")
        for k, c in out["contrasts"].items():
            print(f"  {k:22s} {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
