"""Exploratory re-check of §5.26 (review 6, item 7e): was the neural network's loss partly the training
procedure? Three checks:
  1. early stopping on the most recent 15% of each refit's training matches, with a fall-back to plain
     Dixon-Coles when the network never beats it on those matches;
  2. inputs standardised on training rows only (already the case: run_nn_variant computes mu/sd on tr);
  3. where the original network's loss sits, by how many matches each refit had.
Settings chosen on 2022-24 only; reported on 2024-26 (n = 760). Not pre-registered.

    python prototypes/run_neural_recheck.py      # writes results/neural_net/recheck.json
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
from tabet.experiments import DEV_SEASONS, block_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.neural import NNParams, run_nn_variant  # noqa: E402

ADJ, XI = ["f_xgadj_att_90", "f_xgadj_def_opp_90"], ["f_xi_position", "f_backline_opp"]
TEST = ("2024-25", "2025-26")
OUT = ROOT / "results" / "neural_net"

rows = pd.read_pickle(ROOT / "data/processed/forecast_rows.pkl")
for c in ("cutoff_utc", "kickoff_utc"):
    rows[c] = pd.to_datetime(rows[c], utc=True)
res = {"note": "exploratory re-check of §5.26; settings chosen on 2022-24 only", "variants": {}}

for name, cols, l2 in (("V1x", ADJ, 1.0), ("V2x", ADJ + XI, 100.0)):
    lin = run_variant(rows, cols, l2, TEST)
    ids = lin["match_id"]
    L_lin = per_match_logloss(lin)
    ko = pd.to_datetime(lin["kickoff_utc"], utc=True).reset_index(drop=True)
    # 1. early stopping, tuned on dev
    grid = [NNParams(hidden=8, weight_decay=wd, epochs=600, seeds=3, early_stop=True) for wd in (0.01, 0.1, 1.0)]
    dev = {}
    for p in grid:
        t = time.time()
        dev[p.weight_decay] = float(per_match_logloss(run_nn_variant(rows, cols, DEV_SEASONS, p)).mean())
        print(f"{name} early-stop wd {p.weight_decay}: dev {dev[p.weight_decay]:.4f} ({time.time() - t:.0f}s)", flush=True)
    wd = min(dev, key=dev.get)
    es = run_nn_variant(rows, cols, TEST, NNParams(hidden=8, weight_decay=wd, epochs=600, seeds=5, early_stop=True))
    es = es.set_index("match_id").loc[ids].reset_index()
    L_es = per_match_logloss(es)
    m, lo, hi = block_bootstrap(L_es, L_lin, ko, 14)
    # 3. the original network (no early stopping), split by training size at each refit
    orig = run_nn_variant(rows, cols, TEST, NNParams(hidden=8, weight_decay=0.1, epochs=300, seeds=5))
    orig = orig.set_index("match_id").loc[ids].reset_index()
    d0 = per_match_logloss(orig) - L_lin
    n_tr = orig["stage2_train_matches"].to_numpy()
    bands = {}
    for lo_n, hi_n in ((0, 1000), (1000, 1300), (1300, 1600), (1600, 99999)):
        k = (n_tr >= lo_n) & (n_tr < hi_n)
        if k.any():
            bands[f"{lo_n}-{hi_n}"] = {"n": int(k.sum()), "orig_minus_linear": float(d0[k].mean())}
    res["variants"][name] = {
        "linear": float(L_lin.mean()), "early_stop_wd_chosen": wd, "dev": dev,
        "early_stop": float(L_es.mean()), "early_stop_minus_linear": [m, lo, hi],
        "share_of_refits_falling_back_to_DC": None,
        "original_minus_linear": float(d0.mean()), "original_by_training_size": bands}
    print(f"\n{name}: linear {L_lin.mean():.4f} | early-stop NN {L_es.mean():.4f} diff {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
          f" | original NN diff {d0.mean():+.4f}", flush=True)
    for b, v in bands.items():
        print(f"   original NN - linear, refits with {b} training matches: n={v['n']} {v['orig_minus_linear']:+.4f}")
(OUT / "recheck.json").write_text(json.dumps(res, indent=2))
