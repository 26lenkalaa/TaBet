"""Calibration experiment on the frozen V0-V5 ledger.

Usage: python prototypes/run_calibration.py [LEDGER] [OUT_DIR]
       (defaults: results/fpl_v0_v5/ledger.csv, results/calibration)

Modes: identity -> temperature -> temperature+draw, fitted walk-forward on each
variant's own earlier out-of-sample predictions, identity until 200 known
results. The shrinkage strength is chosen per (variant, mode) on the dev
seasons only. Every mode and variant is scored on the same fixtures.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.backtest import scores  # noqa: E402
from tabet.calibration import MODES, PROB, walk_forward  # noqa: E402
from tabet.experiments import block_bootstrap, paired_bootstrap, per_match_logloss  # noqa: E402

LAMS = (1.0, 10.0, 100.0, 1000.0)
ledger_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/fpl_v0_v5/ledger.csv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/calibration"
out.mkdir(parents=True, exist_ok=True)

led = pd.read_csv(ledger_path, parse_dates=["kickoff_utc", "cutoff_utc"])
for c in ("kickoff_utc", "cutoff_utc"):
    led[c] = pd.to_datetime(led[c], utc=True)
splits = list(dict.fromkeys(led["split"]))

cal, chosen = {}, {}
for v, g in led.groupby("variant", sort=False):
    for mode in MODES:
        if mode == "identity":
            cal[(v, mode)] = walk_forward(g, mode, 0.0)
            chosen[(v, mode)] = None
            continue
        best = None
        for lam in LAMS:
            r = walk_forward(g, mode, lam)
            ll = per_match_logloss(r[r["split"] == "dev"]).mean()
            if best is None or ll < best[0]:
                best = (ll, lam, r)
        chosen[(v, mode)] = best[1]
        cal[(v, mode)] = best[2]


def aligned(v, mode, split):
    r = cal[(v, mode)]
    return r[r["split"] == split].sort_values("match_id").reset_index(drop=True)


results = {"lambda_chosen_on_dev": {f"{v} | {m}": l for (v, m), l in chosen.items()}}
V0, V1, V2 = "V0 Dixon-Coles", "V1 + team xG form", "V2 + XI strength (position prior)"
for split in splits:
    res = {"scores": {}, "calibration_contrasts": {}, "feature_contrasts_by_mode": {}, "draws": {}, "params": {}}
    for (v, mode), r in cal.items():
        a = aligned(v, mode, split)
        s = scores(a[PROB].to_numpy(), a["result"])
        res["scores"][f"{v} | {mode}"] = s
        res["draws"][f"{v} | {mode}"] = {"predicted": float(a["p_draw"].sum()),
                                         "observed": int((a["result"] == "D").sum())}
        if mode != "identity" and a["calib_active"].any():
            last = a.sort_values("cutoff_utc").iloc[-1]
            res["params"][f"{v} | {mode}"] = {"T_last": float(last["calib_T"]), "delta_last": float(last["calib_delta"]),
                                              "T_mean": float(a.loc[a.calib_active, "calib_T"].mean()),
                                              "delta_mean": float(a.loc[a.calib_active, "calib_delta"].mean()),
                                              "share_active": float(a["calib_active"].mean())}
    for v in (V0, V2):
        base = aligned(v, "identity", split)
        ko = base["kickoff_utc"]
        for a_mode, b_mode in (("temperature", "identity"), ("temperature+draw", "temperature"),
                               ("temperature+draw", "identity")):
            la, lb = per_match_logloss(aligned(v, a_mode, split)), per_match_logloss(aligned(v, b_mode, split))
            d, lo, hi = paired_bootstrap(la, lb)
            res["calibration_contrasts"][f"{v}: {a_mode} - {b_mode}"] = {
                "mean": d, "fixture_ci": [lo, hi], "block14d_ci": list(block_bootstrap(la, lb, ko, 14)[1:]),
                "se": float(np.std(la - lb, ddof=1) / np.sqrt(len(la)))}
    for mode in MODES:
        la, lb = per_match_logloss(aligned(V2, mode, split)), per_match_logloss(aligned(V1, mode, split))
        d, lo, hi = paired_bootstrap(la, lb)
        res["feature_contrasts_by_mode"][f"V2 - V1 | {mode}"] = {
            "mean": d, "fixture_ci": [lo, hi],
            "block14d_ci": list(block_bootstrap(la, lb, aligned(V2, mode, split)["kickoff_utc"], 14)[1:])}
    results[split] = res

# reliability of the favourite, V2, test seasons
rel = {}
for mode in MODES:
    a = aligned(V2, mode, "test")
    P = a[PROB].to_numpy()
    fav = P.argmax(axis=1)
    pf = P[np.arange(len(P)), fav]
    hit = (a["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy() == fav)
    b = pd.cut(pf, [0, 0.4, 0.5, 0.6, 0.7, 1.0])
    rel[mode] = pd.DataFrame({"bin": b, "p": pf, "hit": hit}).groupby("bin", observed=True).agg(
        n=("p", "size"), mean_pred=("p", "mean"), observed=("hit", "mean")).round(3).reset_index().astype(str).to_dict("records")
results["favourite_reliability_V2_test"] = rel
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")

pd.set_option("display.width", 200)
print("lambda chosen on dev:", {k: v for k, v in results["lambda_chosen_on_dev"].items() if v is not None and k.split(" |")[0] in (V0, V2)})
for split in splits:
    r = results[split]
    print(f"\n=== {split} ===")
    t = pd.DataFrame({k: v for k, v in r["scores"].items() if k.split(" |")[0] in (V0, V1, V2)}).T
    t["draws_pred"] = [r["draws"][k]["predicted"] for k in t.index]
    t["draws_obs"] = [r["draws"][k]["observed"] for k in t.index]
    print(t[["log_loss", "brier", "rps", "draws_pred", "draws_obs"]].round(4).to_string())
    for k, e in r["calibration_contrasts"].items():
        print(f"  {k}: {e['mean']:+.4f} (SE {e['se']:.4f}) fixture[{e['fixture_ci'][0]:+.4f},{e['fixture_ci'][1]:+.4f}] "
              f"block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")
    for k, e in r["feature_contrasts_by_mode"].items():
        print(f"  {k}: {e['mean']:+.4f} block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")
    for k, e in r["params"].items():
        if k.split(" |")[0] in (V0, V2):
            print(f"  params {k}: T mean {e['T_mean']:.3f} (last {e['T_last']:.3f}), delta mean {e['delta_mean']:+.3f} "
                  f"(last {e['delta_last']:+.3f}), active {e['share_active']:.0%}")
print("\nfavourite reliability, V2 test:")
for mode, rows in rel.items():
    print(" ", mode, [(x["bin"], x["n"], x["mean_pred"], x["observed"]) for x in rows])
print("\nwrote", out / "results.json")
