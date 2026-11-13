"""Experiment: opponent-adjusted xG strength vs raw decayed xG form.

Usage: python prototypes/run_xg_strength.py [OUT_DIR]   (default results/xg_strength)

Pre-specified:
  primary    V1x - V1   (adjusted xG strength *replacing* raw xG form; must beat V1)
  secondary  V1x - V0;  V1+x - V1x (raw and adjusted together);
             V2x - V2   (XI strength with adjusted instead of raw xG; descriptive only,
                         the pre-registered V2 is untouched)
Config (xG half-life x stage-2 L2) is chosen for V1x on dev seasons only, then frozen
and reused for V1+x and V2x. Uncalibrated throughout (calibration chose the identity).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import (DEV_SEASONS, FORWARD_SEASONS, L2_GRID, TEST_SEASONS, XG_FORM,  # noqa: E402
                               block_bootstrap, build_dataset, paired_bootstrap, per_match_logloss, run_variant,
                               summary)
from tabet.fpl import load_appearances  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.xg_strength import add_xg_strength  # noqa: E402

HALF_LIVES = (90.0, 180.0, 365.0)
SPLITS = {"dev": DEV_SEASONS, "test": TEST_SEASONS, "forward_2026_27": FORWARD_SEASONS}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "xg_strength"
out.mkdir(parents=True, exist_ok=True)

m, _ = load_all(ROOT / "res")
app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
rows, _ = build_dataset(m, app)
rows = add_xg_strength(rows, app, HALF_LIVES)

# 1) choose (half-life, L2) for V1x on dev only
grid = []
for hl in HALF_LIVES:
    cols = [f"f_xgadj_att_{int(hl)}", f"f_xgadj_def_opp_{int(hl)}"]
    for l2 in L2_GRID:
        ll = per_match_logloss(run_variant(rows, cols, l2, DEV_SEASONS)).mean()
        grid.append({"half_life": hl, "l2": l2, "dev_log_loss": ll})
grid = pd.DataFrame(grid).sort_values("dev_log_loss")
hl, l2x = grid.iloc[0]["half_life"], grid.iloc[0]["l2"]
ADJ = [f"f_xgadj_att_{int(hl)}", f"f_xgadj_def_opp_{int(hl)}"]
print("dev grid (V1x):\n" + grid.round(5).to_string(index=False))
print(f"chosen: half-life {hl:.0f} d, stage-2 L2 {l2x}")

# frozen reference settings from the V0-V5 ledger
ref = json.loads((ROOT / "results/fpl_v0_v5/manifest.json").read_text())["l2_chosen_on_dev"]
XI = ["f_xi_position", "f_backline_opp"]
VARIANTS = {
    "V0 Dixon-Coles": ([], 0.0),
    "V1 + team xG form": (XG_FORM, ref["V1 + team xG form"]),
    "V1x adjusted xG strength (replaces raw)": (ADJ, l2x),
    "V1+x raw and adjusted xG": (XG_FORM + ADJ, l2x),
    "V2 + XI strength (position prior)": (XG_FORM + XI, ref["V2 + XI strength (position prior)"]),
    "V2x XI strength with adjusted xG": (ADJ + XI, ref["V2 + XI strength (position prior)"]),
}
CONTRASTS = {
    "primary: V1x - V1": ("V1x adjusted xG strength (replaces raw)", "V1 + team xG form"),
    "secondary: V1x - V0": ("V1x adjusted xG strength (replaces raw)", "V0 Dixon-Coles"),
    "secondary: V1+x - V1x": ("V1+x raw and adjusted xG", "V1x adjusted xG strength (replaces raw)"),
    "descriptive: V2x - V2": ("V2x XI strength with adjusted xG", "V2 + XI strength (position prior)"),
}

ledger = pd.read_csv(ROOT / "results/fpl_v0_v5/ledger.csv")
results = {"grid_dev": grid.to_dict("records"), "chosen": {"half_life": hl, "l2": l2x}}
frames = []
for split, seasons in SPLITS.items():
    preds = {n: run_variant(rows, c, l2, seasons) for n, (c, l2) in VARIANTS.items()}
    # sanity: V0/V1 must reproduce the frozen ledger
    for n in ("V0 Dixon-Coles", "V1 + team xG form"):
        old = ledger[(ledger.variant == n) & (ledger.split == split)].set_index("match_id")["p_home"]
        new = preds[n].set_index("match_id")["p_home"].reindex(old.index)
        assert np.allclose(old, new, atol=1e-9), f"{n} {split} does not reproduce the frozen ledger"
    ll = {n: per_match_logloss(p) for n, p in preds.items()}
    ko = pd.to_datetime(preds["V0 Dixon-Coles"]["kickoff_utc"], utc=True)
    res = {"scores": {n: summary(p) for n, p in preds.items()}, "contrasts": {}}
    for name, (a, b) in CONTRASTS.items():
        d, lo, hi = paired_bootstrap(ll[a], ll[b])
        res["contrasts"][name] = {"mean": d, "se": float(np.std(ll[a] - ll[b], ddof=1) / np.sqrt(len(ll[a]))),
                                  "fixture_ci": [lo, hi], "block14d_ci": list(block_bootstrap(ll[a], ll[b], ko, 14)[1:]),
                                  "block28d_ci": list(block_bootstrap(ll[a], ll[b], ko, 28)[1:]), "n": len(ll[a])}
    results[split] = res
    for n, p in preds.items():
        frames.append(p.assign(variant=n, split=split))
    print(f"\n=== {split} ===")
    print(pd.DataFrame(res["scores"]).T[["log_loss", "brier", "rps", "goal_deviance"]].round(4).to_string())
    for name, e in res["contrasts"].items():
        print(f"  {name}: {e['mean']:+.4f} (SE {e['se']:.4f}, n {e['n']}) fixture[{e['fixture_ci'][0]:+.4f},"
              f"{e['fixture_ci'][1]:+.4f}] block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")

# how different are adjusted and raw features?
r1 = rows[rows.season >= "2023-24"]
results["feature_correlation"] = {"att_vs_raw_xgf": float(r1[ADJ[0]].corr(r1["f_xgf"])),
                                  "def_vs_raw_xga": float(r1[ADJ[1]].corr(r1["f_xga_opp"]))}
print("\nfeature correlations (2023-24 on):", results["feature_correlation"])
pd.concat(frames).to_csv(out / "ledger.csv", index=False)
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", out)
