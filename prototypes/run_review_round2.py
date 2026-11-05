"""Review round 2, items #3-#5, all on the V1x base (opponent-adjusted xG).

Usage: python prototypes/run_review_round2.py [OUT_DIR]   (default results/review_round2)

Pre-specified contrasts (log-loss, identical fixtures, settings chosen on dev only):
  #3 lineup      primary   Ld - Lr   departures vs V2-style ratio (both on V1x)
                 secondary Ld - Lc   departures vs continuity-only;  Lc - L0;  Lr - L0
  #4 adaptation  primary   A1 - L0   + recent league scoring level
                 secondary A2 - A1   + recent home advantage
  #5 outcome     primary   M - L0    multinomial on DC log-probs vs the Poisson adjustment with the same features
                 secondary M - V1    vs (calibrated = identity) V1
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import outcome_model  # noqa: E402
from tabet.adaptation import add_scoring_environment  # noqa: E402
from tabet.backtest import scores  # noqa: E402
from tabet.experiments import (DEV_SEASONS, FORWARD_SEASONS, L2_GRID, TEST_SEASONS, XG_FORM,  # noqa: E402
                               block_bootstrap, build_dataset, paired_bootstrap, per_match_logloss, run_variant)
from tabet.fpl import load_appearances  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.lineup import CONTINUITY_COLS, DEPARTURE_COLS, add_lineup_features  # noqa: E402
from tabet.xg_strength import add_xg_strength  # noqa: E402

SPLITS = {"dev": DEV_SEASONS, "test": TEST_SEASONS, "forward_2026_27": FORWARD_SEASONS}
ENV_HL = (30.0, 60.0)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "review_round2"
out.mkdir(parents=True, exist_ok=True)

m, _ = load_all(ROOT / "res")
app = load_appearances(m, ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"], "2026-27", refresh_live=False)
rows, _ = build_dataset(m, app)
rows = add_xg_strength(rows, app, (90.0,))
rows = add_lineup_features(rows, app)
rows = add_scoring_environment(rows, m, ENV_HL)
rows.to_pickle(out / "rows.pkl")
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
L2_V1X = 1.0  # frozen from results/xg_strength
L2_V1 = json.loads((ROOT / "results/fpl_v0_v5/manifest.json").read_text())["l2_chosen_on_dev"]["V1 + team xG form"]


def dev_ll(cols, l2):
    return per_match_logloss(run_variant(rows, cols, l2, DEV_SEASONS)).mean()


def pick_l2(cols):
    return min(L2_GRID, key=lambda l2: dev_ll(cols, l2))


chosen = {}
V = {"L0 = V1x": (ADJ, L2_V1X), "V1 raw xG form": (XG_FORM, L2_V1)}
for name, cols in {"Lc continuity only": ADJ + CONTINUITY_COLS,
                   "Lr ratio (V2-style)": ADJ + ["f_xi_position", "f_backline_opp"],
                   "Ld departures": ADJ + DEPARTURE_COLS}.items():
    V[name] = (cols, pick_l2(cols))
# #4: choose the short half-life and L2 for A1 on dev, freeze for A2
best = min(((hl, l2) for hl in ENV_HL for l2 in L2_GRID), key=lambda t: dev_ll(ADJ + [f"f_lg_{int(t[0])}"], t[1]))
chosen["adaptation_half_life"] = best[0]
V["A1 + league scoring level"] = (ADJ + [f"f_lg_{int(best[0])}"], best[1])
V["A2 + home advantage"] = (ADJ + [f"f_lg_{int(best[0])}", f"f_home_{int(best[0])}"], pick_l2(ADJ + [f"f_lg_{int(best[0])}", f"f_home_{int(best[0])}"]))
# #5
table = outcome_model.match_table(rows, tuple(ADJ))
MCOLS = ["x_diff", "x_sum"]
OM_GRID = (1.0, 10.0, 100.0)
l2m = min(OM_GRID, key=lambda l2: per_match_logloss(outcome_model.run(table, MCOLS, l2, DEV_SEASONS)).mean())
chosen.update({k: v[1] for k, v in V.items()})
chosen["M multinomial"] = l2m
print("chosen on dev:", chosen)

CONTRASTS = {
    "#3 primary: Ld - Lr": ("Ld departures", "Lr ratio (V2-style)"),
    "#3 secondary: Ld - Lc": ("Ld departures", "Lc continuity only"),
    "#3 secondary: Lc - L0": ("Lc continuity only", "L0 = V1x"),
    "#3 secondary: Lr - L0": ("Lr ratio (V2-style)", "L0 = V1x"),
    "#4 primary: A1 - L0": ("A1 + league scoring level", "L0 = V1x"),
    "#4 secondary: A2 - A1": ("A2 + home advantage", "A1 + league scoring level"),
    "#5 primary: M - L0": ("M multinomial", "L0 = V1x"),
    "#5 secondary: M - V1": ("M multinomial", "V1 raw xG form"),
}
xgs = pd.read_csv(ROOT / "results/xg_strength/ledger.csv")
results, frames = {"chosen_on_dev": chosen}, []
for split, seasons in SPLITS.items():
    preds = {n: run_variant(rows, c, l2, seasons) for n, (c, l2) in V.items()}
    preds["M multinomial"] = outcome_model.run(table, MCOLS, l2m, seasons)
    preds = {n: p.sort_values("match_id").reset_index(drop=True) for n, p in preds.items()}
    ref = xgs[(xgs.variant == "V1x adjusted xG strength (replaces raw)") & (xgs.split == split)].set_index("match_id")
    assert np.allclose(ref["p_home"].reindex(preds["L0 = V1x"]["match_id"]).to_numpy(), preds["L0 = V1x"]["p_home"]), "V1x mismatch"
    ll = {n: per_match_logloss(p) for n, p in preds.items()}
    ko = pd.to_datetime(preds["L0 = V1x"]["kickoff_utc"], utc=True)
    res = {"scores": {n: scores(p[["p_home", "p_draw", "p_away"]].to_numpy(), p["result"]) for n, p in preds.items()},
           "contrasts": {}}
    for name, (a, b) in CONTRASTS.items():
        d, lo, hi = paired_bootstrap(ll[a], ll[b])
        res["contrasts"][name] = {"mean": d, "se": float(np.std(ll[a] - ll[b], ddof=1) / np.sqrt(len(ll[a]))),
                                  "fixture_ci": [lo, hi], "block14d_ci": list(block_bootstrap(ll[a], ll[b], ko, 14)[1:]),
                                  "n": len(ll[a])}
    results[split] = res
    for n, p in preds.items():
        frames.append(p.assign(variant=n, split=split))
    print(f"\n=== {split} ===")
    print(pd.DataFrame(res["scores"]).T[["log_loss", "brier", "rps"]].round(4).to_string())
    for name, e in res["contrasts"].items():
        print(f"  {name}: {e['mean']:+.4f} (SE {e['se']:.4f}) block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")
r = rows[rows.season >= "2023-24"]
results["feature_summary"] = r[DEPARTURE_COLS + CONTINUITY_COLS + [f"f_lg_{int(h)}" for h in ENV_HL]].describe().T.round(4).to_dict()
print("\nfeature summary (2023-24 on):\n", r[DEPARTURE_COLS + CONTINUITY_COLS + [f"f_lg_{int(best[0])}"]].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
pd.concat(frames).to_csv(out / "ledger.csv", index=False)
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", out)
