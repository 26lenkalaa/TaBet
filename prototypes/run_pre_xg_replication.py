"""Pre-registered replication of the XI-strength effect on 2016-21 (docs/PREREGISTRATION.md, addendum R).

Usage: python prototypes/run_pre_xg_replication.py [OUT_DIR]   (default results/pre_xg_replication)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.experiments import L2_GRID, block_bootstrap, build_dataset, per_match_logloss, run_variant  # noqa: E402
from tabet.history import load_history_appearances, load_history_matches  # noqa: E402
from tabet.multileague import _sot_sides  # noqa: E402
from tabet.store import RESULT_LAG  # noqa: E402
from tabet.xg_strength import fit_xg_strength  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "pre_xg_replication"
out.mkdir(parents=True, exist_ok=True)
DEV, TEST = ("2017-18",), ("2018-19", "2019-20", "2020-21")
SOT = ["f_sot_att", "f_sot_def_opp"]
XI = ["f_xi_position", "f_backline_opp"]

m = load_history_matches()
app = load_history_appearances(m)
# frozen mapping: threat + creativity stands in for xG + xA (equal weights, scale irrelevant to the ratio)
app["xg"], app["xa"] = app["threat"] / 100.0, app["creativity"] / 100.0

cache = out / "rows.pkl"
if cache.exists():
    rows = pd.read_pickle(cache)
else:
    rows, _ = build_dataset(m, app, first_season="2016-17")
    # opponent-adjusted shots-on-target strength at each row's cutoff
    sides_all = _sot_sides(m[m["status"] == "completed"])
    sot = {}
    for cutoff in rows["cutoff_utc"].unique():
        cutoff = pd.Timestamp(cutoff)
        r, _, _ = fit_xg_strength(sides_all[sides_all["kickoff_utc"] + RESULT_LAG <= cutoff], cutoff, 90.0)
        sot[cutoff] = r
    rows["f_sot_att"] = [float(sot[pd.Timestamp(c)]["att"].get(t, 0.0)) for c, t in zip(rows["cutoff_utc"], rows["team_id"])]
    rows["f_sot_def_opp"] = [float(-sot[pd.Timestamp(c)]["def"].get(o, 0.0)) for c, o in zip(rows["cutoff_utc"], rows["opponent_id"])]
    rows.to_pickle(cache)
print("rows:", rows.shape, rows.groupby("season")["match_id"].nunique().to_dict())

# ---- choose stage-2 L2 on 2017-18 only ----
grid = []
for name, cols in (("B", SOT), ("X", SOT + XI)):
    for l2 in L2_GRID:
        grid.append({"model": name, "l2": l2, "dev_log_loss": per_match_logloss(run_variant(rows, cols, l2, DEV)).mean()})
grid = pd.DataFrame(grid)
l2 = {k: float(g.sort_values("dev_log_loss").iloc[0]["l2"]) for k, g in grid.groupby("model")}
print(grid.round(5).to_string(index=False)); print("chosen L2:", l2)

# ---- score 2018-21 once ----
P = {"DC": run_variant(rows, [], 0.0, TEST), "B": run_variant(rows, SOT, l2["B"], TEST),
     "X": run_variant(rows, SOT + XI, l2["X"], TEST)}
idx = P["B"]["match_id"]
for k in P:
    P[k] = P[k].set_index("match_id").reindex(idx).reset_index()
    P[k]["ll"] = per_match_logloss(P[k])
ko = P["B"]["kickoff_utc"]
res = {"l2_grid_dev": grid.to_dict("records"), "l2_chosen": l2, "n_test": len(idx),
       "stage2_active_test": {k: int(v["stage2_active"].sum()) for k, v in P.items()},
       "log_loss": {k: float(v["ll"].mean()) for k, v in P.items()}}
mean, lo, hi = block_bootstrap(P["X"]["ll"].to_numpy(), P["B"]["ll"].to_numpy(), ko, 14)
d = P["X"]["ll"] - P["B"]["ll"]
res["primary"] = {"contrast": "X - B, 2018-21", "mean": mean, "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                  "paired_sd": float(d.std(ddof=1)), "block14_ci": [lo, hi], "confirmed": bool(mean < 0 and hi < 0)}
print(f"\nPRIMARY X - B (n={len(d)}): {mean:+.4f} (SE {res['primary']['se']:.4f}) block14 [{lo:+.4f}, {hi:+.4f}]"
      f" -> {'CONFIRMED' if res['primary']['confirmed'] else 'not confirmed'}")
print("log-loss:", {k: round(v, 4) for k, v in res["log_loss"].items()})

# ---- reported, not used for the decision ----
season = P["B"]["season"]
res["per_season"] = {}
for s in TEST:
    mk = (season == s).to_numpy()
    a, b = P["X"]["ll"][mk].to_numpy(), P["B"]["ll"][mk].to_numpy()
    mu, l, h = block_bootstrap(a, b, ko[mk].reset_index(drop=True), 14)
    res["per_season"][s] = {"n": int(mk.sum()), "mean": mu, "ci": [l, h]}
    print(f"  {s}{' (no crowds)' if s == '2020-21' else ''}: X - B {mu:+.4f} [{l:+.4f}, {h:+.4f}]")
mu, l, h = block_bootstrap(P["B"]["ll"].to_numpy(), P["DC"]["ll"].to_numpy(), ko, 14)
res["B_minus_DC"] = {"mean": mu, "ci": [l, h]}
print(f"  B - DC: {mu:+.4f} [{l:+.4f}, {h:+.4f}]")
dev_abs = rows.assign(a=rows["f_xi_position"].abs()).groupby("match_id")["a"].sum()
pool = dev_abs[rows.drop_duplicates("match_id").set_index("match_id")["season"].reindex(dev_abs.index) >= "2017-18"]
cuts = np.quantile(pool, [1 / 3, 2 / 3])
band = pd.cut(idx.map(dev_abs), [-np.inf, *cuts, np.inf], labels=["low", "middle", "high"])
res["bands"] = {}
for b_ in ["low", "middle", "high"]:
    mk = (band == b_)
    res["bands"][b_] = {"n": int(mk.sum()), "mean": float(d[mk].mean()), "se": float(d[mk].std(ddof=1) / np.sqrt(mk.sum()))}
    print(f"  band {b_}: n={mk.sum()} X - B {d[mk].mean():+.4f} (SE {d[mk].std(ddof=1) / np.sqrt(mk.sum()):.4f})")
pd.concat([v.assign(variant=k) for k, v in P.items()]).to_csv(out / "ledger.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
