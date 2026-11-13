"""Score-driven dynamic team strengths vs the decayed Dixon-Coles refit (docs/PREREGISTRATION.md, addendum D).

Usage: python prototypes/run_dynamic_ratings.py [OUT_DIR]   (default results/dynamic_ratings)

Models (same goal model: independent Poisson + Dixon-Coles rho):
  D0   frozen Dixon-Coles (xi = 0.004, half-life 173 d, L2 = 10), refit per matchday
  D0s  shorter-half-life challenger (xi = 0.006, 116 d; the shortest value in the original grid)
  S1   score-driven strengths, weekly persistence beta, no summer term
  S2   S1 + summer factor beta_summer (the offseason extension)
Static parameters of S1/S2 are refitted at the start of every evaluated season by maximising the
filtered one-step-ahead log-likelihood on all earlier matches in the same data set. Nothing else
is tuned.
Data sets: (a) clean: 2015-16 -> 2020-21 results (res/history), never used to compare team-strength
models; 2015-17 is burn-in, 2017-18 reported, 2018-21 scored once. (b) recent: 2021-22 -> 2026-27,
2021-22 burn-in (reused validation; reported for information).
Contrasts (log-loss, 14-day block bootstrap, 2,000 resamples, seed 0):
  primary     S2 - D0 on 2018-21 (n = 1,140), all matches
  secondary   S1 - D0, S2 - S1, S2 - D0s on 2018-21; S2 - D0 on early-season matches (kickoff
              within 56 days of the season's first match) on 2018-21; S2 - D0 on 2022-26 and 2026-27
  stage 2     S2 offsets + opponent-adjusted xG (V1x stage 2, L2 = 1) vs frozen V1x, 2022-26 and 2026-27
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import dynamic as D  # noqa: E402
from tabet.backtest import walk_forward  # noqa: E402
from tabet.experiments import block_bootstrap, per_match_logloss, run_variant  # noqa: E402
from tabet.history import load_history_matches  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.model import Hyperparams  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dynamic_ratings"
out.mkdir(parents=True, exist_ok=True)


def ll_of(df):
    return per_match_logloss(df.rename(columns={}))


def run_set(m, eval_seasons):
    P = {"D0": walk_forward(m, eval_seasons, Hyperparams()),
         "D0s": walk_forward(m, eval_seasons, Hyperparams(xi=0.006)),
         "S1": D.walk_forward(m, eval_seasons, summer=False),
         "S2": D.walk_forward(m, eval_seasons, summer=True)}
    ids = P["D0"]["match_id"]
    for k in P:
        P[k] = P[k].set_index("match_id").reindex(ids).reset_index()
        assert P[k]["p_home"].notna().all(), k
        P[k]["ll"] = per_match_logloss(P[k])
    first = P["D0"].groupby("season")["kickoff_utc"].transform("min")
    P["early"] = ((P["D0"]["kickoff_utc"] - first).dt.days < 56).to_numpy()
    return P


def contrast(P, a, b, mask):
    x, y = P[a]["ll"].to_numpy()[mask], P[b]["ll"].to_numpy()[mask]
    mean, lo, hi = block_bootstrap(x, y, P["D0"]["kickoff_utc"][mask].reset_index(drop=True), 14)
    return {"n": int(mask.sum()), "mean": mean, "se": float(np.std(x - y, ddof=1) / np.sqrt(mask.sum())), "ci": [lo, hi]}


res = {"clean_2015_21": {}, "recent_2021_27": {}, "stage2": {}}
# ---- (a) clean data ----
mh = load_history_matches()
Ph = run_set(mh, ("2017-18", "2018-19", "2019-20", "2020-21"))
seas = Ph["D0"]["season"].to_numpy()
test = np.isin(seas, ("2018-19", "2019-20", "2020-21"))
dev = seas == "2017-18"
spec = [("primary", "S2", "D0", test), ("S1-D0", "S1", "D0", test), ("S2-S1", "S2", "S1", test),
        ("S2-D0s", "S2", "D0s", test), ("D0s-D0", "D0s", "D0", test), ("S2-D0 early season", "S2", "D0", test & Ph["early"]),
        ("S2-D0 2017-18 (reported)", "S2", "D0", dev)]
print("clean 2018-21 log-loss:", {k: round(float(Ph[k]["ll"][test].mean()), 4) for k in ("D0", "D0s", "S1", "S2")})
for name, a, b, mk in spec:
    r = contrast(Ph, a, b, mk)
    res["clean_2015_21"][name] = r
    print(f"  {name:28s} {a}-{b}: n={r['n']:4d} {r['mean']:+.4f} (SE {r['se']:.4f}) [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
th = Ph["S2"].groupby("season")[[f"theta_{n}" for n in D.NAMES]].first()
print("  S2 static parameters by season:\n" + th.round(4).to_string())
res["clean_2015_21"]["theta_S2"] = th.reset_index().to_dict("records")

# ---- (b) recent data ----
mr, _ = load_all(ROOT / "res")
seasons_r = ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
Pr = run_set(mr, seasons_r)
sr = Pr["D0"]["season"].to_numpy()
print("\nrecent log-loss:", {k: round(float(Pr[k]["ll"][sr < "2026-27"].mean()), 4) for k in ("D0", "D0s", "S1", "S2")})
for label, mk in (("2022-24", np.isin(sr, ("2022-23", "2023-24"))), ("2024-26", np.isin(sr, ("2024-25", "2025-26"))),
                  ("2026-27 GW1-9", sr == "2026-27"), ("2022-26 early season", (sr < "2026-27") & Pr["early"])):
    for a, b in (("S2", "D0"), ("S2", "S1")):
        r = contrast(Pr, a, b, mk)
        res["recent_2021_27"][f"{label} {a}-{b}"] = r
        print(f"  {label:22s} {a}-{b}: n={r['n']:4d} {r['mean']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
th = Pr["S2"].groupby("season")[[f"theta_{n}" for n in D.NAMES]].first()
print("  S2 static parameters by season:\n" + th.round(4).to_string())
res["recent_2021_27"]["theta_S2"] = th.reset_index().to_dict("records")

# ---- stage 2 with adjusted xG on the S2 offsets ----
rows = pd.read_pickle(ROOT / "results/review_round2/rows.pkl")
s2 = Pr["S2"].set_index("match_id")
dyn = rows.copy()
lam = np.where(dyn["side"] == "H", dyn["match_id"].map(s2["lam_home"]), dyn["match_id"].map(s2["lam_away"]))
keep = ~np.isnan(lam)
dyn = dyn[keep].copy()
dyn["log_dc"], dyn["rho"] = np.log(lam[keep]), dyn["match_id"].map(s2["rho"])
ADJ = ["f_xgadj_att_90", "f_xgadj_def_opp_90"]
for label, seasons in (("2022-24", ("2022-23", "2023-24")), ("2024-26", ("2024-25", "2025-26")), ("2026-27 GW1-9", ("2026-27",))):
    v1x = run_variant(rows, ADJ, 1.0, seasons)
    s2x = run_variant(dyn, ADJ, 1.0, seasons).set_index("match_id").reindex(v1x["match_id"]).reset_index()
    a, b = per_match_logloss(s2x), per_match_logloss(v1x)
    mean, lo, hi = block_bootstrap(a, b, v1x["kickoff_utc"], 14)
    res["stage2"][label] = {"n": len(a), "S2x": float(a.mean()), "V1x": float(b.mean()), "mean": mean, "ci": [lo, hi]}
    print(f"stage 2 {label:14s} S2x - V1x: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  (S2x {a.mean():.4f}, V1x {b.mean():.4f})")

for name, P in (("clean", Ph), ("recent", Pr)):
    pd.concat([P[k].assign(variant=k) for k in ("D0", "D0s", "S1", "S2")]).to_csv(out / f"ledger_{name}.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
