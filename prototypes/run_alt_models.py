"""Compare each alternative team-strength model with the current model (docs/PREREGISTRATION.md, addendum M).

Usage: python prototypes/run_alt_models.py [OUT_DIR]   (default results/alt_models)

Models (files in tabet/alt/; Elo is tabet/elo.py and was run separately, results/elo_blend):
  D0      frozen Dixon-Coles (xi 0.004, L2 10), the current team-strength model
  R2a     returning-team reset: each league spell is a separate team (option 2)
  R2b     R2a + the shared promoted-team offset learned from earlier promoted sides
  BH      hierarchical Bayesian Dixon-Coles, Laplace approximation, spells + group offset + tighter
          prior for teams in their first season (lam_new chosen on dev from {10, 30, 100}),
          forecasts averaged over 300 posterior draws (option 1)
  NB      Dixon-Coles ratings, negative binomial goals (option 4a)
  BP      Dixon-Coles ratings, bivariate Poisson goals (option 4b)
Data: clean = results 2015-16 -> 2020-21 (burn-in 2015-17, dev 2017-18, scored once on 2018-21);
      recent = 2021-22 -> 2026-27 (dev 2022-24; 2024-26 and 2026-27 GW1-9 reported; reused validation).
Contrasts: log-loss, model minus D0, paired, 14-day calendar-block bootstrap (2,000 resamples, seed 0).
  primary (one per model)  model - D0 on clean 2018-21. Five primaries, so a model counts as better
                           only if its mean is below 0 and the Bonferroni 99% block interval
                           (95% / 5 comparisons) excludes 0. The plain 95% interval is reported too.
  secondary                same contrast on early-promoted matches (a side in the first season of a
                           spell, within its first 10 matches) and on recent seasons;
                           BH without averaging (mode) vs D0, to separate averaging from the prior.
  main model               on recent seasons, R2a, R2b and BH-mode rates replace the Dixon-Coles
                           offsets inside the frozen V1x (L2 1) and V2x (L2 100) stage 2, compared
                           with the frozen V1x and V2x. NB, BP and BH averaging change only the score
                           grid, which stage 2 does not use, so they are compared at stage 1 only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.alt import bayes_dc, goal_distributions, returning_reset  # noqa: E402
from tabet.experiments import per_match_logloss, run_variant  # noqa: E402
from tabet.history import load_history_matches  # noqa: E402
from tabet.ingest import load_all  # noqa: E402
from tabet.model import Hyperparams, promoted_team_seasons  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "alt_models"
out.mkdir(parents=True, exist_ok=True)
COLS = ["p_home", "p_draw", "p_away"]


def boot(a, b, ko, days=14, reps=2000, seed=0):
    d = a - b
    block = ((ko - ko.min()).dt.days // days).to_numpy()
    ids = np.unique(block)
    sums = np.array([d[block == i].sum() for i in ids])
    counts = np.array([(block == i).sum() for i in ids])
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(reps, len(ids)))
    bs = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {"n": int(len(d)), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "ci99_bonferroni": [float(np.percentile(bs, 0.5)), float(np.percentile(bs, 99.5))]}


def ll(df, prefix=""):
    y = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    p = df[[prefix + c for c in COLS]].to_numpy()
    return -np.log(p[np.arange(len(y)), y])


def early_promoted(m, ids):
    prom = promoted_team_seasons(m)
    mi = m.set_index("match_id").loc[ids]
    o = pd.concat([mi[["season", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team"}).reset_index(),
                   mi[["season", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team"}).reset_index()])
    full = pd.concat([m[["match_id", "season", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team"}),
                      m[["match_id", "season", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team"})]).sort_values("kickoff_utc")
    full["n"] = full.groupby(["team", "season"]).cumcount() + 1
    full["flag"] = [(t, s) in prom and n <= 10 for t, s, n in zip(full["team"], full["season"], full["n"])]
    return full.groupby("match_id")["flag"].any().reindex(ids).fillna(False).to_numpy()


def run_set(m, eval_seasons, dev_seasons):
    P = {"D0": returning_reset.dc_walk_forward(m, eval_seasons, Hyperparams()),
         "R2a": returning_reset.walk_forward(m, eval_seasons, offset=False),
         "R2b": returning_reset.walk_forward(m, eval_seasons, offset=True)}
    gd = goal_distributions.walk_forward(m, eval_seasons)
    P["NB"] = gd.rename(columns={f"nb_{c}": c for c in COLS})
    P["BP"] = gd.rename(columns={f"bp_{c}": c for c in COLS})
    bh = {lam: bayes_dc.walk_forward(m, eval_seasons, lam) for lam in bayes_dc.LAM_NEW_GRID}
    ids = P["D0"]["match_id"]
    for k in list(P):
        P[k] = P[k].set_index("match_id").loc[ids].reset_index()
    for lam in bh:
        bh[lam] = bh[lam].set_index("match_id").loc[ids].reset_index()
    dev = P["D0"]["season"].isin(dev_seasons).to_numpy()
    dev_ll = {lam: float(ll(b)[dev].mean()) for lam, b in bh.items()}
    lam = min(dev_ll, key=dev_ll.get)
    P["BH"] = bh[lam]
    P["BH-mode"] = bh[lam].assign(**{c: bh[lam][f"mode_{c}"] for c in COLS})
    return P, {"lam_new_dev": dev_ll, "lam_new": lam, "nb_r_last": float(gd["nb_r"].iloc[-1]),
               "bp_l3_last": float(gd["bp_l3"].iloc[-1])}


MODELS = ("R2a", "R2b", "BH", "BH-mode", "NB", "BP")
res = {}
# ---- clean ----
mh = load_history_matches()
Ph, info = run_set(mh, ("2017-18", "2018-19", "2019-20", "2020-21"), ("2017-18",))
seas = Ph["D0"]["season"].to_numpy()
ko = pd.to_datetime(Ph["D0"]["kickoff_utc"], utc=True).reset_index(drop=True)
test = np.isin(seas, ("2018-19", "2019-20", "2020-21"))
ep = early_promoted(mh, Ph["D0"]["match_id"])
L = {k: ll(v) for k, v in Ph.items()}
res["clean"] = {"info": info, "log_loss_2018_21": {k: float(v[test].mean()) for k, v in L.items()}, "contrasts": {}}
print("clean:", info)
print("  2018-21 log-loss:", {k: round(v, 4) for k, v in res["clean"]["log_loss_2018_21"].items()})
for k in MODELS:
    for sub, mk in (("primary 2018-21", test), ("early promoted 2018-21", test & ep)):
        r = boot(L[k][mk], L["D0"][mk], ko[mk].reset_index(drop=True))
        res["clean"]["contrasts"][f"{k} - D0 | {sub}"] = r
        tag = "PRIMARY" if sub.startswith("primary") and k != "BH-mode" else ""
        print(f"  {tag:7s} {k:7s} - D0 {sub:24s} n={r['n']:4d} {r['mean']:+.4f} (SE {r['se']:.4f}) "
              f"95% [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  99% [{r['ci99_bonferroni'][0]:+.4f}, {r['ci99_bonferroni'][1]:+.4f}]")

# ---- recent ----
mr, _ = load_all(ROOT / "res")
Pr, info_r = run_set(mr, ("2022-23", "2023-24", "2024-25", "2025-26", "2026-27"), ("2022-23", "2023-24"))
sr = Pr["D0"]["season"].to_numpy()
kor = pd.to_datetime(Pr["D0"]["kickoff_utc"], utc=True).reset_index(drop=True)
epr = early_promoted(mr, Pr["D0"]["match_id"])
Lr = {k: ll(v) for k, v in Pr.items()}
res["recent"] = {"info": info_r, "contrasts": {}, "main_model": {}}
print("\nrecent:", info_r)
periods = {"2022-24 dev": np.isin(sr, ("2022-23", "2023-24")), "2024-26": np.isin(sr, ("2024-25", "2025-26")),
           "2026-27 GW1-9": sr == "2026-27", "early promoted 2022-26": (sr < "2026-27") & epr}
for k in MODELS:
    for pname, mk in periods.items():
        r = boot(Lr[k][mk], Lr["D0"][mk], kor[mk].reset_index(drop=True))
        res["recent"]["contrasts"][f"{k} - D0 | {pname}"] = r
        print(f"  {k:7s} - D0 {pname:22s} n={r['n']:4d} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")

# ---- inside the main model (stage 2), recent seasons ----
rows = pd.read_pickle(ROOT / "results/review_round2/rows.pkl")
ADJ, XI = ["f_xgadj_att_90", "f_xgadj_def_opp_90"], ["f_xi_position", "f_backline_opp"]
print("\ninside the main model (alternative rates as the stage-2 offset):")
for k in ("R2a", "R2b", "BH-mode"):
    s2 = Pr[k].set_index("match_id")
    alt = rows.copy()
    lam = np.where(alt["side"] == "H", alt["match_id"].map(s2["lam_home"]), alt["match_id"].map(s2["lam_away"]))
    keep = ~np.isnan(lam)
    alt = alt[keep].copy()
    alt["log_dc"], alt["rho"] = np.log(lam[keep]), alt["match_id"].map(s2["rho"])
    for main, cols, l2 in (("V1x", ADJ, 1.0), ("V2x", ADJ + XI, 100.0)):
        for pname, seasons in (("2022-24 dev", ("2022-23", "2023-24")), ("2024-26", ("2024-25", "2025-26")),
                               ("2026-27 GW1-9", ("2026-27",))):
            base = run_variant(rows, cols, l2, seasons)
            new = run_variant(alt, cols, l2, seasons).set_index("match_id").loc[base["match_id"]].reset_index()
            r = boot(per_match_logloss(new), per_match_logloss(base), pd.to_datetime(base["kickoff_utc"], utc=True).reset_index(drop=True))
            res["recent"]["main_model"][f"{main} with {k} - {main} | {pname}"] = r
            print(f"  {main} with {k:7s} - {main} {pname:14s} n={r['n']:4d} {r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")

elo = ROOT / "results/elo_blend/results.json"  # lives on another branch until it is merged
res["elo_blend_reference"] = json.loads(elo.read_text())["clean"]["primary blend-DC"] if elo.exists() else None
for name, P in (("clean", Ph), ("recent", Pr)):
    pd.concat([v.assign(model=k) for k, v in P.items()]).to_csv(out / f"ledger_{name}.csv", index=False)
(out / "results.json").write_text(json.dumps(res, indent=2, default=float))
print(f"wrote {out}")
