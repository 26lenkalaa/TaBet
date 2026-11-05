"""Does pooling La Liga, Serie A, Bundesliga and Ligue 1 improve Premier League forecasts?

Usage: python prototypes/run_multileague.py [OUT_DIR]   (default results/multileague)

Pre-specified (scored on Premier League fixtures only, identical across variants):
  primary    P2 - P1   SoT-strength coefficients learned on all 5 leagues vs PL only
  secondary  P1 - P0   SoT strength (PL-only coefficients) vs Dixon-Coles
             P2 - P0   pooled SoT strength vs Dixon-Coles
             P2 - V1x  pooled SoT strength vs the current lineup-free model (FPL xG, PL only)
             H1        Dixon-Coles settings tuned on all 5 leagues' dev seasons vs PL-tuned, scored on PL
Stage-2 L2 is chosen per variant on PL dev seasons; SoT half-life fixed at 90 days (as for xG).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.backtest import scores  # noqa: E402
from tabet.experiments import (DEV_SEASONS, FORWARD_SEASONS, L2_GRID, TEST_SEASONS, block_bootstrap,  # noqa: E402
                               paired_bootstrap, per_match_logloss)
from tabet.ingest import load_all  # noqa: E402
from tabet.leagues import LEAGUES, load_leagues  # noqa: E402
from tabet.model import Hyperparams  # noqa: E402
from tabet.multileague import SOT, league_rows, run  # noqa: E402

SPLITS = {"dev": DEV_SEASONS, "test": TEST_SEASONS, "forward_2026_27": FORWARD_SEASONS}
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "multileague"
out.mkdir(parents=True, exist_ok=True)

pl, _ = load_all(ROOT / "res")
other = load_leagues()
all_leagues = {"E0": pl.assign(league="E0"), **{d: other[other["league"] == d] for d in LEAGUES}}

# ---- H1: Dixon-Coles hyperparameters tuned on pooled dev vs PL dev ----------
from tabet.backtest import walk_forward  # noqa: E402
grid = [(xi, l2) for xi in (0.002, 0.003, 0.004, 0.006) for l2 in (3.0, 10.0, 30.0)]
h1 = []
for xi, l2 in grid:
    h = Hyperparams(xi=xi, l2=l2)
    ll = {lg: per_match_logloss(walk_forward(mm, DEV_SEASONS, h)).tolist() for lg, mm in all_leagues.items()}
    h1.append({"xi": xi, "l2": l2, "pl_dev": float(np.mean(ll["E0"])),
               "pooled_dev": float(np.mean(sum(ll.values(), [])))})
h1 = pd.DataFrame(h1)
pl_best = h1.sort_values("pl_dev").iloc[0]
pooled_best = h1.sort_values("pooled_dev").iloc[0]
print(h1.round(5).to_string(index=False))
print("PL-chosen:", dict(pl_best[["xi", "l2"]]), "| pooled-chosen:", dict(pooled_best[["xi", "l2"]]))

# ---- rows for every league (default, PL-tuned Dixon-Coles) -------------------
rows = pd.concat([league_rows(mm, lg) for lg, mm in all_leagues.items()], ignore_index=True)
rows.to_pickle(out / "rows.pkl")
print("rows per league:", rows.groupby("league")["match_id"].nunique().to_dict())

E0 = {"E0"}
ALL = set(all_leagues)
pick = lambda tl: min(L2_GRID, key=lambda l2: per_match_logloss(run(rows, SOT, l2, DEV_SEASONS, tl)).mean())  # noqa: E731
l2_p1, l2_p2 = pick(E0), pick(ALL)
print("stage-2 L2 chosen on PL dev: P1", l2_p1, "| P2", l2_p2)

v1x = pd.read_csv(ROOT / "results/xg_strength/ledger.csv")
v0_ref = v1x[v1x.variant == "V0 Dixon-Coles"]
results = {"h1_grid": h1.to_dict("records"), "l2": {"P1": l2_p1, "P2": l2_p2},
           "pl_chosen": dict(pl_best[["xi", "l2"]]), "pooled_chosen": dict(pooled_best[["xi", "l2"]])}
for split, seasons in SPLITS.items():
    P = {"P0 Dixon-Coles": run(rows, [], 0.0, seasons, E0),
         "P1 SoT strength, PL-only coefficients": run(rows, SOT, l2_p1, seasons, E0),
         "P2 SoT strength, pooled coefficients": run(rows, SOT, l2_p2, seasons, ALL)}
    ref = v0_ref[v0_ref.split == split].set_index("match_id")["p_home"].reindex(P["P0 Dixon-Coles"]["match_id"])
    assert np.allclose(ref.to_numpy(), P["P0 Dixon-Coles"]["p_home"].to_numpy()), "P0 does not reproduce the frozen V0"
    P["V1x (FPL xG, PL only)"] = (v1x[(v1x.variant == "V1x adjusted xG strength (replaces raw)") & (v1x.split == split)]
                                  .sort_values("match_id").reset_index(drop=True))
    if (pooled_best["xi"], pooled_best["l2"]) != (pl_best["xi"], pl_best["l2"]):
        from tabet.backtest import walk_forward as wf
        P["H1 DC tuned on pooled dev"] = wf(pl, seasons, Hyperparams(xi=pooled_best["xi"], l2=pooled_best["l2"])) \
            .sort_values("match_id").reset_index(drop=True)
    ll = {n: per_match_logloss(p) for n, p in P.items()}
    ko = pd.to_datetime(P["P0 Dixon-Coles"]["kickoff_utc"], utc=True)
    C = {"primary: P2 - P1": ("P2 SoT strength, pooled coefficients", "P1 SoT strength, PL-only coefficients"),
         "P1 - P0": ("P1 SoT strength, PL-only coefficients", "P0 Dixon-Coles"),
         "P2 - P0": ("P2 SoT strength, pooled coefficients", "P0 Dixon-Coles"),
         "P2 - V1x": ("P2 SoT strength, pooled coefficients", "V1x (FPL xG, PL only)")}
    if "H1 DC tuned on pooled dev" in P:
        C["H1: pooled-tuned DC - PL-tuned DC"] = ("H1 DC tuned on pooled dev", "P0 Dixon-Coles")
    res = {"scores": {n: scores(p[["p_home", "p_draw", "p_away"]].to_numpy(), p["result"]) for n, p in P.items()},
           "stage2_active": {n: int(p["stage2_active"].sum()) for n, p in P.items() if "stage2_active" in p}, "contrasts": {}}
    for name, (a, b) in C.items():
        d, lo, hi = paired_bootstrap(ll[a], ll[b])
        res["contrasts"][name] = {"mean": d, "se": float(np.std(ll[a] - ll[b], ddof=1) / np.sqrt(len(d := ll[a] - ll[b]))),
                                  "block14d_ci": list(block_bootstrap(ll[a], ll[b], ko, 14)[1:])}
    results[split] = res
    print(f"\n=== {split} ({len(ko)} PL matches) ===")
    print(pd.DataFrame(res["scores"]).T[["log_loss", "brier", "accuracy"]].round(4).to_string())
    print("  stage 2 active:", res["stage2_active"])
    for k, e in res["contrasts"].items():
        print(f"  {k}: {e['mean']:+.4f} (SE {e['se']:.4f}) block14[{e['block14d_ci'][0]:+.4f},{e['block14d_ci'][1]:+.4f}]")
(out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
print("wrote", out)
