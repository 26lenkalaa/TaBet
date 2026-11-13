"""Bivariate Poisson vs Dixon-Coles on La Liga, Serie A, Bundesliga and Ligue 1 (docs/PREREGISTRATION.md, addendum B).

Usage: python prototypes/run_bp_four_leagues.py [OUT_DIR]   (default results/bp_four_leagues)

Per league: frozen Dixon-Coles walk-forward by matchday (2021-22 burn-in). DC = its own grid with rho;
BP = same expected goals, bivariate Poisson grid with lambda3 fitted at the cutoff; NB reported too.
Primary: BP - DC pooled over the four leagues on 2022-23 -> 2025-26; confirmed if mean < 0 and the 95%
14-day calendar-block bootstrap interval (2,000 resamples, seed 0) excludes 0.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet.alt import goal_distributions  # noqa: E402
from tabet.experiments import block_bootstrap  # noqa: E402
from tabet.leagues import LEAGUES, load_leagues  # noqa: E402
from tabet.model import DixonColes, Hyperparams, promoted_team_seasons  # noqa: E402
from tabet.store import MatchStore  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "bp_four_leagues"
out.mkdir(parents=True, exist_ok=True)
PRIMARY = ("2022-23", "2023-24", "2024-25", "2025-26")
CURRENT = ("2026-27",)
IDX = {"H": 0, "D": 1, "A": 2}


def ll(df, prefix):
    p = df[[f"{prefix}_p_home", f"{prefix}_p_draw", f"{prefix}_p_away"]].to_numpy()
    return -np.log(np.clip(p[np.arange(len(p)), df["result"].map(IDX).to_numpy()], 1e-15, 1))


def walk(matches, eval_seasons):
    """goal_distributions.walk_forward, plus the frozen Dixon-Coles probabilities from the same fit
    (one Dixon-Coles fit per matchday instead of two)."""
    from tabet.ingest import LOCAL_TZ
    gdm = goal_distributions
    store, promoted = MatchStore(matches), promoted_team_seasons(matches)
    target = matches[matches["season"].isin(eval_seasons) & (matches["status"] == "completed")]
    rows = []
    for _, g in target.groupby(target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date, sort=True):
        cutoff = g["kickoff_utc"].min()
        train = store.completed_before(cutoff)
        dc = DixonColes(Hyperparams()).fit(train, cutoff, promoted)
        lam_tr = np.array([dc.expected_goals(h, a, s) for h, a, s in zip(train["home_id"], train["away_id"], train["season"])])
        x, y = train["home_goals"].to_numpy(int), train["away_goals"].to_numpy(int)
        w = np.exp(-gdm.XI * (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400)
        w = w / w.mean()
        r = gdm.fit_r(lam_tr[:, 0], lam_tr[:, 1], x, y, dc.rho, w)
        l3 = gdm.fit_l3(lam_tr[:, 0], lam_tr[:, 1], x, y, w)
        lam = np.array([dc.expected_goals(h, a, s) for h, a, s in zip(g["home_id"], g["away_id"], g["season"])])
        pnb = gdm._hda3(gdm.nb_grid(lam[:, 0], lam[:, 1], dc.rho, r))
        pbp = gdm._hda3(gdm.bp_grid(lam[:, 0], lam[:, 1], l3))
        pdc = dc.predict_frame(g).to_numpy()
        rows.append(pd.DataFrame({"match_id": g["match_id"].to_numpy(), "season": g["season"].to_numpy(),
                                  "kickoff_utc": g["kickoff_utc"].to_numpy(), "result": g["result"].to_numpy(),
                                  "lam_home": lam[:, 0], "lam_away": lam[:, 1], "rho": dc.rho, "nb_r": r, "bp_l3": l3,
                                  **{f"{k}_p_{o}": P[:, i] for k, P in (("dc", pdc), ("nb", pnb), ("bp", pbp))
                                     for i, o in enumerate(("home", "draw", "away"))}}))
    return pd.concat(rows, ignore_index=True)


allm = load_leagues()
parts = []
for lg in LEAGUES:
    m = allm[allm["league"] == lg].drop(columns="league").reset_index(drop=True)
    cache = out / f"preds_{lg}.csv"
    if cache.exists():
        parts.append(pd.read_csv(cache, parse_dates=["kickoff_utc"]))
        continue
    seasons = PRIMARY + CURRENT
    p = walk(m, seasons).assign(league=lg)
    p.to_csv(cache, index=False)
    print(f"{lg}: {len(p)} matches", flush=True)
    parts.append(p)

p = pd.concat(parts, ignore_index=True)
p["kickoff_utc"] = pd.to_datetime(p["kickoff_utc"], utc=True)
for k in ("dc", "bp", "nb"):
    p[f"ll_{k}"] = ll(p, k)


def contrast(df, a, b="dc"):
    mean, lo, hi = block_bootstrap(df[f"ll_{a}"].to_numpy(), df[f"ll_{b}"].to_numpy(), df["kickoff_utc"], 14)
    d = df[f"ll_{a}"] - df[f"ll_{b}"]
    return {"n": int(len(df)), "mean": mean, "se": float(d.std(ddof=1) / np.sqrt(len(d))), "ci95": [lo, hi]}


res = {}
prim = p[p["season"].isin(PRIMARY)]
res["primary_bp_minus_dc_2022_26"] = contrast(prim, "bp")
r = res["primary_bp_minus_dc_2022_26"]
res["primary_confirmed"] = bool(r["mean"] < 0 and r["ci95"][1] < 0)
res["secondary"] = {
    "nb_minus_dc_2022_26": contrast(prim, "nb"),
    "bp_minus_dc_2026_27_so_far": contrast(p[p["season"].isin(CURRENT)], "bp"),
    "bp_minus_dc_by_league": {lg: contrast(prim[prim["league"] == lg], "bp") for lg in LEAGUES},
    "lambda3_by_league_season": {f"{lg} {s}": float(g["bp_l3"].mean()) for (lg, s), g in p.groupby(["league", "season"])},
    "mean_log_loss": {k: float(prim[f"ll_{k}"].mean()) for k in ("dc", "bp", "nb")},
}
(out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
