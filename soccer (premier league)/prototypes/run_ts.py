"""Addendum TS of docs/PREREGISTRATION.md (team-specific rating speed, F16), frozen in PR #90 before this
file existed and before any SEALED2 data was downloaded.

    python prototypes/run_ts.py dev     # the four development leagues: lambda, withdrawal rule, MDE

The test phase (SEALED2) is added only if the development result passes the withdrawal rule.

Models (tabet/dynamic.py; each league fitted on its own matches only):
  D0  frozen Dixon-Coles (xi = 0.004, L2 = 10) refitted every matchday (tabet.backtest.walk_forward)
  S2  score-driven strengths with the summer factor, one shared alpha_att / alpha_def
  TS  S2 with per-team speeds alpha * exp(u_i), alpha * exp(v_i), penalty lambda * sum(u^2 + v^2),
      u, v in [-2, 2]; S2's and TS's static parameters are refitted at the start of every scored season.

Implementation details the addendum leaves open, fixed here before any development result is seen:
  * Data: football-data season files via tabet.fd_cache. A row is a match if it has both scores.
  * "At least 5 earlier matches": matches in that league's file with an earlier kickoff (any season).
  * Scored seasons: SP1, I1, F1 2007-08 -> 2013-14; D1 2008-09 -> 2013-14 (first two seasons burn-in).
  * TS starts each season from that season's S2 parameters and the previous season's u, v.
  * Lambda: the grid value with the lowest equal-weight mean (over the four leagues) of TS log-loss on
    eligible development matches. Ties go to the larger lambda (closer to S2).
  * MDE: FAT's formula with the block SE of the development TS - D0 difference and 30,000 expected
    test matches.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as DX  # noqa: E402
import run_nlx as N  # noqa: E402
from tabet import dynamic as D  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet.backtest import walk_forward as dc_walk_forward  # noqa: E402
from tabet.experiments import per_match_logloss  # noqa: E402
from tabet.model import Hyperparams  # noqa: E402

OUT = ROOT / "results" / "ts"
LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
DEV = {"SP1": (2005, 2013), "I1": (2005, 2013), "F1": (2005, 2013), "D1": (2006, 2013)}   # first, last start year
BURN_IN = 2
MIN_EARLIER = 5
EXPECTED_TEST_MATCHES = 30000


def _code(y: int) -> str:
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def league_matches(div: str, first: int, last: int) -> pd.DataFrame:
    m = fd_cache.seasons([div], [_code(y) for y in range(first, last + 1)])
    m = m[m["home_goals"].notna() & m["away_goals"].notna()].copy()
    m["status"] = "completed"
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    return m.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def earlier_counts(m: pd.DataFrame) -> pd.DataFrame:
    """Each match's number of earlier matches (strictly earlier kickoff) for the home and the away team."""
    long = pd.concat([m[["match_id", "kickoff_utc", "home_id"]].rename(columns={"home_id": "t"}).assign(side="h"),
                      m[["match_id", "kickoff_utc", "away_id"]].rename(columns={"away_id": "t"}).assign(side="a")])
    long = long.sort_values("kickoff_utc")
    long["n"] = long.groupby("t")["kickoff_utc"].rank(method="min").astype(int) - 1
    return long.pivot(index="match_id", columns="side", values="n")


def one_league(div: str) -> pd.DataFrame:
    """Per-match log-loss of D0, S2 and TS at every lambda, on this league's eligible scored matches."""
    fd_cache._load_env()
    first, last = DEV[div]
    m = league_matches(div, first, last)
    scored = tuple(DX.season_label(y) for y in range(first + BURN_IN, last + 1))
    preds = {"D0": dc_walk_forward(m, scored, Hyperparams())}
    print(f"{div}: D0 done", flush=True)
    preds["S2"] = D.walk_forward(m, scored, summer=True)
    print(f"{div}: S2 done", flush=True)
    for lam in LAMBDAS:
        preds[f"TS_{lam:g}"] = D.walk_forward_ts(m, scored, lam, s2=preds["S2"])
        print(f"{div}: TS lambda={lam:g} done", flush=True)
    n = earlier_counts(m)
    base = m[m["season"].isin(scored)][["match_id", "season", "kickoff_utc", "result"]].copy()
    base["eligible"] = base["match_id"].map((n["h"] >= MIN_EARLIER) & (n["a"] >= MIN_EARLIER)).fillna(False)
    for k, p in preds.items():
        p = p.set_index("match_id")
        base[k] = per_match_logloss(p.reindex(base["match_id"]).reset_index())
        if k.startswith("TS"):
            base[f"uv_{k}"] = (p["uv_home"].reindex(base["match_id"]) + p["uv_away"].reindex(base["match_id"])).to_numpy()
    w = base[base["eligible"] & base[list(preds)].notna().all(axis=1)].drop(columns="eligible")
    return w.assign(div=div, date=pd.to_datetime(w["kickoff_utc"], utc=True))


def dev():
    for div, (first, last) in DEV.items():
        ledger.claim("TS", div, [DX.season_label(y) for y in range(first, last + 1)], "development")
    with ProcessPoolExecutor(len(DEV)) as ex:
        w = pd.concat(ex.map(one_league, DEV), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT / "dev_ledger.csv.gz", index=False)
    ll = {k: DX.league_balanced(w, k) for k in ["D0", "S2"] + [f"TS_{lam:g}" for lam in LAMBDAS]}
    lam = min(LAMBDAS, key=lambda x: (round(ll[f"TS_{x:g}"], 10), -x))
    ts = f"TS_{lam:g}"
    w["ts_d0"], w["ts_s2"], w["s2_d0"] = w[ts] - w["D0"], w[ts] - w["S2"], w["S2"] - w["D0"]
    res = {"addendum": "TS", "phase": "development", "lambda": lam, "n_dev": int(len(w)),
           "eligible_per_league_season": {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])},
           "log_loss_equal_weight": ll}
    for col, name in (("ts_d0", "TS - D0"), ("ts_s2", "TS - S2"), ("s2_d0", "S2 - D0")):
        mean, lo, hi = DX.block_ci(w, col)
        res[name] = {"mean": mean, "ci_95": [lo, hi], "by_league": {d: float(g[col].mean()) for d, g in w.groupby("div")}}
    se = N.block_se(w, "ts_d0")
    res["mde"] = {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
                  "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))}
    res["withdrawal_rule"] = {"dev_mean_TS_minus_D0": res["TS - D0"]["mean"],
                              "withdrawn": not res["TS - D0"]["mean"] < 0}
    (OUT / "dev.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("lambda", "n_dev", "log_loss_equal_weight", "TS - D0", "TS - S2", "S2 - D0",
                                          "mde", "withdrawal_rule")}, indent=2, default=float))


if __name__ == "__main__":
    {"dev": dev}[sys.argv[1]]()
