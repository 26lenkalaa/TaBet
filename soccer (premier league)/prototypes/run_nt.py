"""Addendum NT of docs/PREREGISTRATION.md (teams new to a league), merged in PR #104 before this file
existed and before any of its data was loaded for it.

    python prototypes/run_nt.py dev     # big four 2005-14: N, the withdrawal rule, the MDE
    python prototypes/run_nt.py test    # SEALED2: scored once (only if dev passes the withdrawal rule)

Implementation details the addendum leaves open, fixed here before any result is seen:
  * Data and season handling: addendum TS's (prototypes/run_ts.py): football-data files via fd_cache for
    development; SEALED2's stored files (sealed2/<CODE>.csv) for the test, their SHA-256 checked against
    results/ts/test_check.json before fitting, and TS's scored seasons.
  * S2 and NT are fitted on the same matches; NT starts each season from that season's S2 parameters and
    the previous season's (delta, kappa).
  * "Promoted against expansion clubs": football-data files cannot tell them apart, so this secondary is
    reported as teams **returning** to the file (played in an earlier season of it) against teams **new
    to the file**.
  * Equal-weight means and 14-day block intervals: run_dxi.league_balanced / block_ci, as in TS.
"""
from __future__ import annotations

import hashlib
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
import run_nlx as N_  # noqa: E402
import run_ts as TS  # noqa: E402
from tabet import dynamic as D  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet.experiments import per_match_logloss  # noqa: E402

OUT = ROOT / "results" / "nt"
NS = (10, 19, 38)
EXPECTED_TEST_MATCHES = 44000
TARGET_N = 10


def _score(m: pd.DataFrame, scored: tuple, preds: dict) -> pd.DataFrame:
    """Per-match log-loss of each model on every scored match that all models forecast, with the NT flags."""
    base = m[m["season"].isin(scored)][["match_id", "season", "kickoff_utc", "result"]].copy()
    ref = next(p for k, p in preds.items() if k.startswith("NT")).set_index("match_id")
    for c in ("new_h", "new_a", "n_h", "n_a", "returning_h", "returning_a", "delta_att", "delta_def", "kappa"):
        base[c] = ref[c].reindex(base["match_id"]).to_numpy()
    for k, p in preds.items():
        q = p.set_index("match_id").reindex(base["match_id"]).reset_index()
        q["result"] = base["result"].to_numpy()
        ok = q["p_home"].notna().to_numpy()
        ll = np.full(len(q), np.nan)
        ll[ok] = per_match_logloss(q[ok])
        base[k] = ll
    w = base[base[list(preds)].notna().all(axis=1)].copy()
    w["target"] = (w["new_h"].astype(bool) & (w["n_h"] < TARGET_N)) | (w["new_a"].astype(bool) & (w["n_a"] < TARGET_N))
    return w


# ---------------------------------------------------------------- development

def dev_league(div: str) -> pd.DataFrame:
    import torch
    torch.set_num_threads(1)
    fd_cache._load_env()
    first, last = TS.DEV[div]
    m = TS.league_matches(div, first, last)
    scored = tuple(DX.season_label(y) for y in range(first + TS.BURN_IN, last + 1))
    preds = {"S2": D.walk_forward(m, scored, summer=True)}
    for n in NS:
        preds[f"NT_{n}"] = D.walk_forward_nt(m, scored, n, "both", s2=preds["S2"])
        print(f"{div}: NT N={n} done", flush=True)
    w = _score(m, scored, preds)
    return w.assign(div=div, date=pd.to_datetime(w["kickoff_utc"], utc=True))


def dev():
    for div, (first, last) in TS.DEV.items():
        ledger.claim("NT", div, [DX.season_label(y) for y in range(first, last + 1)], "development")
    with ProcessPoolExecutor(len(TS.DEV)) as ex:
        w = pd.concat(ex.map(dev_league, TS.DEV), ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT / "dev_ledger.csv.gz", index=False)
    ll = {k: DX.league_balanced(w, k) for k in ["S2"] + [f"NT_{n}" for n in NS]}
    n_best = min(NS, key=lambda n: (round(ll[f"NT_{n}"], 10), -n))
    nt = f"NT_{n_best}"
    w["d"] = w[nt] - w["S2"]
    mean, lo, hi = DX.block_ci(w, "d")
    se = N_.block_se(w, "d")
    t = w[w["target"]]
    res = {"addendum": "NT", "phase": "development", "N": n_best, "n_dev": int(len(w)),
           "log_loss_equal_weight": ll,
           "NT - S2": {"mean": mean, "ci_95": [lo, hi], "by_league": {d: float(g["d"].mean()) for d, g in w.groupby("div")}},
           "target group (new team, first 10 matches): NT - S2 (mean, n)": [DX.league_balanced(t, "d"), int(len(t))],
           "fitted delta and kappa by league-season": w.groupby(["div", "season"])[["delta_att", "delta_def", "kappa"]].first()
                                                        .round(4).reset_index().to_dict("records"),
           "mde": {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
                   "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))},
           "withdrawal_rule": {"dev_mean_NT_minus_S2": mean, "withdrawn": not mean < 0}}
    (OUT / "dev.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "fitted delta and kappa by league-season"}, indent=2, default=float))


# ---------------------------------------------------------------- test (SEALED2)

def test_league(args) -> pd.DataFrame:
    code, n, scored = args
    cache = OUT / "test_cache" / f"{code}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    import torch
    torch.set_num_threads(1)
    fd_cache._load_env()
    m, _ = TS.parse_new(code, TS.raw_file(code))
    m = m[m["home_goals"].notna() & m["away_goals"].notna()].copy()
    m["status"] = "completed"
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    scored = tuple(scored)
    preds = {"S2": D.walk_forward(m, scored, summer=True)}
    for v in ("both", "entry", "speed"):
        preds[f"NT_{v}"] = D.walk_forward_nt(m, scored, n, v, s2=preds["S2"])
    print(f"{code}: done", flush=True)
    w = _score(m, scored, preds)
    first = m.groupby("season")["kickoff_utc"].min()
    w["early"] = (w["kickoff_utc"] - w["season"].map(first)).dt.days < 56
    w = w.assign(div=code, date=pd.to_datetime(w["kickoff_utc"], utc=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    w.to_pickle(cache)
    return w


def test():
    choice = json.loads((OUT / "dev.json").read_text())
    if choice["withdrawal_rule"]["withdrawn"]:
        raise SystemExit("NT was withdrawn at development; SEALED2 stays untouched")
    n = int(choice["N"])
    tcheck = json.loads((ROOT / "results" / "ts" / "test_check.json").read_text())
    for code in TS.SEALED2:
        ledger.claim("NT", code, TS.LEDGER_SEASONS, "test")
    fd_cache._load_env()
    plan, check = {}, {"addendum": "NT", "N": n, "leagues": {}}
    for code in TS.SEALED2:
        sha = hashlib.sha256(TS.raw_file(code)).hexdigest()
        rec = tcheck["leagues"][code]
        if sha != rec["sha256"]:
            raise SystemExit(f"{code}: stored file differs from the one TS hashed")
        check["leagues"][code] = {"sha256": sha, "scored_seasons": rec["scored_seasons"], "in_mean": rec["in_mean"]}
        if rec["scored_seasons"]:
            plan[code] = rec["scored_seasons"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_check.json").write_text(json.dumps(check, indent=2))
    with ProcessPoolExecutor(min(10, len(plan))) as ex:
        w = pd.concat(ex.map(test_league, [(c, n, s) for c, s in plan.items()]), ignore_index=True)
    w.to_csv(OUT / "test_ledger.csv.gz", index=False)
    w = w[w["div"].isin([c for c, v in check["leagues"].items() if v["in_mean"]])].copy()
    w["d"], w["d_entry"], w["d_speed"] = w["NT_both"] - w["S2"], w["NT_entry"] - w["S2"], w["NT_speed"] - w["S2"]
    mean, lo, hi = DX.block_ci(w, "d")
    verdict = "confirmed" if (mean <= -0.0002 and hi < 0) else ("closed" if lo > -0.0005 else "inconclusive")
    t = w[w["target"]]
    ret = (w["new_h"].astype(bool) & w["returning_h"].astype(bool)) | (w["new_a"].astype(bool) & w["returning_a"].astype(bool))
    fresh = (w["new_h"].astype(bool) & ~w["returning_h"].astype(bool)) | (w["new_a"].astype(bool) & ~w["returning_a"].astype(bool))
    sec = {"target group (new team, first 10 matches) (mean, n)": [DX.league_balanced(t, "d"), int(len(t))]}
    for col, name in (("d_entry", "entry level only - S2"), ("d_speed", "early speed only - S2")):
        mm, l, h = DX.block_ci(w, col)
        sec[name] = {"mean": mm, "ci_95": [l, h]}
    sec["matches with a returning new team (mean, n)"] = [float(w[ret]["d"].mean()), int(ret.sum())]
    sec["matches with a team new to the file (mean, n)"] = [float(w[fresh]["d"].mean()), int(fresh.sum())]
    sec["early season (mean, n)"] = [DX.league_balanced(w[w["early"]], "d"), int(w["early"].sum())]
    sec["by league (mean, n)"] = {d: [float(g["d"].mean()), int(len(g))] for d, g in w.groupby("div")}
    sec["fitted delta and kappa by league-season"] = w.groupby(["div", "season"])[["delta_att", "delta_def", "kappa"]].first() \
        .round(4).reset_index().to_dict("records")
    sec["development result"] = choice["NT - S2"]
    res = {"addendum": "NT", "phase": "test (SEALED2, scored once)", "N": n, "n_test_matches": int(len(w)),
           "log_loss_equal_weight": {k: DX.league_balanced(w, k) for k in ("S2", "NT_both", "NT_entry", "NT_speed")},
           "primary": {"NT - S2": {"mean": mean, "ci_95": [lo, hi]}, "decision": verdict}, "secondary": sec}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "secondary"}, indent=2, default=float))
    print(json.dumps({k: v for k, v in sec.items() if k != "fitted delta and kappa by league-season"}, indent=2, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
