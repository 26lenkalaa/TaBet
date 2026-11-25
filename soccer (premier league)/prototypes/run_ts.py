"""Addendum TS of docs/PREREGISTRATION.md (team-specific rating speed, F16), frozen in PR #90 before this
file existed and before any SEALED2 data was downloaded.

    python prototypes/run_ts.py dev     # the four development leagues: lambda, withdrawal rule, MDE
    python prototypes/run_ts.py test    # SEALED2, the 13 sealed leagues: scored once (added after dev
                                        # passed the withdrawal rule, before any SEALED2 file was downloaded)

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

Test phase details, fixed before any SEALED2 file was downloaded:
  * Files: football-data's extra-league files https://www.football-data.co.uk/new/<CODE>.csv, one per
    league (all seasons). Each is downloaded once, its SHA-256 recorded, and a copy kept in the private
    bucket football-data (sealed2/<CODE>.csv); later runs read the stored copy.
  * Columns: Season, Date (dd/mm/yyyy), Time (UK time; 15:00 if missing), Home, Away, HG, AG. Season
    "Y/Y+1" and calendar season "Y" are both labelled "Y-(Y+1)". Kickoffs are converted to UTC.
  * Seasons outside the ledger's 2012-13 -> 2025-26 are dropped (recorded). A season counts as completed
    before 2026-11-24 if its last row kicked off before 2026-07-01 (every August-May season through
    2025-26 and calendar season through 2025; a calendar season 2026 is still running). Uncompleted
    seasons are dropped (recorded). The first two remaining seasons are history only.
  * Run condition per league-season, computed before any model is fitted: at least 98% of rows have
    both scores and at least 50 eligible matches (both teams with at least 5 earlier matches).
  * Calendar-year vs August-May (a secondary split): by the file's season format for that season.
  * "The 10% of teams with the largest |u| + |v|": team-seasons, from each season's TS fit; a match is
    in the group if either team is at or above the 90th percentile of |u| + |v| over all test team-seasons.
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
SEALED2 = ("ARG", "BRA", "CHN", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "USA")
LEDGER_SEASONS = tuple(DX.season_label(y) for y in range(2012, 2026))
COMPLETED_BY = pd.Timestamp("2026-07-01", tz="UTC")
MIN_SCORED_SHARE, MIN_ELIGIBLE = 0.98, 50
NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"


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


# ---------------------------------------------------------------- test (SEALED2)

def raw_file(code: str) -> bytes:
    """The league's football-data file: the stored copy if there is one, else downloaded once and stored."""
    import urllib.request
    from tabet import supabase_storage as ss
    from tabet.fpl import _ssl_context
    path = f"sealed2/{code}.csv"
    try:
        return ss.get(fd_cache.BUCKET, path)
    except Exception:
        req = urllib.request.Request(NEW_URL.format(code=code), headers={"User-Agent": "TaBet research (student project)"})
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r:
            data = r.read()
        ss.put(fd_cache.BUCKET, path, data, "text/csv")
        return data


def parse_new(code: str, data: bytes) -> tuple[pd.DataFrame, dict]:
    """A football-data extra-league file in the match schema used here; returns (matches, notes)."""
    import io
    raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["Home"].str.strip() != ""].copy()
    season_raw = raw["Season"].str.strip()
    start = season_raw.str.slice(0, 4).astype(int)
    m = pd.DataFrame({"season": [DX.season_label(y) for y in start],
                      "calendar": ~season_raw.str.contains("/")})
    time = raw["Time"].str.strip().replace("", "15:00") if "Time" in raw else "15:00"
    local = pd.to_datetime(raw["Date"].str.strip() + " " + time, format="%d/%m/%Y %H:%M", errors="coerce")
    bad_dates = int(local.isna().sum())
    m["kickoff_utc"] = local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").to_numpy()
    for side, col in (("home", "Home"), ("away", "Away")):
        names = raw[col].str.strip().to_numpy()
        m[f"{side}_team"] = names
        m[f"{side}_id"] = [code.lower() + "_" + "".join(ch if ch.isalnum() else "-" for ch in n.lower()) for n in names]
    m["home_goals"] = pd.to_numeric(raw["HG"], errors="coerce").to_numpy()
    m["away_goals"] = pd.to_numeric(raw["AG"], errors="coerce").to_numpy()
    m["division"] = code
    notes = {"rows": int(len(m)), "unparsed_dates": bad_dates,
             "seasons_in_file": sorted(m["season"].unique().tolist())}
    m = m[m["kickoff_utc"].notna()]
    out_range = sorted(set(m["season"]) - set(LEDGER_SEASONS))
    m = m[m["season"].isin(LEDGER_SEASONS)]
    last = m.groupby("season")["kickoff_utc"].max()
    running = sorted(last[last >= COMPLETED_BY].index)
    m = m[~m["season"].isin(running)].copy()
    notes |= {"dropped_outside_ledger": out_range, "dropped_not_completed": running}
    m["match_id"] = (code + "-" + m["season"] + "-" + m["kickoff_utc"].dt.strftime("%Y%m%d%H%M") + "-"
                     + m["home_id"] + "-" + m["away_id"])
    m = m.drop_duplicates("match_id")
    return m.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True), notes


def check_league(code: str, m: pd.DataFrame) -> tuple[list[str], dict]:
    """Scored seasons (run condition) and the per-season record. Uses rows and fixture dates only."""
    seasons = sorted(m["season"].unique())
    done = m[m["home_goals"].notna() & m["away_goals"].notna()]
    n = earlier_counts(done)
    elig = done["match_id"].map((n["h"] >= MIN_EARLIER) & (n["a"] >= MIN_EARLIER)).fillna(False)
    rec, scored = {}, []
    for i, s in enumerate(seasons):
        rows = m[m["season"] == s]
        share = float((rows["home_goals"].notna() & rows["away_goals"].notna()).mean())
        ne = int(elig[done["season"] == s].sum())
        if i < BURN_IN:
            why = "history only (first two seasons)"
        elif share < MIN_SCORED_SHARE:
            why = f"history only: {share:.1%} of rows have both scores"
        elif ne < MIN_ELIGIBLE:
            why = f"history only: {ne} eligible matches"
        else:
            why = "scored"
            scored.append(s)
        rec[s] = {"rows": int(len(rows)), "scored_share": round(share, 4), "eligible": ne, "status": why}
    return scored, rec


def test_league(args) -> pd.DataFrame:
    code, lam, scored = args
    import torch
    torch.set_num_threads(1)
    fd_cache._load_env()
    m, _ = parse_new(code, raw_file(code))
    m = m[m["home_goals"].notna() & m["away_goals"].notna()].copy()
    m["status"] = "completed"
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    for c in ("home_team_raw", "away_team_raw"):
        m[c] = m[c.replace("_raw", "")]
    m["kickoff_time_known"] = True
    scored = tuple(scored)
    preds = {"D0": dc_walk_forward(m, scored, Hyperparams())}
    print(f"{code}: D0 done", flush=True)
    preds["S2"] = D.walk_forward(m, scored, summer=True)
    preds["TS"] = D.walk_forward_ts(m, scored, lam, s2=preds["S2"])
    print(f"{code}: S2, TS done", flush=True)
    n = earlier_counts(m)
    base = m[m["season"].isin(scored)][["match_id", "season", "kickoff_utc", "result", "calendar", "home_id", "away_id"]].copy()
    base["eligible"] = base["match_id"].map((n["h"] >= MIN_EARLIER) & (n["a"] >= MIN_EARLIER)).fillna(False)
    for k, p in preds.items():
        p = p.set_index("match_id")
        base[k] = per_match_logloss(p.reindex(base["match_id"]).reset_index())
    ts = preds["TS"].set_index("match_id")
    base["uv_home"] = ts["uv_home"].reindex(base["match_id"]).to_numpy()
    base["uv_away"] = ts["uv_away"].reindex(base["match_id"]).to_numpy()
    first = m.groupby("season")["kickoff_utc"].min()
    base["early"] = (base["kickoff_utc"] - base["season"].map(first)).dt.days < 56
    w = base[base["eligible"] & base[list(preds)].notna().all(axis=1)].drop(columns="eligible")
    return w.assign(div=code, date=pd.to_datetime(w["kickoff_utc"], utc=True))


def test():
    import hashlib
    choice = json.loads((OUT / "dev.json").read_text())          # frozen development choice
    if choice["withdrawal_rule"]["withdrawn"]:
        raise SystemExit("TS was withdrawn at development; SEALED2 stays untouched")
    lam = float(choice["lambda"])
    for code in SEALED2:
        ledger.claim("SEALED2", code, LEDGER_SEASONS, "test")
        ledger.claim("TS", code, LEDGER_SEASONS, "test")
    fd_cache._load_env()
    # 1. download, hash and check every league before any model is fitted
    check, plan = {"addendum": "TS", "lambda": lam, "leagues": {}}, {}
    for code in SEALED2:
        data = raw_file(code)
        m, notes = parse_new(code, data)
        scored, rec = check_league(code, m)
        check["leagues"][code] = {"sha256": hashlib.sha256(data).hexdigest(), **notes, "seasons": rec,
                                  "scored_seasons": scored,
                                  "in_mean": len(scored) >= 2}
        if scored:
            plan[code] = scored
        print(f"{code}: {len(scored)} scored seasons", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_check.json").write_text(json.dumps(check, indent=2))
    # 2. fit and score
    with ProcessPoolExecutor(min(10, len(plan))) as ex:
        w = pd.concat(ex.map(test_league, [(c, lam, s) for c, s in plan.items()]), ignore_index=True)
    w.to_csv(OUT / "test_ledger.csv.gz", index=False)
    in_mean = [c for c, v in check["leagues"].items() if v["in_mean"]]
    wm = w[w["div"].isin(in_mean)].copy()
    wm["ts_d0"], wm["ts_s2"], wm["s2_d0"] = wm["TS"] - wm["D0"], wm["TS"] - wm["S2"], wm["S2"] - wm["D0"]
    res = {"addendum": "TS", "phase": "test (SEALED2, scored once)", "lambda": lam,
           "frozen_dev": {k: choice[k] for k in ("lambda", "mde", "TS - D0")},
           "leagues_in_mean": in_mean, "n_test_matches": int(len(wm)),
           "matches_per_league": {d: int(len(g)) for d, g in wm.groupby("div")},
           "log_loss_equal_weight": {k: DX.league_balanced(wm, k) for k in ("D0", "S2", "TS")}}
    mean, lo, hi = DX.block_ci(wm, "ts_d0")
    verdict = "confirmed" if (mean <= -0.0005 and hi < 0) else ("closed" if lo > -0.0015 else "inconclusive")
    res["primary"] = {"TS - D0": {"mean": mean, "ci_95": [lo, hi]}, "decision": verdict}
    sec = {}
    for col, name in (("ts_s2", "TS - S2"), ("s2_d0", "S2 - D0")):
        mm, l, h = DX.block_ci(wm, col)
        sec[name] = {"mean": mm, "ci_95": [l, h]}
    e = wm[wm["early"]]
    sec["early season TS - D0 (mean, n)"] = [DX.league_balanced(e, "ts_d0"), int(len(e))]
    team_uv = pd.concat([wm[["div", "season", "home_id", "uv_home"]].set_axis(["div", "season", "t", "uv"], axis=1),
                         wm[["div", "season", "away_id", "uv_away"]].set_axis(["div", "season", "t", "uv"], axis=1)]
                        ).groupby(["div", "season", "t"])["uv"].first()
    cut = float(team_uv.quantile(0.9))
    top = (wm["uv_home"] >= cut) | (wm["uv_away"] >= cut)
    sec["matches with a top-10% |u|+|v| team, TS - D0 and TS - S2 (mean, n)"] = {
        "cut": cut, "TS - D0": float(wm[top]["ts_d0"].mean()), "TS - S2": float(wm[top]["ts_s2"].mean()), "n": int(top.sum())}
    sec["by league TS - D0 (mean, n)"] = {d: [float(g["ts_d0"].mean()), int(len(g))] for d, g in wm.groupby("div")}
    cal = wm["calendar"].astype(bool)
    sec["calendar-year vs August-May TS - D0 (mean, n)"] = {
        "calendar": [DX.league_balanced(wm[cal], "ts_d0"), int(cal.sum())],
        "august_may": [DX.league_balanced(wm[~cal], "ts_d0"), int((~cal).sum())] if (~cal).any() else None}
    sec["development result"] = choice["TS - D0"]
    res["secondary"] = sec
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss_equal_weight", "primary", "secondary")},
                     indent=2, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
