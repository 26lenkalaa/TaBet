"""Data for the site's research page (site/research.html): site/data/research.json.

Built from the site's own match files (site/data/matches-<season>.json) and the logger's record files, after
site_data writes them. Descriptive only: the confirmatory reads are the ones in docs/PROSPECTIVE_PROTOCOL.md.

  * backtest: V2x (confirmed-XI backtest) against closing and earlier odds, 2022-23 -> 2025-26:
    cumulative mean log-loss difference by date, per-season means with 14-day block intervals, a
    calibration table (all three outcomes pooled, ten bins), and promoted sides in their first 10 matches.
  * experiments: every pre-registered test's primary result with its 95% interval and what happened to the
    idea (EXPERIMENTS below, copied from docs/IMPROVEMENT_BRIEF.md §5.15-5.41; negative = better).
  * live: logged forecasts (pE, falling back to p0) against the market at the forecast time, with coverage
    of played fixtures since gameweek 11, and counts of shadow (A10) and kickoff (A11) records.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import block_bootstrap

ROOT = Path(__file__).resolve().parent.parent
PROSPECTIVE = ROOT / "results" / "prospective"
OUTCOMES = {"H": 0, "D": 1, "A": 2}
BACKTEST_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
LIVE_FROM_GW = 11
EARLY_N = 10

# Each pre-registered test's primary contrast (log-loss; below 0 = the idea helped), from the brief section
# named. status: "kept" (adopted or moved on to the shadow log), "live" (still being tested in the live log),
# "dropped". Withdrawn before any test (MGR, CAP) are listed in WITHDRAWN instead: they have no interval.
EXPERIMENTS = [
    ("Promoted-team prior (Championship data)", "§5.15", "prior − Dixon-Coles, promoted sides' first 10", -0.0134, -0.058, 0.028, 111, "dropped",
     "Too few matches: the interval is very wide and crosses 0."),
    ("Starting-XI strength, pre-xG seasons", "§5.16", "XI − baseline, PL 2018-21", -0.0009, -0.0022, 0.0004, 1140, "live",
     "Not confirmed on its own; the lineup layer is still being tested in the live log (pE)."),
    ("Transfer priors for new signings", "§5.17", "priors − position prior, PL 2024-26", -0.00025, -0.0010, 0.0005, 760, "dropped",
     "Close to 0 and the interval crosses 0."),
    ("Dynamic ratings (S2), first test", "§5.18", "S2 − Dixon-Coles, PL 2018-21", 0.0009, -0.0051, 0.0066, 1140, "dropped",
     "No gain on that test. Revisited later on more data (below), where it won."),
    ("Elo blend", "§5.20", "blend − Dixon-Coles, PL 2018-21", -0.0009, -0.0030, 0.0009, 1140, "dropped",
     "The interval crosses 0."),
    ("Bivariate Poisson goals", "§5.23", "BP − Dixon-Coles, 4 leagues 2022-26", -0.0006, -0.0017, 0.0004, 5562, "live",
     "Not confirmed; logged in the live test as pBP."),
    ("Table stakes (late-season motivation)", "§5.23", "stakes − baseline, PL 2018-21, gameweek 30+", 0.0035, -0.0063, 0.0149, 273, "dropped",
     "Worse on average, and the interval crosses 0."),
    ("Promoted-team mapping", "§5.28", "mapping − Dixon-Coles, promoted sides, PL 2000-15", -0.0031, -0.0203, 0.0124, None, "dropped",
     "Very wide interval crossing 0; its stop rule closed the line."),
    ("Boosted-tree stage 2", "§5.30", "boosted − linear, big four leagues", 0.0005, -0.0026, 0.0032, None, "dropped",
     "No better than the linear stage."),
    ("Defensive-XI block", "§5.31", "with − without", -0.0003, -0.0014, 0.0007, None, "dropped",
     "Close to 0 and the interval crosses 0."),
    ("Neural-network stage 2", "§5.33", "network − linear", 0.0002, -0.0025, 0.0029, None, "dropped",
     "No better than the linear stage; the line is closed."),
    ("Attacking-XI block", "§5.33", "with − without", -0.0007, -0.0024, 0.0010, None, "dropped",
     "The interval crosses 0."),
    ("Rest and fatigue", "§5.34", "with − without, sealed leagues", 0.00021, -0.00020, 0.00063, None, "dropped",
     "Slightly worse; the interval rules out any useful gain."),
    ("Ratings from all competitions", "§5.35", "all − league only, sealed leagues", -0.00026, -0.0008, 0.0003, None, "dropped",
     "Too small to matter; the interval rules out a useful gain."),
    ("Goalkeeper quality", "§5.37", "with − without, sealed leagues", -0.00025, -0.0006, 0.0002, None, "dropped",
     "Too small to matter; the interval rules out a useful gain."),
    ("Per-team rating speeds", "§5.38", "per-team − one shared speed, 13 new leagues", 0.00009, 0.00002, 0.00017, 44352, "dropped",
     "Slightly worse, and the whole interval is above 0."),
    ("Dynamic ratings (S2), 13 new leagues", "§5.38", "S2 − Dixon-Coles", -0.0024, -0.0037, -0.0010, 44352, "kept",
     "Better: the whole interval is below 0."),
    ("Dynamic ratings (S2), untouched PL seasons", "§5.39", "S2 − Dixon-Coles, PL 1995-2000 and 2008-15", -0.0051, -0.0093, -0.0011, 4492, "kept",
     "Better: the whole interval is below 0. Now in the shadow log against the live model."),
    ("Faster learning for new teams", "§5.40", "with − S2, 13 new leagues", 0.0, -0.00045, 0.00046, None, "dropped",
     "Exactly 0; the interval rules out a useful gain."),
    ("Transfer value (squad cost)", "§5.41", "with − Dixon-Coles, PL 2008-15", -0.0014, -0.0034, 0.0007, None, "live",
     "Promising but the interval crosses 0; now tested in the shadow log."),
]
WITHDRAWN = [("New-manager shifts", "§5.36", "every setting was worse on development data"),
             ("Capping blowouts", "§5.35", "every cap was worse on development data")]


def _experiments() -> dict:
    keys = ("idea", "section", "comparison", "mean", "lo", "hi", "n", "status", "why")
    return {"rows": [dict(zip(keys, e)) for e in EXPERIMENTS],
            "withdrawn": [dict(zip(("idea", "section", "why"), w)) for w in WITHDRAWN]}


def _ll(p, result) -> float:
    return -math.log(max(float(p[OUTCOMES[result]]), 1e-15))


def _first(items, keys):
    d = {x["key"]: x for x in items}
    return next((d[k] for k in keys if k in d), None)


def _ci(a: np.ndarray, b: np.ndarray, when: pd.Series) -> dict:
    """Mean difference with a 14-day block-bootstrap interval; no interval with fewer than 5 blocks (a
    handful of matches in one fortnight would give a degenerate one)."""
    blocks = ((when - when.min()).dt.days // 14).nunique() if len(when) else 0
    if len(a) < 5 or blocks < 5:
        return {"n": int(len(a)), "mean": float(np.mean(a - b)) if len(a) else None, "ci_95": None}
    mean, lo, hi = block_bootstrap(a, b, when.reset_index(drop=True), 14)
    return {"n": int(len(a)), "mean": mean, "ci_95": [lo, hi]}


def _matches(site_data: Path) -> pd.DataFrame:
    index = json.loads((site_data / "index.json").read_text())
    rows = []
    for season, f in index["files"].items():
        for m in json.loads((site_data / f).read_text())["matches"]:
            rows.append({**m, "season": season})
    df = pd.DataFrame(rows)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df.sort_values("kickoff").reset_index(drop=True)


def _promoted_early(df: pd.DataFrame) -> pd.Series:
    """True when either side is in its first 10 league matches of a season it was promoted into."""
    teams = {s: set(g["home"]) | set(g["away"]) for s, g in df.groupby("season")}
    seasons = sorted(teams)
    promoted = {(t, cur) for prev, cur in zip(seasons, seasons[1:]) for t in teams[cur] - teams[prev]}
    played = {}
    flag = []
    for r in df.itertuples():
        f = False
        for t in (r.home, r.away):
            n = played.get((t, r.season), 0)
            f |= (t, r.season) in promoted and n < EARLY_N
        flag.append(f)
        for t in (r.home, r.away):
            played[(t, r.season)] = played.get((t, r.season), 0) + 1
    return pd.Series(flag, index=df.index)


def _backtest(df: pd.DataFrame) -> dict:
    df = df.assign(promoted_early=_promoted_early(df))
    b = df[df["season"].isin(BACKTEST_SEASONS) & (df["status"] == "played") & df["result"].isin(list(OUTCOMES))]
    rows = []
    for r in b.itertuples():
        f = _first(r.forecasts, ["V2x"])
        close, pre = _first(r.market, ["close"]), _first(r.market, ["pre"])
        if f is None or close is None:
            continue
        rows.append({"kickoff": r.kickoff, "season": r.season, "result": r.result, "p": f["p"], "pc": close["p"],
                     "ll": _ll(f["p"], r.result), "ll_close": _ll(close["p"], r.result),
                     "ll_pre": _ll(pre["p"], r.result) if pre else None, "promoted_early": r.promoted_early})
    t = pd.DataFrame(rows)
    t["d_close"] = t["ll"] - t["ll_close"]
    day = t.groupby(t["kickoff"].dt.strftime("%Y-%m-%d"))
    cum = pd.DataFrame({"n": day.size().cumsum(), "sum_close": day["d_close"].sum().cumsum()})
    tp = t.dropna(subset=["ll_pre"])
    dp = tp.groupby(tp["kickoff"].dt.strftime("%Y-%m-%d"))
    cum_pre = (dp.apply(lambda g: (g["ll"] - g["ll_pre"]).sum()).cumsum() / dp.size().cumsum()).rename("vs_pre")
    cum["vs_close"] = cum["sum_close"] / cum["n"]
    cum = cum.join(cum_pre)
    calib = []
    for name, col in (("TaBet (V2x)", "p"), ("Closing odds", "pc")):
        probs = np.array(t[col].tolist())
        y = np.zeros_like(probs)
        y[np.arange(len(t)), t["result"].map(OUTCOMES).to_numpy()] = 1
        pf, yf = probs.ravel(), y.ravel()
        bins = np.minimum((pf * 10).astype(int), 9)
        for k in range(10):
            sel = bins == k
            if sel.sum() >= 20:
                calib.append({"model": name, "bin": k, "predicted": float(pf[sel].mean()), "observed": float(yf[sel].mean()),
                              "n": int(sel.sum())})
    out = {"n": int(len(t)), "log_loss": {"TaBet (V2x)": float(t["ll"].mean()), "Closing odds": float(t["ll_close"].mean()),
                                          "Earlier odds": float(tp["ll_pre"].mean())},
           "vs_close": _ci(t["ll"].to_numpy(), t["ll_close"].to_numpy(), t["kickoff"]),
           "vs_pre": _ci(tp["ll"].to_numpy(), tp["ll_pre"].to_numpy(), tp["kickoff"]),
           "by_season": [{"season": s, **_ci(g["ll"].to_numpy(), g["ll_close"].to_numpy(), g["kickoff"])}
                         for s, g in t.groupby("season")],
           "promoted_early": {k: _ci(g["ll"].to_numpy(), g["ll_close"].to_numpy(), g["kickoff"])
                              for k, g in (("promoted side in its first 10", t[t["promoted_early"]]),
                                           ("all other matches", t[~t["promoted_early"]]))},
           "cumulative": [{"date": d, "n": int(r.n), "vs_close": float(r.vs_close),
                           "vs_pre": None if pd.isna(r.vs_pre) else float(r.vs_pre)} for d, r in cum.iterrows()],
           "calibration": calib}
    return out


def _count(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return len(recs), sum(1 for r in recs if "error" in r)


def _live(df: pd.DataFrame) -> dict:
    live = df[(df["season"] == df["season"].max()) & (df["gameweek"].fillna(0) >= LIVE_FROM_GW)]
    played = live[(live["status"] == "played") & live["result"].isin(list(OUTCOMES))]
    rows, missing = [], 0
    for r in played.itertuples():
        f = _first([x for x in r.forecasts if x.get("kind") == "logged"], ["pE", "p0"])
        if f is None:
            missing += 1
            continue
        mk = _first(r.market, ["matched"])
        rows.append({"kickoff": r.kickoff, "gameweek": int(r.gameweek), "match": f"{r.home} {r.score} {r.away}",
                     "forecast": f["key"], "ll": _ll(f["p"], r.result),
                     "ll_market": _ll(mk["p"], r.result) if mk else None,
                     "role": "fallback" if "fallback" in (f.get("note") or "") else "primary"})
    t = pd.DataFrame(rows)
    shadow, shadow_err = _count(PROSPECTIVE / "shadow.jsonl")
    kick, _ = _count(PROSPECTIVE / "kickoff.jsonl")
    out = {"from_gameweek": LIVE_FROM_GW, "played": int(len(played)), "logged": int(len(t)), "missing": missing,
           "primary": int((t["role"] == "primary").sum()) if len(t) else 0,
           "shadow_records": shadow, "shadow_errors": shadow_err, "kickoff_records": kick, "matches": []}
    if len(t):
        both = t.dropna(subset=["ll_market"])
        out["vs_market"] = _ci(both["ll"].to_numpy(), both["ll_market"].to_numpy(), both["kickoff"])
        out["log_loss"] = {"TaBet (logged)": float(both["ll"].mean()) if len(both) else None,
                           "Market at forecast time": float(both["ll_market"].mean()) if len(both) else None}
        out["matches"] = [{"date": r.kickoff.strftime("%Y-%m-%d"), "gameweek": r.gameweek, "match": r.match,
                           "forecast": r.forecast, "role": r.role, "ll": round(r.ll, 4),
                           "ll_market": None if pd.isna(r.ll_market) else round(r.ll_market, 4)} for r in t.itertuples()]
    return out


def build(site_data: Path) -> dict:
    df = _matches(site_data)
    data = {"built_utc": pd.Timestamp.now(tz="UTC").floor("min").isoformat(),
            "note": "Descriptive. The confirmatory reads are gameweeks 20-38 (docs/PROSPECTIVE_PROTOCOL.md).",
            "backtest": _backtest(df), "live": _live(df), "experiments": _experiments()}
    (site_data / "research.json").write_text(json.dumps(data, separators=(",", ":"), default=float) + "\n")
    return data
