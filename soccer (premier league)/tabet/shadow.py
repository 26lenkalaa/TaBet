"""S2 shadow records (docs/PROSPECTIVE_PROTOCOL.md, amendment A10).

After the logger appends live records, `write_shadow` adds one shadow record per live record to
results/prospective/shadow.jsonl: the same fixture, kickoff, cutoff and record role, with the full model
rebuilt on S2 (score-driven ratings, tabet/dynamic.py) instead of Dixon-Coles:

    p0S2   p0's stage 2 (adjusted xG, L2 = 1) on S2's rates
    pES2   pE's stage 2 (adjusted xG + XI, L2 = 100) on S2's rates, same sources and averaging as pE
    S2     S2 alone (results only)

A12 adds squad cost on the live D0 base (p0C, pEC); A13 adds squad cost on S2's base (p0S2C, pES2C). Each
runs in its own handler, so a failure is stored as cost_error / s2c_error and the earlier candidates stand.

Inputs are the live run's own: the training rows and matches that `forecast()` used (kept in
`forecast.LAST_CONTEXT`), and each live record's stored xG and XI features. Nothing is fetched again.
The live log is never read for writing or changed. Any failure becomes an error record; the logger
carries on.

S2 runs over every PL match from 2015-16 (res/history) to the cutoff. Its static parameters are fitted
once per season, at that season's first kickoff, on all earlier matches, and cached in
results/prospective/shadow_s2_params.json (they never change once fitted).
"""
from __future__ import annotations

import json
import signal
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import dynamic as D

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "prospective"
SHADOW = LOG_DIR / "shadow.jsonl"
PARAMS = LOG_DIR / "shadow_s2_params.json"
TIME_LIMIT_S = 120
SIDES = (("H", "home"), ("A", "away"))


# ---------------------------------------------------------------- S2 rates

def _all_matches(matches: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """PL matches from 2015-16: res/history plus the live run's matches, completed before the cutoff."""
    from .history import load_history_matches
    from .store import MatchStore
    hist = load_history_matches()
    live = MatchStore(matches).completed_before(cutoff)
    m = pd.concat([hist[hist["season"] < live["season"].min()], live], ignore_index=True)
    return m[m["status"] == "completed"]


def _season_starts(matches: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return matches.groupby("season")["kickoff_utc"].min().to_dict()


def season_params(done: pd.DataFrame, seasons: list[str], starts: dict[str, pd.Timestamp],
                  cache: Path = PARAMS) -> dict[str, np.ndarray]:
    """S2's static parameters for each season, fitted at its first kickoff on all earlier matches."""
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    out, theta0 = {}, D.START.copy()
    for s in sorted(seasons):
        if s in saved:
            out[s] = np.array(saved[s]["theta"])
            theta0 = out[s]
            continue
        before = done[done["kickoff_utc"] < starts[s]]
        p, _ = D.prepare(before)
        theta = D.fit(p, len(p.days), theta0)
        saved[s] = {"theta": [float(x) for x in theta], "names": list(D.NAMES), "fitted_on": int(len(before)),
                    "fitted_at": str(starts[s])}
        out[s], theta0 = theta, theta
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(saved, indent=2) + "\n")
    return out


def past_rates(done: pd.DataFrame, thetas: dict[str, np.ndarray]) -> pd.DataFrame:
    """Walk-forward S2 rates (lam_home, lam_away) for every completed match in a season with parameters."""
    p, m = D.prepare(done)
    parts = []
    for s, th in thetas.items():
        sel = (m["season"] == s).to_numpy()
        if not sel.any():
            continue
        lam, _, _ = D.filter_run(th, p)
        parts.append(pd.DataFrame({"match_id": m.loc[sel, "match_id"].to_numpy(),
                                   "lam_home": lam[sel, 0], "lam_away": lam[sel, 1]}))
    return pd.concat(parts, ignore_index=True).set_index("match_id")


def next_rates(done: pd.DataFrame, theta: np.ndarray, home: str, away: str, kickoff: pd.Timestamp,
               season: str) -> tuple[float, float]:
    """S2's pre-match rates for a future fixture: the filter's state after the last result, decayed to
    the fixture's date (and across a summer if the fixture is in a later season)."""
    p, m = D.prepare(done)
    teams = sorted(set(m["home_id"]) | set(m["away_id"]))
    _, (a, d), _ = D.filter_run(theta, p)
    mu, h, _, _, _, beta, bs = theta
    last = m["cutoff_utc"].max()
    f = np.exp(np.log(beta) * (pd.Timestamp(kickoff) - last).total_seconds() / 86400 / 7.0)
    if season != m["season"].iloc[-1]:
        f *= bs
    idx = {t: i for i, t in enumerate(teams)}
    ah, dh = (a[idx[home]] * f, d[idx[home]] * f) if home in idx else (0.0, 0.0)
    aa, da = (a[idx[away]] * f, d[idx[away]] * f) if away in idx else (0.0, 0.0)
    return float(np.exp(mu + h + ah - da)), float(np.exp(mu + aa - dh))


# ---------------------------------------------------------------- stage 2 on a different base

def stage2(rows: pd.DataFrame, log_base: np.ndarray, cols: list[str], l2: float, fill: bool) -> dict:
    """The live stage 2 (forecast.py) with the offset log_base instead of log(D0 rate)."""
    from .experiments import _fit_poisson
    X = rows[cols].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z = ((X.fillna(mu) if fill else X) - mu) / sd
    b = _fit_poisson(Z.to_numpy(), log_base, rows["goals"].to_numpy(float), l2)
    return {"coef": [float(v) for v in b], "mean": [float(v) for v in mu], "sd": [float(v) for v in sd]}


def _mult(st: dict, x: list[float]) -> float:
    z = (np.array(x) - np.array(st["mean"])) / np.array(st["sd"])
    return float(np.exp(st["coef"][0] + z @ np.array(st["coef"][1:])))


def _hda(lh: float, la: float, rho: float) -> dict:
    from .forecast import _hda as hda, _score_matrix
    ph, pd_, pa = hda(_score_matrix(lh, la, rho))
    return {"p_home": ph, "p_draw": pd_, "p_away": pa}


def candidates(rec: dict, base: tuple[float, float], rho: float, s1: dict, sx: dict, extra: dict | None = None) -> dict:
    """p0, pE and the sources rebuilt on `base` rates, from the live record's stored features.
    s1 / sx are the stage-2 fits (adjusted xG; adjusted xG + XI). `extra` = {"home": [...], "away": [...]}
    appends features after the usual ones (A12's dcost), matching extra columns in s1 and sx."""
    c = rec["components"]
    ex = extra or {"home": [], "away": []}
    lam0 = [base[i] * _mult(s1, c[f"xgadj_{side}"] + ex[side]) for i, side in enumerate(("home", "away"))]
    out = {"p0": _hda(lam0[0], lam0[1], rho)}
    for src, feats in (rec.get("xi_features") or {}).items():
        lam = []
        for i, (k, side) in enumerate(SIDES):
            xf = feats[k]
            bl = xf["f_backline_opp"] if xf["f_backline_opp"] is not None else sx["mean"][3]
            lam.append(base[i] * _mult(sx, c[f"xgadj_{side}"] + [xf["f_xi_position"], bl] + ex[side]))
        out[src] = _hda(lam[0], lam[1], rho)
    used = [s for s in rec.get("pE_sources", []) if s in out]
    out["pE"] = ({k: float(np.mean([out[s][k] for s in used])) for k in ("p_home", "p_draw", "p_away")}
                 if used else dict(out["p0"]))      # A2: no valid source -> pE is p0
    return out


# ---------------------------------------------------------------- writing

def _key(r: dict) -> tuple:
    return (int(r["fpl_fixture"]), r["horizon_h"], str(pd.Timestamp(r["kickoff_utc"])))


def _written() -> set:
    if not SHADOW.exists():
        return set()
    return {_key(json.loads(line)) for line in SHADOW.read_text().splitlines() if line.strip()}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout(f"shadow exceeded {TIME_LIMIT_S} s")


def build(records: list[dict], ctx: dict) -> list[dict]:
    from .forecast import ADJ, STAGE2_L2, STAGE2_L2_XI, XI
    matches, rows, cutoff = ctx["matches"], ctx["rows"], ctx["cutoff"]
    done = _all_matches(matches, cutoff)
    starts = _season_starts(pd.concat([done, matches], ignore_index=True))
    fixture_seasons = {r["fpl_fixture"]: _season_of(matches, r) for r in records}
    seasons = sorted(set(rows["season"]) | set(fixture_seasons.values()))
    thetas = season_params(done, seasons, starts)
    rates = past_rates(done, {s: thetas[s] for s in seasons if s in set(rows["season"])})
    tr = rows[rows["match_id"].isin(rates.index)].copy()
    lam = np.where(tr["side"] == "H", tr["match_id"].map(rates["lam_home"]), tr["match_id"].map(rates["lam_away"]))
    log_s2 = np.log(lam.astype(float))
    s1 = stage2(tr, log_s2, ADJ, STAGE2_L2, fill=False)
    sx = stage2(tr, log_s2, ADJ + XI, STAGE2_L2_XI, fill=True)
    out, bases = [], []
    for r in records:
        s = fixture_seasons[r["fpl_fixture"]]
        th = thetas[s]
        base = next_rates(done, th, r["home_id"], r["away_id"], pd.Timestamp(r["kickoff_utc"]), s)
        bases.append((base, float(th[2])))
        cand = candidates(r, base, float(th[2]), s1, sx)
        out.append({**{k: r[k] for k in ("fpl_fixture", "horizon_h", "record_role", "logged_utc", "kickoff_utc",
                                        "gameweek", "home_id", "away_id", "git_commit")},
                    "candidates": {"p0S2": cand["p0"], "pES2": cand["pE"], "S2": _hda(base[0], base[1], float(th[2])),
                                   **{f"{k}S2": v for k, v in cand.items() if k not in ("p0", "pE")}},
                    "pE_sources": r.get("pE_sources", []),
                    "s2": {"season": s, "theta": dict(zip(D.NAMES, map(float, th))), "rate_home": base[0], "rate_away": base[1]},
                    "stage2": {"p0S2": s1, "pES2": {**sx, "cols": ADJ + XI}}, "n_train_rows": int(len(tr))})
    add_cost(out, records, ctx)
    add_s2c(out, records, ctx, tr, log_s2, bases)
    return out


def add_s2c(out: list[dict], records: list[dict], ctx: dict, tr: pd.DataFrame, log_s2: np.ndarray,
            bases: list[tuple]) -> None:
    """A13: squad cost on S2's base (p0S2C, pES2C), added in place; a failure is stored as s2c_error."""
    from .forecast import ADJ, STAGE2_L2, STAGE2_L2_XI, XI
    try:
        got = s2c_candidates(records, ctx, tr, log_s2, bases, ADJ, XI, STAGE2_L2, STAGE2_L2_XI)
        for o, g in zip(out, got):
            o["candidates"].update(g["candidates"])
            o["s2c"] = g["s2c"]
    except Exception as e:  # noqa: BLE001
        for o in out:
            o["s2c_error"] = f"{type(e).__name__}: {e}"[:300]


def add_cost(out: list[dict], records: list[dict], ctx: dict) -> None:
    """A12: squad-cost candidates on the live D0 base, added to the shadow records in place. Its own
    handler, so a failure is stored as cost_error and A10's candidates always stand."""
    from .forecast import ADJ, STAGE2_L2, STAGE2_L2_XI, XI
    try:
        cost = cost_candidates(records, ctx, ADJ, XI, STAGE2_L2, STAGE2_L2_XI)
        for o, r in zip(out, records):
            got = cost.get(r["fpl_fixture"])
            if got:
                o["candidates"].update(got["candidates_cost"])
                o["cost"] = got["cost"]
    except Exception as e:  # noqa: BLE001
        for o in out:
            o["cost_error"] = f"{type(e).__name__}: {e}"[:300]


# ---------------------------------------------------------------- A12: squad cost

COST_CACHE = LOG_DIR / "shadow_cost_cache.json"


def cost_z(moves: pd.DataFrame, clubs: list[str], season: str, day: pd.Timestamp, cache: dict) -> dict:
    """z of log(1 + squad cost) across a season's clubs at a day (moves before that day; point-in-time fee
    index). Cached by season and day: a day's values are those first computed for it."""
    from . import transfers as T
    key = f"{season}|{pd.Timestamp(day):%Y-%m-%d}"
    if key not in cache:
        idx = T.fee_index(moves, before=day)
        v = pd.Series({c: np.log1p(T.squad_cost(moves, c, day, idx)["squad_cost"]) for c in clubs})
        cache[key] = {c: float(x) for c, x in ((v - v.mean()) / (v.std() or 1.0)).items()}
    return cache[key]


def add_dcost(rows: pd.DataFrame, matches: pd.DataFrame, moves: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Training rows with dcost = z(team) - z(opponent) at each row's cutoff."""
    clubs = {s: sorted(set(g["home_id"]) | set(g["away_id"])) for s, g in matches.groupby("season")}
    day = pd.to_datetime(rows["cutoff_utc"], utc=True).dt.normalize()
    out = np.empty(len(rows))
    for (season, d), idx in rows.groupby([rows["season"], day]).groups.items():
        z = cost_z(moves, clubs[season], season, d, cache)
        sub = rows.loc[idx]
        out[rows.index.get_indexer(idx)] = [z.get(t, 0.0) - z.get(o, 0.0) for t, o in zip(sub["team_id"], sub["opponent_id"])]
    return rows.assign(dcost=out)


def cost_candidates(records: list[dict], ctx: dict, ADJ, XI, L2, L2_XI) -> dict:
    """{fixture: {"candidates": {...p0C, pEC, sourcesC}, "cost": {...}}} on the live D0 base."""
    from . import transfers as T
    matches, rows, cutoff = ctx["matches"], ctx["rows"], ctx["cutoff"]
    cache = json.loads(COST_CACHE.read_text()) if COST_CACHE.exists() else {}
    moves = T.load()
    tr = add_dcost(rows.reset_index(drop=True), matches, moves, cache)
    s1 = stage2(tr, tr["log_dc"].to_numpy(), ADJ + ["dcost"], L2, fill=False)
    sx = stage2(tr, tr["log_dc"].to_numpy(), ADJ + XI + ["dcost"], L2_XI, fill=True)
    clubs = {s: sorted(set(g["home_id"]) | set(g["away_id"])) for s, g in matches.groupby("season")}
    out = {}
    for r in records:
        season = _season_of(matches, r)
        z = cost_z(moves, clubs[season], season, pd.Timestamp(r["logged_utc"]).normalize(), cache)
        dh = z.get(r["home_id"], 0.0) - z.get(r["away_id"], 0.0)
        c = r["components"]
        cand = candidates(r, (c["dc_rate_home"], c["dc_rate_away"]), c["rho"], s1, sx, {"home": [dh], "away": [-dh]})
        out[r["fpl_fixture"]] = {"cost": {"dcost_home": dh, "stage2": {"p0C": s1, "pEC": {**sx, "cols": ADJ + XI + ["dcost"]}}},
                                 "candidates_cost": {"p0C": cand["p0"], "pEC": cand["pE"],
                                                     **{f"{k}C": v for k, v in cand.items() if k not in ("p0", "pE")}}}
    COST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COST_CACHE.write_text(json.dumps(cache, separators=(",", ":")) + "\n")
    return out


def s2c_candidates(records: list[dict], ctx: dict, tr: pd.DataFrame, log_s2: np.ndarray, bases: list[tuple],
                   ADJ, XI, L2, L2_XI) -> list[dict]:
    """A13: A10's training rows (offset log S2 rate) with A12's dcost; candidates on each record's S2 rates."""
    from . import transfers as T
    matches = ctx["matches"]
    cache = json.loads(COST_CACHE.read_text()) if COST_CACHE.exists() else {}
    moves = T.load()
    trc = add_dcost(tr.reset_index(drop=True), matches, moves, cache)
    s1 = stage2(trc, log_s2, ADJ + ["dcost"], L2, fill=False)
    sx = stage2(trc, log_s2, ADJ + XI + ["dcost"], L2_XI, fill=True)
    clubs = {s: sorted(set(g["home_id"]) | set(g["away_id"])) for s, g in matches.groupby("season")}
    out = []
    for r, (base, rho) in zip(records, bases):
        season = _season_of(matches, r)
        z = cost_z(moves, clubs[season], season, pd.Timestamp(r["logged_utc"]).normalize(), cache)
        dh = z.get(r["home_id"], 0.0) - z.get(r["away_id"], 0.0)
        cand = candidates(r, base, rho, s1, sx, {"home": [dh], "away": [-dh]})
        out.append({"s2c": {"dcost_home": dh, "stage2": {"p0S2C": s1, "pES2C": {**sx, "cols": ADJ + XI + ["dcost"]}}},
                    "candidates": {"p0S2C": cand["p0"], "pES2C": cand["pE"],
                                   **{f"{k}S2C": v for k, v in cand.items() if k not in ("p0", "pE")}}})
    COST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    COST_CACHE.write_text(json.dumps(cache, separators=(",", ":")) + "\n")
    return out


def _season_of(matches: pd.DataFrame, r: dict) -> str:
    k = pd.Timestamp(r["kickoff_utc"])
    s = matches.loc[matches["kickoff_utc"] <= k, "season"]
    return str(s.max()) if len(s) else str(matches["season"].min())


def write_shadow(records: list[dict], ctx: dict | None = None) -> int:
    """Append shadow records for these live records; never raises. Returns the number written."""
    if not records:
        return 0
    if ctx is None:
        from .forecast import LAST_CONTEXT
        ctx = LAST_CONTEXT
    have = _written()
    todo = [r for r in records if _key(r) not in have]
    if not todo:
        return 0
    old = signal.signal(signal.SIGALRM, _alarm) if hasattr(signal, "SIGALRM") else None
    if old is not None:
        signal.alarm(TIME_LIMIT_S)
    try:
        out = build(todo, ctx)
    except BaseException as e:  # noqa: BLE001 - the live log must never notice
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        err = f"{type(e).__name__}: {e}"[:500]
        print(f"shadow failed ({err})")
        print("".join(traceback.format_exception(e))[-1500:])
        out = [{**{k: r.get(k) for k in ("fpl_fixture", "horizon_h", "record_role", "logged_utc", "kickoff_utc",
                                         "gameweek", "home_id", "away_id", "git_commit")}, "error": err} for r in todo]
    finally:
        if old is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    SHADOW.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW.open("a") as fh:
        for r in out:
            fh.write(json.dumps(r, default=str) + "\n")
    return len(out)
