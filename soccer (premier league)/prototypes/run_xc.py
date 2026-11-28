"""Addenda XC (team ratings from all competitions) and CAP (capping blowouts) of docs/PREREGISTRATION.md,
frozen in PR #82 before this file existed and before any sealed data was loaded.

    python prototypes/run_xc.py dev     # 9 development leagues, 2021-22 -> 2025-26: XC's and CAP's choices, MDEs
    python prototypes/run_xc.py test    # the 9 sealed leagues, 2020-21 -> 2025-26: both families, scored once

Implementation details the addenda leave open, fixed here before any data is loaded:
  * Data: API-Football fixture lists (Supabase), which carry the 90-minute score (score.fulltime) and the
    round. League matches count when status is FT; cup and European matches when FT, AET or PEN.
  * Ratings belong to a team *in a league* ("Burnley in the Championship"), so with both cross weights at 0
    the joint fit is exactly one Dixon-Coles per league (the current model). In a cup or European match a
    team plays as its current league's entity: the league of its most recent domestic league match in the
    previous 365 days. A team-in-league is "rated" with at least 20 matches in that league in the fit window.
  * League strength: each league has a shift s_L added to all its teams' attack and defence (the Premier
    League is the reference, s = 0). It cancels within a league, so only cross-league matches identify it,
    and it is not penalised. Variant A's unrated teams share one more shift, s_U, and an L2 of 100.
  * Weights: each match's time decay exp(-xi * age), divided by the mean decay of that league's league
    matches in the window (league matches; as the current fit, which normalises per league) or of all league
    matches in the window (cup and European matches, then times w_E or w_C).
  * Refits weekly (Monday 00:00 UTC) on the previous 3 years. A league-season match is scored when its league
    has at least 150 league matches in the window. A team-in-league with no matches in the window gets
    rating 0 (league average), as the current fit does, for every model alike.
  * Expected test matches for the MDEs: 15,500.
  * Stage-2 secondary: addendum G's ten team-form features and league indicators (prototypes/run_nlx.py,
    run_fat.py), fitted on top of LO's and of XC's chosen rates, L2 chosen on development.
"""
from __future__ import annotations

import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as D  # noqa: E402
import run_fat as F  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402
from tabet.experiments import _probs  # noqa: E402
from tabet.model import RHO_BOUND, Hyperparams  # noqa: E402

OUT = ROOT / "results" / "xc"
DEV9 = {39: "E0", **F.DEV8}
E1 = {40: "E1"}
SEALED9 = F.SEALED9
EUROPE, CUPS_DEV, CUPS_SEALED = F.EUROPE, F.CUPS_DEV, F.CUPS_SEALED
SEASONS = F.SEASONS                                   # 2015 .. 2025 (API start years)
DEV_SEASONS = F.DEV_SEASONS                           # 2021-22 .. 2025-26
TEST_SEASONS = F.TEST_SEASONS                         # 2020-21 .. 2025-26
WINDOW = pd.Timedelta(days=3 * 365)
CURRENT = pd.Timedelta(days=365)
MIN_LEAGUE, MIN_RATED = 150, 20
L2, L2_UNRATED = Hyperparams().l2, 100.0
XI = Hyperparams().xi
W_E, W_C, VARIANTS, CAPS = (0.25, 0.5, 1.0), (0.0, 0.5, 1.0), ("A", "B"), (3, 4, 5)
EXPECTED_TEST_MATCHES = 15500
MIN_COMPLETE = 0.95
FINISHED_CROSS = {"FT", "AET", "PEN"}


# ---------------------------------------------------------------- loading

def _parse_list(listing: list, league: int, season: int) -> list[dict]:
    out = []
    for f in listing:
        fx, sc = f.get("fixture") or {}, (f.get("score") or {}).get("fulltime") or {}
        out.append({"match_id": str(fx.get("id")), "league": league, "season": D.season_label(season),
                    "kickoff_utc": fx.get("date"), "status_short": (fx.get("status") or {}).get("short"),
                    "home_id": str(f["teams"]["home"]["id"]), "away_id": str(f["teams"]["away"]["id"]),
                    "x": sc.get("home"), "y": sc.get("away"), "round": (f.get("league") or {}).get("round") or ""})
    return out


def load(ids: dict[int, str]) -> pd.DataFrame:
    """Every listed fixture of these competitions, all seasons, cached per competition."""
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = []
    for league, name in ids.items():
        path = f"derived/xc_{league}_{SEASONS[0]}_{SEASONS[-1]}.parquet"
        try:
            t = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        except RuntimeError:
            rows = []
            for season in SEASONS:
                lp = D._retry(store.list_path, league, season)
                if lp is not None:
                    rows += _parse_list(D._retry(store.get, lp).get("response") or [], league, season)
            t = pd.DataFrame(rows)
            buf = io.BytesIO()
            t.to_parquet(buf, index=False)
            D._retry(ss.put, BUCKET, path, buf.getvalue())
        print(f"{name}: {len(t)} fixtures", flush=True)
        parts.append(t)
    m = pd.concat(parts, ignore_index=True)
    m["kickoff_utc"] = pd.to_datetime(m["kickoff_utc"], utc=True)
    return m


def tables(leagues: dict[int, str], comps: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(league and cross-competition matches with a 90-minute score, sorted by kickoff; the raw lists)."""
    lg, cr = load(leagues), load(comps)
    raw = pd.concat([lg, cr], ignore_index=True)
    lg = lg[(lg["status_short"] == "FT") & lg["x"].notna() & lg["y"].notna()].assign(kind="league")
    cr = cr[cr["status_short"].isin(FINISHED_CROSS) & cr["x"].notna() & cr["y"].notna()].copy()
    cr["kind"] = np.where(cr["league"].isin(list(EUROPE)), "europe", "cup")
    m = pd.concat([lg, cr], ignore_index=True).drop_duplicates("match_id")
    m["x"], m["y"] = m["x"].astype(float), m["y"].astype(float)
    r = m["round"].str.lower()
    m["neutral"] = (m["kind"] != "league") & ((r == "final") | ((m["league"] == 45) & r.str.contains("semi")))
    m["div"] = m["league"].map({**leagues})
    return m.sort_values("kickoff_utc").reset_index(drop=True), raw


def current_league(m: pd.DataFrame) -> pd.DataFrame:
    """For each cross-competition match, each side's current league (or None): the league of its most
    recent domestic league match in the previous 365 days."""
    lg = m[m["kind"] == "league"]
    long = pd.concat([lg[["kickoff_utc", "home_id", "league"]].rename(columns={"home_id": "team"}),
                      lg[["kickoff_utc", "away_id", "league"]].rename(columns={"away_id": "team"})]).sort_values("kickoff_utc")
    ns = lambda x: x.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()  # noqa: E731
    by = {t: (ns(g["kickoff_utc"]), g["league"].to_numpy()) for t, g in long.groupby("team")}
    cur = int(CURRENT.value)
    cr = m[m["kind"] != "league"]
    out = {}
    for mid, t0, h, a in zip(cr["match_id"], ns(cr["kickoff_utc"]), cr["home_id"], cr["away_id"]):
        res = []
        for t in (h, a):
            k, l = by.get(t, (np.array([], dtype=np.int64), np.array([])))
            i = int(np.searchsorted(k, t0, "left"))
            res.append(int(l[i - 1]) if i > 0 and t0 - k[i - 1] <= cur else None)
        out[mid] = res
    m = m.copy()
    m["lg_h"] = m.apply(lambda r: r["league"] if r["kind"] == "league" else out[r["match_id"]][0], axis=1)
    m["lg_a"] = m.apply(lambda r: r["league"] if r["kind"] == "league" else out[r["match_id"]][1], axis=1)
    return m


# ---------------------------------------------------------------- the joint fit

def fit(tr: pd.DataFrame, cut: pd.Timestamp, w_e: float, w_c: float, variant: str, cap: int | None) -> dict:
    """Joint Dixon-Coles on the window `tr` (league and cross matches before `cut`). Returns parameters
    for scoring league matches: mu, home and rho per league, attack and defence per (team, league)."""
    lg_rows = tr[tr["kind"] == "league"]
    n_in = pd.concat([lg_rows[["home_id", "league"]].set_axis(["t", "l"], axis=1),
                      lg_rows[["away_id", "league"]].set_axis(["t", "l"], axis=1)]).value_counts()
    rated = {k for k, v in n_in.items() if v >= MIN_RATED}
    rows = [lg_rows]
    cross = tr[(tr["kind"] != "league") & (((tr["kind"] == "europe") & (w_e > 0)) | ((tr["kind"] == "cup") & (w_c > 0)))]
    if len(cross):
        c = cross.copy()
        rh = [(t, l) in rated for t, l in zip(c["home_id"], c["lg_h"])]
        ra = [(t, l) in rated for t, l in zip(c["away_id"], c["lg_a"])]
        c["rh"], c["ra"] = rh, ra
        if variant == "B":
            c = c[c["rh"] & c["ra"]]
        rows.append(c)
    d = pd.concat(rows, ignore_index=True)
    is_lg = (d["kind"] == "league").to_numpy()

    # entities: a team in a league; in cross matches the team's current league if rated, else "U"
    if "rh" not in d:
        d["rh"], d["ra"] = True, True
    d["rh"], d["ra"] = d["rh"].fillna(True).astype(bool), d["ra"].fillna(True).astype(bool)
    eh = [(t, int(l)) if k else ((t, int(c)) if r_ else (t, "U"))
          for t, l, c, r_, k in zip(d["home_id"], d["league"], d["lg_h"].fillna(-1), d["rh"], is_lg)]
    ea = [(t, int(l)) if k else ((t, int(c)) if r_ else (t, "U"))
          for t, l, c, r_, k in zip(d["away_id"], d["league"], d["lg_a"].fillna(-1), d["ra"], is_lg)]
    ents = sorted(set(eh) | set(ea), key=str)
    ei = {e: i for i, e in enumerate(ents)}
    hi = np.array([ei[e] for e in eh])
    ai = np.array([ei[e] for e in ea])
    groups = sorted({e[1] for e in ents}, key=str)
    gi = {g: i for i, g in enumerate(groups)}
    gh = np.array([gi[e[1]] for e in eh])
    ga = np.array([gi[e[1]] for e in ea])
    ctx_names = sorted(set(d.loc[is_lg, "league"].astype(int))) + ["europe", "cup"]
    ci = {c: i for i, c in enumerate(ctx_names)}
    cx = np.array([ci[int(l)] if k else ci[kind] for l, k, kind in zip(d["league"], is_lg, d["kind"])])
    home = (~d["neutral"].to_numpy()).astype(float)
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    if cap is not None:
        x, y = np.minimum(x, cap), np.minimum(y, cap)
    age = (cut - d["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
    dec = np.exp(-XI * age)
    w = np.empty(len(d))
    lg_mean = pd.Series(dec[is_lg]).groupby(d.loc[is_lg, "league"].to_numpy()).mean().to_dict()
    for l, mval in lg_mean.items():
        sel = is_lg & (d["league"].to_numpy() == l)
        w[sel] = dec[sel] / mval
    if (~is_lg).any():
        base = dec[is_lg].mean()
        mult = np.where(d["kind"].to_numpy() == "europe", w_e, w_c)
        w[~is_lg] = mult[~is_lg] * dec[~is_lg] / base
    l2 = np.array([L2_UNRATED if e[1] == "U" else L2 for e in ents])
    n, G, C = len(ents), len(groups), len(ctx_names)
    c00, c01, c10, c11 = [(x == a) & (y == b) & is_lg for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
    ref = gi.get(39)                      # the Premier League is the reference (s = 0) when loaded

    def unpack(p):
        return p[:C], p[C:2 * C], p[2 * C:3 * C], p[3 * C:3 * C + G], p[3 * C + G:3 * C + G + n], p[3 * C + G + n:]

    def nll(p):
        mu, h, rho, s, a, dd = unpack(p)
        lh_ = mu[cx] + h[cx] * home + s[gh] + a[hi] - s[ga] - dd[ai]
        la_ = mu[cx] + s[ga] + a[ai] - s[gh] - dd[hi]
        lh, la = np.exp(lh_), np.exp(la_)
        r = rho[cx]
        tau = np.ones_like(lh)
        tau[c00] = 1 - lh[c00] * la[c00] * r[c00]
        tau[c01] = 1 + lh[c01] * r[c01]
        tau[c10] = 1 + la[c10] * r[c10]
        tau[c11] = 1 - r[c11]
        tau = np.maximum(tau, 1e-10)
        ll = x * lh_ - lh + y * la_ - la + np.log(tau)
        f = -(w * ll).sum() + (l2 * (a * a + dd * dd)).sum()
        g1, g2 = x - lh, y - la
        gr = np.zeros_like(lh)
        gr[c00] = -lh[c00] * la[c00] / tau[c00]
        g1[c00] += -lh[c00] * la[c00] * r[c00] / tau[c00]
        g2[c00] += -lh[c00] * la[c00] * r[c00] / tau[c00]
        gr[c01] = lh[c01] / tau[c01]
        g1[c01] += lh[c01] * r[c01] / tau[c01]
        gr[c10] = la[c10] / tau[c10]
        g2[c10] += la[c10] * r[c10] / tau[c10]
        gr[c11] = -1 / tau[c11]
        g1, g2, gr = w * g1, w * g2, w * gr
        g = np.zeros_like(p)
        g[:C] = np.bincount(cx, g1 + g2, C)
        g[C:2 * C] = np.bincount(cx, g1 * home, C)
        g[2 * C:3 * C] = np.bincount(cx, gr, C)
        g[3 * C:3 * C + G] = np.bincount(gh, g1 - g2, G) + np.bincount(ga, g2 - g1, G)
        g[3 * C + G:3 * C + G + n] = np.bincount(hi, g1, n) + np.bincount(ai, g2, n)
        g[3 * C + G + n:] = -np.bincount(ai, g1, n) - np.bincount(hi, g2, n)
        g = -g
        g[3 * C + G:3 * C + G + n] += 2 * l2 * a
        g[3 * C + G + n:] += 2 * l2 * dd
        return f, g

    p0 = np.zeros(3 * C + G + 2 * n)
    for c, i in ci.items():
        sel = cx == i
        if sel.any():
            p0[i] = np.log(max((x[sel] + y[sel]).mean() / 2, 0.1))
    bounds = [(None, None)] * (2 * C) + [((-RHO_BOUND, RHO_BOUND) if isinstance(c, int) else (0.0, 0.0)) for c in ctx_names]
    linked = (~is_lg).any()
    bounds += [((0.0, 0.0) if (not linked or i == ref or (ref is None and i == 0)) else (None, None)) for i in range(G)]
    bounds += [(None, None)] * (2 * n)
    res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    mu, h, rho, s, a, dd = unpack(res.x)
    return {"mu": {c: float(mu[i]) for c, i in ci.items() if isinstance(c, int)},
            "home": {c: float(h[i]) for c, i in ci.items() if isinstance(c, int)},
            "rho": {c: float(rho[i]) for c, i in ci.items() if isinstance(c, int)},
            "att": {e: float(a[i]) for e, i in ei.items()}, "def": {e: float(dd[i]) for e, i in ei.items()},
            "converged": bool(res.success)}


def rates(par: dict, games: pd.DataFrame) -> pd.DataFrame:
    """log home and away rates and rho for league matches from one fit."""
    out = []
    for g in games.itertuples():
        l = int(g.league)
        if l not in par["mu"]:
            continue
        ah, dh = par["att"].get((g.home_id, l), 0.0), par["def"].get((g.home_id, l), 0.0)
        aa, da = par["att"].get((g.away_id, l), 0.0), par["def"].get((g.away_id, l), 0.0)
        out.append((g.match_id, par["mu"][l] + par["home"][l] + ah - da, par["mu"][l] + aa - dh, par["rho"][l]))
    return pd.DataFrame(out, columns=["match_id", "lh", "la", "rho"])


# ---------------------------------------------------------------- walk-forward

def configs(phase: str, choice: dict | None = None) -> dict[str, tuple]:
    """name -> (w_E, w_C, variant, cap)."""
    if phase == "dev":
        c = {"LO": (0.0, 0.0, "A", None)}
        c.update({f"CAP_{k}": (0.0, 0.0, "A", k) for k in CAPS})
        c.update({f"XC_{v}_{we}_{wc}": (we, wc, v, None) for v in VARIANTS for we in W_E for wc in W_C})
        return c
    xc = (choice["xc"]["w_E"], choice["xc"]["w_C"], choice["xc"]["variant"])
    return {"LO": (0.0, 0.0, "A", None), "CAP": (0.0, 0.0, "A", choice["cap"]), "XC": (*xc, None),
            "XC_CAP": (*xc, choice["cap"])}


def _week(args):
    m, cut, eval_ids, cfgs = args
    tr = m[(m["kickoff_utc"] < cut) & (m["kickoff_utc"] >= cut - WINDOW)]
    games = m[m["match_id"].isin(eval_ids)]
    counts = tr[tr["kind"] == "league"]["league"].value_counts()
    games = games[games["league"].map(counts).fillna(0) >= MIN_LEAGUE]
    if games.empty:
        return []
    out = []
    for name, (we, wc, v, cap) in cfgs.items():
        r = rates(fit(tr, cut, we, wc, v, cap), games).assign(model=name)
        out.append(r)
    return out


def walk_forward(m: pd.DataFrame, eval_divs: list[str], seasons: list[str], cfgs: dict, workers: int = 6,
                 exclude: set | None = None) -> pd.DataFrame:
    """Weekly refits; per scored league match and model, log rates and rho. Long frame."""
    ev = m[(m["kind"] == "league") & m["div"].isin(eval_divs) & m["season"].isin(seasons)]
    if exclude:
        ev = ev[[(d, s) not in exclude for d, s in zip(ev["div"], ev["season"])]]
    week = ev["kickoff_utc"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    jobs = [(m[(m["kickoff_utc"] < c) & (m["kickoff_utc"] >= c - WINDOW) | m["match_id"].isin(set(g["match_id"]))],
             c, set(g["match_id"]), cfgs) for c, g in ev.groupby(week)]
    parts = []
    if workers == 1:
        results = map(_week, jobs)
    else:
        ex = ProcessPoolExecutor(workers)
        results = ex.map(_week, jobs, chunksize=1)
    for i, res in enumerate(results):
        parts += res
        if i % 20 == 0:
            print(f"  week {i + 1}/{len(jobs)}", flush=True)
    if workers != 1:
        ex.shutdown()
    return pd.concat(parts, ignore_index=True)


def scored(m: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Wide per-match log-loss table (one column per model) with div, season, date, early."""
    g = m.set_index("match_id")
    res = np.where(g["x"] > g["y"], 0, np.where(g["x"] < g["y"], 2, 1))
    res = pd.Series(res, index=g.index)
    out = {}
    for name, x in r.groupby("model"):
        p = _probs(np.exp(x["lh"].to_numpy()), np.exp(x["la"].to_numpy()), x["rho"].to_numpy())
        y = res.loc[x["match_id"]].to_numpy()
        out[name] = pd.Series(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)), index=x["match_id"].to_numpy())
    w = pd.DataFrame(out).dropna()
    meta = g.loc[w.index]
    lg = m[m["kind"] == "league"]
    played = pd.concat([lg[["match_id", "season", "home_id"]].rename(columns={"home_id": "t"}),
                        lg[["match_id", "season", "away_id"]].rename(columns={"away_id": "t"})])
    played["n"] = played.groupby(["season", "t"]).cumcount()
    early = played.groupby("match_id")["n"].min() < 10
    w = w.assign(match_id=w.index, div=meta["div"].to_numpy(), season=meta["season"].to_numpy(),
                 date=meta["kickoff_utc"].dt.normalize().to_numpy(), early=early.reindex(w.index).fillna(True).to_numpy())
    return w.reset_index(drop=True)


def check(m_all_lists: pd.DataFrame, m: pd.DataFrame, leagues: dict[int, str], seasons: list[str]) -> dict:
    """Run condition: per scored league-season, the share of finished league matches with a 90-minute score,
    and of finished cross matches involving its teams (between its first and last match) with one."""
    out = {}
    raw = m_all_lists
    for lg, code in leagues.items():
        for s in seasons:
            g = raw[(raw["league"] == lg) & (raw["season"] == s) & (raw["status_short"] == "FT")]
            if g.empty:
                out[f"{code} {s}"] = {"finished": 0, "league_share": 0.0}
                continue
            teams = set(g["home_id"]) | set(g["away_id"])
            o = raw[(~raw["league"].isin(list(SEALED9) + list(E1) + list(DEV9))) & raw["status_short"].isin(FINISHED_CROSS)
                    & (raw["home_id"].isin(teams) | raw["away_id"].isin(teams))
                    & raw["kickoff_utc"].between(g["kickoff_utc"].min(), g["kickoff_utc"].max())]
            out[f"{code} {s}"] = {"finished": int(len(g)), "league_share": round(float(g["x"].notna().mean()), 4),
                                  "cross": int(len(o)), "cross_share": round(float(o["x"].notna().mean()), 4) if len(o) else None}
    return out


def failing(comp: dict) -> dict:
    return {k: v for k, v in comp.items() if v["league_share"] < MIN_COMPLETE
            or (v.get("cross_share") is not None and v["cross_share"] < MIN_COMPLETE)}


# ---------------------------------------------------------------- stage-2 secondary

def stage2(m: pd.DataFrame, r: pd.DataFrame, model: str, leagues: dict[int, str], eval_divs, seasons, l2s) -> pd.DataFrame:
    """G's linear stage 2 on top of one model's rates (secondary): per-match log-loss for each L2."""
    import run_nlx as N
    tb = F.load(leagues)
    lm = tb["matches"]
    lm = lm[(lm["status_short"] == "FT") & lm["home_goals"].notna()].copy()
    lm["date"] = lm["kickoff_utc"].dt.normalize()
    lm["div"] = lm["league"].map(leagues)
    lm["result"] = np.where(lm["home_goals"] > lm["away_goals"], "H", np.where(lm["home_goals"] < lm["away_goals"], "A", "D"))
    played = pd.concat([lm[["match_id", "season", "home_id"]].rename(columns={"home_id": "t"}),
                        lm[["match_id", "season", "away_id"]].rename(columns={"away_id": "t"})])
    played["n"] = played.groupby(["season", "t"]).cumcount()
    lm["early"] = lm["match_id"].map(played.groupby("match_id")["n"].min() < 10).fillna(True)
    form = N.form_features(lm, tb["teams"].set_index(["match_id", "team_id"])["sot"])
    x = r[r["model"] == model].set_index("match_id")
    base = lm[lm["match_id"].isin(x.index)].copy()
    base["lh"], base["la"], base["rho"] = x.loc[base["match_id"], "lh"].to_numpy(), x.loc[base["match_id"], "la"].to_numpy(), x.loc[base["match_id"], "rho"].to_numpy()
    sides = []
    for side, own, opp, goals, lo, lp in (("H", "home_id", "away_id", "home_goals", "lh", "la"),
                                          ("A", "away_id", "home_id", "away_goals", "la", "lh")):
        z = base[["match_id", "div", "season", "date", "early", "result", "rho", own, opp, goals, lo, lp]].rename(
            columns={own: "own_id", opp: "opp_id", goals: "goals", lo: "log_dc", lp: "log_dc_opp"})
        z["side"] = side
        sides.append(z)
    rows = pd.concat(sides, ignore_index=True).merge(form, on=["match_id", "side"])
    rows["f5"] = rows["log_dc"] - rows["log_dc_opp"]
    rows["f10"] = (rows["side"] == "H").astype(float)
    lgs = [f"lg_{c}" for c in sorted(set(leagues.values()))]
    for c in lgs:
        rows[c] = (rows["div"] == c[3:]).astype(float)
    rows = rows[np.isfinite(rows[N.FORM]).all(axis=1)]
    both = rows.groupby("match_id")["side"].nunique()
    rows = rows[rows["match_id"].isin(both[both == 2].index)].sort_values(["date", "match_id", "side"]).reset_index(drop=True)
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    N.EXCLUDED = set()
    w, _ = N.walk_forward(rows, eval_divs, seasons, {f"S2_{l2}": ("lin", N.FORM + lgs, l2) for l2 in l2s})
    return w


# ---------------------------------------------------------------- phases

def _claim(fam: str, include_test: bool):
    for code in DEV9.values():
        ledger.claim(fam, code, ledger.season_range("2015-16", "2020-21"), "training")
        ledger.claim(fam, code, ledger.season_range("2021-22", "2025-26"), "development")
    ledger.claim(fam, "E1", ledger.season_range("2015-16", "2025-26"), "training")
    if include_test:
        for code in ("G1", "T1", "I2", "D2", "F2", "SP2"):
            ledger.claim(fam, code, ledger.season_range("2015-16", "2019-20"), "training")
        for code in ("AUT", "SWZ", "DNK"):
            ledger.claim(fam, code, ledger.season_range("2015-16", "2019-20"), "test")
        for code in SEALED9.values():
            ledger.claim(fam, code, TEST_SEASONS, "test")


def _cached(tag: str, build):
    fd_cache._load_env()
    path = f"derived/xc_rates_{tag}.parquet"
    try:
        return pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
    except RuntimeError:
        r = build()
        buf = io.BytesIO()
        r.to_parquet(buf, index=False)
        D._retry(ss.put, BUCKET, path, buf.getvalue())
        return r


def _mde(w, a, b):
    import run_nlx as N
    se = N.block_se(w.assign(d=w[a] - w[b]), "d")
    return {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
            "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))}


def dev():
    for fam in ("XC", "CAP"):
        _claim(fam, include_test=False)
    leagues = {**DEV9, **E1}
    m, raw = tables(leagues, {**EUROPE, **CUPS_DEV})
    m = current_league(m)
    print(f"matches: {int((m.kind == 'league').sum())} league, {int((m.kind != 'league').sum())} cross", flush=True)
    r = _cached("dev", lambda: walk_forward(m, list(DEV9.values()), DEV_SEASONS, configs("dev")))
    w = scored(m, r)
    sc = {c: D.league_balanced(w, c) for c in w.columns if c.split("_")[0] in ("LO", "CAP", "XC")}
    xc = min((c for c in sc if c.startswith("XC_")), key=sc.get)
    cap = min((c for c in sc if c.startswith("CAP_")), key=sc.get)
    _, v, we, wc = xc.split("_")
    choice = {"xc": {"variant": v, "w_E": float(we), "w_C": float(wc)}, "cap": int(cap.split("_")[1]),
              "dev_log_loss": sc, "dev_xc_minus_lo": sc[xc] - sc["LO"], "dev_cap_minus_lo": sc[cap] - sc["LO"],
              "mde_xc": _mde(w, xc, "LO"), "mde_cap": _mde(w, cap, "LO"),
              "dev_by_league_xc_minus_lo": {d: float((g[xc] - g["LO"]).mean()) for d, g in w.groupby("div")},
              "n_dev_per_league_season": {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])}}
    s2 = {}
    for model, name in (("LO", "LO"), (xc, "XC")):
        ws = stage2(m, r, model, {k: v for k, v in DEV9.items()}, list(DEV9.values()), DEV_SEASONS, (0.1, 1.0, 10.0, 100.0))
        sl = {c: D.league_balanced(ws, c) for c in ws.columns if c.startswith("S2_")}
        best = min(sl, key=sl.get)
        s2[name] = {"l2": float(best.split("_")[1]), "dev_log_loss": sl[best]}
    choice["stage2"] = s2
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2, default=str))
    print(json.dumps({k: choice[k] for k in ("xc", "cap", "dev_xc_minus_lo", "dev_cap_minus_lo", "mde_xc", "mde_cap", "stage2")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())      # frozen development choices
    _claim("XC", include_test=True)              # CAP was withdrawn after development (amendment)
    leagues = {**DEV9, **E1, **SEALED9}
    m, raw = tables(leagues, {**EUROPE, **CUPS_DEV, **CUPS_SEALED})
    comp = check(raw, m, SEALED9, TEST_SEASONS)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_check.json").write_text(json.dumps(comp, indent=2))
    if failing(comp):
        raise SystemExit(f"run condition not met: {failing(comp)}")
    m = current_league(m)
    r = _cached("test", lambda: walk_forward(m, list(SEALED9.values()), TEST_SEASONS, configs("test", choice)))
    w = scored(m, r)
    for a, b, col in (("XC", "LO", "xc"), ("CAP", "LO", "cap"), ("XC_CAP", "XC", "xc_cap")):
        w[col] = w[a] - w[b]
    res = {"addenda": ["XC", "CAP"], "frozen_dev_choice": {k: choice[k] for k in ("xc", "cap", "mde_xc", "mde_cap", "stage2")},
           "n_test_matches": int(len(w)),
           "matches_per_league_season": {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])},
           "log_loss": {c: D.league_balanced(w, c) for c in ("LO", "CAP", "XC", "XC_CAP")}}
    mm, lo, hi = D.block_ci(w, "xc")
    res["XC"] = {"primary": {"XC - LO": {"mean": mm, "ci_95": [lo, hi]}},
                 "decision": "confirmed" if (mm <= -0.001 and hi < 0) else ("closed" if lo > -0.003 else "inconclusive")}
    mm, lo, hi = D.block_ci(w, "cap")
    res["CAP"] = {"withdrawn": "after development (PREREGISTRATION, amendment to CAP); descriptive only",
                  "LO-cap - LO": {"mean": mm, "ci_95": [lo, hi]}}
    sec = {}
    mm, lo, hi = D.block_ci(w, "xc_cap")
    sec["XC with cap - XC"] = {"mean": mm, "ci_95": [lo, hi]}
    top = w["div"].isin(F.TOP_FLIGHTS)
    sec["top flights vs second tiers (XC - LO, n)"] = {"top": [D.league_balanced(w[top], "xc"), int(top.sum())],
                                                       "second": [D.league_balanced(w[~top], "xc"), int((~top).sum())]}
    e = w[w["early"]]
    sec["early season (XC - LO, n)"] = [D.league_balanced(e, "xc"), int(len(e))]
    eu = m[m["kind"] == "europe"]
    long = pd.concat([eu[["kickoff_utc", "home_id"]].rename(columns={"home_id": "t"}),
                      eu[["kickoff_utc", "away_id"]].rename(columns={"away_id": "t"})]).sort_values("kickoff_utc")
    ns = lambda x: x.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()  # noqa: E731
    by = {t: ns(g_["kickoff_utc"]) for t, g_ in long.groupby("t")}
    g = m.set_index("match_id")
    year = 365 * 86_400 * 10**9

    def heavy(mid):
        f = g.loc[mid]
        t0 = int(f["kickoff_utc"].as_unit("ns").value)
        e = np.array([], dtype=np.int64)
        n = lambda t: int(np.searchsorted(by.get(t, e), t0) - np.searchsorted(by.get(t, e), t0 - year))  # noqa: E731
        return max(n(f["home_id"]), n(f["away_id"])) >= 4
    hv = w["match_id"].map(heavy)
    sec["European-heavy matches (XC - LO, n)"] = [D.league_balanced(w[hv], "xc") if hv.any() else None, int(hv.sum())]
    sec["by league (XC - LO, CAP - LO, n)"] = {d: [float(x.xc.mean()), float(x.cap.mean()), int(len(x))] for d, x in w.groupby("div")}
    sec["by season (XC - LO)"] = {s_: D.league_balanced(x, "xc") for s_, x in w.groupby("season")}
    s2 = {}
    for model, name in (("LO", "LO"), ("XC", "XC")):
        ws = stage2(m, r, model, {k: v for k, v in {**DEV9, **SEALED9}.items()}, list(SEALED9.values()), TEST_SEASONS,
                    (choice["stage2"][name]["l2"],))
        s2[name] = ws.rename(columns={c: name for c in ws.columns if c.startswith("S2_")})
    j = s2["XC"][["match_id", "XC"]].merge(s2["LO"][["match_id", "LO", "div", "date"]], on="match_id")
    j["d"] = j["XC"] - j["LO"]
    mm, lo, hi = D.block_ci(j, "d")
    sec["with G's stage 2: XC+LIN - LO+LIN"] = {"mean": mm, "ci_95": [lo, hi], "n": int(len(j))}
    res["secondary"] = sec
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss", "XC", "CAP")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
