"""Addendum MGR of docs/PREREGISTRATION.md (new-manager rating shifts, F5), frozen before this file existed
and before any sealed-league manager data was collected.

    python prototypes/run_mgr.py dev     # 9 development leagues, 2016-17 -> 2025-26: lambda, interim rule, MDE
    python prototypes/run_mgr.py test    # the 9 sealed leagues, 2020-21 -> 2025-26: scored once

Implementation details the addendum leaves open, fixed here before any data is loaded:
  * LO is addendum XC's joint code with no cross-competition matches (run_xc.fit with zero weights), which
    reproduces the current Dixon-Coles; MGR adds the shifts to that same likelihood (one fit per league,
    since without cross matches the leagues share no parameters).
  * An appointment applies from 00:00 UTC on its date (appointment date, else vacancy date); a match that
    day already uses it. Rows of one team with the same date and incoming manager count once.
  * Names: Wikipedia team names are matched to API-Football team ids among the teams that played league
    matches in that league in the season or the one before: first by normalised name (lower case, accents
    and punctuation removed, common club words dropped), then by the closest name (difflib ratio >= 0.6 and
    clearly ahead of the runner-up), then by ALIASES. Unmatched rows are dropped and counted.
  * "First 10 league matches after an appointment": the team's next 10 league matches in that league with
    kickoff at or after the appointment.
  * Development compares settings on the union of all settings' affected matches (a setting that does not
    affect a match forecasts it exactly as LO); the primary uses the chosen setting's own affected matches.
  * Expected affected test matches for the MDE: 6,000.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as D  # noqa: E402
import run_xc as X  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402
from tabet.model import RHO_BOUND  # noqa: E402

OUT = ROOT / "results" / "mgr"
DEV9 = X.DEV9                                   # {39: "E0", 140: "SP1", ...}
SEALED9 = X.SEALED9
DEV_SEASONS = [D.season_label(y) for y in range(2016, 2026)]
TEST_SEASONS = X.TEST_SEASONS
LAMBDAS = (3.0, 10.0, 30.0, 100.0)
AFFECT_N = 10
EXPECTED_TEST_MATCHES = 6000
MIN_MATCHED = 0.80
STOP = {"fc", "cf", "afc", "ac", "sc", "as", "ss", "us", "club", "de", "calcio", "sv", "fk", "sk", "kv", "krc",
        "rsc", "rc", "ud", "cd", "rcd", "sd", "real", "the", "1", "tsg", "vfb", "vfl", "bv", "bsc", "fsv", "ogc",
        "stade", "olympique", "sporting", "cp", "sad", "ksv", "kaa", "rfc", "fcv", "and", "hove", "albion"}
# Written during development from team names only (never from results); extended if names fail to match.
ALIASES: dict[str, str] = {
    "Inter Milan": "Inter", "Internazionale": "Inter", "Athletic Bilbao": "Athletic Club", "AZ": "AZ Alkmaar",
    "PSV": "PSV Eindhoven", "NEC": "NEC Nijmegen", "Wolverhampton Wanderers": "Wolves", "Beveren": "SK Beveren",
    "Mouscron": "Royal Excel Mouscron", "Mouscron-Péruwelz": "Royal Excel Mouscron", "RWD Molenbeek": "RWDM",
    "Roda JC Kerkrade": "Roda", "Estoril Praia": "Estoril", "Estrela da Amadora": "Estrela",
    "Vitória de Guimarães": "Vitória SC", "Brest": "Stade Brestois 29", "Desportivo das Aves": "Aves",
    "B-SAD": "Belenenses", "Académico de Viseu": "Academico Viseu"}
# Third pass: drop one trailing club word ("Leeds United" -> "Leeds") if that gives a unique match.
SUFFIXES = {"united", "city", "town", "hotspur", "wanderers", "athletic", "rovers", "county", "praia"}


# ---------------------------------------------------------------- names and appointments

def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    words = [w for w in s.split() if w not in STOP]
    return " ".join(words) or s.strip()


def team_names(leagues: dict[int, str]) -> pd.DataFrame:
    """(league, season, team_id, name) for every team in the stored league fixture lists; cached."""
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = []
    for league in leagues:
        path = f"derived/mgr_names_{league}.parquet"
        try:
            t = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        except RuntimeError:
            rows = []
            for season in X.SEASONS:
                lp = D._retry(store.list_path, league, season)
                if lp is None:
                    continue
                for f in D._retry(store.get, lp).get("response") or []:
                    for side in ("home", "away"):
                        rows.append({"league": league, "season": D.season_label(season),
                                     "team_id": str(f["teams"][side]["id"]), "name": f["teams"][side]["name"]})
            t = pd.DataFrame(rows).drop_duplicates()
            buf = io.BytesIO()
            t.to_parquet(buf, index=False)
            D._retry(ss.put, BUCKET, path, buf.getvalue())
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def match_names(changes: pd.DataFrame, names: pd.DataFrame, code_to_id: dict[str, int]) -> pd.DataFrame:
    """Adds api_team (or None) to each change row."""
    out = []
    prev = lambda s: f"{int(s[:4]) - 1}-{(int(s[5:]) - 1) % 100:02d}"  # noqa: E731
    for r in changes.itertuples():
        lg = code_to_id[r.league]
        cand = names[(names["league"] == lg) & names["season"].isin([r.season, prev(r.season)])]
        by_norm = {}
        for t, n in zip(cand["team_id"], cand["name"]):
            by_norm.setdefault(norm(n), set()).add(t)
        key = norm(ALIASES.get(r.team_name, r.team_name))
        hit = by_norm.get(key)
        team = next(iter(hit)) if hit and len(hit) == 1 else None
        if team is None and key.split() and key.split()[-1] in SUFFIXES:
            short = by_norm.get(" ".join(key.split()[:-1]))
            team = next(iter(short)) if short and len(short) == 1 else None
        if team is None and by_norm:
            scored = sorted(((difflib.SequenceMatcher(None, key, k).ratio(), k) for k in by_norm), reverse=True)
            best = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if best[0] >= 0.6 and best[0] - second >= 0.1 and len(by_norm[best[1]]) == 1:
                team = next(iter(by_norm[best[1]]))
        out.append(team)
    return changes.assign(api_team=out)


def appointments(changes: pd.DataFrame, code_to_id: dict[str, int], interim: bool) -> pd.DataFrame:
    """(league id, team, date) per counted appointment, deduplicated."""
    c = changes[changes["api_team"].notna()].copy()
    if not interim:
        c = c[~c["interim"].astype(bool)]
    c["date"] = pd.to_datetime(c["appointment_date"].fillna(c["vacancy_date"]), utc=True)
    c = c[c["date"].notna()]
    c["league_id"] = c["league"].map(code_to_id)
    return c.drop_duplicates(["league_id", "api_team", "date", "incoming"])[["league_id", "api_team", "date"]].sort_values("date")


def load_changes(codes: list[str]) -> pd.DataFrame:
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    rows = []
    for c in codes:
        rows += D._retry(store._rows, "manager_changes",
                         "league,season,team_name,incoming,interim,appointment_date,vacancy_date", league=f"eq.{c}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the fit

def fit(tr: pd.DataFrame, cut: pd.Timestamp, events: pd.DataFrame, lam: float) -> dict:
    """League-only Dixon-Coles (as run_xc.fit with zero cross weights) plus, per appointment in `events`
    dated inside the window and before `cut`, shifts (da, dd) on that team's attack and defence in matches
    on or after the date, with L2 penalty lam. No events: exactly LO."""
    d = tr[tr["kind"] == "league"].reset_index(drop=True)
    eh = list(zip(d["home_id"], d["league"].astype(int)))
    ea = list(zip(d["away_id"], d["league"].astype(int)))
    ents = sorted(set(eh) | set(ea), key=str)
    ei = {e: i for i, e in enumerate(ents)}
    hi, ai = np.array([ei[e] for e in eh]), np.array([ei[e] for e in ea])
    leagues = sorted(set(d["league"].astype(int)))
    ci = {l: i for i, l in enumerate(leagues)}
    cx = d["league"].astype(int).map(ci).to_numpy()
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    age = (cut - d["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
    dec = np.exp(-X.XI * age)
    w = dec / pd.Series(dec).groupby(cx).transform("mean").to_numpy()
    ev = events[(events["date"] < cut) & (events["date"] >= cut - X.WINDOW)] if events is not None and len(events) else pd.DataFrame()
    ev = ev[[(t, l) in ei for t, l in zip(ev["api_team"], ev["league_id"])]] if len(ev) else ev
    E = len(ev)
    if E:
        rows_h, cols_h, rows_a, cols_a = [], [], [], []
        for k, (t, l, dt) in enumerate(zip(ev["api_team"], ev["league_id"], ev["date"])):
            after = (d["kickoff_utc"] >= dt).to_numpy()
            mh = after & (hi == ei[(t, l)])
            ma = after & (ai == ei[(t, l)])
            rows_h += list(np.nonzero(mh)[0]); cols_h += [k] * int(mh.sum())
            rows_a += list(np.nonzero(ma)[0]); cols_a += [k] * int(ma.sum())
        H = sparse.csr_matrix((np.ones(len(rows_h)), (rows_h, cols_h)), shape=(len(d), E))
        A = sparse.csr_matrix((np.ones(len(rows_a)), (rows_a, cols_a)), shape=(len(d), E))
    n, C = len(ents), len(leagues)
    c00, c01, c10, c11 = [(x == a) & (y == b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]

    def unpack(p):
        return (p[:C], p[C:2 * C], p[2 * C:3 * C], p[3 * C:3 * C + n], p[3 * C + n:3 * C + 2 * n],
                p[3 * C + 2 * n:3 * C + 2 * n + E], p[3 * C + 2 * n + E:])

    def nll(p):
        mu, h, rho, a, dd, sa, sd = unpack(p)
        lh_ = mu[cx] + h[cx] + a[hi] - dd[ai]
        la_ = mu[cx] + a[ai] - dd[hi]
        if E:
            lh_ = lh_ + H @ sa - A @ sd
            la_ = la_ + A @ sa - H @ sd
        lh, la = np.exp(lh_), np.exp(la_)
        r = rho[cx]
        tau = np.ones_like(lh)
        tau[c00] = 1 - lh[c00] * la[c00] * r[c00]
        tau[c01] = 1 + lh[c01] * r[c01]
        tau[c10] = 1 + la[c10] * r[c10]
        tau[c11] = 1 - r[c11]
        tau = np.maximum(tau, 1e-10)
        ll = x * lh_ - lh + y * la_ - la + np.log(tau)
        f = -(w * ll).sum() + X.L2 * (a @ a + dd @ dd) + lam * (sa @ sa + sd @ sd)
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
        g[C:2 * C] = np.bincount(cx, g1, C)
        g[2 * C:3 * C] = np.bincount(cx, gr, C)
        g[3 * C:3 * C + n] = np.bincount(hi, g1, n) + np.bincount(ai, g2, n)
        g[3 * C + n:3 * C + 2 * n] = -np.bincount(ai, g1, n) - np.bincount(hi, g2, n)
        if E:
            g[3 * C + 2 * n:3 * C + 2 * n + E] = H.T @ g1 + A.T @ g2
            g[3 * C + 2 * n + E:] = -(A.T @ g1 + H.T @ g2)
        g = -g
        g[3 * C:3 * C + n] += 2 * X.L2 * a
        g[3 * C + n:3 * C + 2 * n] += 2 * X.L2 * dd
        if E:
            g[3 * C + 2 * n:3 * C + 2 * n + E] += 2 * lam * sa
            g[3 * C + 2 * n + E:] += 2 * lam * sd
        return f, g

    p0 = np.zeros(3 * C + 2 * n + 2 * E)
    for l, i in ci.items():
        sel = cx == i
        p0[i] = np.log(max((x[sel] + y[sel]).mean() / 2, 0.1))
    bounds = [(None, None)] * (2 * C) + [(-RHO_BOUND, RHO_BOUND)] * C + [(None, None)] * (2 * n + 2 * E)
    res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
    mu, h, rho, a, dd, sa, sd = unpack(res.x)
    shifts = {}
    for k, (t, l, dt) in enumerate(zip(ev["api_team"], ev["league_id"], ev["date"])) if E else []:
        shifts.setdefault((t, int(l)), []).append((dt, float(sa[k]), float(sd[k])))
    return {"mu": {l: float(mu[i]) for l, i in ci.items()}, "home": {l: float(h[i]) for l, i in ci.items()},
            "rho": {l: float(rho[i]) for l, i in ci.items()},
            "att": {e: float(a[i]) for e, i in ei.items()}, "def": {e: float(dd[i]) for e, i in ei.items()},
            "shifts": shifts, "converged": bool(res.success)}


def rates(par: dict, games: pd.DataFrame) -> pd.DataFrame:
    """LO's rates plus the shifts in force at each kickoff."""
    out = []
    for g in games.itertuples():
        l = int(g.league)
        if l not in par["mu"]:
            continue

        def side(t):
            a, d_ = par["att"].get((t, l), 0.0), par["def"].get((t, l), 0.0)
            for dt, sa, sd in par["shifts"].get((t, l), []):
                if g.kickoff_utc >= dt:
                    a, d_ = a + sa, d_ + sd
            return a, d_
        ah, dh = side(g.home_id)
        aa, da = side(g.away_id)
        out.append((g.match_id, par["mu"][l] + par["home"][l] + ah - da, par["mu"][l] + aa - dh, par["rho"][l]))
    return pd.DataFrame(out, columns=["match_id", "lh", "la", "rho"])


# ---------------------------------------------------------------- affected matches and walk-forward

def affected(m: pd.DataFrame, events: pd.DataFrame) -> set[str]:
    """League matches within a team's first AFFECT_N league matches (same league) after an appointment."""
    lg = m[m["kind"] == "league"]
    out = set()
    for t, l, dt in zip(events["api_team"], events["league_id"], events["date"]):
        g = lg[(lg["league"] == l) & ((lg["home_id"] == t) | (lg["away_id"] == t)) & (lg["kickoff_utc"] >= dt)]
        out |= set(g.sort_values("kickoff_utc")["match_id"].head(AFFECT_N))
    return out


def _week(args):
    tr, games, cut, cfgs = args
    out = []
    for name, (events, lam) in cfgs.items():
        out.append(rates(fit(tr, cut, events, lam), games).assign(model=name))
    return out


def walk_forward(m: pd.DataFrame, eval_divs, seasons, eval_ids: set, cfgs: dict, workers: int = 6) -> pd.DataFrame:
    ev = m[(m["kind"] == "league") & m["div"].isin(eval_divs) & m["season"].isin(seasons) & m["match_id"].isin(eval_ids)]
    week = ev["kickoff_utc"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    lg = m[m["kind"] == "league"]
    jobs = []
    for c, g in ev.groupby(week):
        tr = lg[(lg["kickoff_utc"] < c) & (lg["kickoff_utc"] >= c - X.WINDOW)]
        counts = tr["league"].value_counts()
        g = g[g["league"].map(counts).fillna(0) >= X.MIN_LEAGUE]
        if len(g):
            jobs.append((tr, g, c, cfgs))
    parts = []
    results = map(_week, jobs) if workers == 1 else ProcessPoolExecutor(workers).map(_week, jobs, chunksize=1)
    for i, res in enumerate(results):
        parts += res
        if i % 25 == 0:
            print(f"  week {i + 1}/{len(jobs)}", flush=True)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- phases

def _claim(include_test: bool):
    for code in DEV9.values():
        ledger.claim("MGR", code, ["2015-16"], "training")
        ledger.claim("MGR", code, ledger.season_range("2016-17", "2025-26"), "development")
    if include_test:
        for code in ("G1", "T1", "I2", "D2", "F2", "SP2"):
            ledger.claim("MGR", code, ledger.season_range("2015-16", "2019-20"), "training")
        for code in ("AUT", "SWZ", "DNK"):
            ledger.claim("MGR", code, ledger.season_range("2015-16", "2019-20"), "test")
        for code in SEALED9.values():
            ledger.claim("MGR", code, TEST_SEASONS, "test")


def league_tables(leagues: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """League matches with a 90-minute score (run_xc's cached lists, no cups or European matches)."""
    raw = X.load(leagues)
    m = raw[(raw["status_short"] == "FT") & raw["x"].notna() & raw["y"].notna()].assign(kind="league", neutral=False)
    m["x"], m["y"] = m["x"].astype(float), m["y"].astype(float)
    m["div"] = m["league"].map(leagues)
    return m.sort_values("kickoff_utc").reset_index(drop=True), raw


def _prepare(leagues: dict[int, str], show_names: bool):
    m, raw = league_tables(leagues)
    code_to_id = {v: k for k, v in leagues.items()}
    ch = match_names(load_changes(list(leagues.values())), team_names(leagues), code_to_id)
    rate = ch.assign(ok=ch["api_team"].notna()).groupby(["league", "season"])["ok"].mean()
    report = {"rows": int(len(ch)), "matched": int(ch["api_team"].notna().sum()),
              "match_rate_by_league": ch.assign(ok=ch["api_team"].notna()).groupby("league")["ok"].mean().round(3).to_dict()}
    if show_names:
        report["unmatched_names"] = sorted(ch.loc[ch["api_team"].isna(), "team_name"].unique().tolist())[:60]
    return m, raw, ch, code_to_id, rate, report


def dev():
    _claim(include_test=False)
    m, raw, ch, code_to_id, rate, report = _prepare(DEV9, show_names=True)
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    evs = {it: appointments(ch, code_to_id, it) for it in (False, True)}
    cfgs = {"LO": (None, 0.0)}
    cfgs.update({f"MGR_{'int' if it else 'perm'}_{lam:g}": (evs[it], lam) for it in (False, True) for lam in LAMBDAS})
    aff = {it: affected(m, evs[it]) for it in (False, True)}
    union = aff[False] | aff[True]
    r = X._cached("mgr_dev", lambda: walk_forward(m, list(DEV9.values()), DEV_SEASONS, union, cfgs))
    w = X.scored(m, r)
    sc = {c: D.league_balanced(w, c) for c in w.columns if c.split("_")[0] in ("LO", "MGR")}
    best = min((c for c in sc if c.startswith("MGR_")), key=sc.get)
    it = best.split("_")[1] == "int"
    own = w[w["match_id"].isin(aff[it])]
    import run_nlx as N
    se = N.block_se(own.assign(d=own[best] - own["LO"]), "d")
    choice = {"lambda": float(best.split("_")[2]), "interim": it, "names": report,
              "dev_log_loss_union": sc, "dev_mgr_minus_lo_own_affected": D.league_balanced(own.assign(d=own[best] - own["LO"]), "d"),
              "n_affected_dev": int(len(own)),
              "mde": {"dev_block_se": se, "n_dev": int(len(own)), "expected_test": EXPECTED_TEST_MATCHES,
                      "mde": float((1.96 + 0.84) * se * np.sqrt(len(own) / EXPECTED_TEST_MATCHES))},
              "dev_by_league": {d_: float((g[best] - g["LO"]).mean()) for d_, g in own.groupby("div")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2, default=str))
    print(json.dumps({k: choice[k] for k in ("lambda", "interim", "dev_mgr_minus_lo_own_affected", "n_affected_dev", "mde", "dev_by_league")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())        # frozen development choices
    _claim(include_test=True)
    m, raw, ch, code_to_id, rate, report = _prepare(SEALED9, show_names=False)
    comp = X.check(raw, m, SEALED9, TEST_SEASONS)
    short = {k: v for k, v in comp.items() if v["league_share"] < X.MIN_COMPLETE}
    short.update({f"{c} {s} names": float(v) for (c, s), v in rate.items() if s in TEST_SEASONS and v < MIN_MATCHED})
    missing = [f"{c} {s}" for c in SEALED9.values() for s in TEST_SEASONS if (c, s) not in rate.index]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_check.json").write_text(json.dumps({"scores": comp, "names": report, "no_manager_rows": missing}, indent=2))
    if short:
        raise SystemExit(f"run condition not met: {short}")
    ev = appointments(ch, code_to_id, choice["interim"])
    aff = affected(m, ev)
    cfgs = {"LO": (None, 0.0), "MGR": (ev, choice["lambda"])}
    r = X._cached("mgr_test", lambda: walk_forward(m, list(SEALED9.values()), TEST_SEASONS, aff, cfgs))
    w = X.scored(m, r)
    w["d"] = w["MGR"] - w["LO"]
    mm, lo, hi = D.block_ci(w, "d")
    res = {"addendum": "MGR", "frozen_dev_choice": {k: choice[k] for k in ("lambda", "interim", "mde")},
           "n_affected_test": int(len(w)), "per_league": {d_: [float(g.d.mean()), int(len(g))] for d_, g in w.groupby("div")},
           "primary": {"MGR - LO (affected)": {"mean": mm, "ci_95": [lo, hi]},
                       "decision": "confirmed" if (mm <= -0.001 and hi < 0) else ("closed" if lo > -0.003 else "inconclusive")},
           "no_manager_rows": missing}
    lg = m[m["kind"] == "league"]
    k_of, in_season = {}, {}
    span = lg.groupby(["league", "season"])["kickoff_utc"].agg(["min", "max"])
    for t, l, dt in zip(ev["api_team"], ev["league_id"], ev["date"]):
        g = lg[(lg["league"] == l) & ((lg["home_id"] == t) | (lg["away_id"] == t)) & (lg["kickoff_utc"] >= dt)]
        g = g.sort_values("kickoff_utc").head(AFFECT_N)
        for k, (mid, s_) in enumerate(zip(g["match_id"], g["season"]), start=1):
            if k < k_of.get(mid, 99):
                k_of[mid] = k
                lo_, hi_ = span.loc[(l, s_)] if (l, s_) in span.index else (dt, dt)
                in_season[mid] = bool(lo_ < dt <= hi_)
    w["k"] = w["match_id"].map(k_of)
    w["in_season"] = w["match_id"].map(in_season)
    n_all = int(lg[lg["div"].isin(list(SEALED9.values())) & lg["season"].isin(TEST_SEASONS)].shape[0])
    sec = {"all matches (MGR - LO; unaffected matches are identical)": float(w["d"].sum() / n_all) if n_all else None,
           "matches 1-5 vs 6-10 after the appointment": {"1-5": D.league_balanced(w[w["k"] <= 5], "d"),
                                                          "6-10": D.league_balanced(w[w["k"] > 5], "d")},
           "in-season vs off-season appointments": {"in-season": D.league_balanced(w[w["in_season"] == True], "d"),  # noqa: E712
                                                    "off-season": D.league_balanced(w[w["in_season"] == False], "d")},  # noqa: E712
           "by season": {s_: D.league_balanced(g, "d") for s_, g in w.groupby("season")},
           "development E0 (development data)": choice["dev_by_league"].get("E0")}
    res["secondary"] = sec
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_affected_test", "primary")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
