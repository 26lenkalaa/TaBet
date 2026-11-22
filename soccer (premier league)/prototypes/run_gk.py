"""Addendum GK of docs/PREREGISTRATION.md (goalkeeper quality, F7), frozen before this file existed and
before any sealed keeper data was loaded.

    python prototypes/run_gk.py dev     # 9 development leagues, 2021-22 -> 2025-26: k, L2s, MDE
    python prototypes/run_gk.py test    # the 9 sealed leagues, 2020-21 -> 2025-26: scored once

Implementation details the addendum leaves open, fixed here before any data is loaded:
  * Keepers: from each stored match's player statistics, players with position "G". The starting keeper is
    the one with substitute = False (if several, the one with the most minutes). A keeper's line counts
    toward his rating when he played at least 45 minutes.
  * Usual keeper: most starts in the team's previous 10 league matches; a tie goes to the more recent starter.
  * Ratings and the usual keeper update after each match date (state from earlier dates only), within
    the league, as addendum G's team form does.
  * LIN is addendum FAT's LIN (G's ten features plus league indicators); DC rates are FAT's cached ones.
  * Expected test matches for the MDE: 14,000.
"""
from __future__ import annotations

import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as D  # noqa: E402
import run_fat as F  # noqa: E402
import run_nlx as N  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402

OUT = ROOT / "results" / "gk"
DEV9 = {39: "E0", **F.DEV8}
SEALED9 = F.SEALED9
DEV_SEASONS = F.DEV_SEASONS
TEST_SEASONS = F.TEST_SEASONS
EXCLUDED = {("G1", "2020-21"), ("G1", "2021-22")}     # no player statistics (coverage flags), set in advance
KS = (20, 50, 100)
HALF_LIFE = 365.0
MIN_MINUTES = 45
USUAL_N = 10
EXPECTED_TEST_MATCHES = 14000
MIN_COMPLETE = 0.95          # starting keepers identified (amended 2026-11-22)
MIN_LINES = 0.85             # keeper lines complete, per league (amended 2026-11-22)
L2S = N.L2S


# ---------------------------------------------------------------- keeper table

def _keepers_of(body: dict) -> list[dict]:
    out = []
    for f in body.get("response", []):
        mid = str(f["fixture"]["id"])
        for t in f.get("players") or []:
            tid = str(t["team"]["id"])
            for p in t.get("players") or []:
                s = (p.get("statistics") or [{}])[0]
                g = s.get("games") or {}
                if g.get("position") != "G":
                    continue
                gl = s.get("goals") or {}
                out.append({"match_id": mid, "team_id": tid, "player_id": str(p["player"]["id"]),
                            "minutes": g.get("minutes") or 0, "substitute": bool(g.get("substitute")),
                            "saves": gl.get("saves"), "conceded": gl.get("conceded")})
    return out


def keepers(leagues: dict[int, str]) -> pd.DataFrame:
    """Every keeper line of these leagues' stored matches; cached per league."""
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = []
    for league, code in leagues.items():
        path = f"derived/gk_{league}_{F.SEASONS[0]}_{F.SEASONS[-1]}.parquet"
        try:
            t = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        except RuntimeError:
            rows = []
            for season in F.SEASONS:
                objs = D._retry(store._rows, "apif_fixtures", "object_path", league=f"eq.{league}", season=f"eq.{season}")
                paths = sorted({r["object_path"] for r in objs})
                with ThreadPoolExecutor(2) as ex:
                    for body in ex.map(lambda p: D._retry(store.get, p), paths):
                        rows += _keepers_of(body)
            t = pd.DataFrame(rows, columns=["match_id", "team_id", "player_id", "minutes", "substitute", "saves", "conceded"])
            buf = io.BytesIO()
            t.to_parquet(buf, index=False)
            D._retry(ss.put, BUCKET, path, buf.getvalue())
        print(f"{code}: {len(t)} keeper lines", flush=True)
        parts.append(t.assign(league=league))
    k = pd.concat(parts, ignore_index=True)
    for c in ("minutes", "saves", "conceded"):
        k[c] = pd.to_numeric(k[c], errors="coerce")
    return k


def starters(k: pd.DataFrame) -> pd.DataFrame:
    """The starting keeper per (match, team): substitute False, most minutes."""
    s = k[~k["substitute"]].sort_values("minutes", ascending=False)
    return s.drop_duplicates(["match_id", "team_id"])


# ---------------------------------------------------------------- features

def gk_features(m: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """Per (match_id, team_id): the starting keeper's g for each k (g_20, g_50, g_100) and u (not the usual
    keeper), from earlier dates only, within the league. m: league matches with date, league, home/away ids."""
    lam = np.log(2) / HALF_LIFE
    st = starters(k).set_index(["match_id", "team_id"])["player_id"]
    lines = k[k["minutes"] >= MIN_MINUTES].dropna(subset=["saves", "conceded"])
    by_match = {mid: g for mid, g in lines.groupby("match_id")}
    rows = []
    for _, lm in m.groupby("league"):
        rate: dict[str, list] = {}          # keeper -> [t, S, C]
        league_hist: list = []              # (date, saves, conceded)
        recent: dict[str, list] = {}        # team -> list of starting keepers, latest last
        for d, g in lm.groupby("date", sort=True):
            league_hist = [x for x in league_hist if (d - x[0]).days <= 365]
            S_l, C_l = sum(x[1] for x in league_hist), sum(x[2] for x in league_hist)
            qbar = S_l / (S_l + C_l) if S_l + C_l >= 200 else np.nan
            for f in g.itertuples():
                for team in (f.home_id, f.away_id):
                    p = st.get((f.match_id, team))
                    r = {"match_id": f.match_id, "team_id": team, "keeper_known": p is not None and pd.notna(p)}
                    if r["keeper_known"] and np.isfinite(qbar):
                        h = rate.get(p)
                        S = C = 0.0
                        if h is not None:
                            decay = np.exp(-lam * (d - h[0]).days)
                            S, C = h[1] * decay, h[2] * decay
                        for kk in KS:
                            q = (S + kk * qbar) / (S + C + kk)
                            r[f"g_{kk}"] = float(np.log((1 - q) / (1 - qbar)))
                        prev = recent.get(team, [])[-USUAL_N:]
                        # most starts in the previous USUAL_N; a tie goes to the more recent starter
                        usual = max(set(prev), key=lambda q_: (prev.count(q_), max(i for i, v in enumerate(prev) if v == q_))) if prev else None
                        r["u"] = float(usual is not None and p != usual)
                    rows.append(r)
            for f in g.itertuples():           # update after the date
                for team in (f.home_id, f.away_id):
                    p = st.get((f.match_id, team))
                    if p is not None and pd.notna(p):
                        recent.setdefault(team, []).append(p)
                ln = by_match.get(f.match_id)
                if ln is None:
                    continue
                for x in ln.itertuples():
                    h = rate.setdefault(x.player_id, [d, 0.0, 0.0])
                    decay = np.exp(-lam * (d - h[0]).days)
                    h[0], h[1], h[2] = d, h[1] * decay + float(x.saves), h[2] * decay + float(x.conceded)
                    league_hist.append((d, float(x.saves), float(x.conceded)))
    return pd.DataFrame(rows)


def side_rows(tb: dict, k: pd.DataFrame, leagues: dict[int, str]) -> pd.DataFrame:
    """FAT's base rows (DC, G's form, league indicators) plus the opposing keeper's inputs."""
    m = tb["matches"]
    m = m[(m["status_short"] == "FT") & m["home_goals"].notna() & m["league"].isin(list(leagues))].copy()
    m["status"] = "completed"
    m["date"] = m["kickoff_utc"].dt.normalize()
    m["div"] = m["league"].map(leagues)
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    m = m.sort_values("kickoff_utc").reset_index(drop=True)
    played = pd.concat([m[["match_id", "season", "home_id"]].rename(columns={"home_id": "t"}),
                        m[["match_id", "season", "away_id"]].rename(columns={"away_id": "t"})])
    played["n"] = played.groupby(["season", "t"]).cumcount()
    m["early"] = m["match_id"].map(played.groupby("match_id")["n"].min() < 10).fillna(True)
    dc = F.dc_all(m)
    form = N.form_features(m, tb["teams"].set_index(["match_id", "team_id"])["sot"])
    gk = gk_features(m, k)
    print(f"DC {len(dc)}, form {form.match_id.nunique()}, keeper rows {len(gk)}", flush=True)
    base = m[m["match_id"].isin(dc.keys())].copy()
    base[["lh", "la", "rho"]] = pd.DataFrame([dc[i] for i in base["match_id"]], index=base.index)
    sides = []
    for side, own, opp, goals, lo, lp in (("H", "home_id", "away_id", "home_goals", "lh", "la"),
                                          ("A", "away_id", "home_id", "away_goals", "la", "lh")):
        x = base[["match_id", "div", "season", "date", "early", "result", "rho", own, opp, goals, lo, lp]].rename(
            columns={own: "own_id", opp: "opp_id", goals: "goals", lo: "log_dc", lp: "log_dc_opp"})
        x["side"] = side
        sides.append(x)
    r = pd.concat(sides, ignore_index=True)
    go = gk.rename(columns={"team_id": "opp_id", **{f"g_{kk}": f"g_opp_{kk}" for kk in KS}, "u": "u_opp",
                            "keeper_known": "opp_keeper_known"})
    r = r.merge(go, on=["match_id", "opp_id"], how="left").merge(form, on=["match_id", "side"])
    r["f5"] = r["log_dc"] - r["log_dc_opp"]
    r["f10"] = (r["side"] == "H").astype(float)
    for c in sorted(set(leagues.values())):
        r[f"lg_{c}"] = (r["div"] == c).astype(float)
    cols = N.FORM + [f"g_opp_{kk}" for kk in KS] + ["u_opp"]
    r = r[r["opp_keeper_known"].fillna(False).astype(bool) & np.isfinite(r[cols].astype(float)).all(axis=1)]
    both = r.groupby("match_id")["side"].nunique()
    r = r[r["match_id"].isin(both[both == 2].index)]
    return r.sort_values(["date", "match_id", "side"]).reset_index(drop=True)


def check(tb: dict, k: pd.DataFrame, leagues: dict[int, str], seasons: list[str]) -> dict:
    """Run condition (as amended on 2026-11-22): per scored league-season, the share of finished league matches
    with both starting keepers identified, and the share of that league's keeper lines (>= 45 minutes) with
    saves and conceded recorded."""
    m = tb["matches"]
    m = m[m["status_short"] == "FT"]
    st = starters(k)
    known = st.groupby("match_id")["team_id"].nunique()
    out = {}
    for lg, code in leagues.items():
        kl = k[(k["league"] == lg) & (k["minutes"] >= MIN_MINUTES)] if "league" in k else k[k["minutes"] >= MIN_MINUTES]
        lines = float(kl[["saves", "conceded"]].notna().all(axis=1).mean()) if len(kl) else 0.0
        for s in seasons:
            g = m[(m["league"] == lg) & (m["season"] == s)]
            share = float(g["match_id"].map(known).fillna(0).ge(2).mean()) if len(g) else 0.0
            out[f"{code} {s}"] = {"finished": int(len(g)), "keepers_identified_share": round(share, 4),
                                  "league_lines_complete": round(lines, 4)}
    return out


def failing(comp: dict) -> dict:
    return {c: v for c, v in comp.items() if tuple(c.split(" ")) not in EXCLUDED
            and (v["keepers_identified_share"] < MIN_COMPLETE or v["league_lines_complete"] < MIN_LINES)}


# ---------------------------------------------------------------- models and phases

def specs(leagues: dict[int, str], choice=None) -> dict:
    lg = [f"lg_{c}" for c in sorted(set(leagues.values()))]
    if choice is None:
        sp = {f"LIN_{l2}": ("lin", N.FORM + lg, l2) for l2 in L2S}
        sp.update({f"GK{kk}_{l2}": ("lin", N.FORM + lg + [f"g_opp_{kk}", "u_opp"], l2) for kk in KS for l2 in L2S})
        return sp
    kk = choice["k"]
    return {"LIN": ("lin", N.FORM + lg, choice["l2"]["LIN"]),
            "GK": ("lin", N.FORM + lg + [f"g_opp_{kk}", "u_opp"], choice["l2"]["GK"]),
            "G_ONLY": ("lin", N.FORM + lg + [f"g_opp_{kk}"], choice["l2"]["GK"]),
            "U_ONLY": ("lin", N.FORM + lg + ["u_opp"], choice["l2"]["GK"])}


def _claim(include_test: bool):
    for code in DEV9.values():
        ledger.claim("GK", code, ledger.season_range("2015-16", "2020-21"), "training")
        ledger.claim("GK", code, ledger.season_range("2021-22", "2025-26"), "development")
    if include_test:
        for code in ("G1", "T1", "I2", "D2", "F2", "SP2"):
            ledger.claim("GK", code, ledger.season_range("2015-16", "2019-20"), "training")
        for code in ("AUT", "SWZ", "DNK"):
            ledger.claim("GK", code, ledger.season_range("2015-16", "2019-20"), "test")
        for code in SEALED9.values():
            ledger.claim("GK", code, TEST_SEASONS, "test")


def _rows(leagues, tag, check_on=None):
    fd_cache._load_env()
    path = f"derived/gk_rows_{tag}.parquet"
    try:
        rows = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
    except RuntimeError:
        tb, k = F.load(leagues), keepers(leagues)
        if check_on:
            comp = check(tb, k, *check_on)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{tag}_check.json").write_text(json.dumps(comp, indent=2))
            short = failing(comp)
            if tag == "test" and short:
                raise SystemExit(f"run condition not met: {short}")
        rows = side_rows(tb, k, leagues)
        buf = io.BytesIO()
        rows.to_parquet(buf, index=False)
        D._retry(ss.put, BUCKET, path, buf.getvalue())
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    return rows


def dev():
    _claim(include_test=False)
    rows = _rows(DEV9, "dev", check_on=(DEV9, DEV_SEASONS))
    N.EXCLUDED = set()
    w, _ = N.walk_forward(rows, list(DEV9.values()), DEV_SEASONS, specs(DEV9))
    sc = {c: D.league_balanced(w, c) for c in w.columns if c.split("_")[0] in ("DC", "LIN") or c.startswith("GK")}
    lin = min((c for c in sc if c.startswith("LIN_")), key=sc.get)
    gk = min((c for c in sc if c.startswith("GK")), key=sc.get)
    choice = {"k": int(gk.split("_")[0][2:]), "l2": {"LIN": float(lin.split("_")[1]), "GK": float(gk.split("_")[1])},
              "dev_log_loss": sc, "dev_gk_minus_lin": sc[gk] - sc[lin]}
    se = N.block_se(w.assign(d=w[gk] - w[lin]), "d")
    choice["mde"] = {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
                     "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))}
    choice["dev_by_league"] = {d_: float((g[gk] - g[lin]).mean()) for d_, g in w.groupby("div")}
    r = rows[rows["match_id"].isin(set(w["match_id"]))]
    stand_in = set(r.loc[r["u_opp"] == 1, "match_id"])
    s_ = w[w["match_id"].isin(stand_in)]
    choice["dev_stand_in_matches"] = [float((s_[gk] - s_[lin]).mean()) if len(s_) else None, int(len(s_))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2, default=str))
    print(json.dumps({k_: choice[k_] for k_ in ("k", "l2", "dev_gk_minus_lin", "mde", "dev_by_league", "dev_stand_in_matches")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())        # frozen development choices
    _claim(include_test=True)
    leagues = {**DEV9, **SEALED9}
    rows = _rows(leagues, "test", check_on=(SEALED9, TEST_SEASONS))
    N.EXCLUDED = EXCLUDED
    w, _ = N.walk_forward(rows, list(SEALED9.values()), TEST_SEASONS, specs(leagues, choice))
    for a, b, col in (("GK", "LIN", "gk"), ("G_ONLY", "LIN", "g"), ("U_ONLY", "LIN", "u")):
        w[col] = w[a] - w[b]
    m_, lo, hi = D.block_ci(w, "gk")
    res = {"addendum": "GK", "frozen_dev_choice": {k_: choice[k_] for k_ in ("k", "l2", "mde")}, "n_test_matches": int(len(w)),
           "log_loss": {c: D.league_balanced(w, c) for c in ("DC", "LIN", "GK")},
           "primary": {"LIN+GK - LIN": {"mean": m_, "ci_95": [lo, hi]},
                       "decision": "confirmed" if (m_ <= -0.0005 and hi < 0) else ("closed" if lo > -0.0015 else "inconclusive")}}
    r = rows[rows["match_id"].isin(set(w["match_id"]))]
    stand_in = w["match_id"].isin(set(r.loc[r["u_opp"] == 1, "match_id"]))
    top = w["div"].isin(F.TOP_FLIGHTS)
    res["secondary"] = {
        "stand-in keeper matches (mean, n)": [D.league_balanced(w[stand_in], "gk") if stand_in.any() else None, int(stand_in.sum())],
        "g only": D.block_ci(w, "g"), "u only": D.block_ci(w, "u"),
        "top flights vs second tiers": {"top": D.league_balanced(w[top], "gk"), "second": D.league_balanced(w[~top], "gk")},
        "by league (mean, n)": {d_: [float(g.gk.mean()), int(len(g))] for d_, g in w.groupby("div")},
        "by season": {s_: D.league_balanced(g, "gk") for s_, g in w.groupby("season")},
        "development E0 (development data)": choice["dev_by_league"].get("E0")}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k_: res[k_] for k_ in ("n_test_matches", "log_loss", "primary")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
