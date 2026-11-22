"""Addendum FAT of docs/PREREGISTRATION.md (frozen in PR #78 before this file existed and before any
sealed data was loaded): rest, congestion and XI fatigue from all counted competitions.

    python prototypes/run_fat.py dev     # eight development leagues only: each model's L2, the MDE
    python prototypes/run_fat.py test    # the nine sealed leagues, 2020-21 -> 2025-26: scored once

Data: API-Football in Supabase (tabet.api_football). Parsed tables, Dixon-Coles rates and side-rows
are cached there too (derived/fat_*.parquet). Every role is claimed from the ledger first.

Implementation details the addendum leaves open, fixed here before any data is loaded:
  * Seasons: API season y is "y-(y+1)"; every league and competition is loaded for 2015-16 -> 2025-26
    (a season with no stored list is skipped and reported; Dixon-Coles needs 150 earlier matches).
  * Counted matches: status FT, AET or PEN, in the team's league or a counted competition. Days are
    UTC calendar days. "Previous 7/14/4 days" means kickoffs in [t - 7/14/4 days, t).
  * A match "has a lineup" for a team when that team has at least 10 listed starters.
  * Team form, Dixon-Coles and the linear fit are NLX's code (prototypes/run_nlx.py), so G's ten
    features keep their league-only rest (log of days, capped at 21).
  * Expected test matches for the MDE scaling: 15,000 (the addendum's "roughly 15,000").
"""
from __future__ import annotations

import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as D  # noqa: E402
import run_nlx as N  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402

OUT = ROOT / "results" / "fat"
DEV8 = {140: "SP1", 135: "I1", 78: "D1", 61: "F1", 88: "N1", 94: "P1", 144: "B1", 179: "SC0"}
SEALED9 = {218: "AUT", 207: "SWZ", 119: "DNK", 197: "G1", 203: "T1", 136: "I2", 79: "D2", 62: "F2", 141: "SP2"}
TOP_FLIGHTS = {"AUT", "SWZ", "DNK", "G1", "T1"}
EUROPE = {2: "Champions League", 3: "Europa League", 848: "Conference League"}
CUPS_DEV = {45: "FA Cup", 48: "League Cup", 143: "Copa del Rey", 137: "Coppa Italia", 81: "DFB Pokal",
            66: "Coupe de France", 90: "KNVB Beker", 96: "Taca de Portugal", 147: "Belgian Cup",
            181: "Scottish FA Cup", 185: "Scottish League Cup"}
CUPS_SEALED = {220: "Austrian Cup", 209: "Schweizer Cup", 121: "DBU Pokalen", 199: "Greek Cup", 206: "Turkiye Kupasi"}
SEASONS = list(range(2015, 2026))
DEV_SEASONS = [D.season_label(y) for y in range(2021, 2026)]
TEST_SEASONS = [D.season_label(y) for y in range(2020, 2026)]
FINISHED = {"FT", "AET", "PEN"}
REST_CAP, MIN_XI = 14, 10
MIN_LEAGUE_COMPLETE, MIN_OTHER_COMPLETE = 0.95, 0.90
EXPECTED_TEST_MATCHES = 15000
# Amendment of 2026-11-22 (before any model was fitted): these failed the run condition (provider gaps);
# they stay loaded as history but are never scored.
EXCLUDED = {("SWZ", s) for s in ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")} | {
    ("DNK", "2020-21"), ("G1", "2024-25"), ("T1", "2022-23"), ("F2", "2025-26")}
L2S = N.L2S
FORM = N.FORM
F1 = ["rest_own", "rest_opp", "load14_own", "load14_opp"]
F2 = ["xi7_own", "xi7_opp"]
FAT = F1 + F2


# ---------------------------------------------------------------- loading

def _build(store: SupabaseStore, league: int, seasons: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """NLX's builder, except that a season with no stored list is skipped (and returned)."""
    parts, missing = [], []
    for season in seasons:
        lp = D._retry(store.list_path, league, season)
        if lp is None or not D._retry(store.get, lp).get("response"):
            missing.append(season)       # no list, or an empty one (the Conference League began in 2021)
            continue
        parts.append(N._build_league(store, league, [season]))
    if not parts:
        raise RuntimeError(f"no fixture list stored for league {league} in {seasons}")
    return pd.concat(parts, ignore_index=True), missing


def load(ids: dict[int, str]) -> dict[str, pd.DataFrame]:
    """Matches, team rows and starters for these competitions, all seasons; cached per competition."""
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = {"matches": [], "teams": [], "starters": []}
    for league, name in ids.items():
        path = f"derived/fat_{league}_{SEASONS[0]}_{SEASONS[-1]}.parquet"
        try:
            blob = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        except RuntimeError:
            blob, missing = _build(store, league, SEASONS)
            buf = io.BytesIO()
            blob.to_parquet(buf, index=False)
            D._retry(ss.put, BUCKET, path, buf.getvalue())
            if missing:
                print(f"{name}: no list for seasons {missing}", flush=True)
        print(f"{name}: loaded", flush=True)
        for k in parts:
            parts[k].append(blob[blob["_table"] == k].dropna(axis=1, how="all").drop(columns="_table"))
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def counted(leagues: dict, comps: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(finished counted matches with a kind column, their starters)."""
    ms, ss_ = [], []
    for tb, kind in ((leagues, None), (comps, None)):
        m = tb["matches"][tb["matches"]["status_short"].isin(FINISHED)].copy()
        m["kind"] = np.where(m["league"].isin(list(EUROPE)), "europe",
                             np.where(m["league"].isin(list(CUPS_DEV) + list(CUPS_SEALED)), "cup", "league"))
        ms.append(m[["match_id", "league", "season", "kickoff_utc", "home_id", "away_id", "kind"]])
        ss_.append(tb["starters"].reindex(columns=["match_id", "team_id", "player_id"]))
    cm = pd.concat(ms, ignore_index=True).drop_duplicates("match_id")
    cs = pd.concat(ss_, ignore_index=True)
    return cm, cs[cs["match_id"].isin(set(cm["match_id"]))].drop_duplicates()


# ---------------------------------------------------------------- features

def fatigue(targets: pd.DataFrame, cm: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    """For each (match_id, team_id, kickoff_utc) in targets: rest, load14, xi7 and euro4, from
    counted matches strictly before the kickoff. xi7 is NaN without this match's lineup."""
    ns = lambda s: pd.to_datetime(s, utc=True).dt.as_unit("ns").astype("int64").to_numpy()  # noqa: E731
    long = pd.concat([cm.rename(columns={"home_id": "team_id"}), cm.rename(columns={"away_id": "team_id"})])
    long = long.sort_values("kickoff_utc")
    team = {t: (ns(g["kickoff_utc"]), (g["kind"] == "europe").to_numpy()) for t, g in long.groupby("team_id")}
    st = cs.merge(cm[["match_id", "kickoff_utc"]], on="match_id").sort_values("kickoff_utc")
    player = {p: ns(g["kickoff_utc"]) for p, g in st.groupby("player_id")}
    xi = cs.groupby(["match_id", "team_id"])["player_id"].apply(list).to_dict()
    day = 86_400 * 10**9
    out = []
    for r in targets.itertuples(index=False):
        t = int(pd.Timestamp(r.kickoff_utc).as_unit("ns").value)
        k, eu = team.get(r.team_id, (np.array([], dtype=np.int64), np.array([], dtype=bool)))
        i = int(np.searchsorted(k, t, "left"))
        rest = float(REST_CAP) if i == 0 else float(min(t // day - k[i - 1] // day, REST_CAP))
        load14 = float(i - np.searchsorted(k, t - 14 * day, "left"))
        j4 = int(np.searchsorted(k, t - 4 * day, "left"))
        euro4 = bool(eu[j4:i].any())
        ps = xi.get((r.match_id, r.team_id), [])
        if len(ps) >= MIN_XI:
            c = [np.searchsorted(player.get(p, np.array([], dtype=np.int64)), t, "left")
                 - np.searchsorted(player.get(p, np.array([], dtype=np.int64)), t - 7 * day, "left") for p in ps]
            xi7 = float(np.mean(c))
        else:
            xi7 = np.nan
        out.append({"match_id": r.match_id, "team_id": r.team_id, "rest": rest, "load14": load14, "xi7": xi7,
                    "euro4": euro4})
    return pd.DataFrame(out)


def dc_all(m: pd.DataFrame) -> dict:
    """NLX's weekly Dixon-Coles per league, cached under this family's own names."""
    fd_cache._load_env()
    out, todo = {}, []
    for lg, lm in m.groupby("league"):
        path = f"derived/fat_dc_{lg}_{lm['season'].min()}_{lm['season'].max()}.parquet"
        try:
            d = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
            out.update({r.match_id: (r.lh, r.la, r.rho) for r in d.itertuples()})
        except RuntimeError:
            todo.append((lg, lm, path))
    if todo:
        with ProcessPoolExecutor(min(4, len(todo))) as ex:
            for (lg, _, path), res in zip(todo, ex.map(D.dc_rates, [t[1] for t in todo])):
                d = pd.DataFrame([(k, *v) for k, v in res.items()], columns=["match_id", "lh", "la", "rho"])
                buf = io.BytesIO()
                d.to_parquet(buf, index=False)
                D._retry(ss.put, BUCKET, path, buf.getvalue())
                out.update(res)
                print(f"  DC league {lg}: {len(res)} matches", flush=True)
    return out


def side_rows(lt: dict, ct: dict, leagues: dict[int, str]) -> pd.DataFrame:
    m = lt["matches"]
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

    dc = dc_all(m)
    print(f"DC rates: {len(dc)} matches", flush=True)
    sot = lt["teams"].set_index(["match_id", "team_id"])["sot"]
    form = N.form_features(m, sot)
    print(f"form: {form.match_id.nunique()} matches", flush=True)
    cm, cs = counted(lt, ct)
    tg = pd.concat([m[["match_id", "kickoff_utc", "home_id"]].rename(columns={"home_id": "team_id"}),
                    m[["match_id", "kickoff_utc", "away_id"]].rename(columns={"away_id": "team_id"})])
    fat = fatigue(tg, cm, cs)
    print(f"fatigue: {len(fat)} team-matches", flush=True)

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
    fo = fat.rename(columns={"team_id": "own_id", "rest": "rest_own", "load14": "load14_own", "xi7": "xi7_own", "euro4": "euro4_own"})
    fp = fat.rename(columns={"team_id": "opp_id", "rest": "rest_opp", "load14": "load14_opp", "xi7": "xi7_opp", "euro4": "euro4_opp"})
    r = r.merge(fo, on=["match_id", "own_id"]).merge(fp, on=["match_id", "opp_id"]).merge(form, on=["match_id", "side"])
    r["f5"] = r["log_dc"] - r["log_dc_opp"]
    r["f10"] = (r["side"] == "H").astype(float)
    for c in sorted(set(leagues.values())):
        r[f"lg_{c}"] = (r["div"] == c).astype(float)
    ok = np.isfinite(r[FORM + FAT]).all(axis=1)
    r = r[ok]
    both = r.groupby("match_id")["side"].nunique()
    r = r[r["match_id"].isin(both[both == 2].index)]
    return r.sort_values(["date", "match_id", "side"]).reset_index(drop=True)


def check(lt: dict, ct: dict, leagues: dict[int, str], seasons: list[str]) -> dict:
    """The run condition: per league-season, the share of finished league matches with both lineups and
    both teams' shots on target, and of counted non-league fixtures involving its teams (between its
    first and last match) with both lineups."""
    cm, cs = counted(lt, ct)
    n_xi = cs.groupby(["match_id", "team_id"]).size()
    has_xi = lambda mid, t: n_xi.get((mid, t), 0) >= MIN_XI  # noqa: E731
    sot = lt["teams"].set_index(["match_id", "team_id"])["sot"]
    other = cm[cm["kind"] != "league"]
    out = {}
    for lg, code in leagues.items():
        for s in seasons:
            g = cm[(cm["league"] == lg) & (cm["season"] == s) & (cm["kind"] == "league")]
            if g.empty:
                out[f"{code} {s}"] = {"finished": 0, "league_share": 0.0, "other": 0, "other_share": None}
                continue
            ok = [has_xi(f.match_id, f.home_id) and has_xi(f.match_id, f.away_id)
                  and pd.notna(sot.get((f.match_id, f.home_id), np.nan)) and pd.notna(sot.get((f.match_id, f.away_id), np.nan))
                  for f in g.itertuples()]
            teams = set(g["home_id"]) | set(g["away_id"])
            o = other[(other["home_id"].isin(teams) | other["away_id"].isin(teams))
                      & other["kickoff_utc"].between(g["kickoff_utc"].min(), g["kickoff_utc"].max())]
            ok_o = [has_xi(f.match_id, f.home_id) and has_xi(f.match_id, f.away_id) for f in o.itertuples()]
            out[f"{code} {s}"] = {"finished": int(len(g)), "league_share": round(float(np.mean(ok)), 4),
                                  "other": int(len(o)), "other_share": round(float(np.mean(ok_o)), 4) if ok_o else None,
                                  "other_by_kind": o["kind"].value_counts().to_dict()}
    return out


def failing(comp: dict) -> dict:
    return {k: v for k, v in comp.items() if v["league_share"] < MIN_LEAGUE_COMPLETE
            or (v["other_share"] is not None and v["other_share"] < MIN_OTHER_COMPLETE)}


# ---------------------------------------------------------------- models and phases

def specs(leagues: dict[int, str], choice=None) -> dict:
    lg = [f"lg_{c}" for c in sorted(set(leagues.values()))]
    sets = {"LIN": FORM + lg, "FAT": FORM + lg + FAT, "F1": FORM + lg + F1, "F2": FORM + lg + F2}
    if choice is None:
        return {f"{k}_{l2}": ("lin", f, l2) for k, f in sets.items() for l2 in L2S}
    return {k: ("lin", f, choice["l2"][k]) for k, f in sets.items()}


def _claim(include_test: bool):
    for code in DEV8.values():
        ledger.claim("FAT", code, ledger.season_range("2015-16", "2020-21"), "training")
        ledger.claim("FAT", code, ledger.season_range("2021-22", "2025-26"), "development")
    if include_test:
        for code in ("G1", "T1", "I2", "D2", "F2", "SP2"):
            ledger.claim("FAT", code, ledger.season_range("2015-16", "2019-20"), "training")
        for code in ("AUT", "SWZ", "DNK"):
            ledger.claim("FAT", code, ledger.season_range("2015-16", "2019-20"), "test")   # history only
        for code in SEALED9.values():
            ledger.claim("FAT", code, TEST_SEASONS, "test")


def _rows(leagues: dict, comps: dict, tag: str, check_on: dict | None = None):
    """Side-rows (cached per phase) and the completeness report."""
    fd_cache._load_env()
    path = f"derived/fat_rows_{tag}.parquet"
    try:
        rows = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        comp = None
    except RuntimeError:
        lt, ct = load(leagues), load(comps)
        comp = check(lt, ct, *(check_on or (leagues, DEV_SEASONS)))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}_check.json").write_text(json.dumps(comp, indent=2))
        short = {k: v for k, v in failing(comp).items() if tuple(k.split(" ")) not in EXCLUDED}
        if tag == "test" and short:
            raise SystemExit(f"run condition not met: {short}")
        rows = side_rows(lt, ct, leagues)
        buf = io.BytesIO()
        rows.to_parquet(buf, index=False)
        D._retry(ss.put, BUCKET, path, buf.getvalue())
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    return rows, comp


def choose(w: pd.DataFrame) -> dict:
    sc = {c: D.league_balanced(w, c) for c in w.columns if c.split("_")[0] in ("DC", "LIN", "FAT", "F1", "F2")}
    l2 = {k: float(min((c for c in sc if c.split("_")[0] == k), key=sc.get).split("_")[1]) for k in ("LIN", "FAT", "F1", "F2")}
    return {"l2": l2, "dev_log_loss": sc}


def dev():
    _claim(include_test=False)
    rows, comp = _rows(DEV8, {**EUROPE, **CUPS_DEV}, "dev")
    print(f"rows: {rows.match_id.nunique()} eligible matches", flush=True)
    w, _ = N.walk_forward(rows, list(DEV8.values()), DEV_SEASONS, specs(DEV8))
    choice = choose(w)
    lin, fat = f"LIN_{choice['l2']['LIN']}", f"FAT_{choice['l2']['FAT']}"
    se = N.block_se(w.assign(d=w[fat] - w[lin]), "d")
    choice["mde"] = {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
                     "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))}
    choice["dev_fat_minus_lin"] = float(D.league_balanced(w.assign(d=w[fat] - w[lin]), "d"))
    choice["eligible_per_league_season"] = {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])}
    choice["dev_check"] = comp
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2, default=str))
    print(json.dumps({k: choice[k] for k in ("l2", "mde", "dev_fat_minus_lin")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())      # frozen development choices
    _claim(include_test=True)
    leagues = {**DEV8, **SEALED9}
    rows, _ = _rows(leagues, {**EUROPE, **CUPS_DEV, **CUPS_SEALED}, "test", check_on=(SEALED9, TEST_SEASONS))
    N.EXCLUDED = EXCLUDED          # walk_forward skips these when scoring (they stay as history)
    w, _ = N.walk_forward(rows, list(SEALED9.values()), TEST_SEASONS, specs(leagues, choice))
    for a, b, col in (("FAT", "LIN", "fat"), ("F1", "LIN", "f1"), ("F2", "LIN", "f2"), ("LIN", "DC", "lin_dc")):
        w[col] = w[a] - w[b]
    res = {"addendum": "FAT", "excluded_by_amendment": sorted(" ".join(x) for x in EXCLUDED), "frozen_dev_choice": {k: choice[k] for k in ("l2", "mde")},
           "n_test_matches": int(len(w)),
           "matches_per_league_season": {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])},
           "log_loss": {c: D.league_balanced(w, c) for c in ("DC", "LIN", "FAT", "F1", "F2")}}
    m, lo, hi = D.block_ci(w, "fat")
    verdict = "confirmed" if (m <= -0.0005 and hi < 0) else ("closed" if lo > -0.0015 else "inconclusive")
    res["primary"] = {"FAT - LIN": {"mean": m, "ci_95": [lo, hi]}, "decision": verdict}
    sec = {}
    for col, name in (("f1", "F1 only - LIN"), ("f2", "F2 only - LIN"), ("lin_dc", "LIN - DC")):
        mm, l, h = D.block_ci(w, col)
        sec[name] = {"mean": mm, "ci_95": [l, h]}
    top = w["div"].isin(TOP_FLIGHTS)
    sec["top flights vs second tiers (mean, n)"] = {"top": [D.league_balanced(w[top], "fat"), int(top.sum())],
                                                    "second": [D.league_balanced(w[~top], "fat"), int((~top).sum())]}
    eu = rows.groupby("match_id")[["euro4_own", "euro4_opp"]].any().any(axis=1)
    hang = w["match_id"].map(eu).fillna(False).astype(bool)
    sec["European hangover (mean, n)"] = [float(w[hang]["fat"].mean()) if hang.any() else None, int(hang.sum())]
    sec["by league (mean, n)"] = {d: [float(g.fat.mean()), int(len(g))] for d, g in w.groupby("div")}
    sec["by season"] = {s: D.league_balanced(g, "fat") for s, g in w.groupby("season")}
    nc = w[w.season != "2020-21"]
    sec["excluding 2020-21 (mean, n)"] = [D.league_balanced(nc, "fat"), int(len(nc))]
    res["secondary"] = sec
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss", "primary")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
