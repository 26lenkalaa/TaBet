"""Addendum DXI of docs/PREREGISTRATION.md (committed 2026-11-16, before this file existed and
before any player data for it was looked at): does a defensive-XI block add information beyond an
attacking-XI rating?

    python prototypes/run_dxi.py check   # completeness + field definitions, all eight leagues (no results read)
    python prototypes/run_dxi.py dev     # big four 2022-25: choose BASE's then DEF's settings
    python prototypes/run_dxi.py test    # four test leagues 2022-25: scored once with the frozen dev choices

Data: API-Football match details stored in Supabase by tabet.api_football (private bucket
`api-football`). The parsed tables are cached there too (derived/dxi_*.parquet); nothing is kept on
disk except the results JSON. Every scored period is claimed from the data ledger first.

Implementation details the addendum does not spell out, fixed here before any data is loaded:
  * League matches only (status FT). Season = API season y -> "y-(y+1)".
  * Starters come from the lineups' startXI; a starter's position is that match's `games.position`
    (falling back to the lineup's `pos`). Outfielders = D, M, F. Missing counts count as 0.
  * Player ratings use every earlier date in the same league (2022-25 data only, as registered in the
    ledger). Position means are earlier same-league totals over earlier minutes for that position. A
    player with no earlier minutes gets the position mean.
  * Possession adjustment: each match's tackles + interceptions are multiplied by 50 / the opponent's
    possession %. If possession is missing, no adjustment (factor 1); the count is reported.
  * A team-match's block is compared with the mean of that team's previous (up to) 10 league blocks;
    eligibility needs at least 5 earlier matches with player stats for both teams.
  * Stage 2 penalised Poisson: standardised features, unpenalised intercept (as in addendum G).
    A month is scored only if the stage-2 training set has at least 500 earlier side-rows (same
    months for every model, so contrasts stay paired).
  * Development fits train on the four development leagues only: the ledger forbids touching the
    test leagues before the test. The test refits on all eight leagues' earlier rows, as registered.
  * Early season (secondary) = either team has played fewer than 10 league matches that season.
"""
from __future__ import annotations

import gzip
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
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402
from tabet.experiments import _fit_poisson, _probs  # noqa: E402
from tabet.model import DixonColes, Hyperparams  # noqa: E402

OUT = ROOT / "results" / "dxi"
DEV_LEAGUES = {140: "SP1", 135: "I1", 78: "D1", 61: "F1"}
TEST_LEAGUES = {88: "N1", 94: "P1", 144: "B1", 179: "SC0"}
API_SEASONS = (2022, 2023, 2024)
SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in API_SEASONS]
K_ATT, K_DEF, A_DUEL, L2S = (300, 900, 2700), (300, 900, 2700), (20, 60), (1, 10, 100)
MIN_COMPLETE = 0.95
MIN_DC_MATCHES, MIN_PREV, PREV_N, MIN_STAGE2 = 150, 5, 10, 500
PSTATS = ["minutes", "shots", "key_passes", "tackles", "interceptions", "duels", "duels_won"]


def season_label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


# ---------------------------------------------------------------- parsing

def _n(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def parse_fixture(f: dict, league: int, season: int) -> tuple[dict, list[dict], list[dict]]:
    """One API-Football fixture -> (match row, team rows, starter rows)."""
    fx, teams, goals = f["fixture"], f["teams"], f.get("goals") or {}
    h, a = str(teams["home"]["id"]), str(teams["away"]["id"])
    match = {"match_id": str(fx["id"]), "league": league, "season": season_label(season),
             "kickoff_utc": pd.Timestamp(fx["date"]).tz_convert("UTC"), "status_short": fx["status"]["short"],
             "home_id": h, "away_id": a, "home_goals": goals.get("home"), "away_goals": goals.get("away")}
    poss = {}
    for t in f.get("statistics") or []:
        v = {d["type"]: d.get("value") for d in t.get("statistics") or []}.get("Ball Possession")
        if isinstance(v, str) and v.endswith("%"):
            poss[str(t["team"]["id"])] = float(v[:-1])
    team_rows = [{"match_id": match["match_id"], "team_id": t, "possession": poss.get(t, np.nan)} for t in (h, a)]
    stats = {}
    for t in f.get("players") or []:
        for p in t.get("players") or []:
            s = (p.get("statistics") or [{}])[0]
            g, sh, pa, tk, du = (s.get(k) or {} for k in ("games", "shots", "passes", "tackles", "duels"))
            stats[str(p["player"]["id"])] = {
                "position": g.get("position"), "minutes": _n(g.get("minutes")), "shots": _n(sh.get("total")),
                "key_passes": _n(pa.get("key")), "tackles": _n(tk.get("total")),
                "interceptions": _n(tk.get("interceptions")), "duels": _n(du.get("total")), "duels_won": _n(du.get("won"))}
    starters = []
    for t in f.get("lineups") or []:
        tid = str(t["team"]["id"])
        for x in t.get("startXI") or []:
            pid = str((x.get("player") or {}).get("id"))
            s = stats.get(pid)
            pos = (s or {}).get("position") or (x.get("player") or {}).get("pos")
            starters.append({"match_id": match["match_id"], "team_id": tid, "player_id": pid, "pos": pos,
                             "has_stats": s is not None, **{k: (s or {}).get(k, 0.0) for k in PSTATS}})
    return match, team_rows, starters


def load_tables(leagues: dict[int, str]) -> dict[str, pd.DataFrame]:
    """Matches, team rows and starters for these leagues' 2022-25 seasons, from Supabase.
    Cached per league as derived/dxi_<league>.parquet in the api-football bucket."""
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = {"matches": [], "teams": [], "starters": []}
    for league in leagues:
        path = f"derived/dxi_{league}.parquet"
        try:
            blob = pd.read_parquet(io.BytesIO(ss.get(BUCKET, path)))
        except RuntimeError:
            blob = _build_league(store, league)
            buf = io.BytesIO()
            blob.to_parquet(buf, index=False)
            ss.put(BUCKET, path, buf.getvalue())
        for k in parts:
            parts[k].append(blob[blob["_table"] == k].dropna(axis=1, how="all").drop(columns="_table"))
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def _build_league(store: SupabaseStore, league: int) -> pd.DataFrame:
    m, t, s = [], [], []
    for season in API_SEASONS:
        lp = store.list_path(league, season)
        if lp is None:
            raise RuntimeError(f"no fixture list stored for league {league} season {season}")
        listing = store.get(lp)["response"]
        rows = store._rows("apif_fixtures", "fixture_id,object_path", league=f"eq.{league}", season=f"eq.{season}")
        paths = sorted({r["object_path"] for r in rows})
        with ThreadPoolExecutor(8) as ex:
            bodies = list(ex.map(store.get, paths))
        details = {f["fixture"]["id"]: f for b in bodies for f in b.get("response", [])}
        for f in listing:
            d = details.get(f["fixture"]["id"])
            mm, tt, st = parse_fixture(d or f, league, season)
            mm["has_details"] = d is not None
            m.append(mm), t.extend(tt), s.extend(st if d else [])
    out = [pd.DataFrame(m).assign(_table="matches"), pd.DataFrame(t).assign(_table="teams"),
           pd.DataFrame(s).assign(_table="starters")]
    blob = pd.concat(out, ignore_index=True)
    for c in ("home_goals", "away_goals", "possession", *PSTATS):
        blob[c] = pd.to_numeric(blob[c], errors="coerce")
    blob["league"] = blob["league"].astype("Int64")
    blob["has_stats"] = blob["has_stats"].astype("boolean")
    blob["has_details"] = blob["has_details"].astype("boolean")
    return blob


# ---------------------------------------------------------------- check (no results read)

def completeness(tb: dict[str, pd.DataFrame], leagues: dict[int, str]) -> dict:
    m = tb["matches"]
    m = m[m["status_short"] == "FT"]
    st = tb["starters"]
    with_stats = st[st["has_stats"].fillna(False)].groupby("match_id")["team_id"].nunique()
    out = {}
    for lg, code in leagues.items():
        for s in SEASONS:
            g = m[(m["league"] == lg) & (m["season"] == s)]
            ok = g["match_id"].map(with_stats).fillna(0).ge(2) & g["has_details"].fillna(False)
            out[f"{code} {s}"] = {"finished": int(len(g)), "with_player_stats": int(ok.sum()),
                                  "share": round(float(ok.mean()) if len(g) else 0.0, 4)}
    return out


def field_checks(tb: dict[str, pd.DataFrame]) -> dict:
    """Definitions, never outcomes: does `duels` count each contest for both teams (as a two-sided
    contest count must), are won <= total, what positions appear, how often possession is missing."""
    st = tb["starters"][tb["starters"]["has_stats"].fillna(False)]
    team = st.groupby(["match_id", "team_id"])[["duels", "duels_won", "tackles", "interceptions"]].sum().reset_index()
    pair = team.merge(team, on="match_id", suffixes=("", "_opp"))
    pair = pair[pair["team_id"] != pair["team_id_opp"]]
    return {
        "starters_with_stats": int(len(st)),
        "won_gt_total_rows": int((st["duels_won"] > st["duels"]).sum()),
        "positions": st["pos"].value_counts(dropna=False).to_dict(),
        "starter_minutes_zero_share": round(float((st["minutes"] == 0).mean()), 4),
        "possession_missing_share": round(float(tb["teams"]["possession"].isna().mean()), 4),
        "team_duels_vs_opponent_corr (starters only)": round(float(np.corrcoef(pair["duels"], pair["duels_opp"])[0, 1]), 3),
        "team_duels_won_share_mean": round(float((team["duels_won"] / team["duels"].replace(0, np.nan)).mean()), 3),
        "note": "API-Football's duels.total counts ground and aerial duels together (no split is provided)",
    }


# ---------------------------------------------------------------- features

def dc_rates(m: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Weekly Dixon-Coles per league on earlier dates, previous 3 years, at least 150 earlier matches."""
    out = {}
    for _, lm in m.groupby("league"):
        lm = lm.sort_values("kickoff_utc")
        for _, g in lm.groupby(lm["date"].dt.tz_localize(None).dt.to_period("W"), sort=True):
            cut = g["date"].min()
            tr = lm[(lm["date"] < cut) & (lm["date"] >= cut - pd.Timedelta(days=3 * 365))]
            if len(tr) < MIN_DC_MATCHES:
                continue
            try:
                fit = DixonColes(Hyperparams()).fit(tr, cut)
            except Exception:
                continue
            for f in g.itertuples():
                try:
                    lh, la = fit.expected_goals(f.home_id, f.away_id)
                except Exception:   # a team with no earlier matches in the window
                    continue
                out[f.match_id] = (float(np.log(lh)), float(np.log(la)), float(fit.rho))
    return out


def starter_history(tb: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each starter with his earlier same-league totals and the earlier position rates (before this date)."""
    m = tb["matches"][["match_id", "league", "date", "home_id", "away_id"]]
    poss = tb["teams"].set_index(["match_id", "team_id"])["possession"]
    st = tb["starters"][tb["starters"]["has_stats"].fillna(False)].merge(m, on="match_id")
    st["opp_id"] = np.where(st["team_id"] == st["home_id"], st["away_id"], st["home_id"])
    opp_poss = pd.Series(list(zip(st["match_id"], st["opp_id"]))).map(poss).to_numpy(dtype=float)
    st["bw_adj"] = (st["tackles"] + st["interceptions"]) * np.where(np.isnan(opp_poss), 1.0, 50.0 / np.clip(opp_poss, 1, 99))
    st["att"] = st["shots"] + st["key_passes"]
    st = st.sort_values(["league", "date"]).reset_index(drop=True)
    cols = ["minutes", "att", "bw_adj", "duels", "duels_won"]
    # player totals from earlier dates: cumulative by date, shifted to exclude this date
    day = st.groupby(["league", "player_id", "date"])[cols].sum().groupby(level=[0, 1]).cumsum()
    prev = day.groupby(level=[0, 1]).shift(1).fillna(0.0)
    st = st.join(prev.add_suffix("_prev"), on=["league", "player_id", "date"])
    pday = st.groupby(["league", "pos", "date"])[cols].sum().groupby(level=[0, 1]).cumsum()
    pprev = pday.groupby(level=[0, 1]).shift(1)
    st = st.join(pprev.add_suffix("_pos"), on=["league", "pos", "date"])
    return st


def blocks(st: pd.DataFrame, k_att: float, k_def: float, a: float) -> pd.DataFrame:
    """Per team-match: ATT, BW and DUEL (ratings from earlier dates only)."""
    att_rate = st["att_pos"] / st["minutes_pos"]
    bw_rate = st["bw_adj_pos"] / st["minutes_pos"]
    duel_rate = st["duels_won_pos"] / st["duels_pos"]
    r = st[["match_id", "team_id", "league", "date", "pos"]].copy()
    r["att_r"] = 90 * (st["att_prev"] + k_att * att_rate) / (st["minutes_prev"] + k_att)
    r["bw_r"] = 90 * (st["bw_adj_prev"] + k_def * bw_rate) / (st["minutes_prev"] + k_def)
    r["duel_r"] = (st["duels_won_prev"] + a * duel_rate) / (st["duels_prev"] + a)
    out = r[r["pos"].isin(["D", "M", "F"])].groupby(["match_id", "team_id", "league", "date"])["att_r"].mean().rename("ATT").to_frame()
    dm = r[r["pos"].isin(["D", "M"])].groupby(["match_id", "team_id", "league", "date"])[["bw_r", "duel_r"]].mean()
    return out.join(dm.rename(columns={"bw_r": "BW", "duel_r": "DUEL"})).reset_index()


def relative(b: pd.DataFrame) -> pd.DataFrame:
    """log(block / mean of the team's previous <= 10 blocks); n_prev = earlier matches with player stats."""
    b = b.sort_values("date").copy()
    g = b.groupby(["league", "team_id"])
    b["n_prev"] = g.cumcount()
    for c in ("ATT", "BW", "DUEL"):
        base = g[c].transform(lambda s: s.shift(1).rolling(PREV_N, min_periods=1).mean())
        b[c] = np.log(b[c] / base)
    return b


def side_rows(tb, st, dc, k_att, k_def, a) -> pd.DataFrame:
    m = tb["matches"]
    b = relative(blocks(st, k_att, k_def, a)).set_index(["match_id", "team_id"])
    rows = []
    for f in m.itertuples():
        if f.match_id not in dc or (f.match_id, f.home_id) not in b.index or (f.match_id, f.away_id) not in b.index:
            continue
        bh, ba = b.loc[(f.match_id, f.home_id)], b.loc[(f.match_id, f.away_id)]
        if min(bh["n_prev"], ba["n_prev"]) < MIN_PREV:
            continue
        vals = [bh["ATT"], bh["BW"], bh["DUEL"], ba["ATT"], ba["BW"], ba["DUEL"]]
        if not np.all(np.isfinite(vals)):
            continue
        lh, la, rho = dc[f.match_id]
        common = {"match_id": f.match_id, "div": f.div, "season": f.season, "date": f.date, "rho": rho,
                  "result": f.result, "early": f.early}
        rows.append({**common, "side": "H", "log_dc": lh, "goals": f.home_goals,
                     "att": bh["ATT"], "opp_bw": ba["BW"], "opp_duel": ba["DUEL"]})
        rows.append({**common, "side": "A", "log_dc": la, "goals": f.away_goals,
                     "att": ba["ATT"], "opp_bw": bh["BW"], "opp_duel": bh["DUEL"]})
    return pd.DataFrame(rows)


def prepare(tb: dict[str, pd.DataFrame], leagues: dict[int, str]) -> tuple[dict, pd.DataFrame, dict]:
    m = tb["matches"]
    m = m[(m["status_short"] == "FT") & m["home_goals"].notna()].copy()
    m["status"] = "completed"
    m["date"] = m["kickoff_utc"].dt.normalize()
    m["div"] = m["league"].map(leagues)
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    m = m.sort_values("kickoff_utc")
    played = pd.concat([m[["match_id", "season", "home_id"]].rename(columns={"home_id": "t"}),
                        m[["match_id", "season", "away_id"]].rename(columns={"away_id": "t"})])
    played["n"] = played.groupby(["season", "t"]).cumcount()
    early = played.groupby("match_id")["n"].min() < 10
    m["early"] = m["match_id"].map(early).fillna(True)
    tb = {**tb, "matches": m}
    return tb, starter_history(tb), dc_rates(m)


# ---------------------------------------------------------------- stage 2 and scoring

MODELS = {"BASE": ["att"], "DEF": ["att", "opp_bw", "opp_duel"], "BW": ["att", "opp_bw"], "DUEL": ["att", "opp_duel"]}


def fit_lin(tr: pd.DataFrame, feats: list[str], l2: float):
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    b = _fit_poisson(((tr[feats] - mu) / sd).to_numpy(), tr["log_dc"].to_numpy(), tr["goals"].to_numpy(), l2)
    return lambda te: b[0] + ((te[feats] - mu) / sd).to_numpy() @ b[1:]


def log_loss(te: pd.DataFrame, adj: np.ndarray) -> pd.DataFrame:
    t = te.assign(lam=np.exp(te["log_dc"].to_numpy() + adj))
    h = t[t.side == "H"].set_index("match_id")
    aw = t[t.side == "A"].set_index("match_id").reindex(h.index)
    pr = _probs(h["lam"].to_numpy(), aw["lam"].to_numpy(), h["rho"].to_numpy())
    y = h["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return pd.DataFrame({"match_id": h.index, "div": h["div"], "season": h["season"], "date": h["date"],
                         "early": h["early"], "ll": -np.log(np.clip(pr[np.arange(len(y)), y], 1e-15, 1))}).reset_index(drop=True)


def walk_forward(rows: pd.DataFrame, eval_divs, specs: dict[str, tuple[list[str], float]]) -> pd.DataFrame:
    """Monthly refits on all earlier rows; DC plus each named (features, L2) spec. Wide per-match table."""
    sel = rows[rows["div"].isin(eval_divs)]
    out = []
    month = sel["date"].dt.tz_localize(None).dt.to_period("M")
    for mth in sorted(month.unique()):
        start = mth.start_time.tz_localize("UTC")
        te = sel[month == mth]
        tr = rows[rows["date"] < start]
        if len(tr) < MIN_STAGE2:
            continue
        w = log_loss(te, np.zeros(len(te))).rename(columns={"ll": "DC"})
        for name, (feats, l2) in specs.items():
            w[name] = log_loss(te, fit_lin(tr, feats, l2)(te))["ll"].to_numpy()
        out.append(w)
    if not out:
        raise RuntimeError(f"no month had {MIN_STAGE2} earlier stage-2 rows to train on")
    return pd.concat(out, ignore_index=True)


def league_balanced(d: pd.DataFrame, col: str) -> float:
    return float(d.groupby("div")[col].mean().mean())


def block_ci(d: pd.DataFrame, col: str, reps=2000, seed=0):
    """Equal-weight mean of per-league means; 14-day blocks spanning all four leagues together."""
    d = d.assign(block=((d["date"] - d["date"].min()).dt.days // 14))
    g = d.groupby(["block", "div"])[col].agg(["sum", "count"]).unstack("div").fillna(0)
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, len(g), len(g))
        s, c = sums[pick].sum(0), cnts[pick].sum(0)
        boots.append(float(np.mean(s / np.where(c > 0, c, np.nan))))
    return league_balanced(d, col), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


# ---------------------------------------------------------------- phases

def _claim(leagues: dict[int, str], role: str):
    for code in leagues.values():
        ledger.claim("DXI", code, SEASONS, role)


def _require_complete(comp: dict):
    short = {k: v for k, v in comp.items() if v["share"] < MIN_COMPLETE}
    if short:
        raise SystemExit(f"run condition not met (needs >= {MIN_COMPLETE:.0%} with player stats): {short}")


def check():
    leagues = {**DEV_LEAGUES, **TEST_LEAGUES}
    tb = load_tables(leagues)
    res = {"completeness": completeness(tb, leagues), "fields": field_checks(tb)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


def dev():
    _claim(DEV_LEAGUES, "development")
    tb = load_tables(DEV_LEAGUES)
    _require_complete(completeness(tb, DEV_LEAGUES))
    tb, st, dc = prepare(tb, DEV_LEAGUES)
    divs = list(DEV_LEAGUES.values())
    base_scores = {}
    for k_att in K_ATT:
        rows = side_rows(tb, st, dc, k_att, K_DEF[0], A_DUEL[0])
        w = walk_forward(rows, divs, {f"BASE_{l2}": (MODELS["BASE"], l2) for l2 in L2S})
        for l2 in L2S:
            base_scores[f"k_att={k_att} L2={l2}"] = (league_balanced(w, f"BASE_{l2}"), k_att, l2)
        print(f"BASE k_att={k_att}: {len(w)} matches", flush=True)
    _, k_att, base_l2 = min(base_scores.values())
    def_scores = {}
    for k_def in K_DEF:
        for a in A_DUEL:
            rows = side_rows(tb, st, dc, k_att, k_def, a)
            specs = {f"DEF_{l2}": (MODELS["DEF"], l2) for l2 in L2S}
            w = walk_forward(rows, divs, specs)
            for l2 in L2S:
                def_scores[f"k_def={k_def} a={a} L2={l2}"] = (league_balanced(w, f"DEF_{l2}"), k_def, a, l2)
            print(f"DEF k_def={k_def} a={a}: {len(w)} matches", flush=True)
    _, k_def, a, def_l2 = min(def_scores.values())
    choice = {"k_att": k_att, "base_l2": base_l2, "k_def": k_def, "a": a, "def_l2": def_l2,
              "base_dev_log_loss": {k: v[0] for k, v in base_scores.items()},
              "def_dev_log_loss": {k: v[0] for k, v in def_scores.items()}, "n_dev_matches": int(len(w)),
              "dc_dev_log_loss": league_balanced(w, "DC")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps({k: choice[k] for k in ("k_att", "base_l2", "k_def", "a", "def_l2", "n_dev_matches")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())   # the frozen development choices
    _claim(DEV_LEAGUES, "development")
    _claim(TEST_LEAGUES, "test")
    leagues = {**DEV_LEAGUES, **TEST_LEAGUES}
    tb = load_tables(leagues)
    _require_complete(completeness(tb, TEST_LEAGUES))
    tb, st, dc = prepare(tb, leagues)
    rows = side_rows(tb, st, dc, choice["k_att"], choice["k_def"], choice["a"])
    l2b, l2d = choice["base_l2"], choice["def_l2"]
    specs = {"BASE": (MODELS["BASE"], l2b), "DEF": (MODELS["DEF"], l2d),
             "BW": (MODELS["BW"], l2d), "DUEL": (MODELS["DUEL"], l2d)}
    w = walk_forward(rows, list(TEST_LEAGUES.values()), specs)
    w["primary"], w["base_dc"] = w.DEF - w.BASE, w.BASE - w.DC
    w["bw_base"], w["duel_base"] = w.BW - w.BASE, w.DUEL - w.BASE
    m, lo, hi = block_ci(w, "primary")
    res = {"addendum": "DXI", "frozen_dev_choice": {k: choice[k] for k in ("k_att", "base_l2", "k_def", "a", "def_l2")},
           "n_test_matches": int(len(w)), "log_loss": {c: league_balanced(w, c) for c in ("DC", "BASE", "DEF")},
           "primary": {"DEF - BASE": {"mean": m, "ci_95": [lo, hi]}},
           "decision": {"confirmed": bool(m <= -0.001 and hi < 0)}, "secondary": {}}
    sec = res["secondary"]
    for col, name in (("base_dc", "BASE - DC"), ("bw_base", "BW alone - BASE"), ("duel_base", "DUEL alone - BASE")):
        mm, l, h = block_ci(w, col)
        sec[name] = {"mean": mm, "ci_95": [l, h]}
    sec["by league (DEF - BASE, n)"] = {d: [float(g.primary.mean()), int(len(g))] for d, g in w.groupby("div")}
    sec["by season (DEF - BASE)"] = {s: league_balanced(g, "primary") for s, g in w.groupby("season")}
    e = w[w.early]
    sec["early season (DEF - BASE, n)"] = [league_balanced(e, "primary"), int(len(e))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss", "primary", "decision")}, indent=2))


if __name__ == "__main__":
    {"check": check, "dev": dev, "test": test}[sys.argv[1]]()
