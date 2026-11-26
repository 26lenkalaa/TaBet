"""Addenda NLX and XIA of docs/PREREGISTRATION.md (frozen in PR #74 before this file existed and
before their test seasons were loaded).

    python prototypes/run_nlx.py dev     # big four only: grids, the NL choice, the MDE and XIA's bands
    python prototypes/run_nlx.py test    # N1/P1/B1/SC0 2020-22 and 2025-26: scored once, both families

NLX: does a nonlinear stage 2 (GBM or NN, whichever wins development) beat a linear one on identical
inputs that include player-level XI blocks? XIA: does the XI block itself beat the same linear model
without it? Data: API-Football in Supabase (tabet.api_football); parsed tables, Dixon-Coles rates and
side-rows are cached there too (derived/nlx_*.parquet). Every role is claimed from the ledger first.

Implementation details the addenda leave open, fixed here before any data is loaded:
  * Seasons: API season y is "y-(y+1)". Big four: 2015-16 -> 2024-25 (2025-26 is not registered, so not
    loaded). Test leagues (test phase only): 2015-16 -> 2025-26.
  * Team form follows addendum G's code (prototypes/run_g_boosted_stage2.py), with API-Football's
    team "Shots on Goal" for shots on target.
  * XI blocks: prototypes/run_dxi.py with DXI's frozen settings (k_att 300, k_def 2700, a 20).
  * Expected test matches for the MDE scaling: 3,200 (the pre-registration's 3,000-3,500).
  * Development and test score the same model list; LIN-X uses LIN's chosen L2; LIN-XI has its own.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as D  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.api_football import BUCKET, SupabaseStore  # noqa: E402
from tabet.experiments import _fit_poisson  # noqa: E402

OUT = ROOT / "results" / "nlx"
BIG4 = {140: "SP1", 135: "I1", 78: "D1", 61: "F1"}
TEST4 = {88: "N1", 94: "P1", 144: "B1", 179: "SC0"}
ALL8 = {**BIG4, **TEST4}
BIG4_SEASONS = list(range(2015, 2025))           # 2015-16 .. 2024-25
TEST4_SEASONS = list(range(2015, 2026))          # 2015-16 .. 2025-26
DEV_SEASONS = [D.season_label(y) for y in range(2019, 2025)]
TEST_SEASONS = ["2020-21", "2021-22", "2025-26"]
# Amendment of 2026-11-17 (before any model was fitted): B1 2020-21 failed the run condition (provider
# gap, 27% with player stats), so it is training history only, never scored.
EXCLUDED = {("B1", "2020-21")}
K_ATT, K_DEF, A_DUEL = 300, 2700, 20             # DXI's frozen development choices
HALF_LIFE, MIN_PREV_SOT = 120.0, 5               # addendum G
MIN_PREV_XI, MIN_STAGE2, MIN_COMPLETE = 5, 500, 0.95
EXPECTED_TEST_MATCHES = 3200
L2S = (0.1, 1.0, 10.0, 100.0)
GBM_GRID = [(d, leaf) for d in (2, 3) for leaf in (50, 200)]
NN_GRID = [(h, wd) for h in (16, 64) for wd in (0.001, 0.01)]
MAX_ROUNDS, PATIENCE, VAL_FRAC = 1000, 50, 0.15
FORM = [f"f{i}" for i in range(1, 11)]
XI = ["own_att", "own_bw", "own_duel", "opp_att", "opp_bw", "opp_duel"]
LEAGUE_DUMMIES = [f"lg_{c}" for c in ALL8.values()]
PRODUCTS = ["x_att_bw", "x_att_duel"]


def code_seasons(codes) -> dict[str, list[str]]:
    return {c: [D.season_label(y) for y in (BIG4_SEASONS if c in BIG4.values() else TEST4_SEASONS)] for c in codes}


# ---------------------------------------------------------------- loading

def _build_league(store: SupabaseStore, league: int, seasons: list[int]) -> pd.DataFrame:
    m, t, s = [], [], []
    for season in seasons:
        lp = D._retry(store.list_path, league, season)
        if lp is None:
            raise RuntimeError(f"no fixture list stored for league {league} season {season}")
        listing = D._retry(store.get, lp)["response"]
        rows = D._retry(store._rows, "apif_fixtures", "fixture_id,object_path", league=f"eq.{league}", season=f"eq.{season}")
        paths = sorted({r["object_path"] for r in rows})
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            bodies = list(ex.map(lambda p: D._retry(store.get, p), paths))
        details = {f["fixture"]["id"]: f for b in bodies for f in b.get("response", [])}
        for f in listing:
            d = details.get(f["fixture"]["id"])
            mm, tt, st = D.parse_fixture(d or f, league, season)
            sot = {}
            for x in (d or {}).get("statistics") or []:
                v = {q["type"]: q.get("value") for q in x.get("statistics") or []}.get("Shots on Goal")
                sot[str(x["team"]["id"])] = float(v) if isinstance(v, (int, float)) else np.nan
            for r in tt:
                r["sot"] = sot.get(r["team_id"], np.nan)
            mm["has_details"] = d is not None
            m.append(mm), t.extend(tt), s.extend(st if d else [])
    blob = pd.concat([pd.DataFrame(m).assign(_table="matches"), pd.DataFrame(t).assign(_table="teams"),
                      pd.DataFrame(s).assign(_table="starters")], ignore_index=True)
    for c in ("home_goals", "away_goals", "possession", "sot", *D.PSTATS):
        blob[c] = pd.to_numeric(blob[c], errors="coerce")
    blob["league"] = blob["league"].astype("Int64")
    for c in ("has_stats", "has_details"):
        blob[c] = blob[c].astype("boolean")
    return blob


def load_tables(leagues: dict[int, str]) -> dict[str, pd.DataFrame]:
    fd_cache._load_env()
    store = SupabaseStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])
    parts = {"matches": [], "teams": [], "starters": []}
    for league, code in leagues.items():
        seasons = BIG4_SEASONS if league in BIG4 else TEST4_SEASONS
        path = f"derived/nlx_{league}_{seasons[0]}_{seasons[-1]}.parquet"
        try:
            blob = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
        except RuntimeError:
            blob = _build_league(store, league, seasons)
            buf = io.BytesIO()
            blob.to_parquet(buf, index=False)
            D._retry(ss.put, BUCKET, path, buf.getvalue())
        print(f"{code}: loaded", flush=True)
        for k in parts:
            parts[k].append(blob[blob["_table"] == k].dropna(axis=1, how="all").drop(columns="_table"))
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def completeness(tb, leagues: dict[int, str], seasons: list[str]) -> dict:
    m = tb["matches"][tb["matches"]["status_short"] == "FT"]
    st = tb["starters"]
    with_stats = st[st["has_stats"].fillna(False)].groupby("match_id")["team_id"].nunique()
    out = {}
    for lg, code in leagues.items():
        for s in seasons:
            g = m[(m["league"] == lg) & (m["season"] == s)]
            ok = g["match_id"].map(with_stats).fillna(0).ge(2) & g["has_details"].fillna(False)
            out[f"{code} {s}"] = {"finished": int(len(g)), "with_player_stats": int(ok.sum()),
                                  "share": round(float(ok.mean()) if len(g) else 0.0, 4)}
    return out


# ---------------------------------------------------------------- features

def _dc_league(lm: pd.DataFrame) -> dict:
    return D.dc_rates(lm)


def dc_all(m: pd.DataFrame) -> dict:
    """Weekly Dixon-Coles per league (cached per league and season range)."""
    fd_cache._load_env()
    out, todo = {}, []
    for lg, lm in m.groupby("league"):
        path = f"derived/nlx_dc_{lg}_{lm['season'].min()}_{lm['season'].max()}.parquet"
        try:
            d = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
            out.update({r.match_id: (r.lh, r.la, r.rho) for r in d.itertuples()})
        except RuntimeError:
            todo.append((lg, lm, path))
    if todo:
        with ProcessPoolExecutor(min(4, len(todo))) as ex:
            for (lg, _, path), res in zip(todo, ex.map(_dc_league, [t[1] for t in todo])):
                d = pd.DataFrame([(k, *v) for k, v in res.items()], columns=["match_id", "lh", "la", "rho"])
                buf = io.BytesIO()
                d.to_parquet(buf, index=False)
                D._retry(ss.put, BUCKET, path, buf.getvalue())
                out.update(res)
                print(f"  DC league {lg}: {len(res)} matches", flush=True)
    return out


def form_features(m: pd.DataFrame, sot: pd.Series) -> pd.DataFrame:
    """Addendum G's team form, per league, from earlier dates only (state updated after each date).
    Returns one row per (match_id, side) with f1-f4 and f6-f9 (f5 needs DC, f10 is home)."""
    lam = np.log(2) / HALF_LIFE
    rows = []
    for _, lm in m.groupby("league"):
        st, league, last_date, played = {}, [], {}, {}
        for d, g in lm.groupby("date", sort=True):
            league = [x for x in league if (d - x[0]).days <= 365]
            sv = [x[1] for x in league if not np.isnan(x[1])]
            lg_sot = np.mean(sv) if len(sv) >= 50 else np.nan
            lg_g = np.mean([x[2] for x in league]) if len(league) >= 50 else np.nan

            def form(t):
                s = st.get(t)
                if s is None or s["n_sot"] < MIN_PREV_SOT:
                    return None
                return {"sf": s["sf"] / s["ws"], "sa": s["sa"] / s["ws"], "gf": s["gf"] / s["w"], "ga": s["ga"] / s["w"]}

            def rest(t):
                return float(np.log1p(min((d - last_date[t]).days, 21))) if t in last_date else float(np.log1p(21))

            for f in g.itertuples():
                fh, fa = form(f.home_id), form(f.away_id)
                if not (fh and fa and lg_sot > 0 and lg_g > 0):
                    continue
                r = lambda x, b: float(np.log(max(x, 0.05) / b))  # noqa: E731
                n_h = min(played.get((f.home_id, f.season), 0), 10) / 10
                n_a = min(played.get((f.away_id, f.season), 0), 10) / 10
                rows.append({"match_id": f.match_id, "side": "H", "f1": r(fh["sf"], lg_sot), "f2": r(fa["sa"], lg_sot),
                             "f3": r(fh["gf"], lg_g), "f4": r(fa["ga"], lg_g), "f6": rest(f.home_id),
                             "f7": rest(f.away_id), "f8": n_h, "f9": n_a})
                rows.append({"match_id": f.match_id, "side": "A", "f1": r(fa["sf"], lg_sot), "f2": r(fh["sa"], lg_sot),
                             "f3": r(fa["gf"], lg_g), "f4": r(fh["ga"], lg_g), "f6": rest(f.away_id),
                             "f7": rest(f.home_id), "f8": n_a, "f9": n_h})
            for f in g.itertuples():
                hs, as_ = sot.get((f.match_id, f.home_id), np.nan), sot.get((f.match_id, f.away_id), np.nan)
                has = pd.notna(hs) and pd.notna(as_)
                for t, sf, sa, gf, ga in ((f.home_id, hs, as_, f.home_goals, f.away_goals),
                                          (f.away_id, as_, hs, f.away_goals, f.home_goals)):
                    s = st.setdefault(t, {"t": d, "w": 0.0, "ws": 0.0, "sf": 0.0, "sa": 0.0, "gf": 0.0, "ga": 0.0, "n_sot": 0})
                    k = np.exp(-lam * (d - s["t"]).days)
                    for key in ("w", "ws", "sf", "sa", "gf", "ga"):
                        s[key] *= k
                    s["t"] = d
                    s["w"] += 1
                    s["gf"] += float(gf)
                    s["ga"] += float(ga)
                    if has:
                        s["sf"] += float(sf)
                        s["sa"] += float(sa)
                        s["ws"] += 1
                        s["n_sot"] += 1
                    last_date[t] = d
                    played[(t, f.season)] = played.get((t, f.season), 0) + 1
                league.append((d, (float(hs) + float(as_)) / 2 if has else np.nan,
                               (float(f.home_goals) + float(f.away_goals)) / 2))
    return pd.DataFrame(rows)


def side_rows(tb: dict, leagues: dict[int, str]) -> pd.DataFrame:
    m = tb["matches"]
    m = m[(m["status_short"] == "FT") & m["home_goals"].notna()].copy()
    m["status"] = "completed"
    m["date"] = m["kickoff_utc"].dt.normalize()
    m["div"] = m["league"].map(leagues)
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    m = m.sort_values("kickoff_utc").reset_index(drop=True)
    played = pd.concat([m[["match_id", "season", "home_id"]].rename(columns={"home_id": "t"}),
                        m[["match_id", "season", "away_id"]].rename(columns={"away_id": "t"})])
    played["n"] = played.groupby(["season", "t"]).cumcount()
    m["early"] = m["match_id"].map(played.groupby("match_id")["n"].min() < 10).fillna(True)
    tbm = {**tb, "matches": m}

    dc = dc_all(m)
    print(f"DC rates: {len(dc)} matches", flush=True)
    sot = tb["teams"].set_index(["match_id", "team_id"])["sot"]
    form = form_features(m, sot)
    print(f"form: {form.match_id.nunique()} matches", flush=True)
    sh = D.starter_history(tbm)
    b = D.relative(D.blocks(sh, K_ATT, K_DEF, A_DUEL))[["match_id", "team_id", "ATT", "BW", "DUEL", "n_prev"]]

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
    bo = b.rename(columns={"team_id": "own_id", "ATT": "own_att", "BW": "own_bw", "DUEL": "own_duel", "n_prev": "own_n"})
    bp = b.rename(columns={"team_id": "opp_id", "ATT": "opp_att", "BW": "opp_bw", "DUEL": "opp_duel", "n_prev": "opp_n"})
    r = r.merge(bo, on=["match_id", "own_id"]).merge(bp, on=["match_id", "opp_id"]).merge(form, on=["match_id", "side"])
    r["f5"] = r["log_dc"] - r["log_dc_opp"]
    r["f10"] = (r["side"] == "H").astype(float)
    for c in LEAGUE_DUMMIES:
        r[c] = (r["div"] == c[3:]).astype(float)
    r["x_att_bw"] = r["own_att"] * r["opp_bw"]
    r["x_att_duel"] = r["own_att"] * r["opp_duel"]
    ok = (r["own_n"] >= MIN_PREV_XI) & (r["opp_n"] >= MIN_PREV_XI) & np.isfinite(r[FORM + XI + PRODUCTS]).all(axis=1)
    r = r[ok]
    both = r.groupby("match_id")["side"].nunique()
    r = r[r["match_id"].isin(both[both == 2].index)]
    return r.sort_values(["date", "match_id", "side"]).reset_index(drop=True)


# ---------------------------------------------------------------- models

def _std(tr, te, feats):
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1).fillna(1)
    return ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()


def fit_lin(tr, te, feats, l2):
    Xtr, Xte = _std(tr, te, feats)
    b = _fit_poisson(Xtr, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(), l2)
    return b[0] + Xte @ b[1:]


# Known limitation (outside review, 2026-11-26; kept as run, since the test is spent): the GBM split below is
# a row count, which can separate a match's two side-rows, and fit_nn standardises on the whole training
# frame before tabet.neural holds out its validation part. Both affect only early stopping. Reuse should
# split by date with both rows of a match together and fit the scaling on the fitting part only.
def fit_gbm(tr, te, feats, depth, leaf):
    """Addendum G's model: ratio target y/mu_DC with weight mu_DC; chronological stopping on the most
    recent 15% of rows (patience 50, up to 1,000 rounds); built-in early stopping off."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    cut = int(len(tr) * (1 - VAL_FRAC))
    fit, val = tr.iloc[:cut], tr.iloc[cut:]
    Xf, Xv = _std(fit, val, feats)
    _, Xe = _std(fit, te, feats)
    base = np.exp(fit["log_dc"].to_numpy())
    mdl = HistGradientBoostingRegressor(loss="poisson", max_depth=depth, min_samples_leaf=leaf, learning_rate=0.05,
                                        l2_regularization=1.0, max_iter=MAX_ROUNDS, early_stopping=False, random_state=0)
    mdl.fit(Xf, fit["goals"].to_numpy() / base, sample_weight=base)
    vb, vy = np.exp(val["log_dc"].to_numpy()), val["goals"].to_numpy()
    best, best_i, since = np.inf, 0, 0
    for i, ratio in enumerate(mdl.staged_predict(Xv), start=1):
        lamv = vb * ratio
        dev = float(np.mean(lamv - vy * np.log(lamv)))
        if dev < best - 1e-12:
            best, best_i, since = dev, i, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    for i, ratio in enumerate(mdl.staged_predict(Xe), start=1):
        if i == best_i:
            return np.log(ratio), best_i
    return np.zeros(len(te)), 0


def fit_nn(tr, te, feats, hidden, wd):
    from tabet.neural import NNParams, _fit_predict
    Xtr, Xte = _std(tr, te, feats)
    p = NNParams(hidden=hidden, weight_decay=wd, epochs=300, lr=1e-2, seeds=5, early_stop=True, val_frac=VAL_FRAC,
                 fallback=False)
    return _fit_predict(Xtr, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(), Xte, p)


FEATS = FORM + XI + LEAGUE_DUMMIES
FEATS_NOXI = FORM + LEAGUE_DUMMIES
FEATS_X = FEATS + PRODUCTS


def walk_forward(rows: pd.DataFrame, eval_divs, eval_seasons, specs: dict) -> tuple[pd.DataFrame, list]:
    """Monthly refits on all earlier rows. specs: name -> (kind, feats, setting). Wide per-match table."""
    sel = rows[rows["div"].isin(eval_divs) & rows["season"].isin(eval_seasons)]
    sel = sel[[(d, s) not in EXCLUDED for d, s in zip(sel["div"], sel["season"])]]
    month = sel["date"].dt.tz_localize(None).dt.to_period("M")
    out, rounds = [], []
    for mth in sorted(month.unique()):
        start = mth.start_time.tz_localize("UTC")
        te = sel[month == mth]
        tr = rows[rows["date"] < start]
        if len(tr) < MIN_STAGE2:
            continue
        w = D.log_loss(te, np.zeros(len(te))).rename(columns={"ll": "DC"})
        for name, (kind, feats, setting) in specs.items():
            if kind == "lin":
                adj = fit_lin(tr, te, feats, setting)
            elif kind == "gbm":
                adj, n = fit_gbm(tr, te, feats, *setting)
                rounds.append({"month": str(mth), "model": name, "rounds": n})
            else:
                adj = fit_nn(tr, te, feats, *setting)
            w[name] = D.log_loss(te, adj)["ll"].to_numpy()
        out.append(w)
        print(f"  {mth}: {len(w)} matches, trained on {tr.match_id.nunique()}", flush=True)
    if not out:
        raise RuntimeError(f"no month had {MIN_STAGE2} earlier stage-2 rows to train on")
    return pd.concat(out, ignore_index=True), rounds


def all_specs(choice=None) -> dict:
    if choice is None:
        sp = {f"LIN_{l2}": ("lin", FEATS, l2) for l2 in L2S}
        sp.update({f"LINXI_{l2}": ("lin", FEATS_NOXI, l2) for l2 in L2S})
        sp.update({f"GBM_{d}_{lf}": ("gbm", FEATS, (d, lf)) for d, lf in GBM_GRID})
        sp.update({f"NN_{h}_{wd}": ("nn", FEATS, (h, wd)) for h, wd in NN_GRID})
        return sp
    return {"LIN": ("lin", FEATS, choice["lin_l2"]), "LINXI": ("lin", FEATS_NOXI, choice["linxi_l2"]),
            "LINX": ("lin", FEATS_X, choice["lin_l2"]),
            "GBM": ("gbm", FEATS, tuple(choice["gbm"])), "NN": ("nn", FEATS, tuple(choice["nn"]))}


def block_se(d: pd.DataFrame, col: str, reps=2000, seed=0) -> float:
    d = d.assign(block=((d["date"] - d["date"].min()).dt.days // 14))
    g = d.groupby(["block", "div"])[col].agg(["sum", "count"]).unstack("div").fillna(0)
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, len(g), len(g))
        s, c = sums[pick].sum(0), cnts[pick].sum(0)
        boots.append(float(np.mean(s / np.where(c > 0, c, np.nan))))
    return float(np.nanstd(boots))


# ---------------------------------------------------------------- phases

def _claim(include_test: bool):
    for fam in ("NLX", "XIA"):
        for code in BIG4.values():
            ledger.claim(fam, code, ledger.season_range("2015-16", "2018-19"), "training")
            ledger.claim(fam, code, ledger.season_range("2019-20", "2024-25"), "development")
        if include_test:
            for code in TEST4.values():
                ledger.claim(fam, code, ledger.season_range("2015-16", "2019-20"), "training")
                ledger.claim(fam, code, ledger.season_range("2022-23", "2024-25"), "training")
                ledger.claim(fam, code, [s for s in TEST_SEASONS if (code, s) not in EXCLUDED], "test")
                ledger.claim(fam, code, [s for s in TEST_SEASONS if (code, s) in EXCLUDED], "training")


def _rows(leagues, tag):
    """Side-rows, cached in Supabase per phase (deterministic given the stored data)."""
    fd_cache._load_env()
    path = f"derived/nlx_rows_{tag}.parquet"
    try:
        rows = pd.read_parquet(io.BytesIO(D._retry(ss.get, BUCKET, path)))
    except RuntimeError:
        tb = load_tables(leagues)
        if tag == "test":
            comp = {k: v for k, v in completeness(tb, TEST4, TEST_SEASONS).items()
                    if tuple(k.split(" ")) not in EXCLUDED}
            short = {k: v for k, v in comp.items() if v["share"] < MIN_COMPLETE}
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "test_check.json").write_text(json.dumps(comp, indent=2))
            if short:
                raise SystemExit(f"run condition not met (needs >= {MIN_COMPLETE:.0%}): {short}")
        rows = side_rows(tb, leagues)
        buf = io.BytesIO()
        rows.to_parquet(buf, index=False)
        D._retry(ss.put, BUCKET, path, buf.getvalue())
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    return rows


def dev():
    _claim(include_test=False)
    rows = _rows(BIG4, "dev")
    print(f"rows: {rows.match_id.nunique()} eligible matches", flush=True)
    w, rounds = walk_forward(rows, list(BIG4.values()), DEV_SEASONS, all_specs())
    sc = {c: D.league_balanced(w, c) for c in w.columns if c.split("_")[0] in ("DC", "LIN", "LINXI", "GBM", "NN")}
    pick = lambda pre: min((c for c in sc if c.split("_")[0] == pre), key=sc.get)  # noqa: E731
    lin, linxi, gbm, nn = pick("LIN"), pick("LINXI"), pick("GBM"), pick("NN")
    nl = "GBM" if sc[gbm] <= sc[nn] else "NN"
    se = block_se(w.assign(d=w[gbm if nl == "GBM" else nn] - w[lin]), "d")
    mde = (1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES)
    dev_ids = set(w["match_id"])
    r = rows[rows["match_id"].isin(dev_ids)]
    dmax = r.assign(a=r["own_att"].abs()).groupby("match_id")["a"].max()
    choice = {"lin_l2": float(lin.split("_")[1]), "linxi_l2": float(linxi.split("_")[1]),
              "gbm": [int(gbm.split("_")[1]), int(gbm.split("_")[2])],
              "nn": [int(nn.split("_")[1]), float(nn.split("_")[2])], "nl": nl,
              "mde": {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES, "mde": float(mde)},
              "xia_bands": [float(dmax.quantile(1 / 3)), float(dmax.quantile(2 / 3))],
              "dev_log_loss": sc, "gbm_rounds": rounds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps({k: choice[k] for k in ("lin_l2", "linxi_l2", "gbm", "nn", "nl", "mde", "xia_bands")}, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())      # frozen development choices
    _claim(include_test=True)
    rows = _rows(ALL8, "test")
    w, rounds = walk_forward(rows, list(TEST4.values()), TEST_SEASONS, all_specs(choice))
    nl, other = choice["nl"], ("NN" if choice["nl"] == "GBM" else "GBM")
    for a, b, col in ((nl, "LIN", "nlx"), (other, "LIN", "other"), ("LIN", "LINXI", "xia"), ("LINX", "LIN", "linx"),
                      ("LIN", "DC", "lin_dc"), ("GBM", "DC", "gbm_dc"), ("NN", "DC", "nn_dc")):
        w[col] = w[a] - w[b]
    res = {"addenda": ["NLX", "XIA"], "frozen_dev_choice": {k: choice[k] for k in ("lin_l2", "linxi_l2", "gbm", "nn", "nl", "mde", "xia_bands")},
           "n_test_matches": int(len(w)),
           "matches_per_league_season": {f"{d} {s}": int(len(g)) for (d, s), g in w.groupby(["div", "season"])},
           "log_loss": {c: D.league_balanced(w, c) for c in ("DC", "LIN", "LINXI", "LINX", "GBM", "NN")}}
    m, lo, hi = D.block_ci(w, "nlx")
    verdict = "confirmed" if (m <= -0.001 and hi < 0) else ("closed" if lo > -0.003 else "inconclusive")
    res["NLX"] = {"primary": {f"{nl} - LIN": {"mean": m, "ci_95": [lo, hi]}}, "decision": verdict}
    m2, lo2, hi2 = D.block_ci(w, "xia")
    res["XIA"] = {"primary": {"LIN - LIN without XI": {"mean": m2, "ci_95": [lo2, hi2]}},
                  "decision": {"confirmed": bool(m2 <= -0.001 and hi2 < 0)}}
    sec = {}
    for col, name in (("other", f"{other} - LIN"), ("linx", "LIN-X - LIN"), ("lin_dc", "LIN - DC"),
                      ("gbm_dc", "GBM - DC"), ("nn_dc", "NN - DC")):
        mm, l, h = D.block_ci(w, col)
        sec[name] = {"mean": mm, "ci_95": [l, h]}
    sec["by league (NLX, XIA, n)"] = {d: [float(g.nlx.mean()), float(g.xia.mean()), int(len(g))] for d, g in w.groupby("div")}
    sec["by season (NLX, XIA)"] = {s: [D.league_balanced(g, "nlx"), D.league_balanced(g, "xia")] for s, g in w.groupby("season")}
    nc = w[w.season != "2020-21"]
    sec["excluding 2020-21 (NLX, XIA, n)"] = [D.league_balanced(nc, "nlx"), D.league_balanced(nc, "xia"), int(len(nc))]
    e = w[w.early]
    sec["early season (NLX, XIA, n)"] = [D.league_balanced(e, "nlx"), D.league_balanced(e, "xia"), int(len(e))]
    r = rows[rows["match_id"].isin(set(w["match_id"]))]
    dmax = r.assign(a=r["own_att"].abs()).groupby("match_id")["a"].max()
    lo_b, hi_b = choice["xia_bands"]
    band = w["match_id"].map(lambda i: "low" if dmax[i] <= lo_b else ("high" if dmax[i] > hi_b else "mid"))
    sec["XIA by XI-deviation band (mean, n)"] = {b: [D.league_balanced(g, "xia"), int(len(g))] for b, g in w.groupby(band)}
    sec["gbm_rounds"] = rounds
    res["secondary"] = sec
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss", "NLX", "XIA")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
