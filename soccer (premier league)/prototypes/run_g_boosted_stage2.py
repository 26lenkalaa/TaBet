"""Addendum G of docs/PREREGISTRATION.md (committed before this file existed): a shallow boosted
stage 2 vs a pooled linear stage 2 vs Dixon-Coles, on identical inputs, big four 2005-21.

    python prototypes/run_g_boosted_stage2.py dev    # 2014-17: choose LIN's L2 and GBM's settings
    python prototypes/run_g_boosted_stage2.py test   # 2018-21: scored once with the frozen dev choices

Every period is claimed from the data ledger (tabet.ledger) before use. Raw files come from
tabet.fd_cache (Supabase); the derived side-rows are cached there too, so nothing is kept on disk.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import fd_cache, ledger  # noqa: E402
from tabet import supabase_storage as ss  # noqa: E402
from tabet.experiments import _fit_poisson, _probs  # noqa: E402
from tabet.model import DixonColes, Hyperparams  # noqa: E402

OUT = ROOT / "results" / "g_boosted"
DERIVED = "derived/g_boosted_rows.parquet"
FIRST = {"SP1": 2005, "I1": 2005, "D1": 2006, "F1": 2005}
LAST = 2020                                    # 2020-21
TRAIN_UNTIL, DEV, EXTRA, TEST = 2013, (2014, 2016), 2017, (2018, 2020)
HALF_LIFE, MIN_PREV = 120.0, 5
FEATS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]
L2_GRID = (0.1, 1.0, 10.0, 100.0)
GBM_GRID = [(d, leaf) for d in (2, 3) for leaf in (50, 200)]
MAX_ROUNDS, PATIENCE, VAL_FRAC = 1000, 50, 0.15
CLOSED_DOORS = pd.Timestamp("2020-03-01", tz="UTC")


def seasons(y0: int, y1: int) -> list[str]:
    return ledger.season_range(f"{y0}-{(y0 + 1) % 100:02d}", f"{y1}-{(y1 + 1) % 100:02d}")


def code(season: str) -> str:
    return season[2:4] + season[5:7]


# ---------------------------------------------------------------- data (point-in-time, per league)

def league_rows(div: str) -> pd.DataFrame:
    m = fd_cache.seasons([div], [code(s) for s in seasons(FIRST[div], LAST)])
    m = m[m["status"] == "completed"].sort_values("kickoff_utc").reset_index(drop=True)
    m["date"] = m["kickoff_utc"].dt.tz_convert("UTC").dt.normalize()
    # Dixon-Coles, refitted weekly on matches from earlier dates (previous 3 years)
    dc = {}
    for _, g in m.groupby(m["date"].dt.to_period("W"), sort=True):
        cut = g["date"].min()
        tr = m[(m["date"] < cut) & (m["date"] >= cut - pd.Timedelta(days=3 * 365))]
        if tr["home_id"].nunique() < 8 or len(tr) < 150:
            continue
        try:
            fit = DixonColes(Hyperparams()).fit(tr, cut)
        except Exception:
            continue
        for f in g.itertuples():
            lh, la = fit.expected_goals(f.home_id, f.away_id)
            dc[f.match_id] = (np.log(lh), np.log(la), fit.rho)
    # form, rest and games played, from earlier dates only; state updated after each date
    lam = np.log(2) / HALF_LIFE
    st, league, last_date, played, out = {}, [], {}, {}, []
    for d, g in m.groupby("date", sort=True):
        league = [x for x in league if (d - x[0]).days <= 365]
        sot_vals = [x[1] for x in league if not np.isnan(x[1])]
        lg_sot = np.mean(sot_vals) if len(sot_vals) >= 50 else np.nan
        lg_g = np.mean([x[2] for x in league]) if len(league) >= 50 else np.nan

        def form(t):
            s = st.get(t)
            if s is None or s["n_sot"] < MIN_PREV:
                return None
            return {"sf": s["sf"] / s["ws"], "sa": s["sa"] / s["ws"], "gf": s["gf"] / s["w"], "ga": s["ga"] / s["w"]}

        def rest(t):
            return float(np.log1p(min((d - last_date[t]).days, 21))) if t in last_date else float(np.log1p(21))

        for f in g.itertuples():
            if f.match_id not in dc:
                continue
            fh, fa = form(f.home_id), form(f.away_id)
            if not (fh and fa and lg_sot > 0 and lg_g > 0):
                continue
            lh, la, rho = dc[f.match_id]
            r = lambda x, b: float(np.log(max(x, 0.05) / b))  # noqa: E731
            n_h = min(played.get((f.home_id, f.season), 0), 10) / 10
            n_a = min(played.get((f.away_id, f.season), 0), 10) / 10
            common = {"match_id": f.match_id, "div": div, "season": f.season, "date": d, "rho": rho,
                      "result": f.result}
            out.append({**common, "side": "H", "log_dc": lh, "goals": float(f.home_goals),
                        "f1": r(fh["sf"], lg_sot), "f2": r(fa["sa"], lg_sot), "f3": r(fh["gf"], lg_g), "f4": r(fa["ga"], lg_g),
                        "f5": lh - la, "f6": rest(f.home_id), "f7": rest(f.away_id), "f8": n_h, "f9": n_a, "f10": 1.0})
            out.append({**common, "side": "A", "log_dc": la, "goals": float(f.away_goals),
                        "f1": r(fa["sf"], lg_sot), "f2": r(fh["sa"], lg_sot), "f3": r(fa["gf"], lg_g), "f4": r(fh["ga"], lg_g),
                        "f5": la - lh, "f6": rest(f.away_id), "f7": rest(f.home_id), "f8": n_a, "f9": n_h, "f10": 0.0})
        for f in g.itertuples():
            hs, as_ = f.home_shots_on_target, f.away_shots_on_target
            has = pd.notna(hs) and pd.notna(as_)
            for t, sf, sa, gf, ga in ((f.home_id, hs, as_, f.home_goals, f.away_goals), (f.away_id, as_, hs, f.away_goals, f.home_goals)):
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
            league.append((d, (float(hs) + float(as_)) / 2 if has else np.nan, (float(f.home_goals) + float(f.away_goals)) / 2))
    return pd.DataFrame(out)


def load_rows() -> pd.DataFrame:
    fd_cache._load_env()
    try:
        return pd.read_parquet(io.BytesIO(ss.get(fd_cache.BUCKET, DERIVED)))
    except RuntimeError:
        parts = []
        for div in FIRST:
            t = time.time()
            parts.append(league_rows(div))
            print(f"{div}: {parts[-1].match_id.nunique()} usable matches ({time.time() - t:.0f}s)", flush=True)
        rows = pd.concat(parts, ignore_index=True)
        buf = io.BytesIO()
        rows.to_parquet(buf, index=False)
        ss.put(fd_cache.BUCKET, DERIVED, buf.getvalue(), "application/octet-stream")
        return rows


# ---------------------------------------------------------------- models

def fit_lin(tr: pd.DataFrame, l2: float):
    mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
    b = _fit_poisson(((tr[FEATS] - mu) / sd).to_numpy(), tr["log_dc"].to_numpy(), tr["goals"].to_numpy(), l2)
    return lambda te: b[0] + ((te[FEATS] - mu) / sd).to_numpy() @ b[1:]


def fit_gbm(tr: pd.DataFrame, depth: int, leaf: int):
    """Poisson boosting with the DC offset via ratio target and weight; rounds by chronological early
    stopping on the most recent 15% of the training rows (patience 50)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = tr.sort_values("date")
    cut = int(len(tr) * (1 - VAL_FRAC))
    fit, val = tr.iloc[:cut], tr.iloc[cut:]
    mu, sd = fit[FEATS].mean(), fit[FEATS].std().replace(0, 1)
    Z = lambda d: ((d[FEATS] - mu) / sd).to_numpy()  # noqa: E731
    base = np.exp(fit["log_dc"].to_numpy())
    m = HistGradientBoostingRegressor(loss="poisson", max_depth=depth, min_samples_leaf=leaf, learning_rate=0.05,
                                      l2_regularization=1.0, max_iter=MAX_ROUNDS, early_stopping=False, random_state=0)
    m.fit(Z(fit), fit["goals"].to_numpy() / base, sample_weight=base)
    vb, vy = np.exp(val["log_dc"].to_numpy()), val["goals"].to_numpy()
    best, best_i, since = np.inf, 0, 0
    for i, ratio in enumerate(m.staged_predict(Z(val)), start=1):
        lamv = vb * ratio
        dev = float(np.mean(lamv - vy * np.log(lamv)))
        if dev < best - 1e-12:
            best, best_i, since = dev, i, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    def predict(te):
        stages = m.staged_predict(Z(te))
        for i, ratio in enumerate(stages, start=1):
            if i == best_i:
                return np.log(ratio)
        return np.zeros(len(te))
    predict.rounds = best_i
    return predict


def walk_forward(rows: pd.DataFrame, period: tuple[int, int], lin_l2s, gbm_settings) -> pd.DataFrame:
    """Monthly refits over the period's seasons; returns per-match log-loss for DC and each variant."""
    sel = rows[rows["season"].isin(seasons(*period))]
    months = sorted(sel["date"].dt.to_period("M").unique())
    out = []
    for mth in months:
        start = mth.start_time.tz_localize("UTC")
        te = sel[sel["date"].dt.to_period("M") == mth]
        tr = rows[rows["date"] < start]
        adj = {"DC": np.zeros(len(te))}
        for l2 in lin_l2s:
            adj[f"LIN_{l2}"] = fit_lin(tr, l2)(te)
        for depth, leaf in gbm_settings:
            p = fit_gbm(tr, depth, leaf)
            adj[f"GBM_{depth}_{leaf}"] = p(te)
            out.append({"kind": "rounds", "month": str(mth), "setting": f"GBM_{depth}_{leaf}", "rounds": p.rounds})
        for name, a in adj.items():
            t = te.assign(lam=np.exp(te["log_dc"].to_numpy() + a))
            h = t[t.side == "H"].set_index("match_id")
            aw = t[t.side == "A"].set_index("match_id").reindex(h.index)
            pr = _probs(h["lam"].to_numpy(), aw["lam"].to_numpy(), h["rho"].to_numpy())
            y = h["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
            ll = -np.log(np.clip(pr[np.arange(len(y)), y], 1e-15, 1))
            out.extend({"kind": "ll", "model": name, "match_id": mid, "div": dv, "season": s, "date": dt, "ll": v}
                       for mid, dv, s, dt, v in zip(h.index, h["div"], h["season"], h["date"], ll))
        print(f"  {mth}: {te.match_id.nunique()} matches, trained on {tr.match_id.nunique()}", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- inference

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


def wide(ll: pd.DataFrame) -> pd.DataFrame:
    w = ll[ll.kind == "ll"].pivot_table(index=["match_id", "div", "season", "date"], columns="model", values="ll").reset_index()
    return w


# ---------------------------------------------------------------- phases

def claim_training_and_dev():
    for div, y0 in FIRST.items():
        ledger.claim("G", div, seasons(y0, TRAIN_UNTIL), "training")
        ledger.claim("G", div, seasons(*DEV), "development")
        ledger.claim("G", div, seasons(EXTRA, EXTRA), "training")


def dev():
    claim_training_and_dev()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    rows = rows[rows["season"] <= f"{DEV[1]}-{(DEV[1] + 1) % 100:02d}"]   # nothing after development is even loaded into fits
    print(f"rows: {rows.match_id.nunique()} usable matches up to 2016-17", flush=True)
    ll = walk_forward(rows, DEV, L2_GRID, GBM_GRID)
    w = wide(ll)
    scores = {c: league_balanced(w, c) for c in w.columns if c.startswith(("DC", "LIN", "GBM"))}
    lin = min((c for c in scores if c.startswith("LIN")), key=scores.get)
    gbm = min((c for c in scores if c.startswith("GBM")), key=scores.get)
    choice = {"lin_l2": float(lin.split("_")[1]), "gbm_depth": int(gbm.split("_")[1]), "gbm_min_leaf": int(gbm.split("_")[2]),
              "dev_log_loss": scores, "n_dev_matches": int(len(w))}
    (OUT / "dev.json").write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))


def test():
    choice = json.loads((OUT / "dev.json").read_text())   # the frozen development choices
    claim_training_and_dev()
    for div in FIRST:
        ledger.claim("G", div, seasons(*TEST), "test")
    rows = load_rows()
    rows["date"] = pd.to_datetime(rows["date"], utc=True)
    l2, depth, leaf = choice["lin_l2"], choice["gbm_depth"], choice["gbm_min_leaf"]
    ll = walk_forward(rows, TEST, [l2], [(depth, leaf)])
    w = wide(ll).rename(columns={f"LIN_{l2}": "LIN", f"GBM_{depth}_{leaf}": "GBM"})
    w["c1"], w["c2"], w["lin_dc"] = w.GBM - w.LIN, w.GBM - w.DC, w.LIN - w.DC
    res = {"addendum": "G", "frozen_dev_choice": choice, "n_test_matches": int(len(w)),
           "log_loss": {m: league_balanced(w, m) for m in ("DC", "LIN", "GBM")}, "primary": {}, "secondary": {}}
    for k, name in (("c1", "C1 GBM - LIN"), ("c2", "C2 GBM - DC")):
        m, lo, hi = block_ci(w, k)
        res["primary"][name] = {"mean": m, "ci_95": [lo, hi]}
    c1, c2 = res["primary"]["C1 GBM - LIN"], res["primary"]["C2 GBM - DC"]
    res["decision"] = {"confirmed": bool(c1["mean"] <= -0.001 and c1["ci_95"][1] < 0 and c2["ci_95"][1] < 0)}
    m, lo, hi = block_ci(w, "lin_dc")
    res["secondary"]["LIN - DC"] = {"mean": m, "ci_95": [lo, hi]}
    res["secondary"]["by league (C1, C2)"] = {d: [float(g.c1.mean()), float(g.c2.mean()), int(len(g))] for d, g in w.groupby("div")}
    res["secondary"]["by season (C1, C2)"] = {s: [league_balanced(g, "c1"), league_balanced(g, "c2")] for s, g in w.groupby("season")}
    open_doors = w[w.date < CLOSED_DOORS]
    res["secondary"]["excluding behind closed doors (C1, C2)"] = [league_balanced(open_doors, "c1"), league_balanced(open_doors, "c2"), int(len(open_doors))]
    rows_t = rows[rows.match_id.isin(w.match_id)]
    early = set(rows_t.loc[(rows_t.f8 < 1.0) | (rows_t.f9 < 1.0), "match_id"])
    e = w[w.match_id.isin(early)]
    res["secondary"]["early season (C1, C2)"] = [league_balanced(e, "c1"), league_balanced(e, "c2"), int(len(e))]
    res["secondary"]["gbm_rounds"] = ll[ll.kind == "rounds"][["month", "rounds"]].to_dict("records")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("n_test_matches", "log_loss", "primary", "decision")}, indent=2))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
