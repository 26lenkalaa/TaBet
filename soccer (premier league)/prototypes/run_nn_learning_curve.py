"""Exploratory: how much data does a neural network need to match or beat the linear stage 2?
(review 6, item 7e: "learning curves on pooled leagues at 2k, 5k, 10k, 20k, 50k training matches").

Data: football-data via tabet.fd_cache (Supabase), 17 divisions that are neither the Premier League nor
the big four (those stay reserved for confirmatory tests). Shots on target exist for the English
second to fifth tiers and the Scottish Premiership from 2003-04, and for the other leagues from 2017-18.

Per division (ratings never shared across leagues):
  * Dixon-Coles (Hyperparams() defaults) refitted once a week on matches completed before that week's
    first kickoff, over the previous 3 years; each match gets log expected goals per side and rho.
  * Features per side, from that team's and its opponent's earlier matches only (time-decayed, half-life
    120 days, scaled by the division's average over the previous 365 days, logged):
      own shots on target for, opponent's shots on target against, own goals for, opponent's goals against.
    A side-row is used once both teams have at least 5 earlier matches with shots on target.
Learning curve (static fits; nothing from the test period is ever seen in training):
  * test = every usable match from 2018-07-01 on (2018-19 and 2019-20);
  * for N in 2k, 5k, 10k, 20k, all: the most recent N usable matches before 2018-07-01. The most recent
    15% of them choose the linear L2 and the network's weight decay and early stopping; both models are
    then fitted on the other 85% (the same rows), stage 2 on top of the Dixon-Coles offset;
  * log-loss on the test set; network minus linear with a 14-day block bootstrap interval.

    python prototypes/run_nn_learning_curve.py          # writes results/nn_learning_curve/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tabet import fd_cache  # noqa: E402
from tabet.experiments import _fit_poisson, _probs, block_bootstrap  # noqa: E402
from tabet.model import DixonColes, Hyperparams  # noqa: E402
from tabet.neural import NNParams, _fit_predict  # noqa: E402

OUT = ROOT / "results" / "nn_learning_curve"
DIVS = ["E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D2", "I2", "SP2", "F2", "N1", "B1", "P1", "T1", "G1"]
CODES = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2003, 2020)]
TEST_FROM = pd.Timestamp("2018-07-01", tz="UTC")
SIZES = (2000, 5000, 10000, 20000, None)
HALF_LIFE, MIN_PREV = 120.0, 5
FEATS = ["f_sot_for", "f_opp_sot_against", "f_goals_for", "f_opp_goals_against"]


def dc_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Weekly walk-forward Dixon-Coles for one division: log expected goals per side and rho."""
    m = m[m["status"] == "completed"].sort_values("kickoff_utc").reset_index(drop=True)
    week = m["kickoff_utc"].dt.tz_convert("UTC").dt.to_period("W")
    out = []
    for _, g in m.groupby(week, sort=True):
        cutoff = g["kickoff_utc"].min()
        tr = m[(m["kickoff_utc"] < cutoff.normalize()) & (m["kickoff_utc"] >= cutoff - pd.Timedelta(days=3 * 365))]
        if tr["home_id"].nunique() < 8 or len(tr) < 150:
            continue
        try:
            dc = DixonColes(Hyperparams()).fit(tr, cutoff.normalize())
        except Exception:
            continue
        for f in g.itertuples():
            lh, la = dc.expected_goals(f.home_id, f.away_id)
            out.append({"match_id": f.match_id, "log_dc_h": np.log(lh), "log_dc_a": np.log(la), "rho": dc.rho})
    return pd.DataFrame(out)


def form_rows(m: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time decayed form per team, updated only after each date's matches are featured."""
    m = m[(m["status"] == "completed")].sort_values("kickoff_utc")
    lam = np.log(2) / HALF_LIFE
    state, league, out = {}, [], []
    day = m["kickoff_utc"].dt.tz_convert("UTC").dt.normalize()
    for d, g in m.groupby(day, sort=True):
        recent = [x for x in league if (d - x[0]).days <= 365]
        league = recent
        if len(recent) >= 50:
            lg_sot = np.mean([x[1] for x in recent if not np.isnan(x[1])] or [np.nan])
            lg_g = np.mean([x[2] for x in recent])
        else:
            lg_sot = lg_g = np.nan

        def form(t):
            s = state.get(t)
            if s is None or s["n_sot"] < MIN_PREV:
                return None
            # decay cancels in each ratio; shots-on-target averages use only matches that had them
            return {"sf": s["sf"] / s["ws"], "sa": s["sa"] / s["ws"], "gf": s["gf"] / s["w"], "ga": s["ga"] / s["w"]}

        for f in g.itertuples():
            fh, fa = form(f.home_id), form(f.away_id)
            if fh and fa and lg_sot > 0 and lg_g > 0:
                def ratio(x, base):
                    return float(np.log(max(x, 0.05) / base))
                out.append({"match_id": f.match_id,
                            "h": [ratio(fh["sf"], lg_sot), ratio(fa["sa"], lg_sot), ratio(fh["gf"], lg_g), ratio(fa["ga"], lg_g)],
                            "a": [ratio(fa["sf"], lg_sot), ratio(fh["sa"], lg_sot), ratio(fa["gf"], lg_g), ratio(fh["ga"], lg_g)]})
        for f in g.itertuples():   # update after the whole date has been featured
            hs, as_ = f.home_shots_on_target, f.away_shots_on_target
            has_sot = pd.notna(hs) and pd.notna(as_)
            for t, sf, sa, gf, ga in ((f.home_id, hs, as_, f.home_goals, f.away_goals),
                                      (f.away_id, as_, hs, f.away_goals, f.home_goals)):
                s = state.setdefault(t, {"t": d, "w": 0.0, "ws": 0.0, "sf": 0.0, "sa": 0.0, "gf": 0.0, "ga": 0.0, "n_sot": 0})
                decay = np.exp(-lam * (d - s["t"]).days)
                for k in ("w", "ws", "sf", "sa", "gf", "ga"):
                    s[k] *= decay
                s["t"] = d
                s["w"] += 1
                s["gf"] += float(gf)
                s["ga"] += float(ga)
                if has_sot:
                    s["sf"] += float(sf)
                    s["sa"] += float(sa)
                    s["ws"] += 1
                    s["n_sot"] += 1
            league.append((d, np.nan if not has_sot else (float(hs) + float(as_)) / 2, (float(f.home_goals) + float(f.away_goals)) / 2))
    return pd.DataFrame(out)


def build() -> pd.DataFrame:
    rows = []
    for div in DIVS:
        t = time.time()
        m = fd_cache.seasons([div], CODES)
        if m.empty:
            continue
        dc, fm = dc_rows(m), form_rows(m)
        d = m.merge(dc, on="match_id").merge(fm, on="match_id")
        for side, o in (("home", "away"), ("away", "home")):
            k = "h" if side == "home" else "a"
            part = pd.DataFrame({"match_id": d["match_id"], "div": div, "kickoff_utc": d["kickoff_utc"],
                                 "side": k.upper(), "log_dc": d[f"log_dc_{k}"], "rho": d["rho"],
                                 "goals": d[f"{side}_goals"].astype(float), "result": d["result"]})
            part[FEATS] = np.vstack(d[k].to_numpy())
            rows.append(part)
        print(f"{div}: {len(d)} usable matches ({time.time() - t:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True)


def score(rows: pd.DataFrame, adj: np.ndarray) -> pd.DataFrame:
    t = rows.assign(lam=np.exp(rows["log_dc"].to_numpy() + adj))
    h = t[t.side == "H"].set_index("match_id")
    a = t[t.side == "A"].set_index("match_id").reindex(h.index)
    p = _probs(h["lam"].to_numpy(), a["lam"].to_numpy(), h["rho"].to_numpy())
    y = h["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return pd.DataFrame({"match_id": h.index, "kickoff_utc": h["kickoff_utc"].to_numpy(),
                         "ll": -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))})


DERIVED = "derived/nn_learning_curve_rows.parquet"   # in the football-data bucket, not on disk


def load_rows() -> pd.DataFrame:
    import io
    from tabet import supabase_storage as ss
    fd_cache._load_env()
    try:
        return pd.read_parquet(io.BytesIO(ss.get(fd_cache.BUCKET, DERIVED)))
    except RuntimeError:
        rows = build()
        buf = io.BytesIO()
        rows.to_parquet(buf, index=False)
        ss.put(fd_cache.BUCKET, DERIVED, buf.getvalue(), "application/octet-stream")
        return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rows = rows.dropna(subset=["log_dc", "rho", "goals", *FEATS]).sort_values(["kickoff_utc", "match_id", "side"])
    test = rows[rows.kickoff_utc >= TEST_FROM]
    pool = rows[rows.kickoff_utc < TEST_FROM]
    pool_matches = pool.drop_duplicates("match_id")["match_id"].to_numpy()
    print(f"usable matches: training pool {len(pool_matches)}, test {test.match_id.nunique()}", flush=True)
    base = score(test, np.zeros(len(test)))
    res = {"n_pool": int(len(pool_matches)), "n_test": int(test.match_id.nunique()),
           "dixon_coles_only": float(base.ll.mean()), "sizes": {}}
    for N in SIZES:
        ids = pool_matches[-N:] if N else pool_matches
        tr = pool[pool.match_id.isin(ids)]
        cut = tr.drop_duplicates("match_id")["kickoff_utc"].quantile(0.85)
        fit, val = tr[tr.kickoff_utc <= cut], tr[tr.kickoff_utc > cut]
        mu, sd = fit[FEATS].mean(), fit[FEATS].std().replace(0, 1)
        Z = lambda d: ((d[FEATS] - mu) / sd).to_numpy()  # noqa: E731
        # linear: L2 chosen on the validation slice, fitted on the fit slice
        best = None
        for l2 in (0.1, 1.0, 10.0, 100.0):
            b = _fit_poisson(Z(fit), fit.log_dc.to_numpy(), fit.goals.to_numpy(), l2)
            v = score(val, b[0] + Z(val) @ b[1:]).ll.mean()
            if best is None or v < best[0]:
                best = (v, l2, b)
        lin = score(test, best[2][0] + Z(test) @ best[2][1:])
        # network: weight decay chosen on the same slice; early stopping inside on its most recent 15%
        nn_best = None
        for wd in (0.01, 0.1, 1.0):
            p = NNParams(hidden=16, weight_decay=wd, epochs=800, seeds=3, early_stop=True)
            v = score(val, _fit_predict(Z(fit), fit.log_dc.to_numpy(), fit.goals.to_numpy(), Z(val), p)).ll.mean()
            if nn_best is None or v < nn_best[0]:
                nn_best = (v, wd)
        p = NNParams(hidden=16, weight_decay=nn_best[1], epochs=800, seeds=5, early_stop=True)
        nn = score(test, _fit_predict(Z(fit), fit.log_dc.to_numpy(), fit.goals.to_numpy(), Z(test), p))
        m, lo, hi = block_bootstrap(nn.ll.to_numpy(), lin.ll.to_numpy(), pd.to_datetime(lin.kickoff_utc, utc=True), 14)
        key = str(N or len(pool_matches))
        res["sizes"][key] = {"n_train_matches": int(len(ids)), "linear": float(lin.ll.mean()), "linear_l2": best[1],
                             "network": float(nn.ll.mean()), "network_wd": nn_best[1], "network_minus_linear": [m, lo, hi],
                             "linear_minus_dc": float(lin.ll.mean() - base.ll.mean())}
        print(f"N={key:>6}: DC {base.ll.mean():.4f}  linear {lin.ll.mean():.4f}  network {nn.ll.mean():.4f}  "
              f"network-linear {m:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
