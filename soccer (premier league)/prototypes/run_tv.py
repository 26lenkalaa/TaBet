"""Addendum TV of docs/PREREGISTRATION.md (transfer value in the early season), with its amendment of
2026-11-26 (separate development and test series; point-in-time fee index). Written after both were
merged, before any model was fitted.

    python prototypes/run_tv.py dev     # PL 2016-17 -> 2025-26: L2, withdrawal rule, MDE
    python prototypes/run_tv.py test    # PL 2008-09 -> 2014-15: scored once (only if dev passes)

Implementation details left open, fixed here before any result is seen:
  * Matches: football-data E0 files via tabet.fd_cache, team ids from tabet.teams (so they match the
    transfer table's). A match is a row with both scores.
  * D0: tabet.model.DixonColes(Hyperparams()) refitted every matchday on the series' earlier matches; a
    matchday whose fit fails gives no forecast (as in TS).
  * Features at each matchday (the day's first kickoff is the cutoff): for every club in that season,
    log(1 + squad cost) and log(1 + summer fees in), z-scored across the season's clubs; the fee index
    is computed from moves before the cutoff (tabet.transfers.fee_index(before=...)). n = the club's
    league matches this season before the match; w = max(0, 1 - n/10).
  * Stage: b fitted with tabet.experiments._fit_poisson on [dcost, dfees_early] with log(D0 rate) as the
    offset; refitted on the first day of each calendar month on training rows whose kickoff + 2 h 30 is
    before that day (tabet.store.RESULT_LAG), inactive below MIN_TRAIN_MATCHES matches.
  * Probabilities: tabet.experiments._probs with D0's rho for both models.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))
import run_dxi as DX  # noqa: E402
import run_nlx as N_  # noqa: E402
from tabet import fd_cache, ledger  # noqa: E402
from tabet import transfers as T  # noqa: E402
from tabet.experiments import MIN_TRAIN_MATCHES, _fit_poisson, _probs, block_bootstrap  # noqa: E402
from tabet.ingest import LOCAL_TZ  # noqa: E402
from tabet.model import DixonColes, Hyperparams, promoted_team_seasons  # noqa: E402
from tabet.store import RESULT_LAG, MatchStore  # noqa: E402
from tabet.teams import try_resolve_team  # noqa: E402

OUT = ROOT / "results" / "tv"
L2S = (1.0, 10.0, 100.0)
EARLY_N = 10
EXPECTED_TEST_MATCHES = 2660
SERIES = {"dev": (2015, 2025, 2016, 2016), "test": (2005, 2014, 2006, 2008)}   # first, last, rows from, scored from
FEATS = ["dcost", "dfees_early"]


def _label(y: int) -> str:
    return f"{y}-{(y + 1) % 100:02d}"


def matches(first: int, last: int) -> pd.DataFrame:
    fd_cache._load_env()
    m = fd_cache.seasons(["E0"], [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)])
    m = m[m["home_goals"].notna() & m["away_goals"].notna()].copy()
    for side in ("home", "away"):
        ids = m[f"{side}_team"].map(try_resolve_team)
        assert ids.notna().all(), sorted(set(m.loc[ids.isna(), f"{side}_team"]))
        m[f"{side}_id"] = ids
    m["status"] = "completed"
    m["result"] = np.where(m["home_goals"] > m["away_goals"], "H", np.where(m["home_goals"] < m["away_goals"], "A", "D"))
    m["match_id"] = m["season"] + "-" + m["kickoff_utc"].dt.strftime("%Y%m%d") + "-" + m["home_id"] + "-" + m["away_id"]
    return m.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def d0_and_features(args) -> pd.DataFrame:
    """For one season of a series: per match D0's rates and rho, and the two features for both sides."""
    first, last, season = args
    m = matches(first, last)
    store, promoted = MatchStore(m), promoted_team_seasons(m)
    moves = T.load()
    tgt = m[m["season"] == season]
    clubs = sorted(set(tgt["home_id"]) | set(tgt["away_id"]))
    y = int(season[:4])
    played = {c: 0 for c in clubs}
    out = []
    for day, g in tgt.groupby(tgt["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date, sort=True):
        cut = g["kickoff_utc"].min()
        idx = T.fee_index(moves, before=cut)
        cost = pd.Series({c: np.log1p(T.squad_cost(moves, c, cut, idx)["squad_cost"]) for c in clubs})
        fees = pd.Series({c: np.log1p(T.club_features(moves, c, y, cut)["fees_in"]) for c in clubs})
        zc = (cost - cost.mean()) / (cost.std() or 1.0)
        zf = (fees - fees.mean()) / (fees.std() or 1.0)
        try:
            dc = DixonColes(Hyperparams()).fit(store.completed_before(cut), cut, promoted)
        except Exception:
            dc = None
        for r in g.itertuples():
            wh, wa = max(0.0, 1 - played[r.home_id] / EARLY_N), max(0.0, 1 - played[r.away_id] / EARLY_N)
            rec = {"match_id": r.match_id, "season": season, "kickoff_utc": r.kickoff_utc, "result": r.result,
                   "home_id": r.home_id, "away_id": r.away_id, "home_goals": r.home_goals, "away_goals": r.away_goals,
                   "zc_h": zc[r.home_id], "zc_a": zc[r.away_id], "zf_h": zf[r.home_id], "zf_a": zf[r.away_id],
                   "w_h": wh, "w_a": wa, "n_h": played[r.home_id], "n_a": played[r.away_id],
                   "promoted_h": (r.home_id, season) in promoted, "promoted_a": (r.away_id, season) in promoted}
            if dc is not None:
                lh, la = dc.expected_goals(r.home_id, r.away_id, season)
                rec.update(lam_h=lh, lam_a=la, rho=dc.rho)
            out.append(rec)
        for r in g.itertuples():
            played[r.home_id] += 1
            played[r.away_id] += 1
    print(f"{season}: done", flush=True)
    return pd.DataFrame(out)


def build(series: str) -> pd.DataFrame:
    first, last, _, _ = SERIES[series]
    seasons = [_label(y) for y in range(first + 1, last + 1)]          # the first season is D0's burn-in
    with ProcessPoolExecutor(min(6, len(seasons))) as ex:
        return pd.concat(ex.map(d0_and_features, [(first, last, s) for s in seasons]), ignore_index=True)


def side_rows(per: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (each side scoring), with the stage inputs from that side's view."""
    p = per.dropna(subset=["lam_h"])
    h = pd.DataFrame({"match_id": p.match_id, "side": "H", "kickoff_utc": p.kickoff_utc, "season": p.season,
                      "goals": p.home_goals, "log_dc": np.log(p.lam_h),
                      "dcost": p.zc_h - p.zc_a, "dfees_early": p.w_h * p.zf_h - p.w_a * p.zf_a})
    a = pd.DataFrame({"match_id": p.match_id, "side": "A", "kickoff_utc": p.kickoff_utc, "season": p.season,
                      "goals": p.away_goals, "log_dc": np.log(p.lam_a),
                      "dcost": p.zc_a - p.zc_h, "dfees_early": p.w_a * p.zf_a - p.w_h * p.zf_h})
    return pd.concat([h, a], ignore_index=True)


def walk(per: pd.DataFrame, rows_from: int, scored_from: int, l2: float, feats=FEATS) -> pd.DataFrame:
    """Monthly stage refits; per-match log-loss of D0 and TV on the scored seasons."""
    rows = side_rows(per)
    rows = rows[rows["season"].str[:4].astype(int) >= rows_from]
    p = per.dropna(subset=["lam_h"])
    p = p[p["season"].str[:4].astype(int) >= scored_from].copy()
    month = p["kickoff_utc"].dt.tz_convert("UTC").dt.to_period("M")
    p["adj_h"], p["adj_a"] = 0.0, 0.0
    for mth in sorted(month.unique()):
        start = mth.start_time.tz_localize("UTC")
        tr = rows[rows["kickoff_utc"] + RESULT_LAG < start]
        if tr["match_id"].nunique() < MIN_TRAIN_MATCHES:
            continue
        b = _fit_poisson(tr[feats].to_numpy(float), tr["log_dc"].to_numpy(), tr["goals"].to_numpy(float), l2)
        sel = (month == mth).to_numpy()
        q = p[sel]
        xh = np.c_[q["zc_h"] - q["zc_a"], q["w_h"] * q["zf_h"] - q["w_a"] * q["zf_a"]]
        xa = -xh
        if feats != FEATS:                       # a single-feature variant (secondary)
            k = FEATS.index(feats[0])
            xh, xa = xh[:, [k]], xa[:, [k]]
        p.loc[sel, "adj_h"] = b[0] + xh @ b[1:]
        p.loc[sel, "adj_a"] = b[0] + xa @ b[1:]
    y = p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    p0 = _probs(p["lam_h"].to_numpy(), p["lam_a"].to_numpy(), p["rho"].to_numpy())
    p1 = _probs(p["lam_h"].to_numpy() * np.exp(p["adj_h"].to_numpy()), p["lam_a"].to_numpy() * np.exp(p["adj_a"].to_numpy()),
                p["rho"].to_numpy())
    p["D0"] = -np.log(np.clip(p0[np.arange(len(p)), y], 1e-15, 1))
    p["TV"] = -np.log(np.clip(p1[np.arange(len(p)), y], 1e-15, 1))
    p["d"] = p["TV"] - p["D0"]
    p["date"] = pd.to_datetime(p["kickoff_utc"], utc=True)
    p["div"] = "E0"
    return p


def _ci(p: pd.DataFrame) -> dict:
    mean, lo, hi = block_bootstrap(p["TV"].to_numpy(), p["D0"].to_numpy(), p["date"].reset_index(drop=True), 14)
    return {"n": int(len(p)), "mean": mean, "ci_95": [lo, hi]}


def _claim(include_test: bool):
    ledger.claim("TV", "E0", ["2015-16"], "training")              # D0 burn-in of the development series
    ledger.claim("TV", "E0", ledger.season_range("2016-17", "2025-26"), "development")
    if include_test:
        ledger.claim("TV", "E0", ledger.season_range("2005-06", "2007-08"), "training")
        ledger.claim("TV", "E0", ledger.season_range("2008-09", "2014-15"), "test")


def dev():
    _claim(False)
    per = build("dev")
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "dev_rows.csv.gz", index=False)
    _, _, rows_from, scored_from = SERIES["dev"]
    runs = {l2: walk(per, rows_from, scored_from, l2) for l2 in L2S}
    ll = {f"TV_L2_{l2:g}": float(r["TV"].mean()) for l2, r in runs.items()}
    l2 = min(L2S, key=lambda x: (round(ll[f"TV_L2_{x:g}"], 10), -x))
    w = runs[l2]
    early = w[(w["n_h"] < EARLY_N) | (w["n_a"] < EARLY_N)]
    res = {"addendum": "TV", "phase": "development", "l2": l2, "n_dev": int(len(w)),
           "log_loss": {"D0": float(w["D0"].mean()), **ll},
           "TV - D0": _ci(w), "TV - D0, first 10 matches of either club": _ci(early),
           "mde": None}
    se = N_.block_se(w, "d")
    res["mde"] = {"dev_block_se": se, "n_dev": int(len(w)), "expected_test": EXPECTED_TEST_MATCHES,
                  "mde": float((1.96 + 0.84) * se * np.sqrt(len(w) / EXPECTED_TEST_MATCHES))}
    res["withdrawal_rule"] = {"dev_mean_TV_minus_D0": res["TV - D0"]["mean"], "withdrawn": not res["TV - D0"]["mean"] < 0}
    (OUT / "dev.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


def test():
    choice = json.loads((OUT / "dev.json").read_text())
    if choice["withdrawal_rule"]["withdrawn"]:
        raise SystemExit("TV was withdrawn at development; PL 2008-15 stays untouched")
    _claim(True)
    l2 = float(choice["l2"])
    per = build("test")
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "test_rows.csv.gz", index=False)
    _, _, rows_from, scored_from = SERIES["test"]
    w = walk(per, rows_from, scored_from, l2)
    prim = _ci(w)
    mean, (lo, hi) = prim["mean"], prim["ci_95"]
    verdict = "confirmed" if (mean <= -0.0005 and hi < 0) else ("closed" if lo > -0.0015 else "inconclusive")
    early = w[(w["n_h"] < EARLY_N) | (w["n_a"] < EARLY_N)]
    promo = w[(w["promoted_h"] & (w["n_h"] < EARLY_N)) | (w["promoted_a"] & (w["n_a"] < EARLY_N))]
    sec = {"first 10 matches of either club": _ci(early), "a promoted club in its first 10": _ci(promo)}
    for f in FEATS:
        v = walk(per, rows_from, scored_from, l2, feats=[f])
        sec[f"{f} only"] = _ci(v)
    sec["by season (mean, n)"] = {s: [float(g["d"].mean()), int(len(g))] for s, g in w.groupby("season")}
    sec["development result"] = choice["TV - D0"]
    res = {"addendum": "TV", "phase": "test (PL 2008-15, scored once)", "l2": l2, "n_test": int(len(w)),
           "log_loss": {"D0": float(w["D0"].mean()), "TV": float(w["TV"].mean())},
           "primary": {"TV - D0": prim, "decision": verdict}, "secondary": sec}
    w.to_csv(OUT / "test_ledger.csv.gz", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    {"dev": dev, "test": test}[sys.argv[1]]()
