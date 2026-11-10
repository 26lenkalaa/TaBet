"""Market benchmark for the prospective log (docs/PROSPECTIVE_PROTOCOL.md, amendment A6). BENCHMARK ONLY.

Compares the scored TaBet records with bookmaker odds on identical fixtures. Its output is printed by
`python -m tabet score-forecasts` and saved to results/prospective/odds/benchmark.csv. It never
feeds alpha, model choice or any forecast: nothing in tabet/ outside this module, the archiver and
the CLI may import it (tests/test_odds_archive.py).

Snapshots per fixture (amendment A6, item 2):
  matched          the last archived snapshot retrieved at or before the scored record's cutoff
                   (logged_utc); if none, the first one after it and before kickoff, flagged `later`
  last_pre_kickoff the last archived snapshot retrieved before kickoff
  fd_pre_closing   football-data's Avg* columns in the results file (published after the match;
                   collected Friday or Tuesday afternoon), as in prototypes/run_market_comparison.py
  fd_closing       football-data's AvgC* columns in the results file
Archived snapshots use the market average ("Avg") group. Margin is removed proportionally:
p_k = (1/o_k) / sum_j (1/o_j).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import odds_archive
from .forecast import LIVE_SEASON
from .ingest import implied_probabilities, load_all
from .prospective import ROOT, fit_alpha, select_records

SNAPSHOTS = ("matched", "last_pre_kickoff", "fd_pre_closing", "fd_closing")
COLS = ["p_home", "p_draw", "p_away"]
OUT = {"H": 0, "D": 1, "A": 2}


def _implied(o: np.ndarray) -> np.ndarray:
    inv = 1.0 / o
    return inv / inv.sum(axis=1, keepdims=True)


def select_snapshots(records: pd.DataFrame, odds_log: pd.DataFrame, book: str = "Avg") -> pd.DataFrame:
    """One row per (fixture, archived snapshot type). `records`: one scored record per fixture with
    fpl_fixture, logged_utc (the cutoff) and kickoff_utc. `odds_log`: odds_archive.load()."""
    o = odds_log[odds_log["book"] == book].copy()
    o["fpl_fixture"] = pd.to_numeric(o["fpl_fixture"], errors="coerce")
    o = o.dropna(subset=["fpl_fixture"]).sort_values("retrieved_utc")
    out = []
    for r in records.itertuples():
        cut, ko = pd.Timestamp(r.logged_utc), pd.Timestamp(r.kickoff_utc)
        s = o[(o["fpl_fixture"] == r.fpl_fixture) & (o["retrieved_utc"] < ko)]
        if s.empty:
            continue
        before = s[s["retrieved_utc"] <= cut]
        pick = {"matched": (before.iloc[-1], False) if len(before) else (s.iloc[0], True),
                "last_pre_kickoff": (s.iloc[-1], False)}
        for name, (row, later) in pick.items():
            p = _implied(row[["odds_h", "odds_d", "odds_a"]].to_numpy(float)[None, :])[0]
            out.append({"fpl_fixture": int(r.fpl_fixture), "snapshot": name, "retrieved_utc": row["retrieved_utc"],
                        "later": later, "hours_after_cutoff": (row["retrieved_utc"] - cut).total_seconds() / 3600,
                        "p_home": p[0], "p_draw": p[1], "p_away": p[2]})
    return pd.DataFrame(out, columns=["fpl_fixture", "snapshot", "retrieved_utc", "later", "hours_after_cutoff", *COLS])


def results_file_odds(records: pd.DataFrame, res: Path = ROOT / "res") -> pd.DataFrame:
    """football-data's pre-closing and closing averages for the records' fixtures, matched by season,
    home and away team (each pair meets once a season)."""
    m, odds = load_all(res)
    m = m[m["season"] == LIVE_SEASON][["match_id", "home_id", "away_id"]]
    key = records[["fpl_fixture", "home_id", "away_id"]].merge(m, on=["home_id", "away_id"])
    out = []
    for prefix, name in (("Avg", "fd_pre_closing"), ("AvgC", "fd_closing")):
        if not {f"{prefix}{k}" for k in "HDA"} <= set(odds.columns):
            continue
        p = implied_probabilities(odds, prefix).dropna()
        out.append(key.merge(p, on="match_id").assign(snapshot=name, later=False, hours_after_cutoff=np.nan,
                                                       retrieved_utc=pd.NaT))
    if not out:
        return pd.DataFrame(columns=["fpl_fixture", "snapshot", *COLS])
    return pd.concat(out, ignore_index=True)[["fpl_fixture", "snapshot", "retrieved_utc", "later",
                                              "hours_after_cutoff", *COLS]]


def _clustered(d: np.ndarray, gw: np.ndarray) -> tuple[float | None, int]:
    """Gameweek-clustered SE of the mean of d (same estimator as prospective.robustness)."""
    g = pd.Series(d).groupby(gw)
    G = g.ngroups
    if G < 2:
        return None, G
    resid = g.sum() - g.size() * d.mean()
    return float(np.sqrt(G / (G - 1) * (resid ** 2).sum()) / len(d)), G


def tabet_probs(scores: pd.DataFrame) -> pd.DataFrame:
    """p0, pE and p_alpha per scored fixture (the record select_records chooses). alpha is read from
    fit_alpha as it stands; it is provisional until the fitting period (gameweeks 10-19) is complete."""
    s = select_records(scores)
    s = s[s["candidate"].isin(["p0", "pE"])]
    w = s.pivot_table(index=["fpl_fixture", "gameweek", "result", "logged_utc", "kickoff_utc"], columns="candidate",
                      values=COLS, aggfunc="first")
    a = fit_alpha(scores, through_gameweek=19)
    rows = []
    for idx, r in w.iterrows():
        base = dict(zip(["fpl_fixture", "gameweek", "result", "logged_utc", "kickoff_utc"], idx))
        for c in ("p0", "pE"):
            if all(pd.notna(r[(k, c)]) for k in COLS):
                rows.append({**base, "candidate": c, **{k: float(r[(k, c)]) for k in COLS}})
        if all(pd.notna(r[(k, c)]) for k in COLS for c in ("p0", "pE")):
            rows.append({**base, "candidate": "p_alpha",
                         **{k: (1 - a) * float(r[(k, "p0")]) + a * float(r[(k, "pE")]) for k in COLS}})
    return pd.DataFrame(rows).assign(alpha=a)


def benchmark(scores: pd.DataFrame, odds_log: pd.DataFrame | None = None, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (snapshot, candidate): n, coverage, log-loss of each side, paired delta (TaBet - market) with
    paired and gameweek-clustered SEs, agreement and who was right when the picks differ."""
    t = tabet_probs(scores)
    if t.empty:
        return pd.DataFrame()
    recs = t.drop_duplicates("fpl_fixture")
    hid = select_records(scores).drop_duplicates("fpl_fixture").set_index("fpl_fixture")
    if {"home_id", "away_id"} <= set(hid.columns):
        recs = recs.assign(home_id=recs["fpl_fixture"].map(hid["home_id"]), away_id=recs["fpl_fixture"].map(hid["away_id"]))
    odds_log = odds_archive.load() if odds_log is None else odds_log
    snaps = [select_snapshots(recs, odds_log)]
    if market is not None:
        snaps.append(market)
    elif {"home_id", "away_id"} <= set(recs.columns):
        snaps.append(results_file_odds(recs))
    snaps = pd.concat([s for s in snaps if len(s)], ignore_index=True) if any(len(s) for s in snaps) else pd.DataFrame()
    n_fix = recs["fpl_fixture"].nunique()
    out = []
    for snap in SNAPSHOTS:
        m = snaps[snaps["snapshot"] == snap].set_index("fpl_fixture") if len(snaps) else pd.DataFrame()
        for c in ("p0", "pE", "p_alpha"):
            tc = t[t["candidate"] == c].set_index("fpl_fixture")
            ids = tc.index.intersection(m.index) if len(m) else []
            row = {"snapshot": snap, "candidate": c, "n": len(ids), "coverage": len(ids) / n_fix if n_fix else np.nan}
            if len(ids):
                y = tc.loc[ids, "result"].map(OUT).to_numpy()
                pt, pm = tc.loc[ids, COLS].to_numpy(float), m.loc[ids, COLS].to_numpy(float)
                lt, lm = -np.log(pt[np.arange(len(y)), y]), -np.log(pm[np.arange(len(y)), y])
                d = lt - lm
                se_c, G = _clustered(d, tc.loc[ids, "gameweek"].to_numpy())
                kt, km = pt.argmax(1), pm.argmax(1)
                dis = kt != km
                hrs = m.loc[ids, "hours_after_cutoff"].astype(float)
                row.update({"log_loss_tabet": lt.mean(), "log_loss_market": lm.mean(), "tabet_minus_market": d.mean(),
                            "se_paired": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                            "se_gameweek_clustered": se_c, "n_gameweeks": G,
                            "agree_rate": float((~dis).mean()), "disagree": int(dis.sum()),
                            "tabet_only_right": int((dis & (kt == y)).sum()), "market_only_right": int((dis & (km == y)).sum()),
                            "both_wrong": int((dis & (kt != y) & (km != y)).sum()),
                            "n_later": int(m.loc[ids, "later"].astype(bool).sum()),
                            "median_hours_after_cutoff": float(hrs.median()) if hrs.notna().any() else np.nan,
                            "max_abs_hours_from_cutoff": float(hrs.abs().max()) if hrs.notna().any() else np.nan})
            out.append(row)
    return pd.DataFrame(out).assign(alpha=float(t["alpha"].iloc[0]))


def save(b: pd.DataFrame, out_dir: Path = odds_archive.ODDS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "benchmark.csv"
    b.to_csv(p, index=False)
    return p
