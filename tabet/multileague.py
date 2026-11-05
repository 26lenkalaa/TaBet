"""Pooling other leagues to learn shared adjustment coefficients.

Every league gets its own walk-forward Dixon-Coles fit (team ratings never
cross leagues). The shared feature is opponent-adjusted shots-on-target
strength (available in every league and season), fitted within each league
with the same quasi-Poisson model as the xG strength. Stage 2 is then trained
either on Premier League rows only, or on all leagues' rows with a separate
intercept per league, and is always *scored on Premier League fixtures only*.
All training rows respect the target fixture's cutoff, whatever their league.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .experiments import MIN_TRAIN_MATCHES, _fit_poisson, _probs
from .ingest import LOCAL_TZ
from .model import DixonColes, Hyperparams
from .store import RESULT_LAG, MatchStore
from .xg_strength import fit_xg_strength

SOT = ["f_sot_att", "f_sot_def_opp"]


def _sot_sides(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for us, them, v in (("home", "away", "H"), ("away", "home", "A")):
        rows.append(pd.DataFrame({"match_id": m["match_id"], "kickoff_utc": m["kickoff_utc"],
                                  "team_id": m[f"{us}_id"], "opponent_id": m[f"{them}_id"], "venue": v,
                                  "xg": m[f"{us}_shots_on_target"].astype(float)}))
    return pd.concat(rows).dropna(subset=["xg"])


def league_rows(m: pd.DataFrame, league: str, hyper: Hyperparams = Hyperparams(), first_season: str = "2022-23",
                sot_half_life: float = 90.0) -> pd.DataFrame:
    """Walk-forward rows (two per match) for one league: pre-match DC log-rate,
    goals and opponent-adjusted SoT features, all as of each matchday's cutoff."""
    store = MatchStore(m)
    sides_all = _sot_sides(m[m["status"] == "completed"])
    target = m[(m["season"] >= first_season) & (m["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out = []
    for _, fx in target.groupby(day, sort=True):
        cutoff = fx["kickoff_utc"].min()
        train = store.completed_before(cutoff)
        dc = DixonColes(hyper).fit(train, cutoff)
        sides = sides_all[sides_all["kickoff_utc"] + RESULT_LAG <= cutoff]
        r, _, _ = fit_xg_strength(sides, cutoff, sot_half_life)
        for f in fx.itertuples():
            lh, la = dc.expected_goals(f.home_id, f.away_id, f.season)
            for side, t, o, lam, g in (("H", f.home_id, f.away_id, lh, f.home_goals),
                                       ("A", f.away_id, f.home_id, la, f.away_goals)):
                out.append({"league": league, "match_id": f.match_id, "season": f.season, "side": side,
                            "kickoff_utc": f.kickoff_utc, "cutoff_utc": cutoff, "result": f.result,
                            "log_dc": float(np.log(lam)), "goals": float(g), "rho": dc.rho,
                            "f_sot_att": float(r["att"].get(t, 0.0)), "f_sot_def_opp": float(-r["def"].get(o, 0.0))})
    return pd.DataFrame(out)


def run(rows: pd.DataFrame, cols: list[str], l2: float, seasons: tuple[str, ...], train_leagues: set[str],
        eval_league: str = "E0") -> pd.DataFrame:
    """Walk forward over eval-league matchdays; stage 2 trained on rows from
    `train_leagues` whose results were known by each cutoff (league intercepts
    when more than one league is used). Same 300-match activation rule."""
    rows = rows.sort_values(["cutoff_utc", "match_id", "side"]).reset_index(drop=True)
    ev = rows[(rows["league"] == eval_league) & rows["season"].isin(seasons)]
    pool = rows[rows["league"].isin(train_leagues)]
    leagues = sorted(train_leagues)
    preds = []
    for cutoff, te in ev.groupby("cutoff_utc"):
        tr = pool[pool["kickoff_utc"] + RESULT_LAG <= cutoff]
        adj = np.zeros(len(te))
        active = bool(cols) and tr["match_id"].nunique() >= MIN_TRAIN_MATCHES
        if active:
            X = tr[cols].astype(float)
            mu, sd = X.mean(), X.std().replace(0, 1)
            Z = ((X - mu) / sd).to_numpy()
            Zte = ((te[cols].astype(float) - mu) / sd).to_numpy()
            if len(leagues) > 1:  # league intercepts as offsets learned jointly (dummies, eval league = reference)
                D = np.column_stack([(tr["league"] == lg).to_numpy(float) for lg in leagues if lg != eval_league])
                Z = np.hstack([Z, D])
                Zte = np.hstack([Zte, np.zeros((len(te), D.shape[1]))])
            b = _fit_poisson(Z, tr["log_dc"].to_numpy(), tr["goals"].to_numpy(), l2)
            adj = b[0] + Zte @ b[1:]
        t = te.assign(lam=np.exp(te["log_dc"].to_numpy() + adj))
        h = t[t["side"] == "H"].set_index("match_id")
        a = t[t["side"] == "A"].set_index("match_id").reindex(h.index)
        P = _probs(h["lam"].to_numpy(), a["lam"].to_numpy(), h["rho"].to_numpy())
        preds.append(pd.DataFrame({"match_id": h.index, "season": h["season"].to_numpy(), "result": h["result"].to_numpy(),
                                   "kickoff_utc": h["kickoff_utc"].to_numpy(), "stage2_active": active,
                                   "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2]}))
    return pd.concat(preds, ignore_index=True).sort_values("match_id").reset_index(drop=True)
