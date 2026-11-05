"""Results-only Elo ratings with an ordered-logit outcome mapping (for the F14 blend test).

Ratings: every team starts at 1500. After a match, both ratings move by K * G * (S - E), where S is
1 / 0.5 / 0 for a home win / draw / loss, E = 1 / (1 + 10 ** (-(R_home + HOME - R_away) / 400)), and
G is the World Football Elo goal-difference multiplier (1 for a margin of 0-1, 1.5 for 2,
(11 + margin) / 8 for 3 or more). A team that was not in the league the previous season starts
the season at the mean final rating of the teams that left (known before the season starts).

Outcome mapping: an ordered logit on the pre-match rating gap d = (R_home + HOME - R_away) / 400,
    P(away) = sigmoid(c1 - b d),  P(away or draw) = sigmoid(c2 - b d),
fitted at each matchday cutoff on matches already played. Matches are processed by local
matchday, so a same-day result never informs a same-day prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .ingest import LOCAL_TZ

HOME = 65.0


def _g(margin: int) -> float:
    m = abs(margin)
    return 1.0 if m <= 1 else (1.5 if m == 2 else (11 + m) / 8)


def pre_match_gaps(matches: pd.DataFrame, k: float = 20.0, home: float = HOME) -> pd.DataFrame:
    """Pre-match rating gap for every completed match, in chronological matchday order."""
    m = matches[matches["status"] == "completed"].sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    day = m["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    r: dict[str, float] = {}
    prev_teams: set[str] = set()
    season = None
    gaps = np.empty(len(m))
    for d, idx in m.groupby(day, sort=True).groups.items():
        g = m.loc[idx]
        s = g["season"].iloc[0]
        if s != season:  # season boundary: newcomers take the leavers' mean rating
            teams = set(m.loc[m["season"] == s, "home_id"]) | set(m.loc[m["season"] == s, "away_id"])
            leavers = prev_teams - teams
            start = float(np.mean([r[t] for t in leavers])) if leavers else 1500.0
            for t in teams - prev_teams:
                r[t] = start if prev_teams else 1500.0
            prev_teams, season = teams, s
        upd = []
        for i, row in g.iterrows():
            dr = r[row["home_id"]] + home - r[row["away_id"]]
            gaps[i] = dr / 400.0
            e = 1.0 / (1.0 + 10 ** (-dr / 400.0))
            sc = 1.0 if row["home_goals"] > row["away_goals"] else (0.5 if row["home_goals"] == row["away_goals"] else 0.0)
            upd.append((row["home_id"], row["away_id"], k * _g(row["home_goals"] - row["away_goals"]) * (sc - e)))
        for h, a, delta in upd:  # apply after the whole day is predicted
            r[h] += delta
            r[a] -= delta
    m["elo_gap"] = gaps
    m["matchday"] = pd.factorize(day, sort=True)[0]
    return m


def _fit_ologit(d: np.ndarray, y: np.ndarray, x0=(-0.6, 0.4, 1.5)) -> np.ndarray:
    """y: 0 = home, 1 = draw, 2 = away. Returns (c1, c2, b) with c1 < c2."""
    def nll(p):
        c1, c2, b = p[0], p[0] + np.exp(p[1]), p[2]
        pa = expit(c1 - b * d)
        pad = expit(c2 - b * d)
        pr = np.where(y == 2, pa, np.where(y == 1, pad - pa, 1 - pad))
        return -np.log(np.clip(pr, 1e-12, None)).sum()
    res = minimize(nll, np.array([x0[0], np.log(x0[1]), x0[2]]), method="L-BFGS-B")
    return np.array([res.x[0], res.x[0] + np.exp(res.x[1]), res.x[2]])


def probs_from_gap(theta: np.ndarray, d: np.ndarray) -> np.ndarray:
    c1, c2, b = theta
    pa, pad = expit(c1 - b * d), expit(c2 - b * d)
    return np.column_stack([1 - pad, pad - pa, pa])


def walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...], k: float = 20.0) -> pd.DataFrame:
    """Elo H/D/A probabilities for every completed match in `eval_seasons`, with the outcome mapping
    refitted on all earlier matchdays."""
    m = pre_match_gaps(matches, k)
    y = m["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    out = []
    for md in sorted(m.loc[m["season"].isin(eval_seasons), "matchday"].unique()):
        tr = (m["matchday"] < md).to_numpy()
        te = (m["matchday"] == md).to_numpy()
        th = _fit_ologit(m.loc[tr, "elo_gap"].to_numpy(), y[tr])
        p = probs_from_gap(th, m.loc[te, "elo_gap"].to_numpy())
        out.append(pd.DataFrame({"match_id": m.loc[te, "match_id"].to_numpy(), "season": m.loc[te, "season"].to_numpy(),
                                 "kickoff_utc": m.loc[te, "kickoff_utc"].to_numpy(), "result": m.loc[te, "result"].to_numpy(),
                                 "elo_gap": m.loc[te, "elo_gap"].to_numpy(),
                                 "p_home": p[:, 0], "p_draw": p[:, 1], "p_away": p[:, 2]}))
    return pd.concat(out, ignore_index=True)
