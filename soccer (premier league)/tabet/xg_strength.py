"""Opponent-adjusted xG strength (experiment #1 from the external review).

Raw decayed xG form (V1) can mistake an easy run of opponents for real
improvement. This fits, at each cutoff, a regularised log-linear model of
team-match xG:

    E[xG of T vs O] = exp(mu + home * [T at home] + att[T] - def[O])

on past matches only, weighted by exp(-ln2 * age / half_life). xG is
continuous and non-negative, so it is fitted by Poisson *quasi*-likelihood:
the estimating equations sum(w * (y - m) * dlog m) = 0 only need a correct
conditional mean, not integer counts. Ratings get an L2 penalty toward 0, so
teams with little xG history (e.g. promoted sides) start at league average.

The stage-2 features for team T scoring against O are att[T] and -def[O], the
opponent-adjusted analogues of V1's f_xgf and f_xga_opp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import known

LN2 = np.log(2)


def side_xg(app: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match with summed FPL xG (xG era only)."""
    a = app[app["xg"].notna()]
    return (a.groupby(["match_id", "kickoff_utc", "team_id", "opponent_id", "venue"], as_index=False)["xg"].sum())


def fit_xg_strength(sides: pd.DataFrame, cutoff: pd.Timestamp, half_life: float,
                    l2: float = 5.0) -> tuple[pd.DataFrame, float, float]:
    """Returns (ratings indexed by team with columns att/def, mu, home)."""
    if sides.empty:
        return pd.DataFrame(columns=["att", "def"]), 0.0, 0.0
    teams = sorted(set(sides["team_id"]) | set(sides["opponent_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ti = sides["team_id"].map(idx).to_numpy()
    oi = sides["opponent_id"].map(idx).to_numpy()
    home = (sides["venue"] == "H").to_numpy(float)
    y = sides["xg"].to_numpy(float)
    age = (cutoff - sides["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
    w = np.exp(-LN2 * age / half_life)
    w = w / w.mean()

    def f(p):
        mu, h, att, dfn = p[0], p[1], p[2:2 + n], p[2 + n:]
        eta = mu + h * home + att[ti] - dfn[oi]
        m = np.exp(eta)
        loss = (w * (m - y * eta)).sum() + l2 * (att @ att + dfn @ dfn)
        r = w * (m - y)  # d loss / d eta
        g = np.empty_like(p)
        g[0] = r.sum()
        g[1] = (r * home).sum()
        g[2:2 + n] = np.bincount(ti, r, n) + 2 * l2 * att
        g[2 + n:] = -np.bincount(oi, r, n) + 2 * l2 * dfn
        return loss, g

    p0 = np.zeros(2 + 2 * n)
    p0[0] = np.log(max(np.average(y, weights=w), 0.05))
    res = minimize(f, p0, jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"xG strength fit did not converge: {res.message}")
    p = res.x
    ratings = pd.DataFrame({"att": p[2:2 + n], "def": p[2 + n:]}, index=teams)
    return ratings, float(p[0]), float(p[1])


def add_xg_strength(rows: pd.DataFrame, app: pd.DataFrame, half_lives: tuple[float, ...],
                    l2: float = 5.0) -> pd.DataFrame:
    """Add f_xgadj_att_<hl> / f_xgadj_def_opp_<hl> to experiment rows, each
    computed at the row's own cutoff from appearances known by then."""
    sides_all = side_xg(app)
    out = rows.copy()
    for hl in half_lives:
        att_col, def_col = f"f_xgadj_att_{int(hl)}", f"f_xgadj_def_opp_{int(hl)}"
        out[att_col], out[def_col] = 0.0, 0.0
        for cutoff, grp in out.groupby("cutoff_utc"):
            sides = known(sides_all, cutoff)
            r, _, _ = fit_xg_strength(sides, cutoff, hl, l2)
            out.loc[grp.index, att_col] = grp["team_id"].map(r["att"]).fillna(0.0).to_numpy()
            # higher def = concedes less xG, so the scoring side's feature is -def[opponent]
            out.loc[grp.index, def_col] = -grp["opponent_id"].map(r["def"]).fillna(0.0).to_numpy()
    return out
