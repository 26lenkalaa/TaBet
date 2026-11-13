"""Score-driven dynamic team strengths (after Koopman & Lit 2019), same goal model as Dixon-Coles.

    log(home rate) = mu + h + a[home, t] - d[away, t]
    log(away rate) = mu     + a[away, t] - d[home, t]      (+ the Dixon-Coles rho correction)

Strengths are filtered forward through time. Between matches they decay toward 0 (the league
average) with weekly persistence beta, and by an extra factor beta_summer when a season boundary is
crossed. After each result they move along the score (gradient of the Poisson log-likelihood with
respect to the log-rate), scaled by alpha_att / alpha_def:

    a[home] += alpha_att * (x - lam_home)     d[away] -= alpha_def * (x - lam_home)
    a[away] += alpha_att * (y - lam_away)     d[home] -= alpha_def * (y - lam_away)

Static parameters (mu, h, rho, alpha_att, alpha_def, beta, beta_summer) are estimated by maximising
the one-step-ahead (filtered) log-likelihood on earlier matches only. Matches are processed by
matchday: every match of a day is predicted from the state before that day, then the day's results
update the state, so a same-day result never informs another same-day prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .ingest import LOCAL_TZ
from .model import RHO_BOUND

NAMES = ("mu", "home", "rho", "alpha_att", "alpha_def", "beta", "beta_summer")
BOUNDS = [(-1.0, 1.0), (-0.5, 0.8), (-RHO_BOUND, RHO_BOUND), (0.0, 0.2), (0.0, 0.2), (0.9, 1.0), (0.0, 1.0)]
START = np.array([0.3, 0.25, -0.05, 0.03, 0.03, 0.995, 0.8])


@dataclass
class Prepared:
    """Integer-coded match arrays in chronological matchday order."""
    hi: np.ndarray
    ai: np.ndarray
    x: np.ndarray
    y: np.ndarray
    day: np.ndarray       # matchday index (same local date = same index)
    days: np.ndarray      # day number (float) of each matchday
    season: np.ndarray    # season index of each matchday
    n_teams: int


def prepare(matches: pd.DataFrame) -> tuple[Prepared, pd.DataFrame]:
    m = matches[matches["status"] == "completed"].sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)
    teams = sorted(set(m["home_id"]) | set(m["away_id"]))
    idx = {t: i for i, t in enumerate(teams)}
    date = m["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    codes, uniq = pd.factorize(date, sort=True)
    first = m.groupby(codes)["kickoff_utc"].min()
    seasons = m.groupby(codes)["season"].first()
    sidx = pd.factorize(seasons, sort=True)[0]
    p = Prepared(m["home_id"].map(idx).to_numpy(), m["away_id"].map(idx).to_numpy(),
                 m["home_goals"].to_numpy(float), m["away_goals"].to_numpy(float), codes,
                 (first - first.iloc[0]).dt.total_seconds().to_numpy() / 86400, sidx, len(teams))
    m["matchday"] = codes
    m["cutoff_utc"] = m["matchday"].map(first)
    return p, m


def _tau(x, y, lh, la, rho):
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def filter_run(theta: np.ndarray, p: Prepared, upto_day: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the filter. Returns per-match pre-match (lam_home, lam_away) and the summed one-step
    log-likelihood over matches on days < upto_day (all days if None)."""
    mu, h, rho, aa, ad, beta, bs = theta
    a = np.zeros(p.n_teams)
    d = np.zeros(p.n_teams)
    lam = np.empty((len(p.x), 2))
    ll = 0.0
    last_t, last_s = p.days[0], p.season[0]
    n_days = len(p.days) if upto_day is None else upto_day
    start = 0
    lb = np.log(beta)
    for k in range(len(p.days)):
        stop = start
        while stop < len(p.day) and p.day[stop] == k:
            stop += 1
        if k > 0:  # decay since the previous matchday (and across a summer)
            f = np.exp(lb * (p.days[k] - last_t) / 7.0) * (bs if p.season[k] != last_s else 1.0)
            a *= f
            d *= f
            last_t, last_s = p.days[k], p.season[k]
        i, j = p.hi[start:stop], p.ai[start:stop]
        lh = np.exp(mu + h + a[i] - d[j])
        la = np.exp(mu + a[j] - d[i])
        lam[start:stop, 0], lam[start:stop, 1] = lh, la
        xs, ys = p.x[start:stop], p.y[start:stop]
        if k < n_days:
            tau = np.array([_tau(u, v, s, t, rho) for u, v, s, t in zip(xs, ys, lh, la)])
            ll += float((xs * np.log(lh) - lh + ys * np.log(la) - la + np.log(np.maximum(tau, 1e-10))).sum())
        # update with the day's results (np.add.at handles a team twice in one day)
        np.add.at(a, i, aa * (xs - lh))
        np.add.at(d, j, -ad * (xs - lh))
        np.add.at(a, j, aa * (ys - la))
        np.add.at(d, i, -ad * (ys - la))
        start = stop
    return lam, (a, d), ll


def fit(p: Prepared, upto_day: int, theta0: np.ndarray = START) -> np.ndarray:
    """Static parameters by maximising the filtered log-likelihood on days < upto_day."""
    def f(th):
        return -filter_run(th, p, upto_day)[2]
    res = minimize(f, theta0, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 300})
    return res.x


def probs(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    from .experiments import _probs
    return _probs(lh, la, np.full(len(lh), rho))


def walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...], summer: bool = True) -> pd.DataFrame:
    """Static parameters refitted at the start of each evaluated season on all earlier matches;
    the filter then runs through that season using only results known before each matchday."""
    p, m = prepare(matches)
    out = []
    theta = START.copy()
    for s in eval_seasons:
        days = m.loc[m["season"] == s, "matchday"]
        if days.empty:
            continue
        first_day = int(days.min())
        theta = fit(p, first_day, theta) if summer else _fit_no_summer(p, first_day, theta)
        lam, _, _ = filter_run(theta, p)
        sel = (m["season"] == s).to_numpy()
        P = probs(lam[sel, 0], lam[sel, 1], theta[2])
        out.append(pd.DataFrame({"match_id": m.loc[sel, "match_id"].to_numpy(), "season": s,
                                 "kickoff_utc": m.loc[sel, "kickoff_utc"].to_numpy(),
                                 "cutoff_utc": m.loc[sel, "cutoff_utc"].to_numpy(),
                                 "result": m.loc[sel, "result"].to_numpy(),
                                 "lam_home": lam[sel, 0], "lam_away": lam[sel, 1], "rho": theta[2],
                                 "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2],
                                 **{f"theta_{n}": v for n, v in zip(NAMES, theta)}}))
    return pd.concat(out, ignore_index=True)


def _fit_no_summer(p: Prepared, upto_day: int, theta0: np.ndarray) -> np.ndarray:
    b = list(BOUNDS)
    b[6] = (1.0, 1.0)
    th0 = theta0.copy()
    th0[6] = 1.0
    res = minimize(lambda th: -filter_run(th, p, upto_day)[2], th0, method="L-BFGS-B", bounds=b,
                   options={"maxiter": 300})
    return res.x

