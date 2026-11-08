"""Option 4: different goal distributions on top of the same Dixon-Coles team ratings.

To isolate the distribution, both variants keep the frozen Dixon-Coles fit (same ratings, same
expected goals per side) and change only how two expected goal counts become a score grid. The one
extra parameter is fitted at each cutoff by decay-weighted maximum likelihood on the training
matches, using the same fit's in-sample expected goals.

  nb   negative binomial goals with a shared size r (variance = lam + lam^2 / r; r -> infinity is
       Poisson), keeping the Dixon-Coles low-score correction with the fit's rho
  bp   bivariate Poisson (Karlis & Ntzoufras): X = U + W, Y = V + W with a shared component W of rate
       lam3, so the margins keep the Dixon-Coles means (lam_home, lam_away) and lam3 >= 0 adds
       positive correlation. No rho correction (the shared component plays that role).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom

from ..ingest import LOCAL_TZ
from ..model import MAX_GOALS, DixonColes, Hyperparams, promoted_team_seasons
from ..store import MatchStore

G = np.arange(MAX_GOALS + 1)
XI = 0.004


def _tau(lh, la, rho):
    t = np.ones((len(lh), MAX_GOALS + 1, MAX_GOALS + 1))
    t[:, 0, 0] = 1 - lh * la * rho
    t[:, 0, 1] = 1 + lh * rho
    t[:, 1, 0] = 1 + la * rho
    t[:, 1, 1] = 1 - rho
    return t


def nb_grid(lh, la, rho, r):
    ph = nbinom.pmf(G[None, :], r, r / (r + lh[:, None]))
    pa = nbinom.pmf(G[None, :], r, r / (r + la[:, None]))
    m = ph[:, :, None] * pa[:, None, :] * _tau(lh, la, rho)
    return m / m.sum(axis=(1, 2), keepdims=True)


def bp_logpmf(x, y, l1, l2, l3):
    """log P(X = x, Y = y) for the bivariate Poisson, vectorised over matches (goals capped at MAX_GOALS)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.arange(MAX_GOALS + 1, dtype=float)[None, :]
    ok = k <= np.minimum(x, y)[:, None]
    xs, ys = x[:, None], y[:, None]
    terms = (gammaln(xs + 1) - gammaln(k + 1) - gammaln(np.maximum(xs - k, 0) + 1) + gammaln(ys + 1)
             - gammaln(np.maximum(ys - k, 0) + 1) + k * (np.log(l3) - np.log(l1) - np.log(l2))[:, None])
    terms = np.where(ok, terms, -np.inf)
    base = -(l1 + l2 + l3) + x * np.log(l1) - gammaln(x + 1) + y * np.log(l2) - gammaln(y + 1)
    return base + np.logaddexp.reduce(terms, axis=1)


def bp_grid(lh, la, l3):
    x, y = np.meshgrid(G, G, indexing="ij")
    out = np.empty((len(lh), MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(len(lh)):
        l3i = min(l3, 0.95 * min(lh[i], la[i]))
        lp = bp_logpmf(x.ravel(), y.ravel(), np.full(x.size, lh[i] - l3i), np.full(x.size, la[i] - l3i), np.full(x.size, max(l3i, 1e-12)))
        m = np.exp(lp).reshape(x.shape)
        out[i] = m / m.sum()
    return out


def _hda3(m):
    tri_l = np.array([np.tril(mm, -1).sum() for mm in m])
    tri_u = np.array([np.triu(mm, 1).sum() for mm in m])
    return np.column_stack([tri_l, np.trace(m, axis1=1, axis2=2), tri_u])


def fit_r(lh, la, x, y, rho, w):
    tau = _tau(lh, la, rho)[np.arange(len(x)), np.minimum(x, MAX_GOALS), np.minimum(y, MAX_GOALS)]
    def nll(logr):
        r = np.exp(logr)
        ll = nbinom.logpmf(x, r, r / (r + lh)) + nbinom.logpmf(y, r, r / (r + la)) + np.log(np.maximum(tau, 1e-10))
        return -(w * ll).sum()
    res = minimize_scalar(nll, bounds=(np.log(0.5), np.log(1e4)), method="bounded")
    return float(np.exp(res.x))


def fit_l3(lh, la, x, y, w):
    def nll(l3):
        l3v = np.minimum(l3, 0.95 * np.minimum(lh, la))
        return -(w * bp_logpmf(x, y, lh - l3v, la - l3v, np.maximum(l3v, 1e-12))).sum()
    res = minimize_scalar(nll, bounds=(1e-6, 0.4), method="bounded")
    return float(res.x)


def walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...]) -> pd.DataFrame:
    store = MatchStore(matches)
    promoted = promoted_team_seasons(matches)
    target = matches[matches["season"].isin(eval_seasons) & (matches["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out = []
    for _, g in target.groupby(day, sort=True):
        cutoff = g["kickoff_utc"].min()
        train = store.completed_before(cutoff)
        dc = DixonColes(Hyperparams()).fit(train, cutoff, promoted)
        lam_tr = np.array([dc.expected_goals(h, a, s) for h, a, s in zip(train["home_id"], train["away_id"], train["season"])])
        x, y = train["home_goals"].to_numpy(int), train["away_goals"].to_numpy(int)
        age = (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
        w = np.exp(-XI * age)
        w = w / w.mean()
        r = fit_r(lam_tr[:, 0], lam_tr[:, 1], x, y, dc.rho, w)
        l3 = fit_l3(lam_tr[:, 0], lam_tr[:, 1], x, y, w)
        lam = np.array([dc.expected_goals(h, a, s) for h, a, s in zip(g["home_id"], g["away_id"], g["season"])])
        pnb = _hda3(nb_grid(lam[:, 0], lam[:, 1], dc.rho, r))
        pbp = _hda3(bp_grid(lam[:, 0], lam[:, 1], l3))
        for i, f in enumerate(g.itertuples()):
            out.append({"match_id": f.match_id, "season": f.season, "kickoff_utc": f.kickoff_utc, "result": f.result,
                        "lam_home": lam[i, 0], "lam_away": lam[i, 1], "nb_r": r, "bp_l3": l3,
                        "nb_p_home": pnb[i, 0], "nb_p_draw": pnb[i, 1], "nb_p_away": pnb[i, 2],
                        "bp_p_home": pbp[i, 0], "bp_p_draw": pbp[i, 1], "bp_p_away": pbp[i, 2]})
    return pd.DataFrame(out)
