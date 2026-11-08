"""Option 1: hierarchical Bayesian Dixon-Coles (Laplace approximation), with the returning-team reset.

Same goal model as Dixon-Coles (Poisson goals, home advantage, rho low-score correction, 173-day
time decay). What changes:

1. **Spells.** Teams are relabelled by league spell (tabet.alt.returning_reset), so a returning club
   starts fresh.
2. **Hierarchical prior.** Ratings are Gaussian around 0 with precision 2*lam (the L2 penalty,
   lam = 10). A team in the first season of its spell gets a learned group offset (m_att, m_def),
   estimated from earlier promoted sides as in section 5.15. While that first season is under way,
   its own deviation gets a *tighter* prior, precision 2*lam_new with lam_new >= lam, so that two or
   three early results cannot swing it far (the Sunderland / Hull misses). lam_new is chosen on
   development seasons from LAM_NEW_GRID.
3. **Uncertainty.** The posterior is approximated by a Gaussian at the mode, with covariance equal to
   the inverse Hessian of the penalised (decay-weighted) negative log-likelihood. Forecasts average
   the score grid over N_DRAWS draws of (mu, home, m, attack, defence); rho stays at its mode.

This is an approximation, not MCMC: with decay weights the "posterior" is a weighted
pseudo-posterior, so its spread is indicative rather than exact. Refitting a full MCMC model before
every matchday would be far too slow for walk-forward evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ..ingest import LOCAL_TZ
from ..model import MAX_GOALS, RHO_BOUND
from ..store import MatchStore
from .returning_reset import relabel

XI = 0.004
LAM = 10.0
LAM_NEW_GRID = (10.0, 30.0, 100.0)
N_DRAWS = 300
G = np.arange(MAX_GOALS + 1)


class BayesDC:
    def __init__(self, lam_new: float, seed: int = 0):
        self.lam_new = lam_new
        self.rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp, season: str, season_teams: set[str],
            first_season: dict[str, str]) -> "BayesDC":
        """first_season: spell id -> the season its spell started. A team-season is 'new' (gets the
        group offset) when season == first_season[team]. season_teams: teams of the target season
        (unseen newcomers get prior-only ratings)."""
        train = train[train["status"] == "completed"]
        teams = sorted(set(train["home_id"]) | set(train["away_id"]) | season_teams)
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi, ai = train["home_id"].map(idx).to_numpy(), train["away_id"].map(idx).to_numpy()
        x, y = train["home_goals"].to_numpy(float), train["away_goals"].to_numpy(float)
        ph = np.array([first_season.get(t) == s for t, s in zip(train["home_id"], train["season"])], float)
        pa = np.array([first_season.get(t) == s for t, s in zip(train["away_id"], train["season"])], float)
        age = (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
        w = np.exp(-XI * age)
        w = w / w.mean()
        # tighter prior for teams in the first season of their spell, while that season is under way
        lam = np.array([self.lam_new if first_season.get(t) == season else LAM for t in teams])
        c00, c01, c10, c11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)

        def unpack(p):
            return p[0], p[1], p[2], p[3], p[4], p[5:5 + n], p[5 + n:]

        def f(p):
            mu, h, rho, ma, md, a, d = unpack(p)
            lh = np.exp(mu + h + a[hi] + ma * ph - d[ai] - md * pa)
            la = np.exp(mu + a[ai] + ma * pa - d[hi] - md * ph)
            tau = np.ones_like(lh)
            tau[c00] = 1 - lh[c00] * la[c00] * rho
            tau[c01] = 1 + lh[c01] * rho
            tau[c10] = 1 + la[c10] * rho
            tau[c11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = x * np.log(lh) - lh + y * np.log(la) - la + np.log(tau)
            obj = -(w * ll).sum() + (lam * (a * a + d * d)).sum()
            gh, ga, gr = x - lh, y - la, np.zeros_like(lh)
            gr[c00] = -lh[c00] * la[c00] / tau[c00]
            gh[c00] += -lh[c00] * la[c00] * rho / tau[c00]
            ga[c00] += -lh[c00] * la[c00] * rho / tau[c00]
            gr[c01] = lh[c01] / tau[c01]
            gh[c01] += lh[c01] * rho / tau[c01]
            gr[c10] = la[c10] / tau[c10]
            ga[c10] += la[c10] * rho / tau[c10]
            gr[c11] = -1 / tau[c11]
            gh, ga, gr = w * gh, w * ga, w * gr
            g = np.empty_like(p)
            g[0] = gh.sum() + ga.sum()
            g[1] = gh.sum()
            g[2] = gr.sum()
            g[3] = (gh * ph).sum() + (ga * pa).sum()
            g[4] = -(gh * pa).sum() - (ga * ph).sum()
            g[5:5 + n] = np.bincount(hi, gh, n) + np.bincount(ai, ga, n)
            g[5 + n:] = -np.bincount(ai, gh, n) - np.bincount(hi, ga, n)
            g = -g
            g[5:5 + n] += 2 * lam * a
            g[5 + n:] += 2 * lam * d
            return obj, g

        p0 = np.zeros(5 + 2 * n)
        p0[0] = np.log(max((w * (x + y)).sum() / (2 * w.sum()), 0.1))
        bounds = [(None, None), (None, None), (-RHO_BOUND, RHO_BOUND)] + [(None, None)] * (2 + 2 * n)
        res = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
        mode = res.x
        # Laplace: Hessian by central differences of the analytic gradient (rho held at its mode)
        free = np.r_[0, 1, np.arange(3, 5 + 2 * n)]
        H = np.empty((len(free), len(free)))
        eps = 1e-5
        for j, k in enumerate(free):
            e = np.zeros_like(mode)
            e[k] = eps
            H[:, j] = (f(mode + e)[1][free] - f(mode - e)[1][free]) / (2 * eps)
        H = (H + H.T) / 2 + 1e-8 * np.eye(len(free))
        self.cov = np.linalg.inv(H)
        self.mode, self.free, self.idx, self.n = mode, free, idx, n
        self.first_season, self.season = first_season, season
        return self

    def _rates(self, theta: np.ndarray, home: str, away: str) -> tuple[np.ndarray, np.ndarray]:
        n, idx = self.n, self.idx
        mu, h, ma, md = theta[..., 0], theta[..., 1], theta[..., 3], theta[..., 4]
        a, d = theta[..., 5:5 + n], theta[..., 5 + n:]
        newh = float(self.first_season.get(home) == self.season)
        newa = float(self.first_season.get(away) == self.season)
        i, j = idx[home], idx[away]
        lh = np.exp(mu + h + a[..., i] + ma * newh - d[..., j] - md * newa)
        la = np.exp(mu + a[..., j] + ma * newa - d[..., i] - md * newh)
        return np.atleast_1d(lh), np.atleast_1d(la)

    def predict(self, home: str, away: str, average: bool = True) -> tuple[np.ndarray, float, float]:
        """(p_home, p_draw, p_away), and the mode rates."""
        rho = self.mode[2]
        lh0, la0 = self._rates(self.mode, home, away)
        if average:
            draws = np.tile(self.mode, (N_DRAWS, 1))
            draws[:, self.free] = self.rng.multivariate_normal(self.mode[self.free], self.cov, size=N_DRAWS,
                                                               method="cholesky")
            lh, la = self._rates(draws, home, away)
        else:
            lh, la = lh0, la0
        mh, ma = poisson.pmf(G[None, :], lh[:, None]), poisson.pmf(G[None, :], la[:, None])
        m = mh[:, :, None] * ma[:, None, :]
        m[:, 0, 0] *= 1 - lh * la * rho
        m[:, 0, 1] *= 1 + lh * rho
        m[:, 1, 0] *= 1 + la * rho
        m[:, 1, 1] *= 1 - rho
        m /= m.sum(axis=(1, 2), keepdims=True)
        m = m.mean(axis=0)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]), float(lh0[0]), float(la0[0])


def walk_forward(matches: pd.DataFrame, eval_seasons: tuple[str, ...], lam_new: float) -> pd.DataFrame:
    """Per matchday: fit on earlier results (spell-relabelled), predict with and without averaging."""
    m = relabel(matches)
    first_season = m.melt(id_vars="season", value_vars=["home_id", "away_id"]).groupby("value")["season"].min().to_dict()
    store = MatchStore(m)
    target = m[m["season"].isin(eval_seasons) & (m["status"] == "completed")]
    day = target["kickoff_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    out = []
    for _, g in target.groupby(day, sort=True):
        cutoff = g["kickoff_utc"].min()
        season = g["season"].iloc[0]
        teams = set(m.loc[m["season"] == season, "home_id"]) | set(m.loc[m["season"] == season, "away_id"])
        model = BayesDC(lam_new).fit(store.completed_before(cutoff), cutoff, season, teams, first_season)
        for f in g.itertuples():
            p_avg, lh, la = model.predict(f.home_id, f.away_id, average=True)
            p_mode, _, _ = model.predict(f.home_id, f.away_id, average=False)
            out.append({"match_id": f.match_id, "season": f.season, "kickoff_utc": f.kickoff_utc, "result": f.result,
                        "lam_home": lh, "lam_away": la, "rho": float(model.mode[2]),
                        "p_home": p_avg[0], "p_draw": p_avg[1], "p_away": p_avg[2],
                        "mode_p_home": p_mode[0], "mode_p_draw": p_mode[1], "mode_p_away": p_mode[2]})
    return pd.DataFrame(out)
