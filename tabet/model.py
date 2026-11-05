"""Dixon-Coles team-strength model: the baseline from docs/SPEC.md section 5.

Each team has an attack rating a and a defence rating d. Goals are Poisson with

    log(home rate) = mu + home_adv + a[home] - d[away]
    log(away rate) = mu            + a[away] - d[home]

plus the Dixon-Coles correction (rho) for the low scores 0-0, 1-0, 0-1 and 1-1.
Older matches count less, with weight exp(-xi * age_in_days), so xi is the
"weighting the different years" idea as a parameter to tune. An L2 penalty pulls
every rating toward 0, which keeps the fit identifiable and stops a team with a
few lucky results from getting an extreme rating.

Promoted teams: a team's attack and defence get a shared offset (promo_att,
promo_def) in the season it was promoted. The offset is estimated from earlier
promoted sides' first-season matches, so a newly promoted team starts from
"how promoted teams usually do" instead of from league average.

Promoted-team features (optional): with `promo_x` = {(team, season): (x_att,
x_def)} from the team's previous (second-tier) season, the offset becomes
team-specific: promo_att + b_att * x_att and promo_def + b_def * x_def. The
slopes are learned in the same fit from earlier promoted sides' matches, with
the same L2 penalty as the ratings. The team's own ratings still start at 0 and
take over as its matches accumulate, so the offset acts as a prior mean.

The fit only sees the matches it is given. Callers pass
MatchStore.completed_before(cutoff), and stamp the model with that cutoff.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Score grid 0..MAX_GOALS per side; every grid is renormalised to sum to 1. At the largest
# adjusted rate seen (5.41) one side's tail beyond 15 goals is 1.7e-4; across the ledger
# forecasts the largest omitted joint mass is 1.2e-5 (largest rate 4.32).
MAX_GOALS = 15
RHO_BOUND = 0.2


@dataclass(frozen=True)
class Hyperparams:
    # Defaults chosen by walk-forward tuning on 2022-23 + 2023-24 (backtest.tune).
    # The promoted-team prior didn't measurably help, so the simpler model is the default.
    xi: float = 0.004  # time decay per day; half-life = ln 2 / xi (~173 days)
    l2: float = 10.0  # ridge penalty on team ratings
    promoted_prior: bool = False

    @property
    def half_life_days(self) -> float:
        return float(np.log(2) / self.xi) if self.xi > 0 else float("inf")


def promoted_team_seasons(matches: pd.DataFrame) -> set[tuple[str, str]]:
    """(team_id, season) pairs where the team was not in the previous season.
    The earliest season in the data has nothing to compare against, so none of
    its teams are flagged. This only needs each season's team list, which is
    known before that season's first kickoff, so it doesn't leak results."""
    teams = (
        pd.concat([matches[["season", "home_id"]].rename(columns={"home_id": "team"}),
                   matches[["season", "away_id"]].rename(columns={"away_id": "team"})])
        .groupby("season")["team"].agg(set)
    )
    seasons = sorted(teams.index)
    out = set()
    for prev, cur in zip(seasons, seasons[1:]):
        if int(cur[:4]) != int(prev[:4]) + 1:
            continue  # gap in the data: can't tell who was promoted
        out |= {(t, cur) for t in teams[cur] - teams[prev]}
    return out


@dataclass
class DixonColes:
    hyper: Hyperparams = field(default_factory=Hyperparams)
    trained_cutoff: str | None = None
    n_train: int = 0
    mu: float = 0.0
    home_adv: float = 0.0
    rho: float = 0.0
    promo_att: float = 0.0
    promo_def: float = 0.0
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    promoted: list[list[str]] = field(default_factory=list)  # [team_id, season] pairs
    promo_b_att: float = 0.0
    promo_b_def: float = 0.0
    promo_x: dict[str, list[float]] = field(default_factory=dict)  # "team|season" -> [x_att, x_def]

    # ---- fitting -------------------------------------------------------

    def fit(self, train: pd.DataFrame, cutoff: pd.Timestamp,
            promoted: set[tuple[str, str]] | None = None,
            promo_x: dict[tuple[str, str], tuple[float, float]] | None = None) -> "DixonColes":
        """Fit on completed matches, weighting each by its age at `cutoff`.
        `promoted` may include seasons beyond the training data (e.g. the
        season being predicted), which is why it is passed in separately."""
        train = train[train["status"] == "completed"]
        if (train["kickoff_utc"] >= cutoff).any():
            raise ValueError("training data contains matches at or after the cutoff")
        if len(train) == 0:
            raise ValueError("no completed matches to train on")
        promoted = promoted if promoted is not None else promoted_team_seasons(train)
        if not self.hyper.promoted_prior:
            promoted, promo_x = set(), None
        px = promo_x or {}

        teams = sorted(set(train["home_id"]) | set(train["away_id"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = train["home_id"].map(idx).to_numpy()
        ai = train["away_id"].map(idx).to_numpy()
        x = train["home_goals"].to_numpy(dtype=float)
        y = train["away_goals"].to_numpy(dtype=float)
        ph = np.array([(t, s) in promoted for t, s in zip(train["home_id"], train["season"])], dtype=float)
        pa = np.array([(t, s) in promoted for t, s in zip(train["away_id"], train["season"])], dtype=float)
        xh = np.array([px.get((t, s), (0.0, 0.0)) if (t, s) in promoted else (0.0, 0.0)
                       for t, s in zip(train["home_id"], train["season"])], dtype=float).reshape(-1, 2)
        xa = np.array([px.get((t, s), (0.0, 0.0)) if (t, s) in promoted else (0.0, 0.0)
                       for t, s in zip(train["away_id"], train["season"])], dtype=float).reshape(-1, 2)
        age = (cutoff - train["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400
        w = np.exp(-self.hyper.xi * age)
        w = w / w.mean()
        l2 = self.hyper.l2

        c00 = (x == 0) & (y == 0)
        c01 = (x == 0) & (y == 1)
        c10 = (x == 1) & (y == 0)
        c11 = (x == 1) & (y == 1)

        # params: mu, home, rho, promo_att, promo_def, attack[n], defence[n], b_att, b_def
        def unpack(p):
            return p[0], p[1], p[2], p[3], p[4], p[5:5 + n], p[5 + n:5 + 2 * n], p[5 + 2 * n], p[6 + 2 * n]

        def nll(p):
            mu, h, rho, pat, pdf, a, d, ba, bd = unpack(p)
            ll_h = mu + h + a[hi] + pat * ph + ba * xh[:, 0] - d[ai] - pdf * pa - bd * xa[:, 1]  # log home rate
            ll_a = mu + a[ai] + pat * pa + ba * xa[:, 0] - d[hi] - pdf * ph - bd * xh[:, 1]  # log away rate
            lh, la = np.exp(ll_h), np.exp(ll_a)

            tau = np.ones_like(lh)
            tau[c00] = 1 - lh[c00] * la[c00] * rho
            tau[c01] = 1 + lh[c01] * rho
            tau[c10] = 1 + la[c10] * rho
            tau[c11] = 1 - rho
            tau = np.maximum(tau, 1e-10)

            ll = x * ll_h - lh + y * ll_a - la + np.log(tau)
            f = -(w * ll).sum() + l2 * (a @ a + d @ d + ba * ba + bd * bd)

            # d(log-lik)/d(log rate), Poisson part plus tau part
            gh = x - lh
            ga = y - la
            g_rho = np.zeros_like(lh)
            g_rho[c00] = -lh[c00] * la[c00] / tau[c00]
            gh[c00] += -lh[c00] * la[c00] * rho / tau[c00]
            ga[c00] += -lh[c00] * la[c00] * rho / tau[c00]
            g_rho[c01] = lh[c01] / tau[c01]
            gh[c01] += lh[c01] * rho / tau[c01]
            g_rho[c10] = la[c10] / tau[c10]
            ga[c10] += la[c10] * rho / tau[c10]
            g_rho[c11] = -1 / tau[c11]
            gh, ga, g_rho = w * gh, w * ga, w * g_rho

            g = np.empty_like(p)
            g[0] = gh.sum() + ga.sum()
            g[1] = gh.sum()
            g[2] = g_rho.sum()
            g[3] = (gh * ph).sum() + (ga * pa).sum()
            g[4] = -(gh * pa).sum() - (ga * ph).sum()
            g[5:5 + n] = np.bincount(hi, gh, n) + np.bincount(ai, ga, n)
            g[5 + n:5 + 2 * n] = -np.bincount(ai, gh, n) - np.bincount(hi, ga, n)
            g[5 + 2 * n] = (gh * xh[:, 0]).sum() + (ga * xa[:, 0]).sum()
            g[6 + 2 * n] = -(gh * xa[:, 1]).sum() - (ga * xh[:, 1]).sum()
            g = -g
            g[5:5 + n] += 2 * l2 * a
            g[5 + n:5 + 2 * n] += 2 * l2 * d
            g[5 + 2 * n] += 2 * l2 * ba
            g[6 + 2 * n] += 2 * l2 * bd
            return f, g

        p0 = np.zeros(7 + 2 * n)
        p0[0] = np.log(max((w * (x + y)).sum() / (2 * w.sum()), 0.1))
        bounds = [(None, None), (None, None), (-RHO_BOUND, RHO_BOUND)] + [(None, None)] * (4 + 2 * n)
        if not promoted:
            bounds[3] = bounds[4] = (0.0, 0.0)
        if not px:
            bounds[5 + 2 * n] = bounds[6 + 2 * n] = (0.0, 0.0)
        res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000})
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit did not converge: {res.message}")

        mu, h, rho, pat, pdf, a, d, ba, bd = unpack(res.x)
        self.trained_cutoff = cutoff.isoformat()
        self.n_train = int(len(train))
        self.mu, self.home_adv, self.rho = float(mu), float(h), float(rho)
        self.promo_att, self.promo_def = float(pat), float(pdf)
        self.attack = {t: float(v) for t, v in zip(teams, a)}
        self.defence = {t: float(v) for t, v in zip(teams, d)}
        self.promoted = sorted([t, s] for t, s in promoted)
        self.promo_b_att, self.promo_b_def = float(ba), float(bd)
        self.promo_x = {f"{t}|{s}": [float(v) for v in x] for (t, s), x in px.items() if (t, s) in promoted}
        return self

    # ---- prediction ----------------------------------------------------

    def _rating(self, team: str, season: str | None) -> tuple[float, float]:
        # A team never seen in training gets the average rating (0), plus the
        # promoted offset if it's flagged as promoted this season.
        a = self.attack.get(team, 0.0)
        d = self.defence.get(team, 0.0)
        if season is not None and [team, season] in self.promoted:
            xa, xd = self.promo_x.get(f"{team}|{season}", (0.0, 0.0))
            a += self.promo_att + self.promo_b_att * xa
            d += self.promo_def + self.promo_b_def * xd
        return a, d

    def expected_goals(self, home: str, away: str, season: str | None = None) -> tuple[float, float]:
        ah, dh = self._rating(home, season)
        aa, da = self._rating(away, season)
        return (float(np.exp(self.mu + self.home_adv + ah - da)),
                float(np.exp(self.mu + aa - dh)))

    def score_matrix(self, home: str, away: str, season: str | None = None) -> np.ndarray:
        lh, la = self.expected_goals(home, away, season)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        r = self.rho
        m[0, 0] *= 1 - lh * la * r
        m[0, 1] *= 1 + lh * r
        m[1, 0] *= 1 + la * r
        m[1, 1] *= 1 - r
        return m / m.sum()

    def predict_proba(self, home: str, away: str, season: str | None = None) -> tuple[float, float, float]:
        """(p_home, p_draw, p_away), summing to 1."""
        m = self.score_matrix(home, away, season)
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())

    def predict_frame(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        rows = [self.predict_proba(h, a, s)
                for h, a, s in zip(fixtures["home_id"], fixtures["away_id"], fixtures["season"])]
        return pd.DataFrame(rows, columns=["p_home", "p_draw", "p_away"], index=fixtures.index)

    # ---- persistence ---------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"] = "dixon_coles"
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DixonColes":
        d = json.loads(Path(path).read_text())
        if d.pop("model", None) != "dixon_coles":
            raise ValueError(f"{path} is not a dixon_coles model file")
        d["hyper"] = Hyperparams(**d["hyper"])
        return cls(**d)
