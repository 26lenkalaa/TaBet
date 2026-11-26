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


def filter_run(theta: np.ndarray, p: Prepared, upto_day: int | None = None,
               u: np.ndarray | None = None, v: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the filter. Returns per-match pre-match (lam_home, lam_away) and the summed one-step
    log-likelihood over matches on days < upto_day (all days if None). u, v (addendum TS) give each
    team its own speeds alpha_att * exp(u_i), alpha_def * exp(v_i); omitted, every team shares them."""
    mu, h, rho, aa, ad, beta, bs = theta
    ga = aa * np.exp(u) if u is not None else np.full(p.n_teams, aa)
    gd = ad * np.exp(v) if v is not None else np.full(p.n_teams, ad)
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
        np.add.at(a, i, ga[i] * (xs - lh))
        np.add.at(d, j, -gd[j] * (xs - lh))
        np.add.at(a, j, ga[j] * (ys - la))
        np.add.at(d, i, -gd[i] * (ys - la))
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



# ---------------------------------------------------------------- addendum TS: team-specific speeds

UV_BOUND = 2.0


def _ts_loss_torch(p: Prepared, upto_day: int):
    """A torch closure z -> (penalty-free negative filtered log-likelihood on days < upto_day), where
    z = [theta (7), u (n_teams), v (n_teams)]. Same arithmetic as filter_run, differentiable."""
    import torch
    T = torch.float64
    stop = int(np.searchsorted(p.day, upto_day, "left"))
    hi, ai = torch.as_tensor(p.hi[:stop].copy()), torch.as_tensor(p.ai[:stop].copy())
    x, y = torch.as_tensor(p.x[:stop], dtype=T), torch.as_tensor(p.y[:stop], dtype=T)
    gap = torch.as_tensor(np.diff(p.days[:upto_day], prepend=p.days[0]) / 7.0, dtype=T)
    newseason = torch.as_tensor(np.r_[False, p.season[1:upto_day] != p.season[:upto_day - 1]])
    bounds = np.searchsorted(p.day[:stop], np.arange(upto_day + 1), "left")
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
    n = p.n_teams

    def loss(z):
        mu, h, rho, aa, ad, beta, bs = z[:7]
        ga, gd = aa * torch.exp(z[7:7 + n]), ad * torch.exp(z[7 + n:])
        lf = torch.log(beta) * gap
        f = torch.exp(lf) * torch.where(newseason, bs, torch.ones((), dtype=T))
        a = torch.zeros(n, dtype=T)
        d = torch.zeros(n, dtype=T)
        lhs, las = [], []
        for k in range(upto_day):
            s0, s1 = bounds[k], bounds[k + 1]
            if k > 0:
                a, d = a * f[k], d * f[k]
            if s0 == s1:
                continue
            i, j = hi[s0:s1], ai[s0:s1]
            lh = torch.exp(mu + h + a[i] - d[j])
            la = torch.exp(mu + a[j] - d[i])
            lhs.append(lh)
            las.append(la)
            xs, ys = x[s0:s1], y[s0:s1]
            a = a.index_add(0, i, ga[i] * (xs - lh)).index_add(0, j, ga[j] * (ys - la))
            d = d.index_add(0, j, -gd[j] * (xs - lh)).index_add(0, i, -gd[i] * (ys - la))
        lh, la = torch.cat(lhs), torch.cat(las)
        tau = torch.ones_like(lh)
        tau = torch.where(m00, 1 - lh * la * rho, tau)
        tau = torch.where(m01, 1 + lh * rho, tau)
        tau = torch.where(m10, 1 + la * rho, tau)
        tau = torch.where(m11, 1 - rho, tau)
        ll = (x * torch.log(lh) - lh + y * torch.log(la) - la + torch.log(torch.clamp(tau, min=1e-10))).sum()
        return -ll
    return loss


def fit_ts(p: Prepared, upto_day: int, lam: float, theta0: np.ndarray = START,
           u0: np.ndarray | None = None, v0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TS static parameters: maximise the filtered log-likelihood on days < upto_day minus
    lam * sum(u^2 + v^2). Teams with no match before upto_day keep u = v = 0."""
    import torch
    n = p.n_teams
    stop = int(np.searchsorted(p.day, upto_day, "left"))
    seen = np.zeros(n, bool)
    seen[p.hi[:stop]] = True
    seen[p.ai[:stop]] = True
    loss = _ts_loss_torch(p, upto_day)
    z0 = np.r_[theta0, np.zeros(n) if u0 is None else u0, np.zeros(n) if v0 is None else v0]
    z0[7:7 + n][~seen] = 0.0
    z0[7 + n:][~seen] = 0.0
    uvb = [(-UV_BOUND, UV_BOUND) if s else (0.0, 0.0) for s in seen]

    def f(z):
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        val = loss(zt) + lam * (zt[7:] ** 2).sum()
        val.backward()
        return float(val.detach()), zt.grad.numpy().copy()
    res = minimize(f, np.clip(z0, [b[0] for b in BOUNDS + uvb * 2], [b[1] for b in BOUNDS + uvb * 2]),
                   jac=True, method="L-BFGS-B", bounds=BOUNDS + uvb + uvb, options={"maxiter": 500})
    return res.x[:7], res.x[7:7 + n], res.x[7 + n:]


def walk_forward_ts(matches: pd.DataFrame, eval_seasons: tuple[str, ...], lam: float,
                    s2: pd.DataFrame | None = None) -> pd.DataFrame:
    """Addendum TS: like walk_forward (summer=True), with per-team speeds refitted at the start of every
    evaluated season on all earlier matches. s2 (S2's walk-forward output) seeds theta each season."""
    p, m = prepare(matches)
    out = []
    theta, u, v = START.copy(), None, None
    for s in eval_seasons:
        days = m.loc[m["season"] == s, "matchday"]
        if days.empty:
            continue
        first_day = int(days.min())
        if s2 is not None and (s2["season"] == s).any():
            theta = s2.loc[s2["season"] == s, [f"theta_{k}" for k in NAMES]].iloc[0].to_numpy(float)
        theta, u, v = fit_ts(p, first_day, lam, theta, u, v)
        lamr, _, _ = filter_run(theta, p, u=u, v=v)
        sel = (m["season"] == s).to_numpy()
        P = probs(lamr[sel, 0], lamr[sel, 1], theta[2])
        hi, ai = p.hi[sel], p.ai[sel]
        out.append(pd.DataFrame({"match_id": m.loc[sel, "match_id"].to_numpy(), "season": s,
                                 "kickoff_utc": m.loc[sel, "kickoff_utc"].to_numpy(),
                                 "result": m.loc[sel, "result"].to_numpy(),
                                 "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2],
                                 "uv_home": np.abs(u[hi]) + np.abs(v[hi]), "uv_away": np.abs(u[ai]) + np.abs(v[ai]),
                                 **{f"theta_{k}": val for k, val in zip(NAMES, theta)}}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- addendum NT: teams new to a league

NT_BOUNDS = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 5.0)]          # delta_att, delta_def, kappa
NT_START = np.array([-0.1, -0.1, 0.5])


@dataclass
class NewTeams:
    """Per match (prepare's order): whether each side is new to the league this season, and how many
    league matches it had played this season before the match."""
    new_h: np.ndarray
    new_a: np.ndarray
    n_h: np.ndarray
    n_a: np.ndarray
    returning: dict          # (team_id, season) -> True if the team played in an earlier season of the file


def new_teams(m: pd.DataFrame) -> NewTeams:
    """A team is new in a season if it did not play in the previous season of the file (the first season
    flags nobody). Uses only each season's team list, known before the season starts."""
    seasons = sorted(m["season"].unique())
    teams = {s: set(m.loc[m.season == s, "home_id"]) | set(m.loc[m.season == s, "away_id"]) for s in seasons}
    new, returning, seen = set(), {}, set()
    for prev, cur in zip(seasons, seasons[1:]):
        seen |= teams[prev]
        for t in teams[cur] - teams[prev]:
            new.add((t, cur))
            returning[(t, cur)] = t in seen
    long = pd.concat([m[["match_id", "kickoff_utc", "season", "home_id"]].rename(columns={"home_id": "t"}).assign(side="h"),
                      m[["match_id", "kickoff_utc", "season", "away_id"]].rename(columns={"away_id": "t"}).assign(side="a")])
    long = long.sort_values(["kickoff_utc", "match_id"])
    long["n"] = long.groupby(["season", "t"]).cumcount()
    n = long.pivot(index="match_id", columns="side", values="n").reindex(m["match_id"])
    return NewTeams(np.array([(t, s) in new for t, s in zip(m["home_id"], m["season"])]),
                    np.array([(t, s) in new for t, s in zip(m["away_id"], m["season"])]),
                    n["h"].to_numpy(float), n["a"].to_numpy(float), returning)


def _nt_speed(new: np.ndarray, n: np.ndarray, kappa, N: int):
    """Speed multiplier per match side: 1 + kappa * max(0, 1 - n/N) for new teams, else 1."""
    ramp = np.where(new, np.maximum(0.0, 1.0 - n / N), 0.0)
    return 1.0 + kappa * ramp


def filter_run_nt(theta: np.ndarray, nt: np.ndarray, N: int, p: Prepared, info: NewTeams,
                  reset: bool = True, upto_day: int | None = None) -> tuple[np.ndarray, tuple, float]:
    """S2's filter with NT: a new team's ratings are set to (delta_att, delta_def) before its first
    match of the season (if `reset`), and its speeds are multiplied by 1 + kappa * max(0, 1 - n/N)."""
    mu, h, rho, aa, ad, beta, bs = theta
    da_, dd_, kappa = nt
    a, d = np.zeros(p.n_teams), np.zeros(p.n_teams)
    lam = np.empty((len(p.x), 2))
    ll = 0.0
    last_t, last_s = p.days[0], p.season[0]
    n_days = len(p.days) if upto_day is None else upto_day
    gh_all = _nt_speed(info.new_h, info.n_h, kappa, N)
    ga_all = _nt_speed(info.new_a, info.n_a, kappa, N)
    lb = np.log(beta)
    start = 0
    for k in range(len(p.days)):
        stop = start
        while stop < len(p.day) and p.day[stop] == k:
            stop += 1
        if k > 0:
            f = np.exp(lb * (p.days[k] - last_t) / 7.0) * (bs if p.season[k] != last_s else 1.0)
            a *= f
            d *= f
            last_t, last_s = p.days[k], p.season[k]
        i, j = p.hi[start:stop], p.ai[start:stop]
        if reset:
            rh = info.new_h[start:stop] & (info.n_h[start:stop] == 0)
            ra = info.new_a[start:stop] & (info.n_a[start:stop] == 0)
            a[i[rh]], d[i[rh]] = da_, dd_
            a[j[ra]], d[j[ra]] = da_, dd_
        lh = np.exp(mu + h + a[i] - d[j])
        la = np.exp(mu + a[j] - d[i])
        lam[start:stop, 0], lam[start:stop, 1] = lh, la
        xs, ys = p.x[start:stop], p.y[start:stop]
        if k < n_days:
            tau = np.array([_tau(u, v, s, t, rho) for u, v, s, t in zip(xs, ys, lh, la)])
            ll += float((xs * np.log(lh) - lh + ys * np.log(la) - la + np.log(np.maximum(tau, 1e-10))).sum())
        gh, ga = gh_all[start:stop], ga_all[start:stop]
        np.add.at(a, i, aa * gh * (xs - lh))
        np.add.at(d, j, -ad * ga * (xs - lh))
        np.add.at(a, j, aa * ga * (ys - la))
        np.add.at(d, i, -ad * gh * (ys - la))
        start = stop
    return lam, (a, d), ll


def _nt_loss_torch(p: Prepared, info: NewTeams, N: int, upto_day: int, reset: bool):
    """Differentiable negative filtered log-likelihood on days < upto_day; z = [theta (7), delta_att,
    delta_def, kappa]. Same arithmetic as filter_run_nt."""
    import torch
    T = torch.float64
    stop = int(np.searchsorted(p.day, upto_day, "left"))
    hi, ai = torch.as_tensor(p.hi[:stop].copy()), torch.as_tensor(p.ai[:stop].copy())
    x, y = torch.as_tensor(p.x[:stop].copy(), dtype=T), torch.as_tensor(p.y[:stop].copy(), dtype=T)
    gap = torch.as_tensor(np.diff(p.days[:upto_day], prepend=p.days[0]) / 7.0, dtype=T)
    newseason = torch.as_tensor(np.r_[False, p.season[1:upto_day] != p.season[:upto_day - 1]])
    bounds = np.searchsorted(p.day[:stop], np.arange(upto_day + 1), "left")
    ramp_h = torch.as_tensor(np.where(info.new_h[:stop], np.maximum(0.0, 1.0 - info.n_h[:stop] / N), 0.0), dtype=T)
    ramp_a = torch.as_tensor(np.where(info.new_a[:stop], np.maximum(0.0, 1.0 - info.n_a[:stop] / N), 0.0), dtype=T)
    rh = info.new_h[:stop] & (info.n_h[:stop] == 0)
    ra = info.new_a[:stop] & (info.n_a[:stop] == 0)
    m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
    m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
    n = p.n_teams

    def loss(z):
        mu, h, rho, aa, ad, beta, bs, dta, dtd, kappa = z
        f = torch.exp(torch.log(beta) * gap) * torch.where(newseason, bs, torch.ones((), dtype=T))
        gh, ga = 1 + kappa * ramp_h, 1 + kappa * ramp_a
        a, d = torch.zeros(n, dtype=T), torch.zeros(n, dtype=T)
        lhs, las = [], []
        for k in range(upto_day):
            s0, s1 = bounds[k], bounds[k + 1]
            if k > 0:
                a, d = a * f[k], d * f[k]
            if s0 == s1:
                continue
            i, j = hi[s0:s1], ai[s0:s1]
            if reset:
                idx = np.r_[p.hi[s0:s1][rh[s0:s1]], p.ai[s0:s1][ra[s0:s1]]]
                if len(idx):
                    keep = torch.ones(n, dtype=T)
                    keep[torch.as_tensor(idx)] = 0.0
                    put = 1.0 - keep
                    a, d = a * keep + dta * put, d * keep + dtd * put
            lh = torch.exp(mu + h + a[i] - d[j])
            la = torch.exp(mu + a[j] - d[i])
            lhs.append(lh)
            las.append(la)
            xs, ys = x[s0:s1], y[s0:s1]
            g_h, g_a = gh[s0:s1], ga[s0:s1]
            a = a.index_add(0, i, aa * g_h * (xs - lh)).index_add(0, j, aa * g_a * (ys - la))
            d = d.index_add(0, j, -ad * g_a * (xs - lh)).index_add(0, i, -ad * g_h * (ys - la))
        lh, la = torch.cat(lhs), torch.cat(las)
        tau = torch.ones_like(lh)
        tau = torch.where(m00, 1 - lh * la * rho, tau)
        tau = torch.where(m01, 1 + lh * rho, tau)
        tau = torch.where(m10, 1 + la * rho, tau)
        tau = torch.where(m11, 1 - rho, tau)
        return -(x * torch.log(lh) - lh + y * torch.log(la) - la + torch.log(torch.clamp(tau, min=1e-10))).sum()
    return loss


def fit_nt(p: Prepared, info: NewTeams, N: int, upto_day: int, theta0: np.ndarray = START,
           nt0: np.ndarray = NT_START, variant: str = "both") -> tuple[np.ndarray, np.ndarray]:
    """NT's static parameters on days < upto_day. variant: "both", "entry" (kappa = 0) or "speed" (no reset)."""
    import torch
    loss = _nt_loss_torch(p, info, N, upto_day, reset=variant != "speed")
    nb = list(NT_BOUNDS)
    if variant == "entry":
        nb[2] = (0.0, 0.0)
    if variant == "speed":
        nb[0] = nb[1] = (0.0, 0.0)
    b = BOUNDS + nb
    z0 = np.clip(np.r_[theta0, nt0], [x[0] for x in b], [x[1] for x in b])

    def f(z):
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        val = loss(zt)
        val.backward()
        return float(val.detach()), zt.grad.numpy().copy()
    res = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": 500})
    return res.x[:7], res.x[7:]


def walk_forward_nt(matches: pd.DataFrame, eval_seasons: tuple[str, ...], N: int, variant: str = "both",
                    s2: pd.DataFrame | None = None) -> pd.DataFrame:
    """Addendum NT, like walk_forward: static parameters refitted at the start of each evaluated season on
    all earlier matches; s2 (S2's walk-forward output) seeds theta each season."""
    p, m = prepare(matches)
    info = new_teams(m)
    out, theta, nt = [], START.copy(), NT_START.copy()
    for s in eval_seasons:
        days = m.loc[m["season"] == s, "matchday"]
        if days.empty:
            continue
        if s2 is not None and (s2["season"] == s).any():
            theta = s2.loc[s2["season"] == s, [f"theta_{k}" for k in NAMES]].iloc[0].to_numpy(float)
        theta, nt = fit_nt(p, info, N, int(days.min()), theta, nt, variant)
        lam, _, _ = filter_run_nt(theta, nt, N, p, info, reset=variant != "speed")
        sel = (m["season"] == s).to_numpy()
        P = probs(lam[sel, 0], lam[sel, 1], theta[2])
        out.append(pd.DataFrame({"match_id": m.loc[sel, "match_id"].to_numpy(), "season": s,
                                 "kickoff_utc": m.loc[sel, "kickoff_utc"].to_numpy(), "result": m.loc[sel, "result"].to_numpy(),
                                 "p_home": P[:, 0], "p_draw": P[:, 1], "p_away": P[:, 2],
                                 "new_h": info.new_h[sel], "new_a": info.new_a[sel], "n_h": info.n_h[sel], "n_a": info.n_a[sel],
                                 "returning_h": [info.returning.get((t, s), False) for t in m.loc[sel, "home_id"]],
                                 "returning_a": [info.returning.get((t, s), False) for t in m.loc[sel, "away_id"]],
                                 **{f"theta_{k}": v for k, v in zip(NAMES, theta)},
                                 "delta_att": nt[0], "delta_def": nt[1], "kappa": nt[2]}))
    return pd.concat(out, ignore_index=True)
