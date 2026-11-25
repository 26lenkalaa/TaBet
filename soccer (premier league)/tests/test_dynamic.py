import numpy as np
import pandas as pd

from tabet import dynamic as D


def _league(n_rounds=6, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(6)]
    rows, t0 = [], pd.Timestamp("2020-08-15 14:00", tz="UTC")
    for r in range(n_rounds):
        perm = rng.permutation(6)
        for k in range(3):
            h, a = teams[perm[2 * k]], teams[perm[2 * k + 1]]
            ko = t0 + pd.Timedelta(days=7 * r, hours=3 * k)  # three matches on the same local day
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            rows.append({"match_id": f"m{r}{k}", "season": "2020-21", "kickoff_utc": ko, "home_id": h, "away_id": a,
                         "home_goals": hg, "away_goals": ag, "status": "completed",
                         "result": "H" if hg > ag else ("A" if ag > hg else "D")})
    return pd.DataFrame(rows)


def test_predictions_use_only_earlier_matchdays():
    m = _league()
    th = D.START.copy()
    p, mm = D.prepare(m)
    lam, _, _ = D.filter_run(th, p)
    m2 = m.copy()
    m2.loc[m2["match_id"] == "m31", ["home_goals", "away_goals"]] = [7, 0]  # change one round-3 result
    lam2, _, _ = D.filter_run(th, D.prepare(m2)[0])
    day = mm["matchday"].to_numpy()
    k = int(mm.loc[mm["match_id"] == "m31", "matchday"].iloc[0])
    assert np.allclose(lam[day <= k], lam2[day <= k])  # same-day and earlier predictions unchanged
    assert not np.allclose(lam[day > k], lam2[day > k])  # later ones do react


def test_full_summer_reset_starts_the_season_at_league_average():
    m = _league()
    m["season"] = np.where(m.index >= 9, "2021-22", "2020-21")  # rounds 3-5 are a new season
    th = D.START.copy()
    th[3] = th[4] = 0.1
    th[6] = 0.0  # beta_summer = 0: ratings fully reset over the summer
    p, mm = D.prepare(m)
    lam, _, _ = D.filter_run(th, p)
    first_new = (mm["season"] == "2021-22").to_numpy() & (mm["matchday"] == mm.loc[mm.season == "2021-22", "matchday"].min()).to_numpy()
    assert np.allclose(lam[first_new, 0], np.exp(th[0] + th[1]))
    assert np.allclose(lam[first_new, 1], np.exp(th[0]))
    th[6] = 1.0
    lam1, _, _ = D.filter_run(th, p)
    assert not np.allclose(lam1[first_new, 0], np.exp(th[0] + th[1]))


# ---- addendum TS: team-specific speeds

def _two_seasons():
    m = pd.concat([_league(8, 1), _league(8, 2).assign(match_id=lambda d: "n" + d.match_id)], ignore_index=True)
    m.loc[24:, "season"] = "2021-22"
    m.loc[24:, "kickoff_utc"] = m.loc[24:, "kickoff_utc"] + pd.Timedelta(days=365)
    return m


def test_ts_torch_loss_matches_the_numpy_filter():
    import torch
    p, _ = D.prepare(_two_seasons())
    rng = np.random.default_rng(3)
    u, v = rng.normal(0, 0.5, p.n_teams), rng.normal(0, 0.5, p.n_teams)
    th = D.START.copy()
    th[3], th[4] = 0.08, 0.06
    upto = int(p.day.max())       # every day but the last
    want = D.filter_run(th, p, upto, u=u, v=v)[2]
    got = -float(D._ts_loss_torch(p, upto)(torch.tensor(np.r_[th, u, v], dtype=torch.float64)))
    assert np.isclose(got, want, rtol=1e-10)
    assert np.isclose(D.filter_run(th, p, upto, u=0 * u, v=0 * v)[2], D.filter_run(th, p, upto)[2])  # u = v = 0 is S2


def test_ts_predictions_use_only_earlier_matchdays():
    m = _league()
    p, mm = D.prepare(m)
    u, v = np.linspace(-1, 1, p.n_teams), np.linspace(1, -1, p.n_teams)
    lam, _, _ = D.filter_run(D.START, p, u=u, v=v)
    m2 = m.copy()
    m2.loc[m2["match_id"] == "m31", ["home_goals", "away_goals"]] = [7, 0]
    lam2, _, _ = D.filter_run(D.START, D.prepare(m2)[0], u=u, v=v)
    day = mm["matchday"].to_numpy()
    k = int(mm.loc[mm["match_id"] == "m31", "matchday"].iloc[0])
    assert np.allclose(lam[day <= k], lam2[day <= k])
    assert not np.allclose(lam[day > k], lam2[day > k])


def test_ts_fit_keeps_unseen_teams_at_the_shared_speed_and_shrinks_with_lambda():
    m = _two_seasons()
    m.loc[m.index >= 24, "away_id"] = m.loc[m.index >= 24, "away_id"].replace("t5", "newbie")
    p, mm = D.prepare(m)
    first = int(mm.loc[mm.season == "2021-22", "matchday"].min())
    th, u, v = D.fit_ts(p, first, lam=1.0)
    newbie = sorted(set(m.home_id) | set(m.away_id)).index("newbie")
    assert u[newbie] == 0 and v[newbie] == 0
    _, u2, v2 = D.fit_ts(p, first, lam=1000.0)
    assert np.abs(np.r_[u2, v2]).sum() < np.abs(np.r_[u, v]).sum()
    assert np.all(np.abs(np.r_[u, v]) <= D.UV_BOUND + 1e-12)
