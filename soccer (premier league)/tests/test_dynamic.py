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
