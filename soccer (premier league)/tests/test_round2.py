import numpy as np
import pandas as pd
import pytest

from tabet import outcome_model
from tabet.adaptation import add_scoring_environment
from tabet.lineup import add_lineup_features
from tests.test_features import T0, _apps


def _rows_for(app, mids, cutoff):
    g = app[app.match_id.isin(mids)].groupby(["match_id", "team_id", "opponent_id", "venue"], as_index=False).first()
    return pd.DataFrame({"match_id": g.match_id, "team_id": g.team_id, "opponent_id": g.opponent_id,
                         "side": g.venue, "cutoff_utc": cutoff})


def test_unchanged_xi_has_zero_departure_and_full_continuity():
    app = _apps(6)
    cutoff = T0 + pd.Timedelta(days=35)
    f = add_lineup_features(_rows_for(app, ["m5"], cutoff), app)
    assert f[["l_dep_fwd", "l_dep_mid", "l_dep_def"]].abs().max().max() < 1e-9
    assert np.allclose(f["l_own_cont"], 1) and np.allclose(f["l_opp_gk_change"], 0)


def test_dropped_forward_gives_negative_departure_and_future_is_invisible():
    app = _apps(6)
    app.loc[app.player_code == "a11", ["xg", "xa"]] = 0.0  # the usual sub never produces
    # in m5, team a's best forward (a10) is benched and the sub (a11) starts
    sel = (app.match_id == "m5") & (app.team_id == "a")
    app.loc[sel & (app.player_code == "a10"), "started"] = 0.0
    app.loc[sel & (app.player_code == "a11"), "started"] = 1.0
    cutoff = T0 + pd.Timedelta(days=35)
    rows = _rows_for(app, ["m5"], cutoff)
    f = add_lineup_features(rows, app)
    assert f.loc[f.team_id == "a", "l_dep_fwd"].iloc[0] < 0
    assert f.loc[f.team_id == "b", "l_dep_fwd"].iloc[0] == pytest.approx(0, abs=1e-9)
    future = _apps(3, start=T0 + pd.Timedelta(days=42)).assign(match_id=lambda d: "f" + d.match_id, xg=5.0)
    f2 = add_lineup_features(rows, pd.concat([app, future]))
    pd.testing.assert_frame_equal(f, f2)


def test_scoring_environment_zero_when_constant_and_signed_by_side():
    ko = pd.Timestamp("2023-01-01", tz="UTC") + pd.to_timedelta(np.arange(200), unit="D")
    m = pd.DataFrame({"kickoff_utc": ko, "status": "completed", "home_goals": 2, "away_goals": 1})
    rows = pd.DataFrame({"cutoff_utc": [ko[-1] + pd.Timedelta(days=1)] * 2, "side": ["H", "A"]})
    f = add_scoring_environment(rows, m, (30.0,))
    assert np.allclose(f[["f_lg_30", "f_home_30"]], 0)
    m2 = m.copy()
    m2.loc[150:, ["home_goals", "away_goals"]] = [4, 1]  # recent spike in home scoring
    f2 = add_scoring_environment(rows, m2, (30.0,))
    assert f2["f_lg_30"].iloc[0] > 0
    assert f2["f_home_30"].iloc[0] > 0 > f2["f_home_30"].iloc[1]


def test_outcome_model_defaults_to_dc_and_learns_a_shift():
    rng = np.random.default_rng(0)
    n = 3000
    P = rng.dirichlet([4, 3, 3], size=n)
    x = rng.normal(size=(n, 1))
    logp = np.log(P)
    # truth: home logit shifted by +0.5 * x
    z = logp.copy(); z[:, 0] += 0.5 * x[:, 0]
    Q = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=q) for q in Q])
    theta_big = outcome_model.fit(logp, x, y, l2=1e7)
    np.testing.assert_allclose(outcome_model.apply(logp, x, theta_big), P, atol=1e-4)
    theta = outcome_model.fit(logp, x, y, l2=0.01)
    assert theta[1] == pytest.approx(0.5, abs=0.12)  # b_H on x
    assert abs(theta[3]) < 0.15  # b_A on x ~ 0
