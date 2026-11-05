import numpy as np
import pandas as pd
import pytest

from tabet.features import FeatureParams, known, match_features, player_ratings
from tabet.fpl import _infer_missing_starts, _null_unpublished_xg

T0 = pd.Timestamp("2023-01-01 15:00", tz="UTC")


def _apps(n_matches: int = 6, start: pd.Timestamp = T0) -> pd.DataFrame:
    """Synthetic appearances: teams a and b play each other weekly, 11 starters each."""
    rows = []
    for k in range(n_matches):
        ko = start + pd.Timedelta(days=7 * k)
        mid = f"m{k}"
        for team, opp, venue in (("a", "b", "H"), ("b", "a", "A")):
            for j in range(12):
                pos = "GK" if j == 0 else ("DEF" if j < 5 else ("MID" if j < 9 else "FWD"))
                rows.append({"match_id": mid, "season": "2022-23", "kickoff_utc": ko, "team_id": team,
                             "opponent_id": opp, "venue": venue, "player_code": f"{team}{j}",
                             "player_name": f"{team}{j}", "position": pos,
                             "minutes": 90 if j < 11 else 20, "started": float(j < 11),
                             "started_inferred": False, "goals": 0.0, "assists": 0.0,
                             "xg": 0.05 * j, "xa": 0.02 * j, "xgc": 1.0, "threat": 10.0 * j,
                             "creativity": 5.0 * j, "influence": 8.0, "def_contrib": np.nan})
    return pd.DataFrame(rows)


def _matches(app: pd.DataFrame) -> pd.DataFrame:
    g = app.groupby("match_id").first()
    return pd.DataFrame({"match_id": g.index, "kickoff_utc": g["kickoff_utc"].to_numpy(), "season": "2022-23",
                         "home_id": "a", "away_id": "b", "status": "completed",
                         **{c: 10 for c in ("home_shots", "away_shots", "home_corners", "away_corners",
                                             "home_fouls", "away_fouls")}})


def test_future_appearances_do_not_change_ratings_or_features():
    app = _apps(6)
    cutoff = T0 + pd.Timedelta(days=21)  # after m0..m2, before m3
    future = _apps(4, start=T0 + pd.Timedelta(days=21)).assign(
        match_id=lambda d: "f" + d["match_id"], xg=9.9, xa=9.9)  # extreme, should be invisible
    p = FeatureParams()
    a1 = player_ratings(known(app, cutoff), cutoff, p)
    a2 = player_ratings(known(pd.concat([app, future]), cutoff), cutoff, p)
    pd.testing.assert_frame_equal(a1, a2)

    fx = _matches(app).iloc[[3]]
    hist_m = _matches(app).iloc[:3]
    f1 = match_features(fx, app[app.match_id == "m3"], known(app, cutoff), hist_m, cutoff, p)
    f2 = match_features(fx, app[app.match_id == "m3"], known(pd.concat([app, future]), cutoff), hist_m, cutoff, p)
    pd.testing.assert_frame_equal(f1, f2)


def test_known_respects_result_lag():
    app = _apps(2)
    assert known(app, T0 + pd.Timedelta(minutes=100))["match_id"].nunique() == 0
    assert known(app, T0 + pd.Timedelta(hours=2, minutes=30))["match_id"].nunique() == 1


def test_shrinkage_pulls_low_minute_players_to_prior():
    app = _apps(6)
    r = player_ratings(app, T0 + pd.Timedelta(days=60), FeatureParams())
    sub = r.loc["a11"]  # 20-minute sub each week
    starter = r.loc["a10"]
    assert abs(sub["rating"] - sub["prior"]) < abs(sub["rate_raw"] - sub["prior"])
    assert abs(starter["rating"] - starter["rate_raw"]) < abs(sub["rating"] - sub["rate_raw"])


def test_unchanged_xi_gives_zero_xi_feature():
    app = _apps(6)
    cutoff = T0 + pd.Timedelta(days=35)
    f = match_features(_matches(app).iloc[[5]], app[app.match_id == "m5"], known(app, cutoff),
                       _matches(app).iloc[:5], cutoff, FeatureParams())
    assert f["f_xi_position"].abs().max() < 0.05  # same XI as always -> ~usual strength


def test_missing_starts_inferred_from_minutes():
    app = _apps(1).assign(started=0.0)
    out = _infer_missing_starts(app)
    assert out.groupby("team_id")["started"].sum().eq(11).all()
    assert out["started_inferred"].all()
    assert out.loc[out.minutes == 20, "started"].eq(0).all()


def test_unpublished_xg_nulled_but_real_zero_kept():
    app = _apps(2)
    app.loc[app.match_id == "m0", ["xg", "xa", "xgc"]] = 0.0  # unpublished match
    app.loc[(app.match_id == "m1") & (app.team_id == "a"), ["xg", "xa"]] = 0.0  # a genuinely had 0 xG
    out = _null_unpublished_xg(app)
    assert out.loc[out.match_id == "m0", "xg"].isna().all()
    assert out.loc[(out.match_id == "m1") & (out.team_id == "a"), "xg"].eq(0).all()
