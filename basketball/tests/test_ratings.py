"""Offline tests for the efficiency model (v2), using made-up games."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src.models.ratings import (HOME_EDGE, LEAGUE_EFFICIENCY, LEAGUE_PACE, LEARNING_RATE, MARGIN_SD,  # noqa: E402
                                Efficiency, rate_games)

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}


def make_games(results):
    """results: (date, season, home_id, away_id, home_pts, away_pts, possessions) per game."""
    rows = []
    for number, (date, season, home, away, home_pts, away_pts, possessions) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00218{number:05d}", "season": season, "date": date,
            "game_type": "regular", "is_playoffs": False,
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
            "home_possessions": possessions, "away_possessions": possessions,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def games():
    return make_games([
        ("2018-10-16", "2018-19", BOS, NYK, 110, 95, 98.0),
        ("2018-10-17", "2018-19", LAL, GSW, 101, 104, 103.0),
        ("2018-10-19", "2018-19", NYK, LAL, 99, 108, 100.0),
        ("2018-10-20", "2018-19", GSW, BOS, 120, 111, 104.0),
        ("2018-10-22", "2018-19", BOS, LAL, 105, 100, 96.0),
        ("2019-10-22", "2019-20", NYK, GSW, 90, 118, 101.0),
        ("2019-10-24", "2019-20", LAL, BOS, 112, 107, 99.0),
        ("2019-10-26", "2019-20", GSW, NYK, 125, 100, 105.0),
    ])


# --- predicting ------------------------------------------------------------------

def test_average_teams_on_a_neutral_court_are_a_coin_flip():
    model = Efficiency()
    prob, margin = model.predict(BOS, NYK, neutral=True)
    assert prob == pytest.approx(0.5)
    assert margin == pytest.approx(0.0)
    assert model.predict_score(BOS, NYK, neutral=True) == pytest.approx(
        (LEAGUE_EFFICIENCY * LEAGUE_PACE / 100,) * 2)


def test_home_court_favors_the_home_team():
    prob, margin = Efficiency().predict(BOS, NYK)
    assert prob > 0.5
    assert margin == pytest.approx(HOME_EDGE * LEAGUE_PACE / 100)
    assert 2.0 < margin < 3.0   # home court is worth about 2-3 points


def test_offense_meets_the_other_teams_defense():
    model = Efficiency(offense={BOS: 5.0}, defense={NYK: 3.0})
    home_efficiency, away_efficiency, _ = model.expected(BOS, NYK, neutral=True)
    assert home_efficiency == pytest.approx(LEAGUE_EFFICIENCY + 5.0 + 3.0)   # good offense against a leaky defense
    assert away_efficiency == pytest.approx(LEAGUE_EFFICIENCY)


def test_a_faster_expected_pace_means_more_points_for_both_teams():
    slow = Efficiency(pace={BOS: -3.0, NYK: -2.0}).predict_score(BOS, NYK, neutral=True)
    fast = Efficiency(pace={BOS: 3.0, NYK: 2.0}).predict_score(BOS, NYK, neutral=True)
    assert fast[0] > slow[0] and fast[1] > slow[1]
    assert fast[0] == pytest.approx(LEAGUE_EFFICIENCY * (LEAGUE_PACE + 5) / 100)


def test_win_probability_is_the_normal_curve_around_the_margin():
    model = Efficiency(offense={BOS: 13.5 / (LEAGUE_PACE / 100)}, margin_sd=13.5)   # exactly one sd better
    prob, margin = model.predict(BOS, NYK, neutral=True)
    assert margin == pytest.approx(13.5)
    assert prob == pytest.approx(0.8413, abs=1e-4)


def test_swapping_the_teams_on_a_neutral_court_mirrors_the_prediction():
    model = Efficiency(offense={BOS: 4.0, NYK: -2.0}, defense={BOS: -1.5, NYK: 2.5})
    bos_prob, bos_margin = model.predict(BOS, NYK, neutral=True)
    nyk_prob, nyk_margin = model.predict(NYK, BOS, neutral=True)
    assert bos_prob + nyk_prob == pytest.approx(1.0)
    assert bos_margin == pytest.approx(-nyk_margin)


def test_net_rating_is_offense_minus_defense():
    model = Efficiency(offense={BOS: 4.0}, defense={BOS: -1.5})
    assert model.net_rating(BOS) == pytest.approx(5.5)
    assert model.net_rating(NYK) == 0.0


# --- updating --------------------------------------------------------------------

def test_scoring_more_than_expected_raises_the_offense_and_the_opposing_defense():
    model = Efficiency()
    home_expected, away_expected, _ = model.expected(BOS, NYK)
    model.update(BOS, NYK, 125, 100, 100.0, 100.0)   # BOS scored 125 per 100 possessions
    assert model.offense[BOS] == pytest.approx(LEARNING_RATE * (125 - home_expected))
    assert model.defense[NYK] == pytest.approx(model.offense[BOS])   # NYK's defense takes the same blame
    assert model.offense[NYK] == pytest.approx(LEARNING_RATE * (100 - away_expected))
    assert model.net_rating(BOS) > 0 > model.net_rating(NYK)


def test_a_result_exactly_as_expected_changes_nothing():
    model = Efficiency()
    home_efficiency, away_efficiency, possessions = model.expected(BOS, NYK)
    model.update(BOS, NYK, home_efficiency * possessions / 100, away_efficiency * possessions / 100,
                 possessions, possessions)
    assert model.offense[BOS] == pytest.approx(0.0)
    assert model.defense[BOS] == pytest.approx(0.0)
    assert model.pace[BOS] == pytest.approx(0.0)
    assert model.home_edge == pytest.approx(HOME_EDGE)


def test_a_fast_game_raises_both_teams_pace():
    model = Efficiency()
    model.update(BOS, NYK, 120, 118, 109.0, 109.0)
    assert model.pace[BOS] == pytest.approx(LEARNING_RATE * (109.0 - LEAGUE_PACE))
    assert model.pace[NYK] == pytest.approx(model.pace[BOS])
    assert model.league_pace > LEAGUE_PACE


def test_the_two_possession_estimates_are_averaged():
    one, two = Efficiency(), Efficiency()
    one.update(BOS, NYK, 110, 100, 98.0, 102.0)
    two.update(BOS, NYK, 110, 100, 100.0, 100.0)
    assert one.offense[BOS] == pytest.approx(two.offense[BOS])


def test_a_game_without_a_box_score_leaves_the_ratings_alone():
    model = Efficiency()
    model.update(BOS, NYK, 110, 100, float("nan"), float("nan"))
    assert model.offense == {} and model.defense == {} and model.pace == {}
    assert model.league_efficiency == LEAGUE_EFFICIENCY


def test_big_misses_widen_the_bell_curve_and_small_ones_narrow_it():
    wide, narrow = Efficiency(), Efficiency()
    wide.update(BOS, NYK, 140, 90, 100.0, 100.0)
    narrow.update(BOS, NYK, 103, 100, 100.0, 100.0)
    assert wide.margin_sd > MARGIN_SD > narrow.margin_sd


def test_home_teams_winning_by_more_than_expected_raises_the_home_edge():
    model = Efficiency()
    model.update(BOS, NYK, 120, 100, 100.0, 100.0)
    assert model.home_edge > HOME_EDGE


def test_a_new_season_pulls_ratings_back_toward_average():
    model = Efficiency(offense={BOS: 8.0}, defense={BOS: -4.0}, pace={BOS: 2.0})
    model.start_new_season()
    assert (model.offense[BOS], model.defense[BOS], model.pace[BOS]) == pytest.approx((6.0, -3.0, 1.5))


def test_a_team_that_keeps_winning_big_ends_with_the_best_net_rating():
    results = [(f"2018-11-{day:02d}", "2018-19", BOS if day % 2 else NYK, NYK if day % 2 else BOS,
                118 if day % 2 else 98, 98 if day % 2 else 118, 100.0) for day in range(1, 29)]
    _, model = rate_games(make_games(results))
    assert model.net_rating(BOS) > 5 > -5 > model.net_rating(NYK)
    assert model.predict(BOS, NYK, neutral=True)[0] > 0.7


# --- ratings only use earlier games ---------------------------------------------

def test_the_first_game_is_predicted_from_starting_values(games):
    rated, _ = rate_games(games)
    prob, margin = Efficiency().predict(BOS, NYK)
    assert rated.loc[0, "home_win_prob"] == pytest.approx(prob)
    assert rated.loc[0, "predicted_margin"] == pytest.approx(margin)


def test_a_games_own_result_does_not_change_its_prediction(games):
    flipped = games.copy()
    flipped.loc[4, ["home_pts", "away_pts", "home_possessions", "away_possessions"]] = [80, 130, 110.0, 110.0]
    original, _ = rate_games(games)
    changed, _ = rate_games(flipped)
    columns = ["home_win_prob", "predicted_margin", "predicted_home_pts", "predicted_away_pts"]
    pd.testing.assert_frame_equal(changed.loc[:4, columns], original.loc[:4, columns])
    assert not changed.loc[5:, columns].equals(original.loc[5:, columns])   # later games do change


def test_later_games_do_not_change_earlier_predictions(games):
    """Each game's prediction must match what you get from only the games before it."""
    rated, _ = rate_games(games)
    for position in range(len(games)):
        _, model = rate_games(games.iloc[:position])
        game = rated.iloc[position]
        if position and game["season"] != rated.iloc[position - 1]["season"]:
            model.start_new_season()
        prob, margin = model.predict(game["home_team_id"], game["away_team_id"])
        assert game["home_win_prob"] == pytest.approx(prob)
        assert game["predicted_margin"] == pytest.approx(margin)


def test_games_are_replayed_in_date_order_whatever_order_they_arrive_in(games):
    expected, _ = rate_games(games)
    actual, _ = rate_games(games.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(actual, expected)


def test_predicted_scores_and_margin_agree(games):
    rated, _ = rate_games(games)
    assert list(rated["predicted_home_pts"] - rated["predicted_away_pts"]) == pytest.approx(
        list(rated["predicted_margin"]))
    assert rated["home_win_prob"].between(0, 1).all()
    assert len(rated) == len(games)
