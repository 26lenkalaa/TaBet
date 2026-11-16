"""Offline tests for the Elo model and predict.py, using made-up games."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import predict  # noqa: E402
from src.models.elo import HOME_ADVANTAGE, START_RATING, Elo, rate_games  # noqa: E402

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}


def make_games(results):
    """results: (date, season, home_id, away_id, home_pts, away_pts) per game."""
    rows = []
    for number, (date, season, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00218{number:05d}", "season": season, "date": date,
            "game_type": "regular", "is_playoffs": False,
            "home_team_id": home, "home": NAMES[home],
            "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def games():
    return make_games([
        ("2018-10-16", "2018-19", BOS, NYK, 110, 95),
        ("2018-10-17", "2018-19", LAL, GSW, 101, 104),
        ("2018-10-19", "2018-19", NYK, LAL, 99, 108),
        ("2018-10-20", "2018-19", GSW, BOS, 120, 111),
        ("2018-10-22", "2018-19", BOS, LAL, 105, 100),
        ("2019-10-22", "2019-20", NYK, GSW, 90, 118),
        ("2019-10-24", "2019-20", LAL, BOS, 112, 107),
        ("2019-10-26", "2019-20", GSW, NYK, 125, 100),
    ])


# --- the Elo rules ---------------------------------------------------------

def test_equal_teams_on_a_neutral_court_are_a_coin_flip():
    prob, margin = Elo().predict(BOS, NYK, neutral=True)
    assert prob == pytest.approx(0.5)
    assert margin == pytest.approx(0.0)


def test_home_court_favors_the_home_team():
    prob, margin = Elo().predict(BOS, NYK)
    assert prob > 0.5
    assert 2.0 < margin < 3.0   # home court is worth about 2-3 points


def test_the_two_win_probabilities_add_up_to_one():
    model = Elo(ratings={BOS: 1620.0, NYK: 1430.0})
    bos_wins, bos_margin = model.predict(BOS, NYK, neutral=True)
    nyk_wins, nyk_margin = model.predict(NYK, BOS, neutral=True)
    assert bos_wins + nyk_wins == pytest.approx(1.0)
    assert bos_margin == pytest.approx(-nyk_margin)


def test_the_winner_gains_exactly_what_the_loser_loses():
    model = Elo()
    model.update(BOS, NYK, 110, 95)
    assert model.rating(BOS) > START_RATING > model.rating(NYK)
    assert model.rating(BOS) + model.rating(NYK) == pytest.approx(2 * START_RATING)


def test_an_away_win_moves_ratings_the_other_way():
    model = Elo()
    model.update(BOS, NYK, 95, 110)
    assert model.rating(NYK) > START_RATING > model.rating(BOS)


def test_a_bigger_win_moves_ratings_more():
    close, blowout = Elo(), Elo()
    close.update(BOS, NYK, 101, 100)
    blowout.update(BOS, NYK, 130, 100)
    assert blowout.rating(BOS) > close.rating(BOS)


def test_an_upset_moves_ratings_more_than_an_expected_win():
    expected = Elo(ratings={BOS: 1650.0, NYK: 1400.0})
    upset = Elo(ratings={BOS: 1650.0, NYK: 1400.0})
    expected.update(BOS, NYK, 110, 100)
    upset.update(BOS, NYK, 100, 110)
    assert abs(upset.rating(NYK) - 1400.0) > abs(expected.rating(BOS) - 1650.0)


def test_a_new_season_pulls_ratings_back_toward_1500():
    model = Elo(ratings={BOS: 1700.0, NYK: 1300.0})
    model.start_new_season()
    assert model.rating(BOS) == pytest.approx(1650.0)
    assert model.rating(NYK) == pytest.approx(1350.0)


# --- ratings only use earlier games ---------------------------------------

def test_the_first_game_is_predicted_from_starting_ratings(games):
    rated, _ = rate_games(games)
    first = rated.iloc[0]
    assert first["home_elo"] == START_RATING
    assert first["away_elo"] == START_RATING
    assert first["home_win_prob"] == pytest.approx(1 / (1 + 10 ** (-HOME_ADVANTAGE / 400)))


def test_a_games_own_result_does_not_change_its_prediction(games):
    flipped = games.copy()
    flipped.loc[4, ["home_pts", "away_pts"]] = [80, 130]   # game 5: turn a home win into a blowout loss
    original, _ = rate_games(games)
    changed, _ = rate_games(flipped)
    columns = ["home_elo", "away_elo", "home_win_prob", "predicted_margin"]
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
        assert game["home_elo"] == pytest.approx(model.rating(game["home_team_id"]))
        assert game["away_elo"] == pytest.approx(model.rating(game["away_team_id"]))


def test_games_are_replayed_in_date_order_whatever_order_they_arrive_in(games):
    expected, _ = rate_games(games)
    actual, _ = rate_games(games.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(actual, expected)


def test_ratings_regress_when_the_season_changes(games):
    rated, _ = rate_games(games)
    _, after_first_season = rate_games(games.iloc[:5])
    carried = START_RATING + 0.75 * (after_first_season.rating(GSW) - START_RATING)
    assert rated.loc[5, "away_elo"] == pytest.approx(carried)   # GSW's first game of 2019-20


def test_total_rating_stays_constant(games):
    _, model = rate_games(games)
    assert sum(model.ratings.values()) == pytest.approx(4 * START_RATING)


def test_a_team_that_keeps_winning_ends_on_top():
    results = [(f"2018-11-{day:02d}", "2018-19", BOS if day % 2 else NYK, NYK if day % 2 else BOS,
                110 if day % 2 else 95, 95 if day % 2 else 110) for day in range(1, 21)]
    _, model = rate_games(make_games(results))
    assert model.rating(BOS) > 1600 > 1400 > model.rating(NYK)


def test_rate_games_keeps_every_game_and_adds_four_columns(games):
    rated, _ = rate_games(games)
    assert len(rated) == len(games)
    assert list(rated.columns) == list(games.columns) + ["home_elo", "away_elo", "home_win_prob", "predicted_margin"]
    assert rated["home_win_prob"].between(0, 1).all()


# --- predict.py ------------------------------------------------------------

def test_predict_game_uses_the_ratings_after_the_last_game(games):
    _, model = rate_games(games)
    prediction = predict.predict_game(games, "BOS", "NYK")
    prob, margin = model.predict(BOS, NYK)
    assert prediction.home_win_prob == pytest.approx(prob)
    assert prediction.predicted_margin == pytest.approx(margin)
    assert prediction.last_game_date == "2019-10-26"


def test_predict_game_accepts_lower_case_and_neutral_courts(games):
    home = predict.predict_game(games, "bos", "nyk")
    neutral = predict.predict_game(games, "bos", "nyk", neutral=True)
    assert home.home == "BOS"
    assert neutral.home_win_prob < home.home_win_prob


def test_predict_game_rejects_unknown_and_repeated_teams(games):
    with pytest.raises(ValueError, match="Unknown team: XXX"):
        predict.predict_game(games, "XXX", "NYK")
    with pytest.raises(ValueError, match="different teams"):
        predict.predict_game(games, "BOS", "BOS")


def test_command_line_prints_win_probability_and_margin(games, tmp_path, capsys):
    games_file = tmp_path / "games.csv"
    games.to_csv(games_file, index=False)

    assert predict.main(["BOS", "NYK", "--games", str(games_file)]) == 0

    output = capsys.readouterr().out
    assert "BOS win probability:" in output
    assert "NYK win probability:" in output
    assert "Predicted margin: BOS by" in output


def test_command_line_explains_a_missing_games_file(tmp_path, capsys):
    assert predict.main(["BOS", "NYK", "--games", str(tmp_path / "missing.csv")]) == 1
    assert "clean.py" in capsys.readouterr().err


def test_command_line_explains_an_unknown_team(games, tmp_path, capsys):
    games_file = tmp_path / "games.csv"
    games.to_csv(games_file, index=False)
    assert predict.main(["BOS", "ZZZ", "--games", str(games_file)]) == 1
    assert "Unknown team: ZZZ" in capsys.readouterr().err
