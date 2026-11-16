"""Offline tests for predict.py, using made-up games in a temporary database."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import db, predict  # noqa: E402
from src.models import elo, ratings, situational  # noqa: E402

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}


@pytest.fixture
def games():
    results = [
        ("2018-10-16", "2018-19", BOS, NYK, 110, 95),
        ("2018-10-17", "2018-19", LAL, GSW, 101, 104),
        ("2018-10-19", "2018-19", NYK, LAL, 99, 108),
        ("2018-10-20", "2018-19", GSW, BOS, 120, 111),
        ("2018-10-22", "2018-19", BOS, LAL, 105, 100),
        ("2019-10-22", "2019-20", NYK, GSW, 90, 118),
        ("2019-10-24", "2019-20", LAL, BOS, 112, 107),
        ("2019-10-26", "2019-20", GSW, NYK, 125, 100),
    ]
    rows = []
    for number, (date, season, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00218{number:05d}", "season": season, "date": date,
            "game_type": "regular", "is_playoffs": False,
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
            "home_fga": 88, "home_fta": 22, "home_oreb": 10, "home_tov": 13,
            "away_fga": 86, "away_fta": 24, "away_oreb": 9, "away_tov": 14,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def db_file(games, tmp_path):
    file = tmp_path / "basketball.db"
    db.build(db.connect(file), games)
    return file


def test_the_efficiency_model_can_be_chosen(db_file):
    games = predict.load_games(db_file)
    _, model = ratings.rate_games(games)
    prediction = predict.predict_game(games, "BOS", "NYK", model="efficiency")
    prob, margin = model.predict(BOS, NYK)
    assert prediction.model == "efficiency"
    assert prediction.home_win_prob == pytest.approx(prob)
    assert prediction.predicted_margin == pytest.approx(margin)
    assert prediction.home_rating == pytest.approx(model.net_rating(BOS))
    assert prediction.predicted_home_pts - prediction.predicted_away_pts == pytest.approx(margin)
    assert prediction.last_game_date == "2019-10-26"
    assert prediction.adjustment is None


def test_the_elo_model_can_still_be_chosen(db_file):
    games = predict.load_games(db_file)
    _, model = elo.rate_games(games)
    prediction = predict.predict_game(games, "BOS", "NYK", model="elo")
    prob, margin = model.predict(BOS, NYK)
    assert prediction.home_win_prob == pytest.approx(prob)
    assert prediction.predicted_margin == pytest.approx(margin)
    assert prediction.home_rating == pytest.approx(model.rating(BOS))
    assert prediction.predicted_home_pts is None


def test_games_from_the_database_carry_possessions(db_file):
    games = predict.load_games(db_file)
    assert games["home_possessions"].tolist() == pytest.approx([88 + 0.44 * 22 - 10 + 13] * 8)


def test_a_games_table_without_possessions_still_predicts(games):
    prediction = predict.predict_game(games, "BOS", "NYK", model="efficiency")   # ratings cannot move: only home court counts
    assert prediction.home_win_prob > 0.5
    assert prediction.home_rating == 0.0


def test_lower_case_and_neutral_courts(db_file):
    games = predict.load_games(db_file)
    home = predict.predict_game(games, "bos", "nyk", model="efficiency")
    neutral = predict.predict_game(games, "bos", "nyk", neutral=True, model="efficiency")
    assert home.home == "BOS"
    assert neutral.home_win_prob < home.home_win_prob


def test_unknown_teams_models_and_repeated_teams_are_rejected(games):
    with pytest.raises(ValueError, match="Unknown team: XXX"):
        predict.predict_game(games, "XXX", "NYK")
    with pytest.raises(ValueError, match="different teams"):
        predict.predict_game(games, "BOS", "BOS")
    with pytest.raises(ValueError, match="Unknown model"):
        predict.predict_game(games, "BOS", "NYK", model="magic")


def test_command_line_prints_win_probability_margin_and_score(db_file, capsys):
    assert predict.main(["BOS", "NYK", "--model", "efficiency", "--db", str(db_file)]) == 0
    output = capsys.readouterr().out
    assert "Net ratings through 2019-10-26" in output
    assert "BOS win probability:" in output
    assert "NYK win probability:" in output
    assert "Predicted margin:" in output
    assert "Predicted score: BOS" in output


def test_command_line_with_the_elo_model(db_file, capsys):
    assert predict.main(["BOS", "NYK", "--model", "elo", "--db", str(db_file)]) == 0
    output = capsys.readouterr().out
    assert "Elo ratings through 2019-10-26" in output
    assert "Predicted score" not in output


def test_the_two_win_probabilities_printed_add_up_to_100(db_file, capsys):
    predict.main(["GSW", "LAL", "--db", str(db_file)])
    lines = [line for line in capsys.readouterr().out.splitlines() if "win probability" in line]
    assert sum(float(line.split(": ")[1].rstrip("%")) for line in lines) == pytest.approx(100.0, abs=0.11)


def test_command_line_explains_a_missing_database(tmp_path, capsys):
    assert predict.main(["BOS", "NYK", "--db", str(tmp_path / "missing.db")]) == 1
    assert "db.py" in capsys.readouterr().err


def test_command_line_explains_an_unknown_team(db_file, capsys):
    assert predict.main(["BOS", "ZZZ", "--db", str(db_file)]) == 1
    assert "Unknown team: ZZZ" in capsys.readouterr().err


# --- the situational model (the default) -----------------------------------------

EFFECTS = {"back_to_back": -2.0, "long_rest": 0.5, "travel_1000km": -0.2}


def with_effects(games):
    """A replayed v3 model with known effects (these few games are too few to learn any)."""
    fitted = predict.fit(games, "situational")
    fitted.effects = dict(EFFECTS)
    return fitted


def test_the_default_model_is_the_situational_one(db_file):
    games = predict.load_games(db_file)
    prediction = predict.predict_game(games, "BOS", "NYK", date="2019-10-30")
    assert predict.DEFAULT_MODEL == "situational"
    assert prediction.model == "situational"
    assert prediction.game_date == "2019-10-30"
    assert prediction.adjustment == 0.0   # nothing learned from eight games


def test_with_equal_rest_the_situational_model_agrees_with_v2(db_file):
    games = predict.load_games(db_file)
    # Both teams last played more than a month before: fresh, so no adjustment.
    v3 = predict.predict_game(games, "BOS", "NYK", fitted=with_effects(games), date="2020-01-15")
    v2 = predict.predict_game(games, "BOS", "NYK", model="efficiency")
    assert v3.adjustment == 0.0
    assert v3.home_rest_days is None and v3.away_rest_days is None
    assert v3.home_win_prob == pytest.approx(v2.home_win_prob)
    assert v3.predicted_margin == pytest.approx(v2.predicted_margin)


def test_a_back_to_back_moves_the_prediction_against_the_tired_team(db_file):
    games = predict.load_games(db_file)
    fitted = with_effects(games)
    # NYK last played on 2019-10-26 (at GSW), BOS on 2019-10-24 (at LAL).
    tired = predict.predict_game(games, "BOS", "NYK", fitted=fitted, date="2019-10-27")
    rested = predict.predict_game(games, "BOS", "NYK", fitted=fitted, date="2020-01-15")
    assert (tired.home_rest_days, tired.away_rest_days) == (3, 1)
    expected = (-2.0 * -1) + (0.5 * 1) + (-0.2) * (
        situational.distance_km("LAL", "BOS") - situational.distance_km("GSW", "BOS")) / 1000
    assert tired.adjustment == pytest.approx(expected)
    assert tired.predicted_margin == pytest.approx(rested.predicted_margin + expected)
    assert tired.home_win_prob > rested.home_win_prob
    assert tired.predicted_home_pts - tired.predicted_away_pts == pytest.approx(tired.predicted_margin)


def test_the_date_defaults_to_today(db_file):
    import datetime
    prediction = predict.predict_game(predict.load_games(db_file), "BOS", "NYK")
    assert prediction.game_date == datetime.date.today().isoformat()


def test_a_date_before_a_teams_last_game_is_rejected(db_file):
    games = predict.load_games(db_file)
    with pytest.raises(ValueError, match="Choose a later date"):
        predict.predict_game(games, "BOS", "NYK", date="2019-10-24")


def test_describe_rest():
    assert "back-to-back" in predict.describe_rest(1)
    assert predict.describe_rest(2) == "1 day off"
    assert predict.describe_rest(4) == "3 days off"
    assert "fresh" in predict.describe_rest(None)


def test_command_line_default_shows_rest_and_the_game_date(db_file, capsys):
    assert predict.main(["BOS", "NYK", "--date", "2019-10-27", "--db", str(db_file)]) == 0
    output = capsys.readouterr().out
    assert "on 2019-10-27" in output
    assert "Rest: BOS 2 days off; NYK played yesterday (back-to-back)" in output
    assert "Rest and travel:" in output
    assert "Predicted score: BOS" in output


def test_command_line_rejects_a_bad_date(db_file, capsys):
    with pytest.raises(SystemExit):
        predict.main(["BOS", "NYK", "--date", "tomorrow", "--db", str(db_file)])
    assert predict.main(["BOS", "NYK", "--date", "2019-10-01", "--db", str(db_file)]) == 1
    assert "Choose a later date" in capsys.readouterr().err
