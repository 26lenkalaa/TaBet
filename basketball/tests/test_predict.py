"""Offline tests for predict.py, using made-up games in a temporary database."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import db, predict  # noqa: E402
from src.models import elo, ratings  # noqa: E402

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


def test_the_default_model_is_the_efficiency_ratings(db_file):
    games = predict.load_games(db_file)
    _, model = ratings.rate_games(games)
    prediction = predict.predict_game(games, "BOS", "NYK")
    prob, margin = model.predict(BOS, NYK)
    assert prediction.model == "efficiency"
    assert prediction.home_win_prob == pytest.approx(prob)
    assert prediction.predicted_margin == pytest.approx(margin)
    assert prediction.home_rating == pytest.approx(model.net_rating(BOS))
    assert prediction.predicted_home_pts - prediction.predicted_away_pts == pytest.approx(margin)
    assert prediction.last_game_date == "2019-10-26"


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
    prediction = predict.predict_game(games, "BOS", "NYK")   # ratings cannot move, so only home court counts
    assert prediction.home_win_prob > 0.5
    assert prediction.home_rating == 0.0


def test_lower_case_and_neutral_courts(db_file):
    games = predict.load_games(db_file)
    home = predict.predict_game(games, "bos", "nyk")
    neutral = predict.predict_game(games, "bos", "nyk", neutral=True)
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
    assert predict.main(["BOS", "NYK", "--db", str(db_file)]) == 0
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
