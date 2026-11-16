"""Offline tests for db.py: every test builds its own small SQLite file."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import db  # noqa: E402
from src.models.elo import rate_games  # noqa: E402

BOS, NYK, LAL = 1610612738, 1610612752, 1610612747
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL"}


@pytest.fixture
def games():
    results = [
        ("0021800001", "2018-19", "2018-10-16", "regular", BOS, NYK, 105, 87),
        ("0021800002", "2018-19", "2018-10-18", "regular", LAL, BOS, 100, 110),
        ("0021800003", "2018-19", "2018-10-19", "regular", NYK, LAL, 99, 108),
        ("0041800101", "2018-19", "2019-04-13", "playoffs", BOS, LAL, 111, 99),
    ]
    rows = []
    for game_id, season, date, game_type, home, away, home_pts, away_pts in results:
        rows.append({
            "game_id": game_id, "season": season, "date": date, "game_type": game_type,
            "is_playoffs": game_type == "playoffs",
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
            "home_fga": 90, "home_fta": 20, "home_oreb": 10, "home_tov": 12,
            "away_fga": 85, "away_fta": 25, "away_oreb": 8, "away_tov": 15,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def odds():
    return pd.DataFrame({
        "game_id": ["0021800001", "0021800002", "0021800003"],
        "date": ["2018-10-16", "2018-10-18", "2018-10-19"],
        "home": ["BOS", "LAL", "NYK"], "away": ["NYK", "BOS", "LAL"],
        "spread_close": [-5.0, 2.5, None],
        "total_close": [208.5, 215.0, None],
        "home_ml": [-220.0, 120.0, 150.0],
        "away_ml": [180.0, -140.0, -175.0],
    })


@pytest.fixture
def engine(games, odds, tmp_path):
    engine = db.connect(tmp_path / "basketball.db")
    db.build(engine, games, odds)
    return engine


def test_the_five_tables_have_the_planned_columns(engine):
    columns = {table: [column["name"] for column in inspect(engine).get_columns(table)]
               for table in inspect(engine).get_table_names()}
    assert set(columns) == {"teams", "games", "odds", "team_stats", "predictions"}
    assert columns["teams"] == ["id", "abbreviation", "name"]
    assert set(columns["games"]) >= {"id", "season", "date", "home_team_id", "away_team_id",
                                     "home_pts", "away_pts", "is_playoffs"}
    assert set(columns["odds"]) == {"game_id", "source", "home_ml", "away_ml", "spread", "total", "is_closing"}
    assert set(columns["team_stats"]) == {"game_id", "team_id", "possessions", "off_rating", "def_rating",
                                          "rest_days"}
    assert set(columns["predictions"]) == {"game_id", "model_version", "home_win_prob", "predicted_margin",
                                           "created_at"}


def test_every_row_is_loaded(engine):
    assert db.row_counts(engine) == {"teams": 3, "games": 4, "odds": 3, "team_stats": 8, "predictions": 4}


def test_teams_get_their_full_names(games):
    table = db.team_table(games).set_index("abbreviation")
    assert table.loc["BOS", "name"] == "Boston Celtics"
    assert table.loc["LAL", "name"] == "Los Angeles Lakers"
    assert table.loc["NYK", "id"] == NYK


def test_games_come_back_exactly_as_they_went_in(engine, games):
    columns = ["game_id", "season", "date", "game_type", "is_playoffs",
               "home_team_id", "home", "away_team_id", "away", "home_pts", "away_pts"]
    pd.testing.assert_frame_equal(db.read_games(engine), games[columns], check_dtype=False)


def test_game_ids_keep_their_leading_zeros(engine):
    assert list(db.read_games(engine)["game_id"])[0] == "0021800001"


def test_games_come_back_in_date_order(games, tmp_path):
    engine = db.connect(tmp_path / "shuffled.db")
    db.build(engine, games.sample(frac=1, random_state=2))
    assert list(db.read_games(engine)["date"]) == sorted(games["date"])


def test_odds_come_back_with_the_names_the_backtest_uses(engine):
    table = db.read_odds(engine)
    assert list(table.columns) == ["game_id", "spread_close", "total_close", "home_ml", "away_ml"]
    first = table.set_index("game_id").loc["0021800001"]
    assert (first["spread_close"], first["total_close"], first["home_ml"], first["away_ml"]) == (
        -5.0, 208.5, -220.0, 180.0)
    assert pd.isna(table.set_index("game_id").loc["0021800003", "spread_close"])   # empty stays empty


def test_odds_for_games_not_in_the_database_are_skipped(games, odds, tmp_path):
    odds.loc[2, "game_id"] = "0029999999"
    engine = db.connect(tmp_path / "extra.db")
    db.build(engine, games, odds)
    assert db.row_counts(engine)["odds"] == 2


def test_the_database_works_without_odds(games, tmp_path):
    engine = db.connect(tmp_path / "no_odds.db")
    db.build(engine, games)
    assert db.row_counts(engine)["odds"] == 0
    assert db.read_odds(engine).empty


def test_team_stats_possessions_and_ratings(games):
    stats = db.team_stats_table(games).set_index(["game_id", "team_id"])
    home = stats.loc[("0021800001", BOS)]
    away = stats.loc[("0021800001", NYK)]
    assert home["possessions"] == pytest.approx(90 + 0.44 * 20 - 10 + 12)   # FGA + 0.44*FTA - OREB + TOV
    assert away["possessions"] == pytest.approx(85 + 0.44 * 25 - 8 + 15)
    assert home["off_rating"] == pytest.approx(100 * 105 / home["possessions"])
    assert home["def_rating"] == pytest.approx(100 * 87 / away["possessions"])
    assert away["off_rating"] == pytest.approx(home["def_rating"])          # one team's offense is the other's defense


def test_rest_days_count_from_the_teams_previous_game(games):
    stats = db.team_stats_table(games).set_index(["game_id", "team_id"])
    assert pd.isna(stats.loc[("0021800001", BOS), "rest_days"])   # first game we have
    assert stats.loc[("0021800002", BOS), "rest_days"] == 2
    assert stats.loc[("0021800003", LAL), "rest_days"] == 1
    assert stats.loc[("0021800003", NYK), "rest_days"] == 3


def test_team_stats_are_empty_when_the_box_score_is_missing(games):
    stats = db.team_stats_table(games.drop(columns=["home_fga", "away_fga"]))
    assert len(stats) == 8
    assert stats["possessions"].isna().all()
    assert stats["rest_days"].notna().sum() == 5


def test_the_stored_predictions_are_the_pre_game_elo_predictions(engine, games):
    stored = db.read_predictions(engine)
    rated, _ = rate_games(games)
    assert list(stored["game_id"]) == sorted(games["game_id"])
    assert list(stored["home_win_prob"]) == pytest.approx(list(rated.sort_values("game_id")["home_win_prob"]))
    assert stored["created_at"].str.endswith("Z").all()


def test_saving_a_model_again_replaces_its_predictions_only(engine, games):
    rated, _ = rate_games(games)
    db.save_predictions(engine, rated.assign(home_win_prob=0.5), "other-model")
    db.save_predictions(engine, rated.assign(home_win_prob=0.6), "other-model")
    assert db.row_counts(engine)["predictions"] == 8
    assert set(db.read_predictions(engine, "other-model")["home_win_prob"]) == {0.6}
    assert len(db.read_predictions(engine)) == 4   # elo-v1 untouched


def test_rebuilding_gives_the_same_database(engine, games, odds):
    db.build(engine, games, odds)
    assert db.row_counts(engine) == {"teams": 3, "games": 4, "odds": 3, "team_stats": 8, "predictions": 4}


def test_connect_can_insist_the_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="db.py"):
        db.connect(tmp_path / "missing.db", must_exist=True)


def test_main_builds_the_database_from_the_csv_files(games, odds, tmp_path, capsys):
    games.to_csv(tmp_path / "games.csv", index=False)
    odds.to_csv(tmp_path / "odds.csv", index=False)
    db_file = tmp_path / "data" / "basketball.db"

    assert db.main(["--games", str(tmp_path / "games.csv"), "--odds", str(tmp_path / "odds.csv"),
                    "--db", str(db_file)]) == 0

    assert db.row_counts(db.connect(db_file))["games"] == 4
    printed = capsys.readouterr().out
    assert "games" in printed and "predictions" in printed


def test_main_still_builds_without_an_odds_file(games, tmp_path, capsys):
    games.to_csv(tmp_path / "games.csv", index=False)
    db_file = tmp_path / "basketball.db"
    assert db.main(["--games", str(tmp_path / "games.csv"), "--odds", str(tmp_path / "none.csv"),
                    "--db", str(db_file)]) == 0
    assert "fetch_odds.py" in capsys.readouterr().out
    assert db.row_counts(db.connect(db_file))["odds"] == 0


def test_main_needs_games_csv(tmp_path, capsys):
    assert db.main(["--games", str(tmp_path / "none.csv"), "--db", str(tmp_path / "x.db")]) == 1
    assert "clean.py" in capsys.readouterr().err
