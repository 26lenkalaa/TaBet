"""Offline tests for backtest.py, using made-up games and odds."""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import backtest, db, market  # noqa: E402
from src.models.elo import rate_games  # noqa: E402

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}


def make_games(results):
    """results: (date, season, game_type, home_id, away_id, home_pts, away_pts) per game."""
    rows = []
    for number, (date, season, game_type, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00200{number:05d}", "season": season, "date": date,
            "game_type": game_type, "is_playoffs": game_type == "playoffs",
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def games():
    return make_games([
        ("2018-10-16", "2018-19", "regular", BOS, NYK, 110, 95),
        ("2018-10-17", "2018-19", "regular", LAL, GSW, 101, 104),
        ("2019-10-22", "2019-20", "regular", NYK, LAL, 99, 108),
        ("2019-10-23", "2019-20", "regular", GSW, BOS, 120, 111),
        ("2019-10-25", "2019-20", "regular", BOS, LAL, 105, 100),
        ("2020-08-15", "2019-20", "play_in", NYK, GSW, 90, 118),
        ("2020-08-17", "2019-20", "playoffs", LAL, BOS, 112, 107),
        ("2020-10-22", "2020-21", "regular", GSW, NYK, 125, 100),
        ("2020-10-23", "2020-21", "regular", BOS, GSW, 101, 109),
    ])


@pytest.fixture
def odds(games):
    return pd.DataFrame({
        "game_id": games["game_id"],
        "spread_close": [-6.0, 2.5, 4.0, -3.0, -2.0, 9.5, -1.5, -8.0, 1.0],
        "total_close": [215.5] * 9,
        "home_ml": [-240, 120, 160, -150, -130, 380, -120, -350, 100],
        "away_ml": [200, -140, -190, 130, 110, -500, 100, 280, -120],
    })


# --- the scores ----------------------------------------------------------------

def test_brier_score():
    prob, won = pd.Series([0.8, 0.3, 0.5]), pd.Series([1, 0, 1])
    assert backtest.brier_score(prob, won) == pytest.approx((0.04 + 0.09 + 0.25) / 3)
    assert backtest.brier_score(pd.Series([1.0, 0.0]), pd.Series([1, 0])) == 0.0


def test_log_loss():
    prob, won = pd.Series([0.8, 0.3]), pd.Series([1, 0])
    assert backtest.log_loss(prob, won) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert backtest.log_loss(pd.Series([0.5, 0.5]), pd.Series([1, 0])) == pytest.approx(math.log(2))


def test_log_loss_survives_a_certain_prediction_that_was_wrong():
    assert math.isfinite(backtest.log_loss(pd.Series([1.0]), pd.Series([0])))


def test_a_confident_wrong_prediction_scores_worse_than_a_cautious_one():
    won = pd.Series([0])
    assert backtest.log_loss(pd.Series([0.95]), won) > backtest.log_loss(pd.Series([0.6]), won)
    assert backtest.brier_score(pd.Series([0.95]), won) > backtest.brier_score(pd.Series([0.6]), won)


def test_margin_error_ignores_direction():
    assert backtest.margin_error(pd.Series([5.0, -3.0]), pd.Series([9, 1])) == pytest.approx(4.0)


# --- the table -----------------------------------------------------------------

def test_table_keeps_every_game_with_model_and_market_side_by_side(games, odds):
    table = backtest.build_table(games, odds)
    assert len(table) == len(games)
    for column in ["home_win_prob", "predicted_margin", "market_home_win_prob", "market_margin"]:
        assert table[column].notna().all(), column
    assert list(table["home_won"]) == [1, 0, 0, 1, 1, 0, 1, 1, 0]
    assert list(table["margin"]) == [15, -3, -9, 9, 5, -28, 5, 25, -8]


def test_model_columns_are_the_pre_game_elo_predictions(games, odds):
    table = backtest.build_table(games, odds)
    rated, _ = rate_games(games)
    assert list(table["home_win_prob"]) == pytest.approx(list(rated["home_win_prob"]))
    assert list(table["predicted_margin"]) == pytest.approx(list(rated["predicted_margin"]))


def test_changing_a_result_does_not_change_that_games_prediction(games, odds):
    flipped = games.copy()
    flipped.loc[4, ["home_pts", "away_pts"]] = [80, 130]
    before = backtest.build_table(games, odds)
    after = backtest.build_table(flipped, odds)
    assert list(after.loc[:4, "home_win_prob"]) == pytest.approx(list(before.loc[:4, "home_win_prob"]))
    assert after.loc[6, "home_win_prob"] != pytest.approx(before.loc[6, "home_win_prob"])   # next BOS-LAL game


def test_market_columns_come_from_moneylines_and_closing_spread(games, odds):
    table = backtest.build_table(games, odds)
    assert table.loc[0, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(-240, 200)[0])
    assert table.loc[0, "market_margin"] == 6.0
    assert table.loc[5, "market_margin"] == -9.5


def test_play_in_games_are_graded_with_the_playoffs(games, odds):
    table = backtest.build_table(games, odds)
    assert list(table["part"]) == ["regular"] * 5 + ["playoffs"] * 2 + ["regular"] * 2


def test_games_without_odds_stay_in_the_replay_but_have_no_market_prediction(games, odds):
    table = backtest.build_table(games, odds.drop(index=[3, 4]))
    assert len(table) == len(games)
    assert table.loc[[3, 4], "market_home_win_prob"].isna().all()
    assert table["home_win_prob"].notna().all()


# --- grading -------------------------------------------------------------------

def test_grade_scores_model_and_market_on_the_same_games(games, odds):
    table = backtest.build_table(games, odds.drop(index=[3]))
    scores = backtest.grade(table)
    priced = table.dropna(subset=["market_home_win_prob"])
    assert scores["games"] == 8
    assert scores["brier_model"] == pytest.approx(backtest.brier_score(priced["home_win_prob"], priced["home_won"]))
    assert scores["brier_market"] == pytest.approx(
        backtest.brier_score(priced["market_home_win_prob"], priced["home_won"]))
    assert scores["margin_err_market"] == pytest.approx(
        backtest.margin_error(priced["market_margin"], priced["margin"]))


def test_a_perfect_market_beats_the_model(games, odds):
    odds["spread_close"] = -(games["home_pts"] - games["away_pts"]).astype(float)   # spread = real margin
    scores = backtest.grade(backtest.build_table(games, odds))
    assert scores["margin_err_market"] == 0.0
    assert scores["margin_err_model"] > 0.0


def test_summary_skips_the_warm_up_season_and_splits_regular_from_playoffs(games, odds):
    summary = backtest.summarize(backtest.build_table(games, odds), warmup_seasons=1)
    labels = list(zip(summary["part"], summary["season"]))
    assert labels == [("regular", "2019-20"), ("regular", "2020-21"), ("regular", "all"),
                      ("playoffs", "2019-20"), ("playoffs", "all")]
    assert list(summary["games"]) == [3, 2, 5, 2, 2]


def test_summary_with_no_warm_up_grades_the_first_season_too(games, odds):
    summary = backtest.summarize(backtest.build_table(games, odds), warmup_seasons=0)
    assert ("regular", "2018-19") in set(zip(summary["part"], summary["season"]))


def test_the_all_row_is_scored_over_every_graded_game(games, odds):
    table = backtest.build_table(games, odds)
    summary = backtest.summarize(table, warmup_seasons=1).set_index(["part", "season"])
    graded = table[(table["season"] != "2018-19") & (table["part"] == "regular")]
    assert summary.loc[("regular", "all"), "brier_model"] == pytest.approx(
        backtest.brier_score(graded["home_win_prob"], graded["home_won"]))


def test_a_season_with_no_odds_is_left_out(games, odds):
    summary = backtest.summarize(backtest.build_table(games, odds.iloc[:7]), warmup_seasons=1)
    assert "2020-21" not in set(summary["season"])


# --- the command line ----------------------------------------------------------

def make_db(games, odds, tmp_path):
    db_file = tmp_path / "basketball.db"
    db.build(db.connect(db_file), games, odds)
    return db_file


def test_main_prints_scores_per_season_for_model_and_market(games, odds, tmp_path, capsys):
    output = tmp_path / "out" / "summary.csv"

    assert backtest.main(["--db", str(make_db(games, odds, tmp_path)), "--output", str(output)]) == 0

    printed = capsys.readouterr().out
    assert "Warm-up (replayed, not graded): 2018-19" in printed
    assert "Regular season" in printed and "Playoffs" in printed
    for column in backtest.METRICS:
        assert column in printed
    assert "2019-20" in printed and "2020-21" in printed
    saved = pd.read_csv(output)
    assert list(saved.columns) == ["part", "season", "games"] + backtest.METRICS
    assert len(saved) == 5


def test_main_grades_the_database_the_same_as_the_tables_it_was_built_from(games, odds, tmp_path):
    output = tmp_path / "summary.csv"
    backtest.main(["--db", str(make_db(games, odds, tmp_path)), "--output", str(output)])
    expected = backtest.summarize(backtest.build_table(games, odds))
    pd.testing.assert_frame_equal(pd.read_csv(output), expected, check_dtype=False)


def test_main_stores_the_predictions_it_graded(games, odds, tmp_path):
    db_file = make_db(games, odds, tmp_path)
    backtest.main(["--db", str(db_file), "--output", str(tmp_path / "summary.csv")])
    stored = db.read_predictions(db.connect(db_file))
    rated, _ = rate_games(games)
    assert len(stored) == len(games)
    assert list(stored["home_win_prob"]) == pytest.approx(list(rated.sort_values("game_id")["home_win_prob"]))


def test_main_explains_a_missing_database(tmp_path, capsys):
    assert backtest.main(["--db", str(tmp_path / "missing.db")]) == 1
    assert "db.py" in capsys.readouterr().err


def test_main_explains_when_nothing_can_be_graded(games, odds, tmp_path, capsys):
    db_file = make_db(games, odds.iloc[:2], tmp_path)   # odds only for the warm-up season
    assert backtest.main(["--db", str(db_file), "--output", str(tmp_path / "summary.csv")]) == 1
    assert "no games left to grade" in capsys.readouterr().err
