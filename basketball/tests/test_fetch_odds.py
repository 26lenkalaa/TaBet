"""Offline tests for fetch_odds.py, using a small made-up file in Kaggle's layout."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

import src.ingest.fetch_odds as fo  # noqa: E402
from src import market  # noqa: E402


@pytest.fixture
def kaggle_file(tmp_path):
    """Four games the way the Kaggle file lists them."""
    rows = pd.DataFrame({
        "season": [2019, 2019, 2019, 2020],
        "date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
        "regular": [True] * 4,
        "playoffs": [False] * 4,
        "away": ["phi", "okc", "atl", "lac"],
        "home": ["bos", "gs", "ny", "lal"],
        "score_away": [87, 100, 107, 112],
        "score_home": [105, 108, 126, 102],
        "whos_favored": ["home", "home", "home", "away"],
        "spread": [5.0, 12.0, 5.5, 3.5],
        "total": [208.5, 223.5, 217.0, 224.5],
        "moneyline_away": [180, 700, 190, -160],
        "moneyline_home": [-220, -1100, -230, 140],
    })
    file = tmp_path / "nba_2008-2026.csv"
    rows.to_csv(file, index=False)
    return file


@pytest.fixture
def games():
    return pd.DataFrame({
        "game_id": ["0021800001", "0021800002", "0021800003", "0021900001"],
        "season": ["2018-19", "2018-19", "2018-19", "2019-20"],
        "date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
        "home": ["BOS", "GSW", "NYK", "LAL"],
        "away": ["PHI", "OKC", "ATL", "LAC"],
        "home_pts": [105, 108, 126, 102],
        "away_pts": [87, 100, 107, 112],
    })


def change(file, **columns):
    table = pd.read_csv(file)
    for column, values in columns.items():
        table[column] = values
    table.to_csv(file, index=False)


def test_spread_becomes_the_home_teams_line(kaggle_file):
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert odds.loc["BOS", "spread_close"] == -5.0   # home favored -> negative
    assert odds.loc["LAL", "spread_close"] == 3.5    # away favored -> positive


def test_team_codes_become_nba_abbreviations(kaggle_file):
    odds = fo.read_kaggle(kaggle_file)
    assert set(odds["home"]) == {"BOS", "GSW", "NYK", "LAL"}
    assert set(odds["away"]) == {"PHI", "OKC", "ATL", "LAC"}


def test_totals_and_moneylines_are_kept(kaggle_file):
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert odds.loc["GSW", "total_close"] == 223.5
    assert (odds.loc["GSW", "home_ml"], odds.loc["GSW", "away_ml"]) == (-1100, 700)


def test_a_spread_already_written_with_a_sign_is_not_flipped(kaggle_file):
    change(kaggle_file, spread=[-5.0, -12.0, -5.5, -3.5])
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert odds.loc["BOS", "spread_close"] == -5.0
    assert odds.loc["LAL", "spread_close"] == 3.5


def test_a_pick_em_game_has_a_zero_spread(kaggle_file):
    change(kaggle_file, spread=[0.0, 12.0, 5.5, 3.5], whos_favored=[None, "home", "home", "away"])
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert odds.loc["BOS", "spread_close"] == 0.0


def test_a_spread_with_no_favorite_named_is_left_empty(kaggle_file):
    change(kaggle_file, whos_favored=["home", "home", None, "away"])
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert pd.isna(odds.loc["NYK", "spread_close"])


def test_impossible_moneylines_and_totals_are_left_empty(kaggle_file):
    change(kaggle_file, moneyline_home=[0, -1100, -230, 140], total=[0, 223.5, 217.0, 224.5])
    odds = fo.read_kaggle(kaggle_file).set_index("home")
    assert pd.isna(odds.loc["BOS", "home_ml"])
    assert pd.isna(odds.loc["BOS", "total_close"])


def test_a_file_without_moneylines_still_loads(kaggle_file):
    table = pd.read_csv(kaggle_file).drop(columns=["moneyline_home", "moneyline_away"])
    table.to_csv(kaggle_file, index=False)
    odds = fo.read_kaggle(kaggle_file)
    assert "home_ml" not in odds.columns
    assert odds["spread_close"].notna().all()


def test_missing_columns_are_named_along_with_what_the_file_has(kaggle_file):
    table = pd.read_csv(kaggle_file).rename(columns={"whos_favored": "favorite"})
    table.to_csv(kaggle_file, index=False)
    with pytest.raises(ValueError, match="missing columns: whos_favored") as error:
        fo.read_kaggle(kaggle_file)
    assert "favorite" in str(error.value)


def test_an_unknown_team_code_is_an_error(kaggle_file):
    change(kaggle_file, home=["bos", "xx", "ny", "lal"])
    with pytest.raises(ValueError, match="Unknown team name: 'xx'"):
        fo.read_kaggle(kaggle_file)


def test_every_game_gets_a_spread_and_the_right_market_probability(kaggle_file, games):
    odds = fo.read_kaggle(kaggle_file)
    matched = market.match_odds(games, market.standardize_odds(odds))
    assert market.games_without_spread(matched).empty
    assert list(matched["market_margin"]) == [5.0, 12.0, 5.5, -3.5]
    assert matched.loc[3, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(140, -160)[0])
    assert fo.score_mismatches(matched, odds).empty


def test_a_different_final_score_is_reported(kaggle_file, games):
    odds = fo.read_kaggle(kaggle_file)
    games.loc[1, "home_pts"] = 99
    matched = market.match_odds(games, market.standardize_odds(odds))
    assert list(fo.score_mismatches(matched, odds)["game_id"]) == ["0021800002"]


def test_find_raw_file_explains_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        fo.find_raw_file(tmp_path)


def test_find_raw_file_picks_the_csv(kaggle_file):
    assert fo.find_raw_file(kaggle_file.parent) == kaggle_file


def test_main_writes_odds_csv_and_reports_coverage(kaggle_file, games, tmp_path, capsys):
    games_file = tmp_path / "games.csv"
    games.to_csv(games_file, index=False)
    output = tmp_path / "processed" / "odds.csv"

    assert fo.main(["--file", str(kaggle_file), "--games", str(games_file), "--output", str(output)]) == 0

    written = pd.read_csv(output)
    assert list(written.columns) == ["date", "home", "away", "spread_close", "total_close", "home_ml", "away_ml"]
    assert len(written) == 4
    printed = capsys.readouterr().out
    assert "Every game has a closing spread and every score agrees." in printed
    assert "2018-19" in printed


def test_main_lists_games_without_a_spread(kaggle_file, games, tmp_path, capsys):
    extra = games.iloc[[0]].assign(game_id="0021900002", season="2019-20", date="2019-11-01")
    games_file = tmp_path / "games.csv"
    pd.concat([games, extra]).to_csv(games_file, index=False)
    fo.main(["--file", str(kaggle_file), "--games", str(games_file), "--output", str(tmp_path / "odds.csv")])
    printed = capsys.readouterr().out
    assert "1 games have no closing spread" in printed
    assert "0021900002" in printed


def test_main_still_writes_odds_when_games_csv_is_missing(kaggle_file, tmp_path, capsys):
    output = tmp_path / "odds.csv"
    assert fo.main(["--file", str(kaggle_file), "--games", str(tmp_path / "none.csv"), "--output", str(output)]) == 0
    assert output.exists()
    assert "Run clean.py" in capsys.readouterr().out


def test_main_explains_a_missing_kaggle_file(tmp_path, capsys):
    assert fo.main(["--file", str(tmp_path / "missing.csv")]) == 1
    assert "Error" in capsys.readouterr().err
