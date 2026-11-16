"""Offline tests for clean.py, using small made-up game logs."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import clean  # noqa: E402

TEAM_IDS = {"BOS": 1610612738, "NYK": 1610612752, "LAL": 1610612747, "GSW": 1610612744}


def team_row(game_id, date, team, opponent, home, pts, **overrides) -> None:
    """One raw row, shaped like NBA.com's team game log."""
    row = {
        "SEASON_ID": "2" + str(2000 + int(game_id[3:5])),
        "TEAM_ID": TEAM_IDS[team],
        "TEAM_ABBREVIATION": team,
        "TEAM_NAME": team,
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "MATCHUP": f"{team} vs. {opponent}" if home else f"{team} @ {opponent}",
        "WL": "W",
        "MIN": 240,
        "FGM": 40, "FGA": 88, "FG3M": 12, "FG3A": 34, "FTM": 18, "FTA": 22,
        "OREB": 10, "DREB": 34, "REB": 44, "AST": 25, "STL": 7, "BLK": 5, "TOV": 13, "PF": 20,
        "PTS": pts,
        "PLUS_MINUS": 0,
    }
    row.update(overrides)
    return row


def game(game_id, date, home, away, home_pts, away_pts):
    """The two raw rows of one game."""
    return [
        team_row(game_id, date, home, away, True, home_pts),
        team_row(game_id, date, away, home, False, away_pts),
    ]


@pytest.fixture
def raw():
    rows = (
        game("0021800001", "2018-10-16", "BOS", "NYK", 105, 87)
        + game("0021800002", "2018-10-16", "GSW", "LAL", 108, 100)
        + game("0052000001", "2021-05-18", "LAL", "GSW", 103, 100)
        + game("0041800101", "2019-04-13", "NYK", "BOS", 99, 111)
    )
    return pd.DataFrame(rows)


def test_two_team_rows_become_one_game_row(raw):
    games, dropped = clean.clean_games(raw)
    assert len(games) == 4
    assert games["game_id"].is_unique
    assert dropped.empty


def test_home_and_away_come_from_the_matchup_text(raw):
    games, _ = clean.clean_games(raw)
    first = games[games["game_id"] == "0021800001"].iloc[0]
    assert (first["home"], first["away"]) == ("BOS", "NYK")
    assert (first["home_team_id"], first["away_team_id"]) == (TEAM_IDS["BOS"], TEAM_IDS["NYK"])
    assert (first["home_pts"], first["away_pts"]) == (105, 87)


def test_order_of_the_raw_rows_does_not_matter(raw):
    shuffled = raw.sample(frac=1, random_state=3)
    expected, _ = clean.clean_games(raw)
    actual, _ = clean.clean_games(shuffled)
    pd.testing.assert_frame_equal(actual, expected)


def test_box_score_columns_follow_their_team():
    rows = game("0021800001", "2018-10-16", "BOS", "NYK", 105, 87)
    rows[0].update(FGA=90, TOV=11)
    rows[1].update(FGA=80, TOV=17)
    games, _ = clean.clean_games(pd.DataFrame(rows))
    only = games.iloc[0]
    assert (only["home_fga"], only["away_fga"]) == (90, 80)
    assert (only["home_tov"], only["away_tov"]) == (11, 17)


def test_season_and_game_type_come_from_the_game_id(raw):
    games, _ = clean.clean_games(raw)
    by_id = games.set_index("game_id")
    assert by_id.loc["0021800001", "season"] == "2018-19"
    assert by_id.loc["0021800001", "game_type"] == "regular"
    assert by_id.loc["0052000001", "season"] == "2020-21"
    assert by_id.loc["0052000001", "game_type"] == "play_in"
    assert by_id.loc["0041800101", "game_type"] == "playoffs"


def test_only_playoff_games_are_marked_is_playoffs(raw):
    games, _ = clean.clean_games(raw)
    flags = games.set_index("game_id")["is_playoffs"]
    assert flags["0041800101"]
    assert not flags["0021800001"]
    assert not flags["0052000001"]  # play-in games are not playoff games


def test_season_label_handles_the_turn_of_the_century():
    assert clean.season_from_game_id("0029900001") == "2099-00"
    assert clean.season_from_game_id("0022600001") == "2026-27"


def test_dates_are_written_as_year_month_day():
    rows = (
        game("0021800001", "2018-10-16", "BOS", "NYK", 105, 87)
        + game("0021800002", "2018-10-17T00:00:00", "GSW", "LAL", 108, 100)
        + game("0021800003", "OCT 18, 2018", "LAL", "BOS", 101, 99)
    )
    games, _ = clean.clean_games(pd.DataFrame(rows))
    assert list(games["date"]) == ["2018-10-16", "2018-10-17", "2018-10-18"]


def test_games_are_sorted_by_date(raw):
    games, _ = clean.clean_games(raw)
    assert list(games["date"]) == sorted(games["date"])


def test_a_game_in_two_raw_files_is_kept_once(raw):
    doubled = pd.concat([raw, raw], ignore_index=True)
    games, dropped = clean.clean_games(doubled)
    assert len(games) == 4
    assert dropped.empty


def test_a_repeated_row_keeps_the_latest_score():
    old = game("0021800001", "2018-10-16", "BOS", "NYK", 50, 48)
    new = game("0021800001", "2018-10-16", "BOS", "NYK", 105, 87)
    games, _ = clean.clean_games(pd.DataFrame(old + new))
    assert (games.iloc[0]["home_pts"], games.iloc[0]["away_pts"]) == (105, 87)


def test_a_game_with_one_team_row_is_left_out(raw):
    one_sided = raw.drop(index=1)  # removes New York's row of the first game
    games, dropped = clean.clean_games(one_sided)
    assert "0021800001" not in set(games["game_id"])
    assert list(dropped["game_id"]) == ["0021800001"]
    assert "two team rows" in dropped.iloc[0]["reason"]


def test_a_game_without_a_score_is_left_out(raw):
    raw.loc[0, "PTS"] = None
    games, dropped = clean.clean_games(raw)
    assert "0021800001" not in set(games["game_id"])
    assert dropped.iloc[0]["reason"] == "missing score"


def test_a_game_with_two_home_teams_is_left_out():
    rows = game("0021800001", "2018-10-16", "BOS", "NYK", 105, 87)
    rows[1]["MATCHUP"] = "NYK vs. BOS"
    games, dropped = clean.clean_games(pd.DataFrame(rows))
    assert games.empty
    assert "home and away" in dropped.iloc[0]["reason"]


def test_preseason_and_all_star_games_are_ignored(raw):
    extra = game("0011800001", "2018-09-28", "BOS", "NYK", 97, 90) + game(
        "0031800001", "2019-02-17", "LAL", "GSW", 178, 164
    )
    games, dropped = clean.clean_games(pd.concat([raw, pd.DataFrame(extra)], ignore_index=True))
    assert len(games) == 4
    assert dropped.empty


def test_game_ids_read_as_numbers_get_their_zeros_back(raw):
    raw["GAME_ID"] = raw["GAME_ID"].astype(int)
    games, _ = clean.clean_games(raw)
    assert "0021800001" in set(games["game_id"])
    assert games["game_id"].str.len().eq(10).all()


def test_missing_box_score_columns_are_skipped(raw):
    games, _ = clean.clean_games(raw.drop(columns=["STL", "BLK"]))
    assert "home_stl" not in games.columns
    assert "home_fga" in games.columns


def test_scores_and_team_ids_are_whole_numbers(raw):
    games, _ = clean.clean_games(raw)
    for column in ["home_pts", "away_pts", "home_team_id", "away_team_id", "home_fga"]:
        assert pd.api.types.is_integer_dtype(games[column]), column


def test_check_games_passes_a_clean_table(raw):
    games, _ = clean.clean_games(raw)
    clean.check_games(games)


def test_check_games_rejects_duplicates_and_ties(raw):
    games, _ = clean.clean_games(raw)
    with pytest.raises(ValueError, match="duplicate game ids"):
        clean.check_games(pd.concat([games, games.iloc[[0]]], ignore_index=True))
    tied = games.copy()
    tied.loc[0, "away_pts"] = tied.loc[0, "home_pts"]
    with pytest.raises(ValueError, match="tied"):
        clean.check_games(tied)


def test_load_raw_reads_every_csv_and_keeps_game_ids_as_text(raw, tmp_path):
    raw.iloc[:4].to_csv(tmp_path / "regular_2018-19.csv", index=False)
    raw.iloc[4:].to_csv(tmp_path / "other.csv", index=False)
    pd.DataFrame(columns=raw.columns).to_csv(tmp_path / "empty.csv", index=False)
    loaded = clean.load_raw(tmp_path)
    assert len(loaded) == len(raw)
    assert set(loaded["GAME_ID"]) == set(raw["GAME_ID"])  # leading zeros survive


def test_load_raw_explains_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_games.py"):
        clean.load_raw(tmp_path)


def test_load_raw_names_the_file_with_missing_columns(raw, tmp_path):
    raw.drop(columns=["MATCHUP"]).to_csv(tmp_path / "broken.csv", index=False)
    with pytest.raises(ValueError, match="broken.csv is missing columns: MATCHUP"):
        clean.load_raw(tmp_path)


def test_main_writes_games_csv(raw, tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw.to_csv(raw_dir / "games.csv", index=False)
    output = tmp_path / "processed" / "games.csv"

    assert clean.main(["--raw-dir", str(raw_dir), "--output", str(output)]) == 0

    written = pd.read_csv(output, dtype={"game_id": str})
    assert len(written) == 4
    assert list(written.columns[:11]) == [
        "game_id", "season", "date", "game_type", "is_playoffs",
        "home_team_id", "home", "away_team_id", "away", "home_pts", "away_pts",
    ]
    assert "Wrote 4 games" in capsys.readouterr().out
