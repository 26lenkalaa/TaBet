"""Offline tests for fetch_odds.py, using small made-up files in the two Kaggle layouts."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

import src.ingest.fetch_odds as fo  # noqa: E402
from src import market  # noqa: E402


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


@pytest.fixture
def betting_table():
    """Four games the way the betting file (nba_2008-2026.csv) lists them."""
    return pd.DataFrame({
        "season": [2019, 2019, 2019, 2020],
        "date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
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


@pytest.fixture
def moneyline_table():
    """The same four games the way the moneyline file (game_odds.csv) lists them."""
    return pd.DataFrame({
        "game_id": ["0021800001", "0021800002", "0021800003", "0021900001"],
        "game_date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
        "odds_date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
        "last_fetched": ["2024-01-01"] * 4,
        "decimal_home": [1.5, 1.1, 1.4, 2.5],
        "decimal_away": [2.7, 7.5, 3.0, 1.57],
        "moneyline_home": [-200, -1000, -250, 150],
        "moneyline_away": [170, 650, 200, -175],
        "source_url": ["https://example.com/odds"] * 4,
    })


@pytest.fixture
def raw_dir(tmp_path, betting_table, moneyline_table):
    folder = tmp_path / "odds"
    folder.mkdir()
    betting_table.to_csv(folder / "nba_2008-2026.csv", index=False)
    moneyline_table.to_csv(folder / "game_odds.csv", index=False)
    return folder


# --- the betting file ----------------------------------------------------------

def test_spread_becomes_the_home_teams_line(betting_table):
    odds = fo.read_betting(betting_table).set_index("home")
    assert odds.loc["BOS", "spread_close"] == -5.0   # home favored -> negative
    assert odds.loc["LAL", "spread_close"] == 3.5    # away favored -> positive


def test_team_codes_become_nba_abbreviations(betting_table):
    odds = fo.read_betting(betting_table)
    assert set(odds["home"]) == {"BOS", "GSW", "NYK", "LAL"}
    assert set(odds["away"]) == {"PHI", "OKC", "ATL", "LAC"}


def test_totals_and_moneylines_are_kept(betting_table):
    odds = fo.read_betting(betting_table).set_index("home")
    assert odds.loc["GSW", "total_close"] == 223.5
    assert (odds.loc["GSW", "home_ml"], odds.loc["GSW", "away_ml"]) == (-1100, 700)


def test_a_spread_already_written_with_a_sign_is_not_flipped(betting_table):
    betting_table["spread"] = [-5.0, -12.0, -5.5, -3.5]
    odds = fo.read_betting(betting_table).set_index("home")
    assert odds.loc["BOS", "spread_close"] == -5.0
    assert odds.loc["LAL", "spread_close"] == 3.5


def test_a_pick_em_game_has_a_zero_spread(betting_table):
    betting_table["spread"] = [0.0, 12.0, 5.5, 3.5]
    betting_table["whos_favored"] = [None, "home", "home", "away"]
    odds = fo.read_betting(betting_table).set_index("home")
    assert odds.loc["BOS", "spread_close"] == 0.0


def test_a_spread_with_no_favorite_named_is_left_empty(betting_table):
    betting_table["whos_favored"] = ["home", "home", None, "away"]
    odds = fo.read_betting(betting_table).set_index("home")
    assert pd.isna(odds.loc["NYK", "spread_close"])


def test_impossible_moneylines_and_totals_are_left_empty(betting_table):
    betting_table["moneyline_home"] = [0, -1100, -230, 140]
    betting_table["total"] = [0, 223.5, 217.0, 224.5]
    odds = fo.read_betting(betting_table).set_index("home")
    assert pd.isna(odds.loc["BOS", "home_ml"])
    assert pd.isna(odds.loc["BOS", "total_close"])


def test_an_unknown_team_code_is_an_error(betting_table):
    betting_table["home"] = ["bos", "xx", "ny", "lal"]
    with pytest.raises(ValueError, match="Unknown team name: 'xx'"):
        fo.read_betting(betting_table)


# --- the moneyline file --------------------------------------------------------

def test_moneyline_file_gives_one_row_per_game(moneyline_table):
    odds = fo.read_moneylines(moneyline_table).set_index("game_id")
    assert len(odds) == 4
    assert (odds.loc["0021800001", "home_ml"], odds.loc["0021800001", "away_ml"]) == (-200, 170)


def test_game_ids_read_as_numbers_get_their_zeros_back(moneyline_table):
    moneyline_table["game_id"] = moneyline_table["game_id"].astype(int)
    assert set(fo.read_moneylines(moneyline_table)["game_id"]) == set(
        ["0021800001", "0021800002", "0021800003", "0021900001"])


def test_a_game_listed_several_times_keeps_its_latest_odds(moneyline_table):
    earlier = moneyline_table.iloc[[0]].assign(odds_date="2018-10-14", moneyline_home=-150, moneyline_away=130)
    table = pd.concat([moneyline_table, earlier], ignore_index=True)   # the older row comes last in the file
    odds = fo.read_moneylines(table).set_index("game_id")
    assert len(odds) == 4
    assert odds.loc["0021800001", "home_ml"] == -200


def test_decimal_prices_fill_in_for_missing_moneylines(moneyline_table):
    moneyline_table.loc[3, ["moneyline_home", "moneyline_away"]] = [None, None]
    odds = fo.read_moneylines(moneyline_table).set_index("game_id")
    assert odds.loc["0021900001", "home_ml"] == pytest.approx(150)          # 2.50
    assert odds.loc["0021900001", "away_ml"] == pytest.approx(-100 / 0.57)  # 1.57


def test_a_game_with_only_one_price_is_dropped(moneyline_table):
    moneyline_table.loc[2, ["moneyline_away", "decimal_away"]] = [None, None]
    assert "0021800003" not in set(fo.read_moneylines(moneyline_table)["game_id"])


@pytest.mark.parametrize("decimal, american", [(2.5, 150), (2.0, 100), (1.5, -200), (1.4, -250), (11.0, 1000)])
def test_decimal_to_moneyline(decimal, american):
    assert market.decimal_to_moneyline(decimal) == pytest.approx(american)
    assert market.moneyline_to_prob(market.decimal_to_moneyline(decimal)) == pytest.approx(1 / decimal)


def test_decimal_to_moneyline_rejects_impossible_prices():
    with pytest.raises(ValueError):
        market.decimal_to_moneyline(1.0)


# --- telling the files apart ---------------------------------------------------

def test_each_layout_is_recognized(betting_table, moneyline_table):
    assert fo.file_kind(betting_table) == "betting"
    assert fo.file_kind(moneyline_table) == "moneyline"
    assert fo.file_kind(moneyline_table.drop(columns=["moneyline_home", "moneyline_away"])) == "moneyline"
    assert fo.file_kind(pd.DataFrame({"anything": [1]})) == "unknown"


def test_load_raw_reads_both_files(raw_dir):
    betting, moneylines = fo.load_raw(raw_dir)
    assert len(betting) == 4
    assert len(moneylines) == 4


def test_load_raw_works_with_only_one_file(raw_dir):
    (raw_dir / "nba_2008-2026.csv").unlink()
    betting, moneylines = fo.load_raw(raw_dir)
    assert betting is None
    assert len(moneylines) == 4


def test_load_raw_explains_an_unknown_layout(raw_dir):
    pd.DataFrame({"team": ["BOS"], "price": [1.5]}).to_csv(raw_dir / "mystery.csv", index=False)
    with pytest.raises(ValueError, match="mystery.csv is not a layout") as error:
        fo.load_raw(raw_dir)
    assert "It has: team, price" in str(error.value)


def test_load_raw_explains_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        fo.load_raw(tmp_path)


# --- combining them ------------------------------------------------------------

def test_both_files_spread_from_betting_and_its_moneylines_first(games, betting_table, moneyline_table):
    with_odds = fo.attach_odds(games, fo.read_betting(betting_table), fo.read_moneylines(moneyline_table))
    assert list(with_odds["spread_close"]) == [-5.0, -12.0, -5.5, 3.5]
    assert list(with_odds["home_ml"]) == [-220, -1100, -230, 140]   # the betting file's prices
    assert market.games_without_spread(with_odds).empty


def test_moneyline_file_fills_games_the_betting_file_has_no_prices_for(games, betting_table, moneyline_table):
    betting_table.loc[1, ["moneyline_home", "moneyline_away"]] = [None, None]   # GSW game: no prices
    betting_table.loc[2, "moneyline_away"] = None                               # NYK game: half a pair
    with_odds = fo.attach_odds(games, fo.read_betting(betting_table), fo.read_moneylines(moneyline_table))
    assert (with_odds.loc[1, "home_ml"], with_odds.loc[1, "away_ml"]) == (-1000, 650)
    assert (with_odds.loc[2, "home_ml"], with_odds.loc[2, "away_ml"]) == (-250, 200)   # the pair stays together
    assert with_odds.loc[0, "home_ml"] == -220


def test_moneyline_file_covers_games_missing_from_the_betting_file(games, betting_table, moneyline_table):
    with_odds = fo.attach_odds(games, fo.read_betting(betting_table.iloc[:3]), fo.read_moneylines(moneyline_table))
    assert pd.isna(with_odds.loc[3, "spread_close"])
    assert with_odds.loc[3, "home_ml"] == 150
    assert with_odds.loc[3, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(150, -175)[0])
    assert pd.isna(with_odds.loc[3, "market_margin"])


def test_moneyline_file_alone_gives_win_probabilities_but_no_spreads(games, moneyline_table):
    with_odds = fo.attach_odds(games, None, fo.read_moneylines(moneyline_table))
    assert with_odds["spread_close"].isna().all()
    assert with_odds["market_home_win_prob"].notna().all()
    assert with_odds.loc[0, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(-200, 170)[0])


def test_betting_file_alone_still_works(games, betting_table):
    with_odds = fo.attach_odds(games, fo.read_betting(betting_table), None)
    assert list(with_odds["market_margin"]) == [5.0, 12.0, 5.5, -3.5]
    assert with_odds.loc[3, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(140, -160)[0])


def test_moneylines_for_games_we_do_not_have_are_ignored(games, moneyline_table):
    moneyline_table["game_id"] = ["0021800001", "0021800002", "0021800003", "0029999999"]
    with_odds = fo.attach_odds(games, None, fo.read_moneylines(moneyline_table))
    assert len(with_odds) == 4
    assert pd.isna(with_odds.loc[3, "home_ml"])


def test_a_different_final_score_is_reported(games, betting_table):
    betting = fo.read_betting(betting_table)
    games.loc[1, "home_pts"] = 99
    with_odds = fo.attach_odds(games, betting, None)
    assert list(fo.score_mismatches(with_odds, betting)["game_id"]) == ["0021800002"]
    assert fo.score_mismatches(with_odds, None).empty


# --- the command line ----------------------------------------------------------

def run(raw_dir, games, tmp_path):
    games_file = tmp_path / "games.csv"
    games.to_csv(games_file, index=False)
    output = tmp_path / "processed" / "odds.csv"
    code = fo.main(["--raw-dir", str(raw_dir), "--games", str(games_file), "--output", str(output)])
    return code, output


def test_main_writes_one_row_per_game_keyed_by_game_id(raw_dir, games, tmp_path, capsys):
    code, output = run(raw_dir, games, tmp_path)
    assert code == 0
    written = pd.read_csv(output, dtype={"game_id": str})
    assert list(written.columns) == ["game_id", "date", "home", "away",
                                     "spread_close", "total_close", "home_ml", "away_ml"]
    assert list(written["game_id"]) == list(games["game_id"])
    printed = capsys.readouterr().out
    assert "Every game has a closing spread and every score agrees." in printed
    assert "with_moneyline" in printed
    assert "4 of its 4 games are in games.csv" in printed


def test_main_with_only_the_moneyline_file_reports_no_spreads(raw_dir, games, tmp_path, capsys):
    (raw_dir / "nba_2008-2026.csv").unlink()
    code, output = run(raw_dir, games, tmp_path)
    assert code == 0
    written = pd.read_csv(output)
    assert written["spread_close"].isna().all()
    assert written["home_ml"].notna().all()
    assert "4 games have no closing spread" in capsys.readouterr().out


def test_main_leaves_out_games_with_no_odds_at_all(raw_dir, games, tmp_path, capsys):
    extra = games.iloc[[0]].assign(game_id="0021900002", season="2019-20", date="2019-11-01")
    code, output = run(raw_dir, pd.concat([games, extra], ignore_index=True), tmp_path)
    assert code == 0
    assert len(pd.read_csv(output)) == 4
    printed = capsys.readouterr().out
    assert "Wrote odds for 4 of 5 games" in printed
    assert "0021900002" in printed


def test_main_says_when_no_moneyline_ids_match(raw_dir, games, tmp_path, capsys):
    table = pd.read_csv(raw_dir / "game_odds.csv")
    table["game_id"] = ["a1", "a2", "a3", "a4"]
    table.to_csv(raw_dir / "game_odds.csv", index=False)
    run(raw_dir, games, tmp_path)
    assert "None match" in capsys.readouterr().out


def test_main_needs_games_csv(raw_dir, tmp_path, capsys):
    assert fo.main(["--raw-dir", str(raw_dir), "--games", str(tmp_path / "none.csv")]) == 1
    assert "clean.py" in capsys.readouterr().err


def test_main_explains_an_empty_odds_folder(games, tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _ = run(empty, games, tmp_path)
    assert code == 1
    assert "Kaggle" in capsys.readouterr().err
