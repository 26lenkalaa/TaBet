"""Offline tests for export_site.py, using made-up games in a temporary database."""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import backtest, db, export_site, market  # noqa: E402

BOS, NYK, LAL, GSW = 1, 2, 3, 4
NAMES = {BOS: "BOS", NYK: "NYK", LAL: "LAL", GSW: "GSW"}
AS_OF = date(2020, 12, 24)


@pytest.fixture
def games():
    results = [
        ("2018-10-16", "2018-19", "regular", BOS, NYK, 110, 95),
        ("2018-10-17", "2018-19", "regular", LAL, GSW, 101, 104),
        ("2019-10-22", "2019-20", "regular", NYK, LAL, 99, 108),
        ("2019-10-22", "2019-20", "regular", GSW, BOS, 120, 111),
        ("2020-08-17", "2019-20", "playoffs", LAL, BOS, 112, 107),
        ("2020-12-22", "2020-21", "regular", GSW, NYK, 125, 100),
        ("2020-12-23", "2020-21", "regular", BOS, LAL, 101, 109),
    ]
    rows = []
    for number, (day, season, game_type, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00200{number:05d}", "season": season, "date": day,
            "game_type": game_type, "is_playoffs": game_type == "playoffs",
            "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
            "home_pts": home_pts, "away_pts": away_pts,
            "home_possessions": 100.0, "away_possessions": 100.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def odds(games):
    return pd.DataFrame({
        "game_id": games["game_id"].iloc[:5],                # nothing for 2020-21
        "spread_close": [-6.0, 2.5, 4.0, -3.0, -1.5],
        "total_close": [215.5] * 5,
        "home_ml": [-240, 120, 160, -150, None],
        "away_ml": [200, -140, -190, 130, None],
    })


@pytest.fixture
def schedule():
    rows = [
        ("0022000100", "2020-12-22", GSW, NYK, True),     # already played: must not be forecast
        ("0022000101", "2020-12-24", NYK, BOS, False),    # BOS played on the 23rd: back-to-back
        ("0022000102", "2020-12-25", LAL, NYK, False),    # NYK back-to-back, counted from the schedule
        ("0022000103", "2021-03-01", GSW, LAL, False),    # beyond the window
    ]
    return pd.DataFrame([{"game_id": game_id, "date": day, "game_type": "regular",
                          "home_team_id": home, "home": NAMES[home], "away_team_id": away, "away": NAMES[away],
                          "played": played} for game_id, day, home, away, played in rows])


def test_payload_has_the_newest_two_seasons_newest_first(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF)
    assert [report["summary"]["season"] for report in payload["reports"]] == ["2020-21", "2019-20"]
    assert payload["as_of"] == "2020-12-24"
    assert payload["model"] == "sit-v3"
    assert payload["results_through"] == "2020-12-23"


def test_chosen_seasons_can_be_exported(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF, seasons=["2018-19"])
    assert [report["summary"]["season"] for report in payload["reports"]] == ["2018-19"]
    with pytest.raises(ValueError, match="No games for season 2030-31"):
        export_site.build_payload(games, odds, None, AS_OF, seasons=["2030-31"])


def test_each_game_has_forecast_result_and_market(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF)
    table = backtest.build_table(games, odds, "situational").set_index("game_id")
    season = next(report for report in payload["reports"] if report["summary"]["season"] == "2019-20")
    first = season["games"][0]
    assert first["game_id"] == "0020000003"
    assert (first["home_team"], first["away_team"], first["home_score"], first["away_score"]) == ("NYK", "LAL", 99, 108)
    assert first["p_home"] == pytest.approx(table.loc["0020000003", "home_win_prob"], abs=1e-4)
    assert first["margin"] == pytest.approx(table.loc["0020000003", "predicted_margin"], abs=0.05)
    assert first["market_p_home"] == pytest.approx(market.moneylines_to_probs(160, -190)[0], abs=1e-4)
    assert first["spread"] == 4.0
    assert [game["game_type"] for game in season["games"]] == ["regular", "regular", "playoffs"]


def test_a_game_with_a_spread_but_no_moneyline_still_has_a_market_probability(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF)
    playoff = payload["reports"][1]["games"][-1]
    assert playoff["spread"] == -1.5
    assert 0.5 < playoff["market_p_home"] < 0.6


def test_summary_scores_model_and_market_on_the_same_games(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF)
    summary = payload["reports"][1]["summary"]
    table = backtest.build_table(games, odds, "situational")
    season = table[table["season"] == "2019-20"]
    assert (summary["games"], summary["scored_games"], summary["missing_odds"]) == (3, 3, 0)
    assert summary["model_brier"] == pytest.approx(
        backtest.brier_score(season["home_win_prob"], season["home_won"]), abs=1e-4)
    assert summary["market_brier"] == pytest.approx(
        backtest.brier_score(season["market_home_win_prob"], season["home_won"]), abs=1e-4)


def test_a_season_without_odds_has_no_market_score_but_keeps_the_models(games, odds):
    summary = export_site.build_payload(games, odds, None, AS_OF)["reports"][0]["summary"]
    assert (summary["games"], summary["scored_games"], summary["missing_odds"]) == (2, 0, 2)
    assert summary["model_brier"] is None and summary["market_brier"] is None
    assert 0 < summary["model_brier_all"] < 1
    game = export_site.build_payload(games, odds, None, AS_OF)["reports"][0]["games"][0]
    assert game["market_p_home"] is None and game["spread"] is None


def test_games_on_or_after_the_snapshot_date_are_not_used(games, odds):
    payload = export_site.build_payload(games, odds, None, date(2020, 12, 23))
    assert payload["results_through"] == "2020-12-22"
    assert payload["reports"][0]["summary"]["games"] == 1
    with pytest.raises(ValueError, match="No games before"):
        export_site.build_payload(games, odds, None, date(2018, 1, 1))


def test_without_a_schedule_there_are_no_upcoming_games(games, odds):
    payload = export_site.build_payload(games, odds, None, AS_OF)
    assert payload["upcoming"] == []
    assert payload["has_schedule"] is False


def test_upcoming_games_are_the_unplayed_ones_inside_the_window(games, odds, schedule):
    payload = export_site.build_payload(games, odds, schedule, AS_OF, days=14)
    assert payload["has_schedule"] is True
    assert [game["game_id"] for game in payload["upcoming"]] == ["0022000101", "0022000102"]
    first = payload["upcoming"][0]
    assert (first["home_team"], first["away_team"], first["game_date"]) == ("NYK", "BOS", "2020-12-24")
    assert 0 < first["p_home"] < 1
    assert isinstance(first["home_pts"], int) and isinstance(first["away_pts"], int)


def test_upcoming_rest_follows_the_schedule_not_only_played_games(games, odds, schedule):
    upcoming = export_site.build_payload(games, odds, schedule, AS_OF)["upcoming"]
    assert (upcoming[0]["home_rest_days"], upcoming[0]["away_rest_days"]) == (2, 1)   # NYK 22nd, BOS 23rd
    assert upcoming[1]["away_rest_days"] == 1     # NYK plays on the 24th (scheduled), then the 25th
    assert upcoming[1]["home_rest_days"] == 2     # LAL last played on the 23rd


def test_a_longer_window_reaches_later_games(games, odds, schedule):
    payload = export_site.build_payload(games, odds, schedule, AS_OF, days=90)
    assert "0022000103" in [game["game_id"] for game in payload["upcoming"]]


def test_main_writes_compact_json_the_page_can_read(games, odds, schedule, tmp_path, capsys):
    db_file = tmp_path / "basketball.db"
    box = games.drop(columns=["home_possessions", "away_possessions"]).assign(
        home_fga=88, home_fta=22, home_oreb=10, home_tov=13, away_fga=86, away_fta=24, away_oreb=9, away_tov=14)
    db.build(db.connect(db_file), box, odds)
    schedule_file = tmp_path / "schedule.csv"
    schedule.to_csv(schedule_file, index=False)
    output = tmp_path / "site" / "data" / "basketball.json"

    assert export_site.main(["--db", str(db_file), "--schedule", str(schedule_file), "--output", str(output),
                             "--as-of", "2020-12-24"]) == 0

    payload = json.loads(output.read_text())
    assert set(payload) == {"as_of", "model", "results_through", "reports", "upcoming", "has_schedule"}
    assert len(payload["upcoming"]) == 2
    assert "NaN" not in output.read_text()
    assert "upcoming games: 2" in capsys.readouterr().out


def test_main_works_without_a_schedule_and_says_so(games, odds, tmp_path, capsys):
    db_file = tmp_path / "basketball.db"
    db.build(db.connect(db_file), games.drop(columns=["home_possessions", "away_possessions"]), odds)
    output = tmp_path / "basketball.json"
    assert export_site.main(["--db", str(db_file), "--schedule", str(tmp_path / "none.csv"),
                             "--output", str(output), "--as-of", "2020-12-24"]) == 0
    assert json.loads(output.read_text())["upcoming"] == []
    assert "fetch_schedule.py" in capsys.readouterr().out


def test_main_explains_a_missing_database(tmp_path, capsys):
    assert export_site.main(["--db", str(tmp_path / "missing.db"), "--output", str(tmp_path / "x.json")]) == 1
    assert "db.py" in capsys.readouterr().err
