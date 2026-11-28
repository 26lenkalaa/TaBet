"""Offline tests for fetch_schedule.py: NBA.com is never called."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

import src.ingest.fetch_schedule as fs  # noqa: E402


def game(game_id, day, home, away, status=1, home_id=1610612738, away_id=1610612752):
    """One game the way NBA.com's schedule table lists it."""
    return {
        "gameId": game_id, "gameStatus": status, "gameStatusText": "7:30 pm ET",
        "gameDateEst": f"{day}T00:00:00Z", "gameDateTimeUTC": f"{day}T23:30:00Z",
        "homeTeam_teamId": home_id, "homeTeam_teamTricode": home, "homeTeam_teamName": home,
        "awayTeam_teamId": away_id, "awayTeam_teamTricode": away, "awayTeam_teamName": away,
    }


@pytest.fixture
def raw():
    return pd.DataFrame([
        game("0022600300", "2026-11-18", "BOS", "NYK"),
        game("0022600250", "2026-11-16", "NYK", "BOS", status=3),
        game("0012600010", "2026-10-05", "BOS", "NYK", status=3),            # preseason
        game("0032600001", "2027-02-14", "EST", "WST"),                      # All-Star
        game("0042600101", "2027-04-17", None, None, home_id=0, away_id=0),  # playoff slot, teams unknown
        game("0052600001", "2027-04-13", "MIA", "CHI", home_id=1610612748, away_id=1610612741),
    ])


def test_regular_season_and_play_in_games_are_kept(raw):
    schedule = fs.parse_schedule(raw)
    assert list(schedule["game_id"]) == ["0022600250", "0022600300", "0052600001"]   # date order
    assert list(schedule["game_type"]) == ["regular", "regular", "play_in"]
    assert list(schedule.columns) == fs.COLUMNS


def test_each_row_has_date_teams_and_whether_it_was_played(raw):
    schedule = fs.parse_schedule(raw).set_index("game_id")
    row = schedule.loc["0022600300"]
    assert row["date"] == "2026-11-18"
    assert (row["home"], row["away"]) == ("BOS", "NYK")
    assert (row["home_team_id"], row["away_team_id"]) == (1610612738, 1610612752)
    assert not row["played"]
    assert schedule.loc["0022600250", "played"]


def test_other_date_spellings_are_read(raw):
    raw["gameDateEst"] = raw["gameDateEst"].str[:10]
    assert fs.parse_schedule(raw).set_index("game_id").loc["0022600300", "date"] == "2026-11-18"


def test_game_ids_keep_ten_digits(raw):
    raw["gameId"] = raw["gameId"].astype(int)
    assert "0022600300" in set(fs.parse_schedule(raw)["game_id"])


def test_a_game_listed_twice_is_kept_once(raw):
    assert fs.parse_schedule(pd.concat([raw, raw]))["game_id"].is_unique


def test_a_changed_layout_is_reported_clearly(raw):
    with pytest.raises(ValueError, match="missing columns: homeTeam_teamTricode"):
        fs.parse_schedule(raw.drop(columns="homeTeam_teamTricode"))


class FakeSchedule:
    """Stands in for scheduleleaguev2.ScheduleLeagueV2."""

    calls: list[dict] = []
    fail_times = 0
    frame = pd.DataFrame()

    def __init__(self, **kwargs):
        FakeSchedule.calls.append(kwargs)
        if FakeSchedule.fail_times:
            FakeSchedule.fail_times -= 1
            raise TimeoutError("NBA.com timed out")
        self.season_games = self

    def get_data_frame(self):
        return FakeSchedule.frame


@pytest.fixture
def fake_nba(raw, monkeypatch):
    FakeSchedule.calls, FakeSchedule.fail_times, FakeSchedule.frame = [], 0, raw
    monkeypatch.setattr(fs.scheduleleaguev2, "ScheduleLeagueV2", FakeSchedule)
    monkeypatch.setattr(fs.time, "sleep", lambda seconds: None)
    return FakeSchedule


def test_download_asks_for_the_current_season_by_default(fake_nba, monkeypatch):
    monkeypatch.setattr(fs, "current_season", lambda: "2026-27")
    fs.download()
    fs.download("2025-26")
    assert [call["season"] for call in fake_nba.calls] == ["2026-27", "2025-26"]


def test_download_retries_after_a_timeout(fake_nba):
    fake_nba.fail_times = 2
    assert len(fs.download("2026-27")) == 6
    assert len(fake_nba.calls) == 3


def test_download_gives_up_after_the_last_attempt(fake_nba):
    fake_nba.fail_times = 99
    with pytest.raises(OSError, match="TimeoutError"):
        fs.download("2026-27")
    assert len(fake_nba.calls) == fs.MAX_ATTEMPTS


def test_main_writes_schedule_csv(fake_nba, tmp_path, capsys):
    output = tmp_path / "raw" / "schedule.csv"
    assert fs.main(["--season", "2026-27", "--output", str(output)]) == 0
    written = pd.read_csv(output, dtype={"game_id": str})
    assert list(written.columns) == fs.COLUMNS
    assert len(written) == 3
    assert "2 still to play" in capsys.readouterr().out


def test_main_reports_a_failed_download(fake_nba, tmp_path, capsys):
    fake_nba.fail_times = 99
    assert fs.main(["--output", str(tmp_path / "schedule.csv")]) == 1
    assert "could not get the schedule" in capsys.readouterr().err


def test_main_reports_an_empty_schedule(fake_nba, raw, tmp_path, capsys):
    fake_nba.frame = raw.iloc[2:4]   # only preseason and All-Star
    assert fs.main(["--output", str(tmp_path / "schedule.csv")]) == 1
    assert "no regular-season" in capsys.readouterr().err
