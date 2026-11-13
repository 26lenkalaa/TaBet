"""Tests for fetch_games.py. NBA.com is never called: a fake endpoint stands in.

Run from the repo root:
    python3 -m pytest basketball/tests -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ingest"))

import fetch_games as fg  # noqa: E402


def team_rows(game_ids: list[str]) -> pd.DataFrame:
    """Two rows per game, like NBA.com returns."""
    rows = []
    for game_id in game_ids:
        rows.append({"GAME_ID": game_id, "TEAM_ABBREVIATION": "BOS", "MATCHUP": "BOS vs. NYK", "PTS": 110})
        rows.append({"GAME_ID": game_id, "TEAM_ABBREVIATION": "NYK", "MATCHUP": "NYK @ BOS", "PTS": 104})
    return pd.DataFrame(rows)


class FakeLog:
    """Stands in for leaguegamelog.LeagueGameLog."""

    calls: list[dict[str, object]] = []
    fail_times = 0
    frame = team_rows(["001", "002"])

    def __init__(self, **kwargs: object) -> None:
        FakeLog.calls.append(kwargs)
        if FakeLog.fail_times > 0:
            FakeLog.fail_times -= 1
            raise TimeoutError("NBA.com timed out")

    def get_data_frames(self) -> list[pd.DataFrame]:
        return [FakeLog.frame]


@pytest.fixture(autouse=True)
def fake_nba(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.calls = []
    FakeLog.fail_times = 0
    FakeLog.frame = team_rows(["001", "002"])
    monkeypatch.setattr(fg.leaguegamelog, "LeagueGameLog", FakeLog)
    monkeypatch.setattr(fg.time, "sleep", lambda _: None)
    monkeypatch.setattr(fg, "RAW_DIR", tmp_path)


def run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["fetch_games.py", *args])
    fg.main()


def test_saves_one_file_per_season_and_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run(monkeypatch, "--seasons", "2021-22", "2022-23")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "games_2021-22_playoffs.csv", "games_2021-22_regular.csv",
        "games_2022-23_playoffs.csv", "games_2022-23_regular.csv",
    ]
    assert len(pd.read_csv(tmp_path / "games_2022-23_regular.csv")) == 4


def test_asks_nba_com_for_team_rows_and_right_season_type(monkeypatch: pytest.MonkeyPatch) -> None:
    run(monkeypatch, "--seasons", "2022-23", "--types", "playoffs")
    assert FakeLog.calls == [{
        "season": "2022-23",
        "season_type_all_star": "Playoffs",
        "player_or_team_abbreviation": "T",
        "timeout": fg.TIMEOUT_SECONDS,
    }]


def test_skips_existing_files_unless_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    run(monkeypatch, "--seasons", "2022-23", "--types", "regular")
    run(monkeypatch, "--seasons", "2022-23", "--types", "regular")
    assert len(FakeLog.calls) == 1
    run(monkeypatch, "--seasons", "2022-23", "--types", "regular", "--force")
    assert len(FakeLog.calls) == 2


def test_retries_after_timeouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.fail_times = 2
    run(monkeypatch, "--seasons", "2022-23", "--types", "regular")
    assert len(FakeLog.calls) == 3
    assert (tmp_path / "games_2022-23_regular.csv").exists()


def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.fail_times = fg.MAX_ATTEMPTS
    with pytest.raises(SystemExit):
        run(monkeypatch, "--seasons", "2022-23", "--types", "regular")
    assert not any(tmp_path.iterdir())


def test_rejects_game_with_one_team_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLog.frame = team_rows(["001"]).iloc[:1]
    with pytest.raises(SystemExit):
        run(monkeypatch, "--seasons", "2022-23", "--types", "regular")
    assert not any(tmp_path.iterdir())


def test_empty_play_in_is_fine_but_empty_regular_season_is_not() -> None:
    empty = pd.DataFrame(columns=["GAME_ID"])
    fg.check(empty, "2018-19", "playin")
    with pytest.raises(ValueError):
        fg.check(empty, "2018-19", "regular")
