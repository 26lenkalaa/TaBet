import pandas as pd
import pytest

from tabet.store import RESULT_LAG, MatchStore


@pytest.fixture(scope="module")
def store(real):
    return MatchStore(real[0])


def test_no_leakage_across_many_cutoffs(store, real):
    m = real[0]
    cutoffs = pd.date_range("2021-08-01", "2026-06-30", freq="37h", tz="UTC")
    for cutoff in cutoffs[::5]:
        seen = store.completed_before(cutoff)
        assert (seen["kickoff_utc"] + RESULT_LAG <= cutoff).all()
        # ...and nothing that *was* available is withheld
        expected = (m["kickoff_utc"] + RESULT_LAG <= cutoff).sum()
        assert len(seen) == expected


def test_match_in_progress_at_cutoff_excluded(store):
    kickoff = pd.Timestamp("2021-08-13 19:00", tz="UTC")  # Brentford v Arsenal
    during = store.get_history("Arsenal", kickoff + pd.Timedelta(minutes=100))
    after = store.get_history("Arsenal", kickoff + RESULT_LAG)
    assert len(during) == 0
    assert len(after) == 1


def test_history_from_team_perspective(store):
    h = store.get_history("Arsenal", "2021-08-15T00:00Z").iloc[0]
    assert (h["venue"], h["opponent_id"], h["goals_for"], h["goals_against"], h["outcome"], h["points"]) == (
        "A", "brentford", 0, 2, "L", 0)


def test_last_n_and_season_totals(store):
    h = store.get_history("Man City", "2022-07-01T00:00Z")
    assert len(h) == 38  # full 2021-22 season
    assert len(store.get_history("Man City", "2022-07-01T00:00Z", last=5)) == 5
    assert h["points"].sum() == 93  # Man City 2021-22 title total


def test_naive_cutoff_rejected(store):
    with pytest.raises(ValueError, match="timezone"):
        store.get_history("Arsenal", "2022-01-01 12:00")
