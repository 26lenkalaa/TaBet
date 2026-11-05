import pandas as pd
import pytest

from tabet.leagues import DIR, load_leagues, validate_leagues

pytestmark = pytest.mark.skipif(not any(DIR.glob("*.csv")), reason="other-league files not downloaded")


@pytest.fixture(scope="module")
def other():
    return load_leagues()


def test_complete_double_round_robins(other):
    done = other[other["season"] < "2026-27"]
    for (lg, season), s in done.groupby(["league", "season"]):
        n = len(set(s["home_id"]) | set(s["away_id"]))
        assert len(s) == n * (n - 1), (lg, season)


def test_team_ids_are_league_prefixed_and_disjoint(other):
    for lg, s in other.groupby("league"):
        ids = set(s["home_id"]) | set(s["away_id"])
        assert all(i.startswith(lg.lower() + "_") for i in ids)


def test_validation_catches_repeated_fixture(other):
    s = other[(other["league"] == "SP1") & (other["season"] == "2022-23")]
    dup = pd.concat([other, s.iloc[[0]].assign(match_id="dup")])
    assert any("repeated fixture" in p for p in validate_leagues(dup))
