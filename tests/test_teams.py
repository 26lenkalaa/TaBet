import pytest

from tabet.teams import UnknownTeamError, resolve_team


@pytest.mark.parametrize("name,expected", [
    ("Nott'm Forest", "nottm_forest"),
    ("Nottingham Forest", "nottm_forest"),
    ("  man   united ", "man_united"),
    ("Manchester United", "man_united"),
    ("Spurs", "tottenham"),
    ("wolves", "wolves"),
])
def test_aliases(name, expected):
    assert resolve_team(name) == expected


def test_unknown_team_raises():
    with pytest.raises(UnknownTeamError):
        resolve_team("Real Madrid")
