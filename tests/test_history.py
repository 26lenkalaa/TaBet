import itertools

import pandas as pd
import pytest

from tabet.fpl import FplError
from tabet.history import _fixture_teams, _map_teams


def _toy(n=20):
    teams = [f"t{i:02d}" for i in range(n)]
    rows, gw, fid = [], [], 0
    day = pd.Timestamp("2017-08-12 14:00", tz="UTC")
    for k, (h, a) in enumerate(itertools.permutations(range(n), 2)):
        fid += 1
        ko = day + pd.Timedelta(days=k // 10)
        hg, ag = (k * 7) % 4, (k * 3) % 3
        rows.append({"season": "2017-18", "home_id": teams[h], "away_id": teams[a], "kickoff_utc": ko,
                     "home_goals": hg, "away_goals": ag, "status": "completed"})
        # FPL ids are a permutation of ours
        fh, fa = (h * 7) % n + 1, (a * 7) % n + 1
        for home, opp in ((True, fa), (False, fh)):
            gw.append({"fixture": fid, "kickoff_time": ko.isoformat(), "team_h_score": hg, "team_a_score": ag,
                       "was_home": home, "opponent_team": opp})
    return pd.DataFrame(rows), pd.DataFrame(gw), teams


def test_fpl_team_ids_recovered_from_opponents_and_scores():
    ours, gw, teams = _toy()
    fx = _fixture_teams(gw)
    mp = _map_teams(fx, ours, "2017-18")
    assert all(mp[(i * 7) % 20 + 1] == teams[i] for i in range(20))


def test_mapping_fails_loudly_when_teams_cannot_be_identified():
    ours, gw, _ = _toy()
    gw = gw[gw["opponent_team"] != 3]  # one club's opponents never appear
    with pytest.raises(FplError):
        _map_teams(_fixture_teams(gw.assign(opponent_team=1)), ours, "2017-18")
