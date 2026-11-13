import numpy as np
import pandas as pd

from tabet.alt import returning_reset as R


def _seasons(rows):
    out = []
    for i, (s, teams) in enumerate(rows):
        for k, (h, a) in enumerate(zip(teams, teams[1:] + teams[:1])):
            out.append({"match_id": f"{s}-{k}", "season": s, "home_id": h, "away_id": a,
                        "kickoff_utc": pd.Timestamp(f"20{s[2:4]}-09-01", tz="UTC") + pd.Timedelta(days=k),
                        "home_goals": 1, "away_goals": 0, "status": "completed", "result": "H"})
    return pd.DataFrame(out)


def test_a_club_that_misses_a_season_starts_a_new_spell():
    m = _seasons([("2021-22", ["a", "b", "c"]), ("2022-23", ["a", "b", "d"]), ("2023-24", ["a", "c", "d"])])
    sp = R.spell_ids(m)
    assert sp[("c", "2021-22")] == "c" and sp[("c", "2023-24")] == "c#2"
    assert sp[("d", "2022-23")] == "d" and sp[("d", "2023-24")] == "d"  # consecutive seasons: same spell
    assert R.returning_spells(m) == ["c#2"]


def test_relabel_only_changes_ids():
    m = _seasons([("2021-22", ["a", "b", "c"]), ("2022-23", ["a", "b", "d"]), ("2023-24", ["a", "c", "d"])])
    r = R.relabel(m)
    assert (r.drop(columns=["home_id", "away_id"]).equals(m.drop(columns=["home_id", "away_id"])))
    assert set(r.loc[r.season == "2023-24", "home_id"]) == {"a", "c#2", "d"}
