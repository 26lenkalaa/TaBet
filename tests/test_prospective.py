import numpy as np
import pandas as pd
import pytest

from tabet.prospective import fit_alpha, summarise


def _scores(n=400, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p0 = rng.dirichlet([5, 3, 3])
        truth = rng.dirichlet([5, 3, 3]) if informative else p0
        pe = 0.5 * p0 + 0.5 * truth  # pE carries half of the true signal
        y = "HDA"[rng.choice(3, p=truth)]
        for name, p in (("p0", p0), ("pE", pe)):
            rows.append({"fpl_fixture": i, "gameweek": 10 + i // 40, "horizon_h": 24.0, "candidate": name,
                         "result": y, "log_loss": -np.log(p["HDA".index(y)]),
                         "p_home": p[0], "p_draw": p[1], "p_away": p[2]})
    return pd.DataFrame(rows)


def test_alpha_moves_toward_informative_source_and_stays_bounded():
    s = _scores()
    a = fit_alpha(s, through_gameweek=19)
    assert 0.5 < a <= 1.0
    assert fit_alpha(_scores(informative=False), through_gameweek=19) == pytest.approx(0.0, abs=0.35)


def test_alpha_uses_only_earlier_gameweeks():
    s = _scores()
    a1 = fit_alpha(s, through_gameweek=12)
    later = s[s.gameweek > 12].copy()
    later["result"] = "D"  # rewrite the future
    s2 = pd.concat([s[s.gameweek <= 12], later])
    assert fit_alpha(s2, through_gameweek=12) == a1


def test_summary_counts_each_fixture_once_and_pairs_with_p0():
    s = _scores(n=50)
    t = summarise(s).set_index("candidate")
    assert t.loc["p0", "n"] == 50 and t.loc["pE", "n"] == 50
    assert t.loc["p0", "minus_p0"] == pytest.approx(0)


# ---- amendment A2 rules ----
from tabet.lineup_sources import availability_field, unavailable_codes  # noqa: E402
from tabet.prospective import ALPHA_MIN_FIXTURES, _ko  # noqa: E402


def _boot():
    return {"events": [{"id": 10, "deadline_time": "2026-11-06T18:30:00Z"},
                       {"id": 11, "deadline_time": "2026-11-21T11:00:00Z"}]}


def test_availability_field_follows_the_fixtures_own_gameweek():
    before = pd.Timestamp("2026-11-06T12:00Z")
    after = pd.Timestamp("2026-11-07T14:00Z")  # a Sunday GW10 fixture logged after the GW10 deadline
    assert availability_field(_boot(), 10, before) == "chance_next"
    assert availability_field(_boot(), 10, after) == "chance_this"
    assert availability_field(_boot(), 11, after) == "chance_next"


def test_missing_chance_counts_as_zero_unless_available():
    fp = pd.DataFrame({"player_code": [1, 2, 3, 4, 5], "status": ["i", "s", "a", "d", "d"],
                       "chance": [None, None, None, 75, 25]})
    assert unavailable_codes(fp, "chance") == {1, 2, 5}


def test_moved_kickoff_is_a_new_logging_key():
    assert _ko("2026-11-08T14:00:00Z") == _ko(pd.Timestamp("2026-11-08 14:00", tz="UTC"))
    assert _ko("2026-11-08T14:00:00Z") != _ko("2026-11-09T20:00:00Z")


def test_alpha_defaults_to_zero_when_degenerate():
    s = _scores(n=ALPHA_MIN_FIXTURES - 1)
    assert fit_alpha(s, through_gameweek=19) == 0.0
    flat = _scores(n=100, informative=False)
    flat.loc[flat.candidate == "pE", ["p_home", "p_draw", "p_away"]] = \
        flat.loc[flat.candidate == "p0", ["p_home", "p_draw", "p_away"]].to_numpy()
    assert fit_alpha(flat, through_gameweek=19) == 0.0


def test_superseded_records_are_left_out_of_the_summary():
    s = _scores(n=50).assign(superseded=False)
    s.loc[s.fpl_fixture < 5, "superseded"] = True
    t = summarise(s).set_index("candidate")
    assert t.loc["p0", "n"] == 45
