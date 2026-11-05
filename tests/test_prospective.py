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
