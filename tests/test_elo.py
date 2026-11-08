import numpy as np
import pandas as pd

from tabet import elo


def _league(seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(6)]
    rows = []
    t0 = pd.Timestamp("2020-08-15 14:00", tz="UTC")
    for r in range(8):
        perm = rng.permutation(6)
        for k in range(3):
            h, a = teams[perm[2 * k]], teams[perm[2 * k + 1]]
            hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
            rows.append({"match_id": f"m{r}{k}", "season": "2020-21", "kickoff_utc": t0 + pd.Timedelta(days=7 * r, hours=2 * k),
                         "home_id": h, "away_id": a, "home_goals": hg, "away_goals": ag, "status": "completed",
                         "result": "H" if hg > ag else ("A" if ag > hg else "D")})
    return pd.DataFrame(rows)


def test_gaps_use_only_earlier_matchdays():
    m = _league()
    g1 = elo.pre_match_gaps(m).set_index("match_id")["elo_gap"]
    m2 = m.copy()
    m2.loc[m2.match_id == "m41", ["home_goals", "away_goals", "result"]] = [6, 0, "H"]
    g2 = elo.pre_match_gaps(m2).set_index("match_id")["elo_gap"]
    same_or_earlier = [i for i in g1.index if int(i[1]) <= 4]
    later = [i for i in g1.index if int(i[1]) > 4]
    assert np.allclose(g1[same_or_earlier], g2[same_or_earlier])
    assert not np.allclose(g1[later], g2[later])


def test_ordered_logit_probabilities_are_valid_and_monotone():
    th = np.array([-0.5, 0.6, 2.0])
    p = elo.probs_from_gap(th, np.linspace(-1, 1, 21))
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    assert np.all(np.diff(p[:, 0]) > 0) and np.all(np.diff(p[:, 2]) < 0)


def test_newcomers_start_at_the_leavers_mean():
    m = _league()
    late = m["kickoff_utc"] >= m["kickoff_utc"].min() + pd.Timedelta(days=28)
    m.loc[late, "season"] = "2021-22"
    m.loc[late, "home_id"] = m.loc[late, "home_id"].replace({"t5": "new"})
    m.loc[late, "away_id"] = m.loc[late, "away_id"].replace({"t5": "new"})
    g = elo.pre_match_gaps(m)
    first_new = g[(g.season == "2021-22") & ((g.home_id == "new") | (g.away_id == "new"))].iloc[0]
    # t5's final 2020-21 rating is what "new" inherits, so the gap equals the one t5 would have had
    m_alt = m.copy()
    m_alt[["home_id", "away_id"]] = m_alt[["home_id", "away_id"]].replace({"new": "t5"})
    g_alt = elo.pre_match_gaps(m_alt).set_index("match_id")["elo_gap"]
    assert np.isclose(first_new["elo_gap"], g_alt[first_new["match_id"]])
