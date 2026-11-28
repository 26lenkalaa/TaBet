import json

import numpy as np
import pandas as pd

from tabet import dynamic as D
from tabet import shadow as S


def _rec(fixture=1, sources=("ffs", "rotowire")):
    feats = {"H": {"f_xi_position": 0.05, "f_backline_opp": 0.7}, "A": {"f_xi_position": -0.1, "f_backline_opp": None}}
    return {"fpl_fixture": fixture, "horizon_h": 24.0, "record_role": "primary", "logged_utc": "2026-11-28T15:00:00+00:00",
            "kickoff_utc": "2026-11-29T15:00:00+00:00", "gameweek": 13, "home_id": "arsenal", "away_id": "chelsea",
            "git_commit": "x", "components": {"xgadj_home": [0.1, 0.05], "xgadj_away": [-0.02, 0.0]},
            "xi_features": {s: feats for s in sources}, "pE_sources": list(sources)}


ST1 = {"coef": [0.01, 0.1, 0.05], "mean": [0.0, 0.0], "sd": [1.0, 1.0]}
STX = {"coef": [0.0, 0.0, 0.05, 0.05, -0.01], "mean": [0.0, 0.0, 0.0, 0.6], "sd": [1.0, 1.0, 0.07, 0.13]}


def test_candidates_are_probabilities_and_pE_averages_its_sources():
    c = S.candidates(_rec(), (1.6, 1.1), -0.05, ST1, STX)
    for k in ("p0", "pE", "ffs", "rotowire"):
        assert abs(sum(c[k].values()) - 1) < 1e-9
    assert np.isclose(c["pE"]["p_home"], (c["ffs"]["p_home"] + c["rotowire"]["p_home"]) / 2)


def test_pE_falls_back_to_p0_without_a_valid_source():
    c = S.candidates(_rec(sources=()), (1.6, 1.1), -0.05, ST1, STX)
    assert c["pE"] == c["p0"]


def test_a_failure_becomes_an_error_record_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SHADOW", tmp_path / "shadow.jsonl")
    assert S.write_shadow([_rec()], ctx={}) == 1          # missing inputs: an error record, no exception
    row = json.loads((tmp_path / "shadow.jsonl").read_text())
    assert row["fpl_fixture"] == 1 and "error" in row
    assert S.write_shadow([_rec()], ctx={}) == 0          # never written twice for the same key


def test_next_rates_carry_the_filter_state_forward():
    rng = np.random.default_rng(0)
    teams = [f"t{i}" for i in range(6)]
    rows, t0 = [], pd.Timestamp("2020-08-15 14:00", tz="UTC")
    for r in range(10):
        perm = rng.permutation(6)
        for k in range(3):
            hg, ag = rng.poisson(1.5), rng.poisson(1.1)
            rows.append({"match_id": f"m{r}{k}", "season": "2020-21", "kickoff_utc": t0 + pd.Timedelta(days=7 * r),
                         "home_id": teams[perm[2 * k]], "away_id": teams[perm[2 * k + 1]], "home_goals": hg,
                         "away_goals": ag, "status": "completed"})
    done = pd.DataFrame(rows)
    th = D.START.copy()
    th[5] = th[6] = 1.0                                    # no decay: the next rates are the filter's end state
    lh, la = S.next_rates(done, th, "t0", "t1", t0 + pd.Timedelta(days=80), "2020-21")
    p, m = D.prepare(done)
    _, (a, d), _ = D.filter_run(th, p)
    assert np.isclose(lh, np.exp(th[0] + th[1] + a[0] - d[1])) and np.isclose(la, np.exp(th[0] + a[1] - d[0]))
    lh2, la2 = S.next_rates(done, th, "newbie", "t1", t0 + pd.Timedelta(days=80), "2020-21")
    assert np.isclose(lh2, np.exp(th[0] + th[1] - d[1]))   # an unseen team starts at league average


def test_extra_feature_with_zero_value_leaves_candidates_unchanged():
    s1x = {"coef": ST1["coef"] + [0.3], "mean": ST1["mean"] + [0.0], "sd": ST1["sd"] + [1.0]}
    sxx = {"coef": STX["coef"] + [0.3], "mean": STX["mean"] + [0.0], "sd": STX["sd"] + [1.0]}
    base = S.candidates(_rec(), (1.6, 1.1), -0.05, ST1, STX)
    zero = S.candidates(_rec(), (1.6, 1.1), -0.05, s1x, sxx, {"home": [0.0], "away": [0.0]})
    assert all(np.isclose(base[k]["p_home"], zero[k]["p_home"]) for k in base)
    up = S.candidates(_rec(), (1.6, 1.1), -0.05, s1x, sxx, {"home": [1.0], "away": [-1.0]})
    assert up["pE"]["p_home"] > base["pE"]["p_home"]            # a costlier home squad raises the home chance


def test_a_squad_cost_failure_keeps_the_s2_candidates(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no transfer table")
    monkeypatch.setattr(S, "cost_candidates", boom)
    out = [{"candidates": {"pES2": {"p_home": 0.5}}}]
    S.add_cost(out, [_rec()], {})
    assert out[0]["candidates"] == {"pES2": {"p_home": 0.5}}
    assert out[0]["cost_error"].startswith("RuntimeError")


def test_an_s2c_failure_keeps_the_earlier_candidates(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no transfer table")
    monkeypatch.setattr(S, "s2c_candidates", boom)
    out = [{"candidates": {"pES2": {"p_home": 0.5}, "pEC": {"p_home": 0.6}}}]
    S.add_s2c(out, [_rec()], {}, pd.DataFrame(), np.array([]), [((1.6, 1.1), -0.05)])
    assert out[0]["candidates"] == {"pES2": {"p_home": 0.5}, "pEC": {"p_home": 0.6}}
    assert out[0]["s2c_error"].startswith("RuntimeError")
