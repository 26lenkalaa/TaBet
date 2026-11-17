"""Addenda NLX/XIA script on synthetic leagues: features from earlier dates only, every model runs
end to end, and the chosen settings drive the test specs (the test runs once, so it must not be
the first time the code runs)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROTO = Path(__file__).resolve().parent.parent / "prototypes"
sys.path.insert(0, str(PROTO))
spec = importlib.util.spec_from_file_location("run_nlx", PROTO / "run_nlx.py")
nlx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nlx)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dxi import fixture  # noqa: E402


def synthetic(leagues=((140, 0), (135, 1)), seasons=(2018, 2019, 2020, 2021), teams=10):
    m, t, s = [], [], []
    for league, seed in leagues:
        rng = np.random.default_rng(seed)
        fid = league * 100000
        for y in seasons:
            day = pd.Timestamp(f"{y}-08-06", tz="UTC")
            pairs = [(h, a) for h in range(1, teams + 1) for a in range(1, teams + 1) if h != a]
            rng.shuffle(pairs)
            for i in range(0, len(pairs), teams // 2):
                for h, a in pairs[i:i + teams // 2]:
                    hh, aa = league * 100 + h, league * 100 + a
                    f = fixture(fid, day.isoformat(), hh, aa, int(rng.poisson(1.5)), int(rng.poisson(1.1)),
                                poss_h=int(rng.integers(35, 66)), rng=rng)
                    f["statistics"][0]["statistics"].append({"type": "Shots on Goal", "value": int(rng.poisson(5))})
                    f["statistics"][1]["statistics"].append({"type": "Shots on Goal", "value": int(rng.poisson(4))})
                    mm, tt, st = nlx.D.parse_fixture(f, league, y)
                    for r in tt:
                        r["sot"] = float({str(x["team"]["id"]): x["statistics"][1]["value"] for x in f["statistics"]}[r["team_id"]])
                    mm["has_details"] = True
                    m.append(mm), t.extend(tt), s.extend(st)
                    fid += 1
                day += pd.Timedelta(days=7)
    tb = {"matches": pd.DataFrame(m), "teams": pd.DataFrame(t), "starters": pd.DataFrame(s)}
    tb["starters"]["has_stats"] = tb["starters"]["has_stats"].astype("boolean")
    tb["matches"]["has_details"] = tb["matches"]["has_details"].astype("boolean")
    return tb


@pytest.fixture(scope="module")
def rows():
    tb = synthetic()
    nlx.dc_all = lambda m: {k: v for lg, lm in m.groupby("league") for k, v in nlx.D.dc_rates(lm).items()}   # no Supabase
    return tb, nlx.side_rows(tb, {140: "SP1", 135: "I1"})


def test_completeness(rows):
    comp = nlx.completeness(rows[0], {140: "SP1"}, ["2019-20"])
    assert comp["SP1 2019-20"]["share"] == 1.0


def test_side_rows_are_complete_pairs_with_every_input(rows):
    r = rows[1]
    assert len(r) and (r.groupby("match_id")["side"].nunique() == 2).all()
    assert np.isfinite(r[nlx.FEATS_X].to_numpy()).all()
    assert set(r.loc[r["div"] == "SP1", "lg_SP1"]) == {1.0} and set(r.loc[r["div"] == "SP1", "lg_I1"]) == {0.0}
    assert (r[["own_n", "opp_n"]] >= nlx.MIN_PREV_XI).all().all()
    assert (r.loc[r.side == "H", "f10"] == 1).all() and (r.loc[r.side == "A", "f10"] == 0).all()
    h = r[r.side == "H"].set_index("match_id")
    a = r[r.side == "A"].set_index("match_id").reindex(h.index)
    assert np.allclose(h["own_att"], a["opp_att"]) and np.allclose(h["f5"], -a["f5"])


def test_every_model_runs_and_choice_drives_the_test_specs(rows, monkeypatch):
    monkeypatch.setattr(nlx, "MIN_STAGE2", 100)
    monkeypatch.setattr(nlx, "L2S", (1.0, 10.0))
    monkeypatch.setattr(nlx, "GBM_GRID", [(2, 50)])
    monkeypatch.setattr(nlx, "NN_GRID", [(16, 0.01)])
    monkeypatch.setattr(nlx, "MAX_ROUNDS", 60)
    r = rows[1]
    w, rounds = nlx.walk_forward(r, ["SP1", "I1"], ["2021-22"], nlx.all_specs())
    cols = ["DC", "LIN_1.0", "LIN_10.0", "LINXI_1.0", "GBM_2_50", "NN_16_0.01"]
    assert len(w) and np.isfinite(w[cols].to_numpy()).all() and rounds
    choice = {"lin_l2": 1.0, "linxi_l2": 10.0, "gbm": [2, 50], "nn": [16, 0.01], "nl": "GBM"}
    sp = nlx.all_specs(choice)
    assert sp["LINX"][1] == nlx.FEATS_X and sp["LINXI"][1] == nlx.FEATS_NOXI and sp["GBM"][2] == (2, 50)
    se = nlx.block_se(w.assign(d=w["GBM_2_50"] - w["LIN_1.0"]), "d", reps=100)
    assert se >= 0


def test_nn_fallback_switch():
    """The most recent 15% reverse the pattern the network learns, so it never beats Dixon-Coles on them:
    with the fallback it returns no adjustment; with fallback=False (NLX) it keeps its adjustment."""
    from tabet.neural import NNParams, _fit_predict
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 1))
    sign = np.where(np.arange(n) < int(n * 0.85), 1.0, -1.0)
    y = rng.poisson(np.exp(0.8 * sign * X[:, 0])).astype(float)
    off = np.zeros(n)
    kw = dict(hidden=8, epochs=100, lr=0.05, seeds=2, early_stop=True)
    assert np.allclose(_fit_predict(X, off, y, X[:5], NNParams(**kw)), 0)
    assert np.abs(_fit_predict(X, off, y, X[:5], NNParams(**kw, fallback=False))).max() > 0
