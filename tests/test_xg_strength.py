import numpy as np
import pandas as pd
import pytest

from tabet.features import known
from tabet.xg_strength import fit_xg_strength

T0 = pd.Timestamp("2023-08-01 15:00", tz="UTC")


def _sides(n_rounds=40, seed=0, start=T0):
    rng = np.random.default_rng(seed)
    teams = [f"t{i}" for i in range(8)]
    att = dict(zip(teams, np.linspace(-0.4, 0.4, 8)))
    dfn = dict(zip(teams, np.linspace(0.3, -0.3, 8)))
    rows = []
    for r in range(n_rounds):
        perm = rng.permutation(teams)
        ko = start + pd.Timedelta(days=7 * r)
        for h, a in zip(perm[::2], perm[1::2]):
            mid = f"{r}-{h}-{a}"
            for t, o, v in ((h, a, "H"), (a, h, "A")):
                mean = np.exp(0.3 + 0.15 * (v == "H") + att[t] - dfn[o])
                xg = rng.gamma(shape=4.0, scale=mean / 4.0)  # continuous, non-integer
                rows.append({"match_id": mid, "kickoff_utc": ko, "team_id": t, "opponent_id": o,
                             "venue": v, "xg": xg})
    return pd.DataFrame(rows), att, dfn


def test_recovers_opponent_adjusted_strengths():
    sides, att, dfn = _sides()
    cutoff = sides["kickoff_utc"].max() + pd.Timedelta(days=1)
    r, mu, home = fit_xg_strength(sides, cutoff, half_life=1e6, l2=0.01)
    a_hat = r["att"] - r["att"].mean()
    d_hat = r["def"] - r["def"].mean()
    a_true = pd.Series(att) - np.mean(list(att.values()))
    d_true = pd.Series(dfn) - np.mean(list(dfn.values()))
    assert np.corrcoef(a_hat[a_true.index], a_true)[0, 1] > 0.95
    assert np.corrcoef(d_hat[d_true.index], d_true)[0, 1] > 0.95
    assert home == pytest.approx(0.15, abs=0.06)


def test_penalty_shrinks_toward_average():
    sides, _, _ = _sides(n_rounds=4)
    cutoff = sides["kickoff_utc"].max() + pd.Timedelta(days=1)
    loose, _, _ = fit_xg_strength(sides, cutoff, 1e6, l2=0.01)
    tight, _, _ = fit_xg_strength(sides, cutoff, 1e6, l2=1e4)
    assert tight["att"].abs().max() < 1e-3 < loose["att"].abs().max()


def test_future_matches_do_not_change_ratings():
    sides, _, _ = _sides(n_rounds=20)
    cutoff = T0 + pd.Timedelta(days=7 * 10)
    future, _, _ = _sides(n_rounds=5, seed=9, start=cutoff)
    future = future.assign(match_id="f" + future["match_id"], xg=10.0)
    r1, _, _ = fit_xg_strength(known(sides, cutoff), cutoff, 180)
    r2, _, _ = fit_xg_strength(known(pd.concat([sides, future]), cutoff), cutoff, 180)
    pd.testing.assert_frame_equal(r1, r2)
