import numpy as np
import pandas as pd
import pytest

from tabet.calibration import apply, fit, walk_forward


def _synthetic(n=6000, T_true=1.3, delta_true=0.0, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.7, size=(n, 3))
    q = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # true probabilities
    y = np.array([rng.choice(3, p=row) for row in q])
    # reported p such that apply(p, T_true, delta_true) == q (i.e. miscalibrated by that map)
    z = np.log(q)
    z[:, 1] -= delta_true
    z *= T_true
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return p, y


def test_identity_and_normalisation():
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    np.testing.assert_allclose(apply(p), p)
    q = apply(p, T=1.7, delta=0.3)
    np.testing.assert_allclose(q.sum(axis=1), 1)
    assert (q[:, 1] > p[:, 1]).all()  # positive delta adds draw mass
    flat = apply(p, T=2.0)
    assert flat.max() < p.max()  # T > 1 softens


def test_fit_recovers_temperature_and_draw_shift():
    p, y = _synthetic(T_true=1.3)
    T, d = fit(p, y, "temperature", lam=0.0)
    assert T == pytest.approx(1.3, rel=0.08) and d == 0.0
    p, y = _synthetic(T_true=1.0, delta_true=0.25, seed=1)
    T, d = fit(p, y, "temperature+draw", lam=0.0)
    assert T == pytest.approx(1.0, abs=0.08) and d == pytest.approx(0.25, abs=0.08)


def test_penalty_shrinks_to_identity():
    p, y = _synthetic(n=300, T_true=1.3)
    T_free, _ = fit(p, y, "temperature", lam=0.0)
    T_pen, _ = fit(p, y, "temperature", lam=1e6)
    assert abs(T_pen - 1) < 1e-3 < abs(T_free - 1)


def test_walk_forward_has_no_lookahead():
    p, y = _synthetic(n=600, T_true=1.3, seed=2)
    ko = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.arange(600) // 5, unit="D")
    df = pd.DataFrame({"match_id": [f"m{i}" for i in range(600)], "kickoff_utc": ko, "cutoff_utc": ko,
                       "result": np.array(["H", "D", "A"])[y], "p_home": p[:, 0], "p_draw": p[:, 1],
                       "p_away": p[:, 2]})
    a = walk_forward(df, "temperature", lam=1.0, min_matches=200)
    flipped = df.copy()
    flipped.loc[400:, "result"] = "D"  # rewrite the future
    b = walk_forward(flipped, "temperature", lam=1.0, min_matches=200)
    first_future_cutoff = df.loc[400, "cutoff_utc"]
    early = a["cutoff_utc"] <= first_future_cutoff
    np.testing.assert_allclose(a.loc[early, "p_home"], b.loc[early, "p_home"])
    assert not a.loc[:199, "calib_active"].any()  # identity until 200 known results
    assert a["calib_active"].iloc[-1]
