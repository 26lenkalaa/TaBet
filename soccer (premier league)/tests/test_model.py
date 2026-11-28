import numpy as np
import pandas as pd
import pytest

from tabet.backtest import scores, walk_forward
from tabet.model import DixonColes, Hyperparams, promoted_team_seasons
from tabet.store import MatchStore

CUT = pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture(scope="module")
def fitted(real):
    m = real[0]
    train = MatchStore(m).completed_before(CUT)
    return DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promoted_team_seasons(m))


def test_probabilities_valid(fitted):
    for h, a in [("arsenal", "fulham"), ("luton", "man_city"), ("unknown_fc", "chelsea")]:
        p = fitted.predict_proba(h, a, "2023-24")
        assert all(0 < x < 1 for x in p)
        assert sum(p) == pytest.approx(1)


def test_home_advantage_and_strength_order(fitted):
    assert fitted.home_adv > 0
    ph, _, pa = fitted.predict_proba("man_city", "sheffield_united", "2023-24")
    assert ph > 0.6 and pa < 0.15


def test_promoted_flags(real):
    promo = promoted_team_seasons(real[0])
    assert {("brentford", "2021-22")} & promo == set()  # first season: unknown
    assert {("luton", "2023-24"), ("burnley", "2023-24"), ("sheffield_united", "2023-24")} <= promo
    assert sum(s == "2023-24" for _, s in promo) == 3


def test_fit_rejects_data_after_cutoff(real):
    with pytest.raises(ValueError, match="cutoff"):
        DixonColes().fit(real[0], CUT)


def test_save_load_roundtrip(fitted, tmp_path):
    fitted.save(tmp_path / "m.json")
    loaded = DixonColes.load(tmp_path / "m.json")
    assert loaded.predict_proba("arsenal", "luton", "2023-24") == fitted.predict_proba("arsenal", "luton", "2023-24")


def test_walk_forward_uses_only_past_results(real):
    preds = walk_forward(real[0], ("2023-24",), Hyperparams())
    assert len(preds) == 380
    assert (preds["cutoff_utc"] <= preds["kickoff_utc"]).all()
    assert np.allclose(preds[["p_home", "p_draw", "p_away"]].sum(axis=1), 1)


def test_scores_perfect_and_uniform():
    r = pd.Series(["H", "D", "A"])
    assert scores(np.eye(3), r)["log_loss"] == pytest.approx(0, abs=1e-9)
    assert scores(np.full((3, 3), 1 / 3), r)["log_loss"] == pytest.approx(np.log(3))


def test_promoted_features_learn_a_slope_and_round_trip(real, tmp_path):
    """Synthetic second-tier features equal to each promoted side's later strength
    should get a positive attack slope; zero features reduce to the shared offset."""
    m = real[0]
    train = m[(m["kickoff_utc"] < CUT) & (m["status"] == "completed")]
    promo = promoted_team_seasons(m)
    base = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo)
    zero = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo, {k: (0.0, 0.0) for k in promo})
    assert zero.promo_b_att == 0.0 and abs(zero.promo_att - base.promo_att) < 1e-6
    x = {k: (base.attack.get(k[0], 0.0) * 5, base.defence.get(k[0], 0.0) * 5) for k in promo}
    fx = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo, x)
    assert fx.promo_b_att > 0
    fx.save(tmp_path / "m.json")
    again = DixonColes.load(tmp_path / "m.json")
    t, s = sorted(promo)[0]
    assert again.expected_goals(t, "arsenal", s) == fx.expected_goals(t, "arsenal", s)


def test_fixed_promoted_offsets_act_as_given_prior_means(real):
    # addendum P (R6): mapped offsets are added as given, slopes 1 and no shared offset
    m = real[0]
    train = MatchStore(m).completed_before(CUT)
    promo = promoted_team_seasons(m)
    mapped = {k: (0.3, -0.2) for k in promo if k[1] == "2023-24"}
    dc = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo, mapped, fixed_promo=True)
    assert dc.promo_b_att == pytest.approx(1.0) and dc.promo_b_def == pytest.approx(1.0)
    assert dc.promo_att == 0.0 and dc.promo_def == 0.0
    a, d = dc._rating("luton", "2023-24")
    assert a == pytest.approx(dc.attack.get("luton", 0.0) + 0.3)
    assert d == pytest.approx(dc.defence.get("luton", 0.0) - 0.2)
    # the default (learned) prior is unchanged by the new option
    base = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo)
    again = DixonColes(Hyperparams(promoted_prior=True)).fit(train, CUT, promo, fixed_promo=True)
    assert base.promo_att == pytest.approx(again.promo_att)


def test_low_score_factors_never_make_a_probability_negative():
    from scipy.stats import poisson
    from tabet.model import MAX_GOALS, RHO_BOUND, dc_correct
    g = np.arange(MAX_GOALS + 1)
    for lh, la, rho in ((6.0, 1.0, -RHO_BOUND), (3.0, 2.5, RHO_BOUND), (1.4, 1.1, -0.05)):
        m = dc_correct(np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)), lh, la, rho)
        assert (m >= 0).all() and np.isclose(m.sum(), 1.0)


def test_dixon_coles_gradient_matches_finite_differences_even_when_tau_is_clipped(monkeypatch):
    import types
    import tabet.model as M
    captured = {}

    def fake_minimize(fun, x0, jac, method, bounds, options):
        captured["f"] = fun
        return types.SimpleNamespace(success=True, x=x0, message="")
    monkeypatch.setattr(M, "minimize", fake_minimize)
    t0 = pd.Timestamp("2025-08-01", tz="UTC")
    rows = [("a", "b", 0, 1), ("b", "c", 1, 0), ("c", "a", 0, 0), ("a", "c", 1, 1), ("b", "a", 3, 0), ("c", "b", 0, 1)]
    train = pd.DataFrame({"match_id": [f"m{i}" for i in range(6)], "season": "2025-26",
                          "kickoff_utc": [t0 + pd.Timedelta(days=7 * i) for i in range(6)],
                          "home_id": [r[0] for r in rows], "away_id": [r[1] for r in rows],
                          "home_goals": [r[2] for r in rows], "away_goals": [r[3] for r in rows], "status": "completed"})
    M.DixonColes(M.Hyperparams()).fit(train, t0 + pd.Timedelta(days=60), set())
    f = captured["f"]
    n = 3
    for p in (np.r_[0.2, 0.25, -0.05, 0, 0, 0.1, -0.2, 0.05, 0.0, 0.1, -0.1, 0, 0],
              np.r_[1.8, 0.3, -0.2, 0, 0, 0.5, 0.0, 0.0, -0.4, 0.0, 0.0, 0, 0]):   # second: home rate > 5, tau clipped
        _, g = f(p)
        num = np.array([(f(p + e)[0] - f(p - e)[0]) / 2e-6 for e in np.eye(len(p)) * 1e-6])
        free = [0, 1, 2] + list(range(5, 5 + 2 * n))                                  # promo and slope params are fixed at 0
        assert np.allclose(g[free], num[free], rtol=1e-4, atol=1e-4)
