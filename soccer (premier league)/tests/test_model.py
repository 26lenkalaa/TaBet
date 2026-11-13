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
