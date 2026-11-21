"""Offline tests for market.py: odds conversions and matching odds to games."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import market  # noqa: E402


# --- moneylines ----------------------------------------------------------------

@pytest.mark.parametrize("moneyline, prob", [
    (-150, 0.6), (+150, 0.4), (-100, 0.5), (+100, 0.5), (-400, 0.8), (+400, 0.2), (-110, 110 / 210),
])
def test_moneyline_to_prob(moneyline, prob):
    assert market.moneyline_to_prob(moneyline) == pytest.approx(prob)


@pytest.mark.parametrize("bad", [0, 50, -99, float("nan"), float("inf")])
def test_moneyline_to_prob_rejects_impossible_odds(bad):
    with pytest.raises(ValueError):
        market.moneyline_to_prob(bad)


@pytest.mark.parametrize("prob, moneyline", [(0.6, -150), (0.4, +150), (0.5, -100), (0.8, -400), (0.2, +400)])
def test_prob_to_moneyline(prob, moneyline):
    assert market.prob_to_moneyline(prob) == pytest.approx(moneyline)


@pytest.mark.parametrize("moneyline", [-900, -250, -110, +105, +320, +1500])
def test_moneyline_survives_a_round_trip(moneyline):
    assert market.prob_to_moneyline(market.moneyline_to_prob(moneyline)) == pytest.approx(moneyline)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.3])
def test_prob_to_moneyline_rejects_impossible_probabilities(bad):
    with pytest.raises(ValueError):
        market.prob_to_moneyline(bad)


def test_bookmaker_prices_add_up_to_more_than_one_until_the_margin_is_removed():
    raw_total = market.moneyline_to_prob(-110) + market.moneyline_to_prob(-110)
    assert raw_total == pytest.approx(1.0476, abs=1e-4)
    assert market.moneylines_to_probs(-110, -110) == pytest.approx((0.5, 0.5))


def test_margin_free_probabilities_add_up_to_one_and_keep_the_favorite():
    home, away = market.moneylines_to_probs(-250, +200)
    assert home + away == pytest.approx(1.0)
    assert home > away
    assert home == pytest.approx((250 / 350) / (250 / 350 + 100 / 300))


def test_remove_margin_keeps_the_ratio_between_the_two_sides():
    home, away = market.remove_margin(0.63, 0.42)
    assert home / away == pytest.approx(0.63 / 0.42)


# --- spreads -------------------------------------------------------------------

def test_a_pick_em_spread_is_a_coin_flip():
    assert market.spread_to_prob(0) == pytest.approx(0.5)


def test_a_negative_home_spread_means_the_home_team_is_favored():
    assert market.spread_to_prob(-5.5) > 0.5 > market.spread_to_prob(+5.5)
    assert market.spread_to_margin(-5.5) == 5.5


def test_opposite_spreads_give_probabilities_that_add_up_to_one():
    assert market.spread_to_prob(-7) + market.spread_to_prob(7) == pytest.approx(1.0)


def test_spread_to_prob_matches_the_normal_curve():
    assert market.spread_to_prob(-12, margin_sd=12) == pytest.approx(0.8413, abs=1e-4)   # one sd
    assert market.spread_to_prob(-6, margin_sd=12) == pytest.approx(0.6915, abs=1e-4)    # half an sd


def test_a_wider_spread_of_results_pulls_probabilities_toward_one_half():
    assert market.spread_to_prob(-6, margin_sd=15) < market.spread_to_prob(-6, margin_sd=10)


def test_spread_to_prob_works_on_a_whole_column():
    probs = market.spread_to_prob(pd.Series([-12.0, 0.0, 12.0, None]), margin_sd=12)
    assert probs[:3] == pytest.approx([0.8413, 0.5, 0.1587], abs=1e-4)
    assert pd.isna(probs[3])


def test_spread_to_prob_rejects_a_zero_sd():
    with pytest.raises(ValueError):
        market.spread_to_prob(-3, margin_sd=0)


def test_estimate_margin_sd_measures_the_miss_between_spread_and_result():
    games = pd.DataFrame({
        "home_pts": [110, 100, 95], "away_pts": [100, 104, 95 - 9],
        "spread_close": [-4.0, 2.0, -3.0],   # market margins 4, -2, 3; real margins 10, -4, 9
    })
    assert market.estimate_margin_sd(games) == pytest.approx(((6 ** 2 + 2 ** 2 + 6 ** 2) / 3) ** 0.5)


# --- team names ----------------------------------------------------------------

@pytest.mark.parametrize("name, abbreviation", [
    ("BOS", "BOS"), ("Boston", "BOS"), ("boston celtics", "BOS"), ("GoldenState", "GSW"),
    ("Golden State", "GSW"), ("L.A. Clippers", "LAC"), ("LA Lakers", "LAL"), ("BRK", "BKN"),
    ("PHO", "PHX"), ("CHO", "CHA"), ("New Orleans", "NOP"), ("NewYork", "NYK"), ("Oklahoma City", "OKC"),
    ("76ers", "PHI"), (" San Antonio ", "SAS"),
])
def test_team_abbreviation_accepts_common_spellings(name, abbreviation):
    assert market.team_abbreviation(name) == abbreviation


def test_team_abbreviation_covers_all_thirty_teams_without_clashes():
    assert len(market.TEAM_NAMES) == 30
    spellings = [market._key(s) for abbreviation, names in market.TEAM_NAMES.items() for s in [abbreviation, *names]]
    assert len(spellings) == len(set(spellings))


def test_team_abbreviation_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown team name: 'Los Angeles'"):
        market.team_abbreviation("Los Angeles")   # could be either L.A. team


# --- matching ------------------------------------------------------------------

@pytest.fixture
def games():
    return pd.DataFrame({
        "game_id": ["0021800001", "0021800002", "0021800003", "0021900001"],
        "season": ["2018-19", "2018-19", "2018-19", "2019-20"],
        "date": ["2018-10-16", "2018-10-16", "2018-10-17", "2019-10-22"],
        "home": ["BOS", "GSW", "NYK", "LAL"],
        "away": ["PHI", "OKC", "ATL", "LAC"],
        "home_pts": [105, 108, 126, 102],
        "away_pts": [87, 100, 107, 112],
    })


@pytest.fixture
def odds():
    return pd.DataFrame({
        "date": ["10/16/2018", "10/16/2018", "10/17/2018", "10/22/2019"],
        "home": ["Boston", "Golden State", "New York", "LA Lakers"],
        "away": ["Philadelphia", "Oklahoma City", "Atlanta", "LA Clippers"],
        "spread_open": [-4.5, -11.0, -5.0, 1.0],
        "spread_close": [-5.0, -12.0, -5.5, 3.5],
        "total_close": [208.5, 223.5, 217.0, 224.5],
    })


def test_every_game_gets_its_own_closing_spread(games, odds):
    matched = market.match_odds(games, odds)
    assert len(matched) == len(games)
    assert list(matched["spread_close"]) == [-5.0, -12.0, -5.5, 3.5]
    assert list(matched["total_close"]) == [208.5, 223.5, 217.0, 224.5]
    assert market.games_without_spread(matched).empty


def test_matching_does_not_depend_on_the_order_of_the_odds(games, odds):
    expected = market.match_odds(games, odds)
    actual = market.match_odds(games, odds.sample(frac=1, random_state=5))
    pd.testing.assert_frame_equal(actual, expected)


def test_market_predictions_come_from_the_closing_spread(games, odds):
    matched = market.match_odds(games, odds, margin_sd=12)
    assert list(matched["market_margin"]) == [5.0, 12.0, 5.5, -3.5]
    assert matched.loc[1, "market_home_win_prob"] == pytest.approx(0.8413, abs=1e-4)
    assert matched.loc[3, "market_home_win_prob"] < 0.5


def test_moneylines_are_used_for_win_probability_when_present(games, odds):
    odds["home_ml"] = [-200.0, None, -220.0, +140.0]
    odds["away_ml"] = [+170.0, None, +180.0, -160.0]
    matched = market.match_odds(games, odds)
    assert matched.loc[0, "market_home_win_prob"] == pytest.approx(market.moneylines_to_probs(-200, +170)[0])
    assert matched.loc[1, "market_home_win_prob"] == pytest.approx(market.spread_to_prob(-12.0))   # falls back


def test_a_game_with_no_odds_is_kept_and_reported(games, odds):
    matched = market.match_odds(games, odds.iloc[:3])
    assert len(matched) == 4
    assert pd.isna(matched.loc[3, "spread_close"])
    assert pd.isna(matched.loc[3, "market_home_win_prob"])
    missing = market.games_without_spread(matched)
    assert list(missing["game_id"]) == ["0021900001"]


def test_home_and_away_must_both_match(games, odds):
    odds.loc[0, ["home", "away"]] = ["Philadelphia", "Boston"]   # same teams, other arena
    matched = market.match_odds(games, odds)
    assert pd.isna(matched.loc[0, "spread_close"])


@pytest.mark.parametrize("odds_date", ["2019-10-23", "2019-10-21"])
def test_odds_dated_one_day_off_still_match(games, odds, odds_date):
    odds.loc[3, "date"] = odds_date
    matched = market.match_odds(games, odds)
    assert matched.loc[3, "spread_close"] == 3.5
    assert matched.loc[3, "total_close"] == 224.5


def test_odds_dated_two_days_off_do_not_match(games, odds):
    odds.loc[3, "date"] = "2019-10-24"
    matched = market.match_odds(games, odds)
    assert pd.isna(matched.loc[3, "spread_close"])


def test_back_to_back_games_between_the_same_teams_keep_their_own_odds():
    games = pd.DataFrame({
        "game_id": ["0022000001", "0022000002"], "season": ["2020-21"] * 2,
        "date": ["2021-01-06", "2021-01-07"], "home": ["BOS", "BOS"], "away": ["MIA", "MIA"],
        "home_pts": [107, 99], "away_pts": [105, 104],
    })
    odds = pd.DataFrame({
        "date": ["2021-01-06", "2021-01-07"], "home": ["BOS", "BOS"], "away": ["MIA", "MIA"],
        "spread_close": [-3.0, -1.5],
    })
    assert list(market.match_odds(games, odds)["spread_close"]) == [-3.0, -1.5]
    # With only the second day's odds, the first game must not borrow them and leave the second empty.
    late_only = market.match_odds(games, odds.iloc[[1]])
    assert pd.isna(late_only.loc[0, "spread_close"])
    assert late_only.loc[1, "spread_close"] == -1.5


def test_unreadable_spreads_become_missing(games, odds):
    odds["spread_close"] = odds["spread_close"].astype(object)
    odds.loc[2, "spread_close"] = "NL"
    matched = market.match_odds(games, odds)
    assert pd.isna(matched.loc[2, "spread_close"])


def test_standardize_odds_names_missing_columns(odds):
    with pytest.raises(ValueError, match="missing columns: spread_close"):
        market.standardize_odds(odds.drop(columns="spread_close"))


def test_standardize_odds_rejects_a_game_listed_twice(odds):
    with pytest.raises(ValueError, match="more than once"):
        market.standardize_odds(pd.concat([odds, odds.iloc[[0]]]))


def test_coverage_counts_games_with_and_without_a_spread_per_season(games, odds):
    table = market.coverage(market.match_odds(games, odds.iloc[:2]))
    assert table.loc["2018-19"].tolist() == [3, 2, 1]
    assert table.loc["2019-20"].tolist() == [1, 0, 1]
