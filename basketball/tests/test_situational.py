"""Offline tests for the situational model (v3), using made-up schedules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src.models import ratings, situational  # noqa: E402
from src.models.situational import FEATURES, Situational, add_situation, distance_km, learn_effects, rate_games  # noqa: E402

TEAMS = {"BOS": 1, "NYK": 2, "LAL": 3, "GSW": 4, "MIA": 5, "DEN": 6}


def make_games(results):
    """results: (date, season, home, away, home_pts, away_pts) per game."""
    rows = []
    for number, (date, season, home, away, home_pts, away_pts) in enumerate(results, start=1):
        rows.append({
            "game_id": f"00218{number:05d}", "season": season, "date": date,
            "game_type": "regular", "is_playoffs": False,
            "home_team_id": TEAMS[home], "home": home, "away_team_id": TEAMS[away], "away": away,
            "home_pts": home_pts, "away_pts": away_pts,
            "home_possessions": 100.0, "away_possessions": 100.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def schedule():
    return make_games([
        ("2018-10-16", "2018-19", "BOS", "NYK", 110, 95),    # 0: first game for both
        ("2018-10-17", "2018-19", "LAL", "BOS", 101, 104),   # 1: BOS back-to-back after flying to L.A.
        ("2018-10-20", "2018-19", "NYK", "LAL", 99, 108),    # 2: NYK 4 days rest, LAL 3 days and a long trip
        ("2018-10-22", "2018-19", "BOS", "LAL", 105, 100),   # 3: BOS 5 days rest, LAL 2 days
        ("2019-10-22", "2019-20", "NYK", "BOS", 90, 118),    # 4: new season
    ])


def simulated_seasons(back_to_back_cost, seasons=4, days=250, seed=0):
    """Even teams, real schedule quirks: a team on a back-to-back scores `back_to_back_cost` fewer points."""
    rng = np.random.default_rng(seed)
    names = list(TEAMS)
    results, last_played = [], {}
    for season_number in range(seasons):
        season = f"{2018 + season_number}-{(19 + season_number) % 100:02d}"
        for day in pd.date_range(f"{2018 + season_number}-10-20", periods=days):
            order = rng.permutation(names)
            for home, away in zip(order[:3], order[3:]):
                if rng.random() < 0.25:
                    continue   # not every team plays every day
                points = {}
                for team in (home, away):
                    tired = last_played.get(team) == day - pd.Timedelta(days=1)
                    points[team] = 110 + rng.normal(0, 8) - (back_to_back_cost if tired else 0)
                    last_played[team] = day
                home_pts, away_pts = round(points[home] + 1.3), round(points[away] - 1.3)
                results.append((day.strftime("%Y-%m-%d"), season, home, away, home_pts + (home_pts == away_pts), away_pts))
    return make_games(results)


# --- distance ------------------------------------------------------------------

def test_every_team_has_an_arena():
    assert len(situational.ARENAS) == 30


def test_distances_are_about_right():
    assert distance_km("BOS", "BOS") == 0.0
    assert distance_km("BOS", "LAL") == pytest.approx(4170, abs=60)      # coast to coast
    assert distance_km("NYK", "BKN") < 12                                # across the river
    assert distance_km("LAL", "BOS") == pytest.approx(distance_km("BOS", "LAL"))
    assert distance_km("BOS", "XXX") == 0.0                              # unknown arena


# --- the situation before each game ------------------------------------------

def test_rest_days_count_from_the_teams_previous_game(schedule):
    table = add_situation(schedule)
    assert pd.isna(table.loc[0, "home_rest_days"]) and pd.isna(table.loc[0, "away_rest_days"])   # first games
    assert table.loc[1, "away_rest_days"] == 1      # BOS played yesterday
    assert table.loc[2, "home_rest_days"] == 4      # NYK last played on the 16th
    assert table.loc[2, "away_rest_days"] == 3
    assert table.loc[3, "home_rest_days"] == 5


def test_travel_is_measured_from_where_the_team_last_played(schedule):
    table = add_situation(schedule)
    assert table.loc[1, "away_travel_km"] == pytest.approx(distance_km("BOS", "LAL"))   # Boston to L.A.
    assert table.loc[2, "home_travel_km"] == pytest.approx(distance_km("BOS", "NYK"))   # NYK home from Boston
    assert table.loc[2, "away_travel_km"] == pytest.approx(distance_km("LAL", "NYK"))
    assert table.loc[3, "home_travel_km"] == pytest.approx(distance_km("LAL", "BOS"))   # BOS back from L.A.
    assert table.loc[0, "home_travel_km"] == 0.0


def test_the_gaps_are_home_minus_away(schedule):
    table = add_situation(schedule)
    assert table.loc[1, "back_to_back"] == -1.0     # only the away team is on a back-to-back
    assert table.loc[1, "long_rest"] == 0.0
    assert table.loc[2, "long_rest"] == 0.0         # both rested 3+ days
    assert table.loc[3, "long_rest"] == 1.0         # BOS 5 days, LAL 2
    assert table.loc[1, "travel_1000km"] == pytest.approx(-distance_km("BOS", "LAL") / 1000)


def test_an_off_season_gap_resets_rest_and_travel(schedule):
    table = add_situation(schedule)
    assert pd.isna(table.loc[4, "home_rest_days"]) and pd.isna(table.loc[4, "away_rest_days"])
    assert table.loc[4, "away_travel_km"] == 0.0
    assert (table.loc[4, FEATURES] == 0).all()


def test_the_situation_does_not_depend_on_any_result(schedule):
    changed = schedule.assign(home_pts=schedule["away_pts"] + 40)
    columns = ["home_rest_days", "away_rest_days", "home_travel_km", "away_travel_km"] + FEATURES
    pd.testing.assert_frame_equal(add_situation(changed)[columns], add_situation(schedule)[columns])


# --- learning the effects -----------------------------------------------------

def test_too_few_games_means_no_effects(schedule):
    table = add_situation(schedule)
    assert learn_effects(table, pd.Series([5.0] * len(table))) == dict.fromkeys(FEATURES, 0.0)


def test_learn_effects_recovers_a_known_pattern():
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"back_to_back": rng.choice([-1.0, 0.0, 1.0], 4000),
                          "long_rest": rng.choice([-1.0, 0.0, 1.0], 4000),
                          "travel_1000km": rng.normal(0, 1.5, 4000)})
    misses = -2.0 * table["back_to_back"] + 0.5 * table["long_rest"] - 0.3 * table["travel_1000km"] \
        + rng.normal(0, 12, 4000)
    effects = learn_effects(table, misses)
    assert effects["back_to_back"] == pytest.approx(-2.0, abs=0.6)
    assert effects["long_rest"] == pytest.approx(0.5, abs=0.6)
    assert effects["travel_1000km"] == pytest.approx(-0.3, abs=0.4)


def test_adjustment_adds_up_the_effects():
    model = Situational(effects={"back_to_back": -2.0, "long_rest": 0.5, "travel_1000km": -0.3})
    assert model.adjustment(back_to_back=-1.0, long_rest=1.0, travel_1000km=2.0) == pytest.approx(2.0 + 0.5 - 0.6)
    assert Situational().adjustment(1.0, 1.0, 1.0) == 0.0


# --- the model ------------------------------------------------------------------

def test_the_first_season_is_plain_v2(schedule):
    rated, _ = rate_games(schedule)
    base, _ = ratings.rate_games(schedule)
    first_season = rated["season"] == "2018-19"
    assert list(rated.loc[first_season, "predicted_margin"]) == pytest.approx(
        list(base.loc[first_season, "predicted_margin"]))
    assert list(rated.loc[first_season, "home_win_prob"]) == pytest.approx(
        list(base.loc[first_season, "home_win_prob"]))


def test_it_learns_that_back_to_backs_hurt():
    games = simulated_seasons(back_to_back_cost=4.0)
    rated, model = rate_games(games)
    assert model.effects["back_to_back"] == pytest.approx(-4.0, abs=1.2)

    later = rated[rated["season"] > "2018-19"]
    tired_home = later[later["back_to_back"] == 1]
    tired_away = later[later["back_to_back"] == -1]
    assert (tired_home["predicted_margin"] - tired_home["base_margin"]).mean() < -2
    assert (tired_away["predicted_margin"] - tired_away["base_margin"]).mean() > 2

    actual = later["home_pts"] - later["away_pts"]
    v3_error = (later["predicted_margin"] - actual).abs().mean()
    v2_error = (later["base_margin"] - actual).abs().mean()
    assert v3_error < v2_error   # knowing who is tired helps


def test_with_no_real_effect_it_stays_close_to_v2():
    rated, model = rate_games(simulated_seasons(back_to_back_cost=0.0, seed=3))
    assert abs(model.effects["back_to_back"]) < 1.2
    assert (rated["predicted_margin"] - rated["base_margin"]).abs().mean() < 0.6


def test_each_seasons_effects_come_only_from_earlier_seasons():
    games = simulated_seasons(back_to_back_cost=4.0)
    rated, _ = rate_games(games)
    # Rewrite the last season's results: predictions in the last season must not move.
    last = games["season"] == games["season"].max()
    changed = games.copy()
    changed.loc[last, "home_pts"] = games.loc[last, "away_pts"] + 30
    rated_changed, _ = rate_games(changed)
    last_rated = rated["season"] == rated["season"].max()
    first_of_last = rated[last_rated].index[0]
    # The first game of the last season depends on nothing in that season.
    assert rated_changed.loc[first_of_last, "predicted_margin"] == pytest.approx(
        rated.loc[first_of_last, "predicted_margin"])
    # Later games differ only through v2's ratings, never through the learned effects.
    adjustment = (rated["predicted_margin"] - rated["base_margin"])[last_rated]
    adjustment_changed = (rated_changed["predicted_margin"] - rated_changed["base_margin"])[last_rated]
    assert list(adjustment_changed) == pytest.approx(list(adjustment))


def test_a_games_own_result_does_not_change_its_prediction():
    games = simulated_seasons(back_to_back_cost=4.0)
    rated, _ = rate_games(games)
    position = len(games) - 10
    game_id = rated.loc[position, "game_id"]
    changed = games.copy()
    changed.loc[changed["game_id"] == game_id, ["home_pts", "away_pts"]] = [60, 140]
    rated_changed, _ = rate_games(changed)
    columns = ["home_win_prob", "predicted_margin"]
    pd.testing.assert_frame_equal(rated_changed.loc[:position, columns], rated.loc[:position, columns])


def test_output_has_the_columns_the_backtest_needs(schedule):
    rated, model = rate_games(schedule)
    assert len(rated) == len(schedule)
    for column in ["home_win_prob", "predicted_margin", "base_margin", "predicted_home_pts",
                   "predicted_away_pts"] + FEATURES:
        assert column in rated.columns
    assert rated["home_win_prob"].between(0, 1).all()
    assert list(rated["predicted_home_pts"] - rated["predicted_away_pts"]) == pytest.approx(
        list(rated["predicted_margin"]))
    assert set(model.effects) == set(FEATURES)
