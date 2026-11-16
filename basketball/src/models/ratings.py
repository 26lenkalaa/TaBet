"""v2 model: efficiency ratings.

Each team has three ratings, all measured against the league average:

  * offense: points scored per 100 possessions;
  * defense: points allowed per 100 possessions (negative is good);
  * pace: possessions per game.

Before a game, the home offense meets the away defense (and the other way
round) to give each team's expected points per 100 possessions; expected pace
turns that into a predicted score for each team. The predicted margin is the
gap between the two scores, and the win probability is the share of a bell
curve around that margin that lies above zero.

After the game, each rating moves a small step toward what just happened, so
recent games count more than old ones. Ratings are only updated after a game
is predicted, so a prediction never uses that game's result or any later one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import norm

LEARNING_RATE = 0.04       # share of each game's surprise that moves a team's ratings
LEAGUE_RATE = 0.01         # same, for the league-wide numbers (they drift slowly)
SEASON_CARRYOVER = 0.75    # share of each rating kept between seasons

# Starting values for the league-wide numbers; each one is then learned from the games.
LEAGUE_EFFICIENCY = 112.0  # points per 100 possessions
LEAGUE_PACE = 99.0         # possessions per team per game
HOME_EDGE = 2.5            # points per 100 possessions the home team gains over the away team
MARGIN_SD = 13.5           # typical gap, in points, between predicted and real margin


@dataclass
class Efficiency:
    """Current ratings plus the rules for predicting and updating."""

    learning_rate: float = LEARNING_RATE
    league_rate: float = LEAGUE_RATE
    season_carryover: float = SEASON_CARRYOVER
    league_efficiency: float = LEAGUE_EFFICIENCY
    league_pace: float = LEAGUE_PACE
    home_edge: float = HOME_EDGE
    margin_sd: float = MARGIN_SD
    offense: dict[int, float] = field(default_factory=dict)
    defense: dict[int, float] = field(default_factory=dict)
    pace: dict[int, float] = field(default_factory=dict)

    def expected(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float, float]:
        """(home points per 100 possessions, away points per 100 possessions, possessions)."""
        edge = 0.0 if neutral else self.home_edge / 2
        home_efficiency = self.league_efficiency + self.offense.get(home_id, 0.0) + self.defense.get(away_id, 0.0) + edge
        away_efficiency = self.league_efficiency + self.offense.get(away_id, 0.0) + self.defense.get(home_id, 0.0) - edge
        possessions = self.league_pace + self.pace.get(home_id, 0.0) + self.pace.get(away_id, 0.0)
        return home_efficiency, away_efficiency, possessions

    def predict_score(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        """(predicted home points, predicted away points)."""
        home_efficiency, away_efficiency, possessions = self.expected(home_id, away_id, neutral)
        return home_efficiency * possessions / 100, away_efficiency * possessions / 100

    def predict(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        """(home win probability, predicted home margin in points)."""
        home_pts, away_pts = self.predict_score(home_id, away_id, neutral)
        margin = home_pts - away_pts
        return float(norm.cdf(margin / self.margin_sd)), margin

    def net_rating(self, team_id: int) -> float:
        """Points per 100 possessions better than an average team (offense minus defense)."""
        return self.offense.get(team_id, 0.0) - self.defense.get(team_id, 0.0)

    def update(self, home_id: int, away_id: int, home_pts: float, away_pts: float,
               home_possessions: float, away_possessions: float) -> None:
        """Move the ratings a step toward what happened in a finished game."""
        home_efficiency, away_efficiency, expected_possessions = self.expected(home_id, away_id)

        # Learn how wide the bell curve around the predicted margin should be.
        _, predicted_margin = self.predict(home_id, away_id)
        miss = (home_pts - away_pts) - predicted_margin
        self.margin_sd = math.sqrt((1 - self.league_rate) * self.margin_sd ** 2 + self.league_rate * miss ** 2)

        # Both teams have the same number of possessions in a game, give or take one; average the two estimates.
        counts = [count for count in (home_possessions, away_possessions) if count == count and count > 0]
        if not counts:
            return   # no box score: nothing to learn about efficiency or pace
        possessions = sum(counts) / len(counts)

        home_surprise = 100 * home_pts / possessions - home_efficiency
        away_surprise = 100 * away_pts / possessions - away_efficiency
        pace_surprise = possessions - expected_possessions

        rate = self.learning_rate
        # A team scoring more than expected says its offense is better and the other defense is worse.
        self.offense[home_id] = self.offense.get(home_id, 0.0) + rate * home_surprise
        self.defense[away_id] = self.defense.get(away_id, 0.0) + rate * home_surprise
        self.offense[away_id] = self.offense.get(away_id, 0.0) + rate * away_surprise
        self.defense[home_id] = self.defense.get(home_id, 0.0) + rate * away_surprise
        self.pace[home_id] = self.pace.get(home_id, 0.0) + rate * pace_surprise
        self.pace[away_id] = self.pace.get(away_id, 0.0) + rate * pace_surprise

        self.league_efficiency += self.league_rate * (home_surprise + away_surprise) / 2
        self.league_pace += self.league_rate * pace_surprise
        self.home_edge += self.league_rate * (home_surprise - away_surprise)

    def start_new_season(self) -> None:
        """Pull every team part of the way back to average: rosters change over the summer."""
        for ratings in (self.offense, self.defense, self.pace):
            for team_id in ratings:
                ratings[team_id] *= self.season_carryover


def rate_games(games: pd.DataFrame, model: Efficiency | None = None) -> tuple[pd.DataFrame, Efficiency]:
    """Replay games in date order, predicting each one before learning its result.

    `games` needs the columns clean.py writes plus home_possessions and
    away_possessions. Returns (rated, model): `rated` is `games` in date order
    with what the model predicted BEFORE each game (home_win_prob,
    predicted_margin, predicted_home_pts, predicted_away_pts, and margin_sd,
    the width of the bell curve it used), and `model` holds the ratings after
    the last game.
    """
    model = model or Efficiency()
    ordered = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    before: list[tuple[float, float, float, float, float]] = []
    season = None
    for game in ordered.itertuples(index=False):
        if season is not None and game.season != season:
            model.start_new_season()
        season = game.season

        home_win_prob, predicted_margin = model.predict(game.home_team_id, game.away_team_id)
        predicted_home_pts, predicted_away_pts = model.predict_score(game.home_team_id, game.away_team_id)
        before.append((home_win_prob, predicted_margin, predicted_home_pts, predicted_away_pts, model.margin_sd))
        model.update(game.home_team_id, game.away_team_id, game.home_pts, game.away_pts,
                     game.home_possessions, game.away_possessions)

    columns = ["home_win_prob", "predicted_margin", "predicted_home_pts", "predicted_away_pts", "margin_sd"]
    rated = pd.concat([ordered, pd.DataFrame(before, columns=columns, index=ordered.index)], axis=1)
    return rated, model
