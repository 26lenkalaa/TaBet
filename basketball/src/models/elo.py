"""v1 model: Elo ratings with margin of victory and home court.

Every team starts at 1500. After each game the winner takes rating points
from the loser: more for an upset, more for a big win. Before a game, the
rating gap (plus a home-court bonus) gives the home team's win probability
and the predicted point margin.

Ratings are only ever updated after a game is predicted, so a prediction
never uses that game's result or any later one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

START_RATING = 1500.0
K_FACTOR = 20.0            # how fast ratings move after each game
HOME_ADVANTAGE = 70.0      # home-court bonus in rating points (about 2.5 points on the scoreboard)
SEASON_CARRYOVER = 0.75    # share of a team's distance from 1500 kept between seasons
ELO_PER_POINT = 28.0       # rating points that equal one point of margin


@dataclass
class Elo:
    """Current ratings plus the rules for predicting and updating."""

    k_factor: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    season_carryover: float = SEASON_CARRYOVER
    elo_per_point: float = ELO_PER_POINT
    ratings: dict[int, float] = field(default_factory=dict)

    def rating(self, team_id: int) -> float:
        """A team's current rating; teams not seen yet start at 1500."""
        return self.ratings.get(team_id, START_RATING)

    def rating_gap(self, home_id: int, away_id: int, neutral: bool = False) -> float:
        """Home rating minus away rating, including the home-court bonus."""
        bonus = 0.0 if neutral else self.home_advantage
        return self.rating(home_id) + bonus - self.rating(away_id)

    def predict(self, home_id: int, away_id: int, neutral: bool = False) -> tuple[float, float]:
        """(home win probability, predicted home margin in points)."""
        gap = self.rating_gap(home_id, away_id, neutral)
        home_win_prob = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
        return home_win_prob, gap / self.elo_per_point

    def update(self, home_id: int, away_id: int, home_pts: int, away_pts: int, neutral: bool = False) -> None:
        """Move both ratings after a finished game."""
        gap = self.rating_gap(home_id, away_id, neutral)
        home_win_prob, _ = self.predict(home_id, away_id, neutral)
        home_won = home_pts > away_pts
        margin = abs(home_pts - away_pts)

        # Bigger wins move ratings more, but less so when the favorite won
        # (otherwise strong teams would climb forever by beating weak ones).
        winner_gap = gap if home_won else -gap
        margin_multiplier = (margin + 3.0) ** 0.8 / (7.5 + 0.006 * winner_gap)

        change = self.k_factor * margin_multiplier * ((1.0 if home_won else 0.0) - home_win_prob)
        self.ratings[home_id] = self.rating(home_id) + change
        self.ratings[away_id] = self.rating(away_id) - change

    def start_new_season(self) -> None:
        """Pull every team part of the way back to 1500: rosters change over the summer."""
        for team_id, rating in self.ratings.items():
            self.ratings[team_id] = START_RATING + self.season_carryover * (rating - START_RATING)


def rate_games(games: pd.DataFrame, model: Elo | None = None) -> tuple[pd.DataFrame, Elo]:
    """Replay games in date order, predicting each one before learning its result.

    `games` is the table clean.py writes. Returns (rated, model): `rated` is
    `games` in date order with four new columns holding what the model knew
    BEFORE each game (home_elo, away_elo, home_win_prob, predicted_margin), and
    `model` holds the ratings after the last game.
    """
    model = model or Elo()
    ordered = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    before: list[tuple[float, float, float, float]] = []
    season = None
    for game in ordered.itertuples(index=False):
        if season is not None and game.season != season:
            model.start_new_season()
        season = game.season

        home_win_prob, predicted_margin = model.predict(game.home_team_id, game.away_team_id)
        before.append((model.rating(game.home_team_id), model.rating(game.away_team_id),
                       home_win_prob, predicted_margin))
        model.update(game.home_team_id, game.away_team_id, game.home_pts, game.away_pts)

    columns = ["home_elo", "away_elo", "home_win_prob", "predicted_margin"]
    rated = pd.concat([ordered, pd.DataFrame(before, columns=columns, index=ordered.index)], axis=1)
    return rated, model
