"""v3 model: the efficiency ratings (v2) plus the situation each team is in.

Three things are known before tip-off and can tilt a game:

  * back-to-back: the team also played yesterday;
  * long rest: the team has had three or more days off;
  * travel: how far the team came from its previous game.

For each one the model looks at the home team's situation minus the away
team's, and learns how many points of margin it has been worth. It learns that
from how wrong the v2 model was in EARLIER seasons only: the effects used in a
season are fitted on the seasons before it, so a prediction never uses that
game's result or any later one. The first season has no earlier seasons, so its
predictions are plain v2.

Injuries ("key players out") are not included: there is no injury data in the project yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models import ratings

FEATURES = ["back_to_back", "long_rest", "travel_1000km"]
LONG_REST_DAYS = 3         # this many days since the last game counts as long rest
FRESH_START_DAYS = 30      # a gap this long is an off-season: no rest or travel effect
MIN_GAMES_TO_LEARN = 500   # with fewer earlier games than this, effects stay at zero
RIDGE = 10.0               # keeps the learned effects steady when there is little data

# Home arena of each team: (latitude, longitude).
ARENAS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975), "CHA": (35.225, -80.839),
    "CHI": (41.881, -87.674), "CLE": (41.497, -81.688), "DAL": (32.790, -96.810), "DEN": (39.749, -105.008),
    "DET": (42.341, -83.055), "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (33.945, -118.343), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051), "MIA": (25.781, -80.187),
    "MIL": (43.045, -87.917), "MIN": (44.979, -93.276), "NOP": (29.949, -90.082), "NYK": (40.751, -73.994),
    "OKC": (35.463, -97.515), "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438), "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}


def distance_km(place: str, other: str) -> float:
    """Straight-line distance between two teams' arenas; 0 if either is unknown."""
    if place not in ARENAS or other not in ARENAS:
        return 0.0
    (lat1, lon1), (lat2, lon2) = ARENAS[place], ARENAS[other]
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def add_situation(games: pd.DataFrame) -> pd.DataFrame:
    """Add each team's situation before every game, and the home-minus-away gaps the model uses.

    New columns: home_rest_days, away_rest_days, home_travel_km, away_travel_km
    (rest is empty and travel 0 for a team's first game or after an off-season),
    plus back_to_back, long_rest and travel_1000km (home team's value minus away team's).
    Everything comes from the schedule before the game, never from its result.
    """
    ordered = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    dates = pd.to_datetime(ordered["date"])
    last_game: dict[int, tuple[pd.Timestamp, str]] = {}   # team id -> (date, arena) of its previous game

    rows = []
    for position, game in enumerate(ordered.itertuples(index=False)):
        today, arena = dates[position], game.home
        row = {}
        for side, team_id in (("home", game.home_team_id), ("away", game.away_team_id)):
            rest, travel = float("nan"), 0.0
            if team_id in last_game:
                previous_date, previous_arena = last_game[team_id]
                days = (today - previous_date).days
                if days < FRESH_START_DAYS:
                    rest, travel = float(days), distance_km(previous_arena, arena)
            row[f"{side}_rest_days"], row[f"{side}_travel_km"] = rest, travel
            last_game[team_id] = (today, arena)
        rows.append(row)

    situation = pd.DataFrame(rows, index=ordered.index)
    table = pd.concat([ordered, situation], axis=1)
    for side in ("home", "away"):
        table[f"_{side}_b2b"] = (table[f"{side}_rest_days"] == 1).astype(float)
        table[f"_{side}_long"] = (table[f"{side}_rest_days"] >= LONG_REST_DAYS).astype(float)
    table["back_to_back"] = table["_home_b2b"] - table["_away_b2b"]
    table["long_rest"] = table["_home_long"] - table["_away_long"]
    table["travel_1000km"] = (table["home_travel_km"] - table["away_travel_km"]) / 1000
    return table.drop(columns=[column for column in table.columns if column.startswith("_")])


def learn_effects(situations: pd.DataFrame, misses: pd.Series) -> dict[str, float]:
    """How many points of margin each situation gap has been worth, from games already played.

    `misses` is the real margin minus the v2 model's predicted margin for the same games.
    """
    if len(situations) < MIN_GAMES_TO_LEARN:
        return dict.fromkeys(FEATURES, 0.0)
    x = situations[FEATURES].to_numpy(dtype=float)
    y = misses.to_numpy(dtype=float)
    effects = np.linalg.solve(x.T @ x + RIDGE * np.eye(len(FEATURES)), x.T @ y)
    return dict(zip(FEATURES, map(float, effects)))


@dataclass
class Situational:
    """The v2 ratings after the last game, plus the effects learned from every season played."""

    base: ratings.Efficiency = field(default_factory=ratings.Efficiency)
    effects: dict[str, float] = field(default_factory=lambda: dict.fromkeys(FEATURES, 0.0))

    def adjustment(self, back_to_back: float, long_rest: float, travel_1000km: float) -> float:
        """Points added to the home margin for a game with these home-minus-away gaps."""
        gaps = {"back_to_back": back_to_back, "long_rest": long_rest, "travel_1000km": travel_1000km}
        return sum(self.effects[name] * gaps[name] for name in FEATURES)


def rate_games(games: pd.DataFrame, model: Situational | None = None) -> tuple[pd.DataFrame, Situational]:
    """Replay games in date order, predicting each one before learning its result.

    `games` needs the same columns the v2 model needs. Returns (rated, model):
    `rated` has the v3 prediction made BEFORE each game (home_win_prob,
    predicted_margin), the v2 prediction it started from (base_margin) and the
    situation columns from add_situation().
    """
    model = model or Situational()
    base, model.base = ratings.rate_games(games, model.base)
    rated = add_situation(base)
    rated["base_margin"] = rated["predicted_margin"]
    miss = (rated["home_pts"] - rated["away_pts"]) - rated["base_margin"]

    adjustment = pd.Series(0.0, index=rated.index)
    seasons = sorted(rated["season"].unique())
    for season in seasons:
        earlier = rated["season"] < season
        effects = learn_effects(rated[earlier], miss[earlier])   # only seasons before this one
        this_season = rated["season"] == season
        adjustment[this_season] = sum(effects[name] * rated.loc[this_season, name] for name in FEATURES)

    rated["predicted_margin"] = rated["base_margin"] + adjustment
    rated["home_win_prob"] = norm.cdf(rated["predicted_margin"] / rated["margin_sd"])
    for side, sign in (("home", 0.5), ("away", -0.5)):
        rated[f"predicted_{side}_pts"] = rated[f"predicted_{side}_pts"] + sign * adjustment
    model.effects = learn_effects(rated, miss)   # for games after the last one: learned from everything played
    return rated, model
