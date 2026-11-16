"""Results-only Elo. Odds stay outside the model."""

from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from typing import Iterator

from data import Game


START_RATING = 1500.0
HOME_ADVANTAGE = 55.0
UPDATE_RATE = 20.0
SEASON_RETENTION = 0.75


@dataclass
class Elo:
    ratings: dict[str, float] = field(default_factory=dict)
    season: int | None = None

    def start_season(self, season: int) -> None:
        if self.season is not None and season < self.season:
            raise ValueError("Seasons must be processed in order.")
        if self.season is not None and season != self.season:
            retention = SEASON_RETENTION ** (season - self.season)
            for team, rating in self.ratings.items():
                self.ratings[team] = START_RATING + retention * (rating - START_RATING)
        self.season = season

    def home_probability(self, home: str, away: str, neutral: bool = False) -> float:
        gap = self.ratings.get(home, START_RATING) - self.ratings.get(away, START_RATING)
        if not neutral:
            gap += HOME_ADVANTAGE
        return 1.0 / (1.0 + 10.0 ** (-gap / 400.0))

    def update(self, game: Game, home_probability: float) -> None:
        change = UPDATE_RATE * (game.home_result - home_probability)
        self.ratings[game.home_team] = self.ratings.get(game.home_team, START_RATING) + change
        self.ratings[game.away_team] = self.ratings.get(game.away_team, START_RATING) - change


def replay(games: list[Game], model: Elo) -> Iterator[tuple[Game, float]]:
    ordered = sorted(games, key=lambda game: (game.game_date, game.game_id))
    for _, day in groupby(ordered, key=lambda game: game.game_date):
        batch = list(day)
        model.start_season(batch[0].season)
        forecasts = [
            (game, model.home_probability(game.home_team, game.away_team, game.neutral))
            for game in batch
        ]
        # Forecast the whole day before using any of its results.
        yield from forecasts
        for game, probability in forecasts:
            model.update(game, probability)


def forecast(
    games: list[Game], home: str, away: str, game_date: date, season: int,
    neutral: bool = False,
) -> dict[str, object]:
    if home == away:
        raise ValueError("Choose two different teams.")
    # NFL seasons can finish in January of the next calendar year.
    expected_season = game_date.year - (game_date.month < 8)
    if season != expected_season:
        raise ValueError(f"Use season {expected_season} for this date.")
    history = [game for game in games if game.game_date < game_date]
    model = Elo()
    for _ in replay(history, model):
        pass
    if model.season is None:
        raise ValueError("No games before the requested date.")
    for team in (home, away):
        if team not in model.ratings:
            raise ValueError(f"No earlier games for {team}. Use its nflverse team code.")
    model.start_season(season)
    probability = model.home_probability(home, away, neutral)
    return {
        "sport": "football",
        "league": "NFL",
        "model": "elo",
        "game_date": game_date.isoformat(),
        "season": season,
        "home_team": home,
        "away_team": away,
        "neutral": neutral,
        "probability_basis": "conditional_on_no_tie",
        "outcome_probabilities": {"home": probability, "away": 1 - probability},
        "history_through": max(game.game_date for game in history).isoformat(),
    }
