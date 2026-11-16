"""Predict one NBA game with the Elo model.

Give the home team first, then the away team, as three-letter abbreviations.
Run it from the repo root (after fetch_games.py and clean.py):

    python basketball/src/predict.py BOS NYK
    python basketball/src/predict.py BOS NYK --neutral   # no home-court bonus
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src.clean import OUTPUT_FILE as GAMES_FILE  # noqa: E402
from src.models.elo import Elo, rate_games  # noqa: E402


@dataclass
class Prediction:
    home: str
    away: str
    home_elo: float
    away_elo: float
    home_win_prob: float
    predicted_margin: float   # positive = home team favored
    last_game_date: str
    neutral: bool


def load_games(games_file: Path = GAMES_FILE) -> pd.DataFrame:
    if not Path(games_file).exists():
        raise FileNotFoundError(f"{games_file} not found. Run basketball/src/clean.py first.")
    return pd.read_csv(games_file, dtype={"game_id": str})


def team_ids(games: pd.DataFrame) -> dict[str, int]:
    """Abbreviation -> team id, for every team in the games table."""
    home = games[["home", "home_team_id"]].set_axis(["team", "team_id"], axis=1)
    away = games[["away", "away_team_id"]].set_axis(["team", "team_id"], axis=1)
    teams = pd.concat([home, away]).drop_duplicates()
    return {str(team): int(team_id) for team, team_id in zip(teams["team"], teams["team_id"])}


def predict_game(games: pd.DataFrame, home: str, away: str, neutral: bool = False,
                 model: Elo | None = None) -> Prediction:
    """Rate every game in `games`, then predict home vs away with the final ratings."""
    home, away = home.upper(), away.upper()
    ids = team_ids(games)
    unknown = [team for team in (home, away) if team not in ids]
    if unknown:
        raise ValueError(f"Unknown team: {', '.join(unknown)}. Choose from: {', '.join(sorted(ids))}")
    if home == away:
        raise ValueError("Home and away must be different teams.")

    _, model = rate_games(games, model)
    home_win_prob, predicted_margin = model.predict(ids[home], ids[away], neutral)
    return Prediction(
        home=home,
        away=away,
        home_elo=model.rating(ids[home]),
        away_elo=model.rating(ids[away]),
        home_win_prob=home_win_prob,
        predicted_margin=predicted_margin,
        last_game_date=str(games["date"].max()),
        neutral=neutral,
    )


def describe(prediction: Prediction) -> str:
    p = prediction
    place = "neutral court" if p.neutral else f"{p.home} at home"
    if abs(p.predicted_margin) < 0.05:
        margin = "even"
    else:
        favorite = p.home if p.predicted_margin > 0 else p.away
        margin = f"{favorite} by {abs(p.predicted_margin):.1f} points"
    return "\n".join([
        f"{p.home} vs {p.away} ({place})",
        f"Elo ratings through {p.last_game_date}: {p.home} {p.home_elo:.0f}, {p.away} {p.away_elo:.0f}",
        f"{p.home} win probability: {p.home_win_prob:.1%}",
        f"{p.away} win probability: {1 - p.home_win_prob:.1%}",
        f"Predicted margin: {margin}",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict one NBA game with the Elo model.")
    parser.add_argument("home", help="home team abbreviation, e.g. BOS")
    parser.add_argument("away", help="away team abbreviation, e.g. NYK")
    parser.add_argument("--neutral", action="store_true", help="neutral court: no home-court bonus")
    parser.add_argument("--games", type=Path, default=GAMES_FILE, help="cleaned games table")
    args = parser.parse_args(argv)

    try:
        prediction = predict_game(load_games(args.games), args.home, args.away, args.neutral)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(describe(prediction))
    return 0


if __name__ == "__main__":
    sys.exit(main())
