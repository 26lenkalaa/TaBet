"""Predict one NBA game.

Give the home team first, then the away team, as three-letter abbreviations.
Run it from the repo root (after fetch_games.py, clean.py and db.py):

    python3 basketball/src/predict.py BOS NYK
    python3 basketball/src/predict.py BOS NYK --neutral     # no home-court bonus
    python3 basketball/src/predict.py BOS NYK --model elo   # the v1 Elo model

The default model is the efficiency ratings (v2), which beat Elo in the backtest.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the basketball/ folder

from src import db  # noqa: E402
from src.models import elo, ratings  # noqa: E402

MODELS = ["efficiency", "elo"]
DEFAULT_MODEL = "efficiency"


@dataclass
class Prediction:
    home: str
    away: str
    model: str
    home_rating: float        # Elo rating, or net rating (points per 100 possessions above average)
    away_rating: float
    home_win_prob: float
    predicted_margin: float   # positive = home team favored
    last_game_date: str
    neutral: bool
    predicted_home_pts: float | None = None   # efficiency model only
    predicted_away_pts: float | None = None


def load_games(db_file: Path = db.DB_FILE) -> pd.DataFrame:
    """Every game in the database, with each team's possessions."""
    return db.read_games(db.connect(db_file, must_exist=True), possessions=True)


def team_ids(games: pd.DataFrame) -> dict[str, int]:
    """Abbreviation -> team id, for every team in the games table."""
    home = games[["home", "home_team_id"]].set_axis(["team", "team_id"], axis=1)
    away = games[["away", "away_team_id"]].set_axis(["team", "team_id"], axis=1)
    teams = pd.concat([home, away]).drop_duplicates()
    return {str(team): int(team_id) for team, team_id in zip(teams["team"], teams["team_id"])}


def predict_game(games: pd.DataFrame, home: str, away: str, neutral: bool = False,
                 model: str = DEFAULT_MODEL) -> Prediction:
    """Replay every game in `games`, then predict home vs away with the final ratings."""
    home, away = home.upper(), away.upper()
    ids = team_ids(games)
    unknown = [team for team in (home, away) if team not in ids]
    if unknown:
        raise ValueError(f"Unknown team: {', '.join(unknown)}. Choose from: {', '.join(sorted(ids))}")
    if home == away:
        raise ValueError("Home and away must be different teams.")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Choose from: {', '.join(MODELS)}")

    common = dict(home=home, away=away, model=model, neutral=neutral, last_game_date=str(games["date"].max()))
    if model == "elo":
        _, rated = elo.rate_games(games)
        home_win_prob, predicted_margin = rated.predict(ids[home], ids[away], neutral)
        return Prediction(home_rating=rated.rating(ids[home]), away_rating=rated.rating(ids[away]),
                          home_win_prob=home_win_prob, predicted_margin=predicted_margin, **common)

    for column in ("home_possessions", "away_possessions"):
        if column not in games.columns:
            games = games.assign(**{column: float("nan")})
    _, rated = ratings.rate_games(games)
    home_win_prob, predicted_margin = rated.predict(ids[home], ids[away], neutral)
    home_pts, away_pts = rated.predict_score(ids[home], ids[away], neutral)
    return Prediction(home_rating=rated.net_rating(ids[home]), away_rating=rated.net_rating(ids[away]),
                      home_win_prob=home_win_prob, predicted_margin=predicted_margin,
                      predicted_home_pts=home_pts, predicted_away_pts=away_pts, **common)


def describe(prediction: Prediction) -> str:
    p = prediction
    place = "neutral court" if p.neutral else f"{p.home} at home"
    if abs(p.predicted_margin) < 0.05:
        margin = "even"
    else:
        favorite = p.home if p.predicted_margin > 0 else p.away
        margin = f"{favorite} by {abs(p.predicted_margin):.1f} points"

    lines = [f"{p.home} vs {p.away} ({place})"]
    if p.model == "elo":
        lines.append(f"Elo ratings through {p.last_game_date}: {p.home} {p.home_rating:.0f}, "
                     f"{p.away} {p.away_rating:.0f}")
    else:
        lines.append(f"Net ratings through {p.last_game_date}: {p.home} {p.home_rating:+.1f}, "
                     f"{p.away} {p.away_rating:+.1f} (points per 100 possessions against an average team)")
    lines += [
        f"{p.home} win probability: {p.home_win_prob:.1%}",
        f"{p.away} win probability: {1 - p.home_win_prob:.1%}",
        f"Predicted margin: {margin}",
    ]
    if p.predicted_home_pts is not None and p.predicted_away_pts is not None:
        lines.append(f"Predicted score: {p.home} {p.predicted_home_pts:.0f}, {p.away} {p.predicted_away_pts:.0f} "
                     f"(total {p.predicted_home_pts + p.predicted_away_pts:.0f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict one NBA game.")
    parser.add_argument("home", help="home team abbreviation, e.g. BOS")
    parser.add_argument("away", help="away team abbreviation, e.g. NYK")
    parser.add_argument("--neutral", action="store_true", help="neutral court: no home-court bonus")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                        help=f"model to use (default {DEFAULT_MODEL})")
    parser.add_argument("--db", type=Path, default=db.DB_FILE, help="database file built by db.py")
    args = parser.parse_args(argv)

    try:
        prediction = predict_game(load_games(args.db), args.home, args.away, args.neutral, args.model)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(describe(prediction))
    return 0


if __name__ == "__main__":
    sys.exit(main())
