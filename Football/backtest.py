"""Compare Elo and the market on the same games."""

import csv
from math import isfinite
from pathlib import Path

from data import Game
from model import Elo, replay


def implied_probability(value: str) -> float:
    odds = float(value)
    if not isfinite(odds) or abs(odds) < 100:
        raise ValueError("Expected American moneyline odds.")
    return 100 / (odds + 100) if odds > 0 else -odds / (100 - odds)


def market_home_probability(game: Game) -> float | None:
    try:
        home = implied_probability(game.home_moneyline)
        away = implied_probability(game.away_moneyline)
    except ValueError:
        return None
    return home / (home + away)


def backtest(games: list[Game], season: int, output: Path) -> dict[str, object]:
    if not any(game.season < season for game in games):
        raise ValueError("Include at least one earlier season to warm up Elo.")
    rows: list[dict[str, object]] = []
    model_losses: list[float] = []
    market_losses: list[float] = []
    ties = missing_odds = 0

    for game, home in replay([game for game in games if game.season <= season], Elo()):
        if game.season != season:
            continue
        actual = game.home_result
        market = market_home_probability(game)
        if actual == 0.5:
            status = "tie"
            ties += 1
        elif market is None:
            status = "missing_odds"
            missing_odds += 1
        else:
            status = "scored"
            model_losses.append((home - actual) ** 2)
            market_losses.append((market - actual) ** 2)
        rows.append({
            "game_id": game.game_id,
            "game_date": game.game_date.isoformat(),
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_score": game.home_score,
            "away_score": game.away_score,
            "p_home": home,
            "p_away": 1 - home,
            "market_p_home": market,
            "status": status,
        })

    if not rows:
        raise ValueError(f"No completed games for season {season}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count = len(model_losses)
    return {
        "season": season,
        "games": len(rows),
        "scored_games": count,
        "ties": ties,
        "missing_odds": missing_odds,
        "elo_brier": sum(model_losses) / count if count else None,
        "market_brier": sum(market_losses) / count if count else None,
        "output": str(output),
    }
