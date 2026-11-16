"""Run from the Football folder: python cli.py --help."""

import argparse
import json
from datetime import date
from pathlib import Path

from backtest import backtest
from data import load_games, team_code
from model import forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="TaBet NFL forecasts")
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "res/games.csv")
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="Forecast a matchup")
    predict.add_argument("home")
    predict.add_argument("away")
    predict.add_argument("date", type=date.fromisoformat, metavar="YYYY-MM-DD")
    predict.add_argument("--season", required=True, type=int)
    predict.add_argument("--neutral", action="store_true")

    evaluate = commands.add_parser("backtest", help="Score a regular season")
    evaluate.add_argument("season", type=int)
    evaluate.add_argument("--output", type=Path, default=Path("results/backtest.csv"))

    args = parser.parse_args()
    try:
        games = load_games(args.data)
        if args.command == "predict":
            result = forecast(
                games, team_code(args.home), team_code(args.away),
                args.date, args.season, args.neutral,
            )
        else:
            result = backtest(games, args.season, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
